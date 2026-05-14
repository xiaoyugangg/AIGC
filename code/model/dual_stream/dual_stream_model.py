"""
双流（Video-Action）完整模型。

整合：
  1. 官方 Cosmos-Predict2.5-2B Video2World 模型（视频生成分支）
  2. 视觉桥接模块（VisualBridge）
  3. 1D U-Net 动作去噪器（ActionDenoiser1DUNet）
  4. 轻量文本编码器接口

推理流程：
  frames → [Video Diffusion] → z0 → [VisualBridge] → visual_cond
                                                          ↓
  actions_noisy + sigma → [ActionDenoiser] ← text_cond
                              ↓
                        actions_pred (x0)

训练流程（扩散目标）：
  Video Loss: 标准 EDM/Flow Matching 损失（来自官方 Video2World）
  Action Loss: MSE(x0_pred, x0_gt) 或 v-prediction

注意事项：
  - 视频分支使用官方预训练权重，根据策略选择是否冻结
  - 动作分支从随机初始化开始训练
  - 视觉桥接模块也从随机初始化开始
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 确保官方源码在路径中
COSMOS_ROOT = Path(__file__).parent.parent.parent / "cosmos-predict2.5-1.5.0"
if str(COSMOS_ROOT) not in sys.path:
    sys.path.insert(0, str(COSMOS_ROOT))

from .visual_bridge import build_visual_bridge, VisualBridge
from .action_denoiser import ActionDenoiser1DUNet


# ============================================================
# 扩散辅助函数（参考 Cosmos 官方 EDM 格式）
# ============================================================

def add_noise(x0: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    EDM 风格加噪：x_t = x0 + sigma * ε
    sigma: (B,) → broadcast 到 (B, T, D)
    """
    eps = torch.randn_like(x0)
    sigma_expanded = sigma.view(-1, 1, 1)  # (B, 1, 1) broadcast
    return x0 + sigma_expanded * eps, eps


def sigma_to_precond(sigma: torch.Tensor, sigma_data: float = 0.5):
    """
    EDM 预条件化系数（参考 Karras et al. 2022）：
    c_skip, c_out, c_in, c_noise
    """
    c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
    c_out = sigma * sigma_data / (sigma**2 + sigma_data**2).sqrt()
    c_in = 1 / (sigma**2 + sigma_data**2).sqrt()
    c_noise = sigma.log() / 4
    return c_skip, c_out, c_in, c_noise


def sample_sigma(batch_size: int, sigma_min: float = 0.002, sigma_max: float = 80.0) -> torch.Tensor:
    """
    从 EDM 的 log-normal 分布采样噪声水平。
    P(sigma) ∝ LogNormal(0, 1.2)，截断到 [sigma_min, sigma_max]
    """
    log_sigma = torch.randn(batch_size) * 1.2
    sigma = log_sigma.exp().clamp(sigma_min, sigma_max)
    return sigma


# ============================================================
# 双流模型核心
# ============================================================

class DualStreamModel(nn.Module):
    """
    双流视频-动作生成模型。
    
    Args:
        video_model: 已加载的官方 Video2World 模型
        action_dim: 动作维度（AIGC 数据集=26）
        visual_dim: 视觉桥接输出维度（论文中=132）
        text_dim: 文本编码维度（T5-base=768）
        pool_out_size: 3D 池化输出尺寸 (T', H', W')
        action_steps: 动作预测步数（=50）
        freeze_video_model: 是否冻结视频生成分支（推荐初始冻结）
        bridge_type: 视觉桥接类型 ("simple" 或 "attention")
        use_text_cond: 是否在动作分支使用文本条件
    """

    def __init__(
        self,
        video_model: nn.Module,
        action_dim: int = 26,
        visual_dim: int = 256,          # 可任意设定，不必是 132
        text_dim: int = 896,            # Qwen2.5-0.5B 隐藏层维度
        pool_out_size: Tuple[int, int, int] = (1, 2, 4),
        action_steps: int = 50,
        freeze_video_model: bool = True,
        bridge_type: str = "simple",
        use_text_cond: bool = True,
        action_denoiser_channels: int = 128,
        action_denoiser_mults: Tuple[int, ...] = (1, 2, 4),
        sigma_data: float = 0.5,
    ):
        super().__init__()

        self.video_model = video_model
        self.action_dim = action_dim
        self.visual_dim = visual_dim
        self.action_steps = action_steps
        self.sigma_data = sigma_data

        # 冻结视频分支
        if freeze_video_model:
            for p in self.video_model.parameters():
                p.requires_grad_(False)
            self.video_model.eval()

        # 视觉桥接模块（从 VAE 隐空间通道数获取）
        # Cosmos VAE 隐空间通道数通常为 16
        vae_channels = getattr(video_model, "latent_channels", 16)
        self.visual_bridge = build_visual_bridge(
            bridge_type=bridge_type,
            in_channels=vae_channels,
            visual_dim=visual_dim,
            pool_out_size=pool_out_size,
        )

        # 动作去噪器
        self.action_denoiser = ActionDenoiser1DUNet(
            action_dim=action_dim,
            visual_dim=visual_dim,
            text_dim=text_dim,
            base_channels=action_denoiser_channels,
            channel_mults=action_denoiser_mults,
            use_text_cond=use_text_cond,
        )

    def get_z0_from_video_model(
        self,
        data_batch: Dict[str, Any],
        noise_x: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        拦截视频扩散分支的 z0 预测。
        
        通过调用官方 denoise() 函数，并从 DenoisePrediction 中取出 .x0。
        
        Args:
            data_batch: 官方格式的数据批次（含 video, text embeddings 等）
            noise_x: (B, C, T_lat, H_lat, W_lat) — 加噪的视频隐变量
            sigma: (B,) — 噪声水平
            
        Returns:
            z0: (B, C, T_lat, H_lat, W_lat) — 预测的干净视频隐变量
        """
        # 获取条件
        condition, _ = self.video_model.conditioner.get_condition_uncondition(data_batch)
        
        # 调用官方 denoise
        denoise_pred = self.video_model.denoise(noise_x, sigma, condition)
        
        return denoise_pred.x0  # (B, C, T_lat, H_lat, W_lat)

    def encode_video_to_latent(self, frames: torch.Tensor) -> torch.Tensor:
        """
        将像素帧编码为 VAE 隐变量。
        
        Args:
            frames: (B, C, T, H, W) uint8 or float [0, 255]
            
        Returns:
            latent: (B, C_lat, T_lat, H_lat, W_lat)
        """
        with torch.no_grad():
            latent = self.video_model.tokenizer.encode(frames.to(torch.bfloat16))
        return latent.float()

    def forward_video_only(
        self,
        data_batch: Dict[str, Any],
        guidance: float = 7.0,
        num_steps: int = 35,
        seed: int = 42,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        仅运行视频分支推理，同时记录最终去噪步的 z0（用于视觉桥接）。
        
        Returns:
            video: 生成的像素视频
            z0: 最后一步的干净隐变量（用于传递给动作分支）
        """
        # 存储 z0 的钩子
        captured_z0 = {}

        original_denoise = self.video_model.denoise

        def hooked_denoise(xt, sigma, condition):
            result = original_denoise(xt, sigma, condition)
            captured_z0["z0"] = result.x0.detach()
            return result

        # 挂载钩子
        self.video_model.denoise = hooked_denoise

        # 运行推理（调用官方 generate_samples_from_batch）
        sample = self.video_model.generate_samples_from_batch(
            data_batch,
            n_sample=1,
            guidance=guidance,
            seed=seed,
            is_negative_prompt=True,
            num_steps=num_steps,
        )

        # 恢复原始方法
        self.video_model.denoise = original_denoise

        # 解码视频
        video = self.video_model.decode(sample)
        z0 = captured_z0.get("z0", None)

        return video, z0

    def forward_action_from_z0(
        self,
        z0: torch.Tensor,
        sigma_action: torch.Tensor,
        x_t_action: Optional[torch.Tensor] = None,
        text_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        从 z0 生成动作预测（单步去噪）。
        
        Args:
            z0: (B, C, T_lat, H_lat, W_lat) — 视频分支的干净隐变量
            sigma_action: (B,) — 动作扩散的噪声水平
            x_t_action: (B, T_act, action_dim) — 带噪动作（推理时从噪声开始）
            text_cond: (B, seq_len, text_dim) — 文本条件
            
        Returns:
            x0_action: (B, T_act, action_dim) — 预测的干净动作
        """
        B = z0.shape[0]

        # 通过视觉桥接提取视觉条件
        visual_cond = self.visual_bridge(z0)  # (B, visual_dim)

        # 如果没有提供带噪动作，从纯噪声开始
        if x_t_action is None:
            x_t_action = torch.randn(B, self.action_steps, self.action_dim, device=z0.device)

        # 动作去噪预测
        x0_action = self.action_denoiser(
            x_t=x_t_action,
            sigma=sigma_action,
            visual_cond=visual_cond,
            text_cond=text_cond,
        )

        return x0_action

    def compute_action_loss(
        self,
        actions_gt: torch.Tensor,
        z0_video: torch.Tensor,
        text_cond: Optional[torch.Tensor] = None,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
    ) -> Dict[str, torch.Tensor]:
        """
        计算动作分支的扩散训练损失。
        
        使用 EDM 损失：L = E_{σ, ε}[λ(σ) * ||x0_pred - x0_gt||²]
        
        Args:
            actions_gt: (B, T_act, action_dim) — 真实动作序列
            z0_video: (B, C, T_lat, H_lat, W_lat) — 视频分支 z0（已停止梯度）
            text_cond: (B, seq_len, text_dim) — 文本条件
            
        Returns:
            loss_dict: {'action_loss': tensor, 'action_loss_unweighted': tensor}
        """
        B = actions_gt.shape[0]

        # 采样噪声水平
        sigma = sample_sigma(B, sigma_min, sigma_max).to(actions_gt.device)

        # 加噪
        x_t, eps = add_noise(actions_gt, sigma)

        # EDM 预条件化
        c_skip, c_out, c_in, _ = sigma_to_precond(sigma, self.sigma_data)
        c_skip = c_skip.view(B, 1, 1)
        c_out = c_out.view(B, 1, 1)
        c_in = c_in.view(B, 1, 1)

        x_t_scaled = x_t * c_in  # (B, T_act, action_dim)

        # 视觉条件（停止梯度，不让 video 分支的梯度流入）
        z0_video_sg = z0_video.detach()
        visual_cond = self.visual_bridge(z0_video_sg)  # (B, visual_dim)

        # 动作去噪预测
        net_out = self.action_denoiser(
            x_t=x_t_scaled,
            sigma=sigma,
            visual_cond=visual_cond,
            text_cond=text_cond,
        )

        # 重建 x0 预测
        x0_pred = c_skip * x_t + c_out * net_out

        # EDM 损失权重：λ(σ) = (σ² + σ_data²) / (σ * σ_data)²
        loss_weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data)**2
        loss_weight = loss_weight.view(B, 1, 1)

        loss_unweighted = F.mse_loss(x0_pred, actions_gt, reduction="none")  # (B, T, D)
        loss = (loss_weight * loss_unweighted).mean()

        return {
            "action_loss": loss,
            "action_loss_unweighted": loss_unweighted.mean(),
        }


# ============================================================
# 模型加载工具
# ============================================================

def load_dual_stream_model(
    ckpt_path: str,
    experiment_name: str,
    config_file: str = "cosmos_predict2/_src/predict2/configs/video2world/config.py",
    action_dim: int = 26,
    visual_dim: int = 132,
    text_dim: int = 768,
    freeze_video: bool = True,
    device: str = "cuda",
) -> DualStreamModel:
    """
    加载完整的双流模型（视频分支 + 动作分支）。
    
    Args:
        ckpt_path: Cosmos 预训练权重路径
        experiment_name: 实验名称
        ...
        
    Returns:
        model: DualStreamModel
    """
    import os
    os.chdir(str(COSMOS_ROOT))

    from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint

    # 加载官方视频模型
    video_model, config = load_model_from_checkpoint(
        experiment_name=experiment_name,
        s3_checkpoint_dir=ckpt_path,
        config_file=config_file,
        load_ema_to_reg=True,
        experiment_opts=["~data_train"],
        to_device=device,
    )

    # 构建双流模型
    dual_model = DualStreamModel(
        video_model=video_model,
        action_dim=action_dim,
        visual_dim=visual_dim,
        text_dim=text_dim,
        freeze_video_model=freeze_video,
    )

    return dual_model


if __name__ == "__main__":
    """简单的形状验证测试（不需要真实 checkpoint）。"""
    print("=== DualStreamModel Component Test ===")

    from .visual_bridge import VisualBridge
    from .action_denoiser import ActionDenoiser1DUNet

    # 模拟一个最小化的"视频模型"占位符
    class MockVideoModel(nn.Module):
        latent_channels = 16
        def forward(self, *args, **kwargs): pass

    mock_video = MockVideoModel()

    # 只测试新增的两个模块
    bridge = VisualBridge(in_channels=16, pool_out_size=(1, 2, 4), visual_dim=132)
    denoiser = ActionDenoiser1DUNet(
        action_dim=26, visual_dim=132, text_dim=768,
        base_channels=64, channel_mults=(1, 2),
    )

    B, C, T_lat, H_lat, W_lat = 2, 16, 20, 24, 40
    T_act, D_act = 50, 26

    z0 = torch.randn(B, C, T_lat, H_lat, W_lat)
    visual = bridge(z0)
    print(f"VisualBridge output: {visual.shape}")

    x_t = torch.randn(B, T_act, D_act)
    sigma = torch.ones(B)
    text = torch.randn(B, 77, 768)
    x0 = denoiser(x_t, sigma, visual, text)
    print(f"ActionDenoiser output: {x0.shape}")

    assert visual.shape == (B, 132)
    assert x0.shape == (B, T_act, D_act)
    print("✅ All component shapes verified!")
