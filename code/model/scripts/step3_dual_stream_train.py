"""
第三步：双流模型训练脚本。

训练策略：
  Phase 1（第一步已完成）：验证官方推理
  Phase 2（第二步已完成）：准备轻量文本 embedding
  Phase 3（本脚本）：冻结视频分支，只训练视觉桥接 + 动作去噪器

用法示例：
  cd d:/ZTE
  python scripts/step3_dual_stream_train.py \
      --ckpt_path /path/to/Cosmos-Predict2.5-2B \
      --data_root AIGC/release/train \
      --embedding_dir outputs/text_embeddings \
      --output_dir outputs/step3_dual_stream \
      --batch_size 4 \
      --max_steps 10000 \
      --learning_rate 1e-4
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger

# 确保双流模块在路径中
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
COSMOS_ROOT = ROOT / "cosmos-predict2.5-1.5.0"
sys.path.insert(0, str(COSMOS_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="双流模型训练（第三步）")
    # 数据相关
    parser.add_argument("--data_root", type=str, default=str(ROOT / "autodl-tmp/AIGC/release/train"))
    parser.add_argument("--embedding_dir", type=str, default=str(ROOT / "outputs/text_embeddings"))
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "outputs/step3_dual_stream"))

    # 模型相关
    parser.add_argument("--ckpt_path", type=str, default="nvidia/Cosmos-Predict2.5-2B")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--config_file", type=str,
                        default="cosmos_predict2/_src/predict2/configs/video2world/config.py")

    # 双流架构超参数
    parser.add_argument("--action_dim", type=int, default=26, help="动作维度")
    parser.add_argument("--visual_dim", type=int, default=256, help="视觉条件维度（可任意设定）")
    parser.add_argument("--text_dim", type=int, default=896, help="文本编码维度（Qwen2.5-0.5B=896）")
    parser.add_argument("--action_steps", type=int, default=50, help="动作预测步数")
    parser.add_argument("--bridge_type", type=str, default="simple", choices=["simple", "attention"])
    parser.add_argument("--pool_out_size", type=str, default="1,2,4",
                        help="3D池化输出尺寸 T,H,W（用逗号分隔）")

    # 数据超参数
    parser.add_argument("--num_cond_frames", type=int, default=16, help="视频条件帧数")
    parser.add_argument("--image_size", type=str, default="192,320", help="H,W")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader \u5b50\u8fdb\u7a0b\u6570\uff080=\u4e3b\u8fdb\u7a0b\u52a0\u8f7d\uff0c\u907f\u514d fork \u5d29\u6e83\uff1b\u7a33\u5b9a\u540e\u53ef\u8c03\u81f3 4\uff09")

    # 训练超参数
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=10,
                        help="\u6bcf N \u6b65\u6253\u5370\u4e00\u6b21 loss\uff08\u9ed8\u8ba4 10\uff0c\u5c0f\u4e8e max_steps \u5373\u53ef\u770b\u5230\u8f93\u51fa\uff09")
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--use_amp", action="store_true", default=True, help="使用混合精度训练")

    # 视频分支策略
    parser.add_argument("--freeze_video", action="store_true", default=True,
                        help="冻结视频分支（推荐在初始阶段冻结）")
    parser.add_argument("--video_loss_weight", type=float, default=0.0,
                        help="视频分支损失权重（0=不计算视频损失）")

    # 是否跳过视频模型加载（用于快速测试动作分支）
    parser.add_argument("--skip_video_model", action="store_true", default=False,
                        help="跳过视频模型加载，只测试动作分支（开发调试用）")

    # mock 数据模式：完全跳过真实数据集，用随机张量验证训练流程
    parser.add_argument("--mock_data", action="store_true", default=False,
                        help="用随机张量代替真实数据，完全不需要数据集或 ffmpeg（验证代码流程用）")

    return parser.parse_args()


def build_mock_dataloader(
    batch_size: int,
    action_dim: int,
    action_steps: int,
    text_dim: int,
    num_cond_frames: int,
    image_size: tuple,
    max_steps: int,
) -> DataLoader:
    """
    结构化的 Mock DataLoader，完全不需要真实数据集。
    每个 batch 返回和真实数据相同 shape 的随机张量。
    """
    from torch.utils.data import IterableDataset

    class MockDataset(IterableDataset):
        def __init__(self, total_batches):
            self.total_batches = total_batches

        def __iter__(self):
            H, W = image_size
            for _ in range(self.total_batches):
                yield {
                    "frames":         torch.randn(batch_size, 3, num_cond_frames, H, W),
                    "text":           ["mock instruction"] * batch_size,
                    "actions":        torch.randn(batch_size, action_steps, action_dim),
                    "text_embedding": torch.randn(batch_size, 128, text_dim),
                    "sample_name":    ["mock"] * batch_size,
                }

    dataset = MockDataset(total_batches=max_steps)
    return DataLoader(dataset, batch_size=None)  # batch_size=None 表示数据集已经是批形式


def build_optimizer_and_scheduler(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    max_steps: int,
    warmup_steps: int,
):
    """构建优化器和学习率调度器。"""
    # 只优化非冻结参数
    params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Trainable parameters: {sum(p.numel() for p in params):,}")

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # cosine LR + warmup
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        import math
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def train_step_action_only(
    batch: dict,
    visual_bridge: nn.Module,
    action_denoiser: nn.Module,
    video_model: Optional[nn.Module],
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    sigma_data: float = 0.5,
    device: str = "cuda",
) -> dict:
    """
    单个训练步骤（仅动作分支，视频分支冻结/跳过）。
    
    这里用一个关键的简化：在初始训练时，先用视频帧的 VAE 编码
    直接作为 z0（而不是运行完整的扩散去噪），避免在每个训练步都运行扩散采样。
    
    正式双流联合训练时，z0 应来自视频扩散分支的去噪预测。
    """
    actions_gt = batch["actions"].to(device)  # (B, T_act, 26)
    frames = batch["frames"].to(device)        # (B, 3, T_cond, H, W)
    B = actions_gt.shape[0]

    # 方案A（简化）：用 VAE 编码帧作为 z0 近似
    if video_model is not None:
        with torch.no_grad():
            # 转为官方期望的格式 (B, C, T, H, W) uint8
            frames_uint8 = (frames * 255).clamp(0, 255).to(torch.uint8)
            # 使用官方 tokenizer 编码
            try:
                z0 = video_model.tokenizer.encode(frames_uint8.bfloat16())
                z0 = z0.float()
            except Exception as e:
                logger.warning(f"Tokenizer encode failed: {e}, using random z0")
                z0 = torch.randn(B, 16, 5, 24, 40, device=device)
    else:
        # 纯 mock 模式：随机生成 z0
        z0 = torch.randn(B, 16, 5, 24, 40, device=device)

    # 获取文本条件
    text_cond = batch.get("text_embedding")
    if text_cond is not None:
        text_cond = text_cond.to(device)  # (B, seq_len, text_dim)

    # 采样噪声水平
    sigma = sample_sigma(B, sigma_min, sigma_max).to(device)

    # 加噪
    eps = torch.randn_like(actions_gt)
    sigma_exp = sigma.view(B, 1, 1)
    x_t = actions_gt + sigma_exp * eps

    # EDM 预条件化
    c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
    c_out = sigma * sigma_data / (sigma**2 + sigma_data**2).sqrt()
    c_in = 1 / (sigma**2 + sigma_data**2).sqrt()
    c_skip = c_skip.view(B, 1, 1)
    c_out = c_out.view(B, 1, 1)
    c_in = c_in.view(B, 1, 1)

    x_t_scaled = x_t * c_in

    # 视觉桥接
    visual_cond = visual_bridge(z0.detach())  # (B, visual_dim)

    # 动作去噪
    net_out = action_denoiser(
        x_t=x_t_scaled,
        sigma=sigma,
        visual_cond=visual_cond,
        text_cond=text_cond,
    )

    x0_pred = c_skip * x_t + c_out * net_out  # (B, T_act, action_dim)

    # EDM 损失
    loss_weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data)**2
    loss_weight = loss_weight.view(B, 1, 1)
    loss = (loss_weight * (x0_pred - actions_gt)**2).mean()

    return {
        "loss": loss,
        "sigma_mean": sigma.mean().item(),
    }


def sample_sigma(batch_size: int, sigma_min: float, sigma_max: float) -> torch.Tensor:
    """从 log-normal 分布采样噪声水平。"""
    import math
    log_sigma = torch.randn(batch_size) * 1.2
    return log_sigma.exp().clamp(sigma_min, sigma_max)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置
    with open(output_dir / "train_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # ---- 加载数据集 ----
    image_size = tuple(int(x) for x in args.image_size.split(","))

    if args.mock_data:
        # Mock 模式：不需要真实数据集和 ffmpeg
        logger.info("🧹 Mock data mode: using random tensors (no real dataset needed)")
        dataloader = build_mock_dataloader(
            batch_size=args.batch_size,
            action_dim=args.action_dim,
            action_steps=args.action_steps,
            text_dim=args.text_dim,
            num_cond_frames=args.num_cond_frames,
            image_size=image_size,
            max_steps=args.max_steps,
        )
        embedding_dir = None  # mock 模式下 embedding 已内置在 batch 中
        logger.info(f"Mock dataloader ready: {args.max_steps} batches of size {args.batch_size}")
    else:
        # 真实数据集
        from dual_stream.dataset import get_aigc_dataloader
        embedding_dir = args.embedding_dir if Path(args.embedding_dir).exists() else None
        dataloader = get_aigc_dataloader(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_cond_frames=args.num_cond_frames,
            num_action_steps=args.action_steps,
            image_size=image_size,
            embedding_dir=embedding_dir,
            shuffle=True,
        )
        logger.info(f"Dataset: {len(dataloader.dataset)} samples, {len(dataloader)} batches/epoch")

    # ---- 加载视频模型（可选）----
    video_model = None
    if not args.skip_video_model:
        try:
            os.chdir(str(COSMOS_ROOT))
            from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint
            from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey

            pre_trained_key = ModelKey(post_trained=False)
            checkpoint_config = MODEL_CHECKPOINTS[pre_trained_key]
            experiment_name = args.experiment or checkpoint_config.experiment
            ckpt_path = args.ckpt_path if args.ckpt_path != "nvidia/Cosmos-Predict2.5-2B" else checkpoint_config.s3.uri

            logger.info(f"Loading video model from {ckpt_path}...")
            video_model, _ = load_model_from_checkpoint(
                experiment_name=experiment_name,
                s3_checkpoint_dir=ckpt_path,
                config_file=args.config_file,
                load_ema_to_reg=True,
                experiment_opts=["~data_train"],
                to_device=device,
            )
            video_model.eval()
            for p in video_model.parameters():
                p.requires_grad_(False)
            logger.success("Video model loaded and frozen.")
        except Exception as e:
            logger.warning(f"Could not load video model: {e}")
            logger.warning("Running in mock mode (random z0).")
    else:
        logger.info("Skipping video model load (--skip_video_model).")

    os.chdir(str(ROOT))

    # ---- 构建动作分支模块 ----
    pool_out_size = tuple(int(x) for x in args.pool_out_size.split(","))

    from dual_stream.visual_bridge import build_visual_bridge
    from dual_stream.action_denoiser import ActionDenoiser1DUNet

    visual_bridge = build_visual_bridge(
        bridge_type=args.bridge_type,
        in_channels=16,  # Cosmos VAE 隐空间通道数
        visual_dim=args.visual_dim,
        pool_out_size=pool_out_size,
    ).to(device)

    action_denoiser = ActionDenoiser1DUNet(
        action_dim=args.action_dim,
        visual_dim=args.visual_dim,
        text_dim=args.text_dim,
        base_channels=128,
        channel_mults=(1, 2, 4),
        use_text_cond=(embedding_dir is not None or args.mock_data),  # mock_data \u6a21\u5f0f\u4e0b\u603b\u6709 text_embedding
    ).to(device)

    logger.info(f"VisualBridge params: {sum(p.numel() for p in visual_bridge.parameters()):,}")
    logger.info(f"ActionDenoiser params: {sum(p.numel() for p in action_denoiser.parameters()):,}")

    # 合并可训练参数
    trainable_modules = nn.ModuleList([visual_bridge, action_denoiser])
    optimizer, scheduler = build_optimizer_and_scheduler(
        trainable_modules, args.learning_rate, args.weight_decay,
        args.max_steps, args.warmup_steps,
    )

    scaler = torch.amp.GradScaler("cuda") if args.use_amp and device == "cuda" else None

    # ---- 训练循环 ----
    step = 0
    epoch = 0
    loss_history = []

    while step < args.max_steps:
        epoch += 1
        for batch in dataloader:
            if step >= args.max_steps:
                break

            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    result = train_step_action_only(
                        batch, visual_bridge, action_denoiser, video_model, device=device
                    )
                loss = result["loss"]
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_modules.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                result = train_step_action_only(
                    batch, visual_bridge, action_denoiser, video_model, device=device
                )
                loss = result["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_modules.parameters(), args.grad_clip)
                optimizer.step()

            scheduler.step()
            step += 1

            # 日志
            if step % args.log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                loss_val = loss.item()
                loss_history.append({"step": step, "loss": loss_val, "lr": lr})
                logger.info(
                    f"Step {step}/{args.max_steps} | "
                    f"Loss: {loss_val:.4f} | "
                    f"σ_mean: {result['sigma_mean']:.3f} | "
                    f"LR: {lr:.2e}"
                )

            # 保存 checkpoint
            if step % args.save_interval == 0 or step == args.max_steps:
                ckpt = {
                    "step": step,
                    "visual_bridge_state": visual_bridge.state_dict(),
                    "action_denoiser_state": action_denoiser.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "args": vars(args),
                }
                ckpt_path = output_dir / f"checkpoint_step{step:06d}.pt"
                torch.save(ckpt, ckpt_path)
                logger.success(f"Saved checkpoint to {ckpt_path}")

    # 保存最终模型
    final_ckpt = {
        "step": step,
        "visual_bridge_state": visual_bridge.state_dict(),
        "action_denoiser_state": action_denoiser.state_dict(),
        "args": vars(args),
    }
    torch.save(final_ckpt, output_dir / "final_model.pt")

    # 保存损失曲线
    with open(output_dir / "loss_history.json", "w") as f:
        json.dump(loss_history, f, indent=2)

    logger.success(
        f"Training complete! "
        + (f"Final loss: {loss_history[-1]['loss']:.4f}" if loss_history else f"(run {args.max_steps} steps, loss not logged — increase --max_steps or reduce --log_interval)")
    )


if __name__ == "__main__":
    main()
