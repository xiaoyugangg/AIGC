"""
第四步：双流模型推理脚本。
读取 step3 训练好的权重，输入视频和文本，生成动作序列。

用法示例：
  cd d:/ZTE
  python scripts/step4_dual_stream_inference.py \
      --ckpt_path outputs/step3_dual_stream/final_model.pt \
      --video_path AIGC/release/test/1_1/video.mp4 \
      --instruction_file AIGC/release/test/1_1/instruction.txt \
      --output_dir outputs/step4_inference
"""

import argparse
import os
import sys
from pathlib import Path
import json

import torch
import numpy as np
import cv2
from loguru import logger

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
COSMOS_ROOT = ROOT / "cosmos-predict2.5-1.5.0"
sys.path.insert(0, str(COSMOS_ROOT))

from dual_stream.visual_bridge import build_visual_bridge
from dual_stream.action_denoiser import ActionDenoiser1DUNet
from dual_stream.dual_stream_model import sigma_to_precond

def load_video_frames(video_path: str, max_frames: int = 16, target_size=(192, 320)):
    """读取视频并返回 (1, 3, T, H, W) 张量"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (target_size[1], target_size[0]))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    
    # 填充到所需帧数
    while len(frames) < max_frames:
        frames.append(frames[-1])
        
    frames_np = np.stack(frames) # (T, H, W, C)
    frames_t = torch.from_numpy(frames_np).permute(3, 0, 1, 2) # (C, T, H, W)
    return frames_t.unsqueeze(0) # (1, C, T, H, W)


def edm_euler_sampler(
    denoiser: torch.nn.Module,
    visual_cond: torch.Tensor,
    text_cond: torch.Tensor,
    action_shape: tuple,
    device: str,
    num_steps: int = 50,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    sigma_data: float = 0.5,
):
    """EDM 采样器 (Euler integration)"""
    B = action_shape[0]
    
    # Karras 时间步表
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]).float()
    
    # 从纯噪声开始
    x = torch.randn(action_shape, device=device) * sigma_max
    
    for i in range(num_steps):
        t_cur = t_steps[i]
        t_next = t_steps[i + 1]
        
        sigma = torch.full((B,), t_cur, device=device)
        c_skip, c_out, c_in, _ = sigma_to_precond(sigma, sigma_data)
        c_skip = c_skip.view(B, 1, 1)
        c_out = c_out.view(B, 1, 1)
        c_in = c_in.view(B, 1, 1)
        
        x_scaled = x * c_in
        net_out = denoiser(x_t=x_scaled, sigma=sigma, visual_cond=visual_cond, text_cond=text_cond)
        x0_pred = c_skip * x + c_out * net_out
        
        d_x = (x - x0_pred) / t_cur
        x = x + (t_next - t_cur) * d_x
        
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", default=str(ROOT / "outputs/step3_dual_stream/final_model.pt"))
    parser.add_argument("--video_path", default=str(ROOT / "AIGC/release/test/1_1/video.mp4"))
    parser.add_argument("--instruction_file", default=str(ROOT / "AIGC/release/test/1_1/instruction.txt"))
    parser.add_argument("--output_dir", default=str(ROOT / "outputs/step4_inference"))
    parser.add_argument("--tokenizer_path", default=str(COSMOS_ROOT.parent / "cosmos-predict2.5/tokenizer.pth"))
    
    # 这些必须和 step3 的 args 保持一致
    parser.add_argument("--action_dim", type=int, default=26)
    parser.add_argument("--action_steps", type=int, default=50)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("1. Loading custom dual-stream weights...")
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    train_args = ckpt["args"]
    
    visual_bridge = build_visual_bridge(
        bridge_type=train_args["bridge_type"],
        in_channels=16, # VAE latent
        visual_dim=train_args["visual_dim"],
        pool_out_size=tuple(map(int, train_args["pool_out_size"].split(","))),
    ).to(device)
    
    action_denoiser = ActionDenoiser1DUNet(
        action_dim=train_args["action_dim"],
        visual_dim=train_args["visual_dim"],
        text_dim=train_args["text_dim"],
        base_channels=128,
        channel_mults=(1, 2, 4),
        use_text_cond=True,
    ).to(device)
    
    visual_bridge.load_state_dict(ckpt["visual_bridge_state"])
    action_denoiser.load_state_dict(ckpt["action_denoiser_state"])
    visual_bridge.eval()
    action_denoiser.eval()
    logger.success("Model loaded successfully.")
    
    logger.info("2. Loading Video VAE Tokenizer...")
    from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface
    tokenizer = Wan2pt1VAEInterface(vae_pth=args.tokenizer_path, temporal_window=16)
    tokenizer.model.model = tokenizer.model.model.to(device)
    
    logger.info("3. Loading and encoding video frames...")
    # 解析 step3 的 image_size
    H, W = map(int, train_args["image_size"].split(","))
    frames = load_video_frames(args.video_path, max_frames=train_args["num_cond_frames"], target_size=(H, W))
    frames = frames.to(device)
    
    with torch.no_grad():
        # 转为 bfloat16 [0, 255] 以符合官方 tokenizer 预期 (见 step3 train_step_action_only)
        frames_uint8 = (frames * 255).clamp(0, 255).to(torch.bfloat16)
        z0 = tokenizer.encode(frames_uint8[0].unsqueeze(0)).float() # (1, 16, T_lat, H_lat, W_lat)
        
        logger.info("4. Extracting visual condition...")
        visual_cond = visual_bridge(z0) # (1, visual_dim)
        
        logger.info("5. Preparing Text Condition (Mocked for test)...")
        # 由于我们没有在此加载 7B 模型，用全零替代 (或者你应该加载 Step2 预提取的 embeddings)
        # 这里为了演示测试流程不崩溃，使用随机张量代替
        text_cond = torch.randn(1, 128, train_args["text_dim"], device=device)
        
        logger.info("6. Sampling actions with EDM Euler...")
        action_shape = (1, train_args["action_steps"], train_args["action_dim"])
        predicted_actions = edm_euler_sampler(
            denoiser=action_denoiser,
            visual_cond=visual_cond,
            text_cond=text_cond,
            action_shape=action_shape,
            device=device,
            num_steps=50,
        )
        
    actions_np = predicted_actions[0].cpu().numpy()
    out_file = os.path.join(args.output_dir, "predicted_actions.npy")
    np.save(out_file, actions_np)
    logger.success(f"Action generation complete! Saved to {out_file}")
    logger.info(f"Generated actions shape: {actions_np.shape}")
    logger.info(f"Sample action step 0: {actions_np[0][:5]}...")

if __name__ == "__main__":
    main()
