"""
VAE Encode-Decode Roundtrip Diagnostic
======================================
This script isolates the VAE to verify it can correctly encode and decode video frames.
If this produces clean output, the issue is in the diffusion model.
If this also produces distorted colors, the issue is in the VAE.

Run on AutoDL:
    python scripts/diagnose_vae.py \
        --input_video /root/autodl-tmp/AIGC/release/train/1_1/video.mp4 \
        --tokenizer_path /root/autodl-tmp/cosmos-predict2.5/tokenizer.pth \
        --output_dir /root/autodl-tmp/outputs/vae_diag
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

# Set up project path
COSMOS_ROOT = "/root/autodl-tmp/cosmos-predict2.5-1.5.0"
sys.path.insert(0, COSMOS_ROOT)


def load_video_frames(video_path: str, max_frames: int = 9) -> np.ndarray:
    """Load first N frames from video, returns (T, H, W, C) uint8 RGB."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return np.stack(frames)


def save_frames_as_video(frames_uint8: np.ndarray, output_path: str, fps: int = 4):
    """Save (T, H, W, C) uint8 RGB frames as mp4."""
    T, H, W, C = frames_uint8.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    for frame in frames_uint8:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr)
    out.release()
    print(f"[SAVED] {output_path} ({T} frames @ {fps}fps)")


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """Convert (C, T, H, W) in [-1, 1] to (T, H, W, C) uint8."""
    t = t.float().clamp(-1, 1)
    t = (t + 1.0) / 2.0 * 255.0
    arr = t.permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--tokenizer_path", default="/root/autodl-tmp/cosmos-predict2.5/tokenizer.pth")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/outputs/vae_diag")
    parser.add_argument("--num_frames", type=int, default=9, help="Must be (4k+1): 1,5,9,13...")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    # ── Step 1: Load input video ──────────────────────────────────────────────
    print(f"\n[STEP 1] Loading input video: {args.input_video}")
    frames_np = load_video_frames(args.input_video, max_frames=args.num_frames)
    T = frames_np.shape[0]
    H, W = frames_np.shape[1], frames_np.shape[2]
    print(f"  Loaded {T} frames, shape: {frames_np.shape}, dtype: {frames_np.dtype}")

    # Save original frames
    orig_path = os.path.join(args.output_dir, "00_original.mp4")
    save_frames_as_video(frames_np, orig_path, fps=4)

    # ── Step 2: Convert to tensor [-1, 1] in (B, C, T, H, W) ─────────────────
    print(f"\n[STEP 2] Converting to tensor...")
    video_tensor = torch.from_numpy(frames_np).float() / 255.0  # [0,1]
    video_tensor = video_tensor * 2.0 - 1.0  # [-1,1]
    video_tensor = video_tensor.permute(3, 0, 1, 2)  # (C, T, H, W)
    video_tensor = video_tensor.unsqueeze(0)  # (1, C, T, H, W)
    video_tensor = video_tensor.to(device, dtype=torch.bfloat16)
    print(f"  Tensor: {video_tensor.shape}, range: [{video_tensor.min():.3f}, {video_tensor.max():.3f}]")

    # ── Step 3: Load WanVAE tokenizer ─────────────────────────────────────────
    print(f"\n[STEP 3] Loading WanVAE tokenizer: {args.tokenizer_path}")
    from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface

    tokenizer = Wan2pt1VAEInterface(
        vae_pth=args.tokenizer_path,
        temporal_window=16,
    )
    tokenizer.model.model = tokenizer.model.model.to(device)
    print(f"  WanVAE loaded. Parameters: {tokenizer.model.count_param():,}")

    # ── Step 4: ENCODE ─────────────────────────────────────────────────────────
    print(f"\n[STEP 4] Encoding video to latent space...")
    with torch.no_grad():
        # encode expects (B, C, T, H, W) but Wan2pt1VAEInterface.encode takes (C, T, H, W)
        # Let's use the model directly
        x = video_tensor[0]  # (C, T, H, W)
        x_for_enc = x.unsqueeze(0)  # (1, C, T, H, W) - batch=1

        # scale from [-1,1] to [0,1] for VAE input? Let's check what the VAE expects
        # The WanVAE was trained on [0,1]? or [-1,1]?
        # Looking at the code: input to Decoder3d has no explicit range constraint.
        # The encoder input in training is typically normalized to [-1,1] via augmentors.
        # Let's try both and see.

        # Method A: pass as-is in [-1,1]
        latent_a = tokenizer.encode(x_for_enc)
        print(f"  Latent A shape: {latent_a.shape}, range: [{latent_a.min():.4f}, {latent_a.max():.4f}]")
        print(f"  Latent A mean: {latent_a.mean():.4f}, std: {latent_a.std():.4f}")

    # ── Step 5: DECODE ─────────────────────────────────────────────────────────
    print(f"\n[STEP 5] Decoding latent back to pixel space...")
    with torch.no_grad():
        recon_a = tokenizer.decode(latent_a)
        print(f"  Recon A shape: {recon_a.shape}, range: [{recon_a.min():.4f}, {recon_a.max():.4f}]")
        print(f"  Recon A mean: {recon_a.mean():.4f}, std: {recon_a.std():.4f}")

    # ── Step 6: Save reconstruction ───────────────────────────────────────────
    print(f"\n[STEP 6] Saving results...")
    recon_frames = tensor_to_uint8(recon_a[0])  # (T, H, W, C)
    recon_path = os.path.join(args.output_dir, "01_vae_roundtrip.mp4")
    save_frames_as_video(recon_frames, recon_path, fps=4)

    # ── Step 7: Per-channel statistics of latent ───────────────────────────────
    print(f"\n[STEP 7] Per-channel latent statistics:")
    latent_np = latent_a[0].float().cpu().numpy()  # (C, T, H, W)
    for ch in range(latent_np.shape[0]):
        ch_data = latent_np[ch]
        print(f"  ch{ch:02d}: mean={ch_data.mean():.4f}  std={ch_data.std():.4f}  "
              f"min={ch_data.min():.4f}  max={ch_data.max():.4f}")

    # ── Step 8: Save side-by-side comparison frames ────────────────────────────
    print(f"\n[STEP 8] Saving pixel-level comparison...")
    orig_resized = []
    for frame in frames_np:
        resized = cv2.resize(frame, (recon_frames.shape[2], recon_frames.shape[1]))
        orig_resized.append(resized)
    orig_resized = np.stack(orig_resized)

    # Compute PSNR per frame
    n_compare = min(len(orig_resized), len(recon_frames))
    print(f"  Comparing {n_compare} frames:")
    for i in range(n_compare):
        mse = np.mean((orig_resized[i].astype(float) - recon_frames[i].astype(float)) ** 2)
        psnr = 10 * np.log10(255**2 / (mse + 1e-8)) if mse > 1e-8 else float('inf')
        print(f"  Frame {i:02d}: MSE={mse:.2f}  PSNR={psnr:.2f}dB")

    # Save side-by-side
    comparison_frames = []
    for i in range(n_compare):
        side_by_side = np.concatenate([orig_resized[i], recon_frames[i]], axis=1)
        comparison_frames.append(side_by_side)
    comparison_path = os.path.join(args.output_dir, "02_comparison.mp4")
    save_frames_as_video(np.stack(comparison_frames), comparison_path, fps=2)

    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC COMPLETE")
    print(f"  Original:          {orig_path}")
    print(f"  VAE Roundtrip:     {recon_path}")
    print(f"  Side-by-side:      {comparison_path}")
    print(f"\nInterpretation:")
    print(f"  - If VAE roundtrip looks similar to original → VAE is fine → issue in diffusion model")
    print(f"  - If VAE roundtrip has color distortion → VAE/tokenizer has a problem")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
