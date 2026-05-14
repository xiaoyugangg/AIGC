"""
Official Pipeline Inference Test
=================================
Mirrors exactly what 'examples/inference.py' does internally,
but with explicit local paths — no cosmos_oss / git LFS required.

Run on AutoDL:
    python scripts/official_pipeline_test.py \
        --input_video /root/autodl-tmp/AIGC/release/train/1_1/video.mp4 \
        --prompt "A robotic arm performing a precise manipulation task." \
        --output_dir /root/autodl-tmp/outputs/official_pipeline \
        --model_size 2B \
        --pretrained  # use pre-trained weights (not post-trained)
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

COSMOS_ROOT = "/root/autodl-tmp/cosmos-predict2.5-1.5.0"
sys.path.insert(0, COSMOS_ROOT)

# ── match exactly what cosmos_predict2/config.py does for ModelKey(post_trained=False)
PRE_TRAINED_CKPT_UUID = "d20b7120-df3e-4911-919d-db6e08bad31c"
POST_TRAINED_CKPT_UUID = "81edfebe-bd6a-4039-8c1d-737df1a790bf"

# Official experiment name for the 2B rectified-flow pre-trained model
OFFICIAL_EXPERIMENT = (
    "Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720"
    "-Fps-16-Note-T2V_high_sigma_loss_reweighted_1_1_rectified_flow_only_resume2"
)


def save_video_cv2(video_tensor: torch.Tensor, path: str, fps: int = 16):
    """Save (C, T, H, W) tensor in [-1,1] as mp4 using OpenCV."""
    # → (T, H, W, C) uint8
    video = video_tensor.float().clamp(-1, 1)
    video = ((video + 1.0) / 2.0 * 255.0).to(torch.uint8)
    video = video.permute(1, 2, 3, 0).cpu().numpy()
    T, H, W, C = video.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (W, H))
    for frame in video:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"[SAVED] {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--prompt", default="A robotic arm performing precise manipulation.")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/outputs/official_pipeline")
    parser.add_argument("--resolution", default="192,320")
    parser.add_argument("--num_steps", type=int, default=35)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_latent_conditional_frames", type=int, default=1)
    parser.add_argument("--pretrained", action="store_true",
                        help="Use pre-trained weights (d20b...). Default uses post-trained (81ed...)")
    parser.add_argument("--ckpt_base", default="/root/autodl-tmp/cosmos-predict2.5/base")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Select checkpoint ──────────────────────────────────────────────────────
    ckpt_uuid = PRE_TRAINED_CKPT_UUID if args.pretrained else POST_TRAINED_CKPT_UUID
    # The checkpoint_db.py patch maps UUID → local file
    # pre-trained  → .../base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt
    # post-trained → .../base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt
    if args.pretrained:
        ckpt_path = os.path.join(args.ckpt_base, "pre-trained",
                                 f"{PRE_TRAINED_CKPT_UUID}_ema_bf16.pt")
    else:
        ckpt_path = os.path.join(args.ckpt_base, "post-trained",
                                 f"{POST_TRAINED_CKPT_UUID}_ema_bf16.pt")
    print(f"[INFO] Checkpoint: {ckpt_path}")
    print(f"[INFO] Exists: {os.path.exists(ckpt_path)}, "
          f"Size: {os.path.getsize(ckpt_path)/1e9:.2f} GB" if os.path.exists(ckpt_path)
          else "[WARN] Checkpoint not found!")

    # ── Initialise Video2WorldInference (identical to cosmos_predict2.inference.Inference) ──
    print(f"\n[INIT] Loading Video2WorldInference...")
    from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference

    pipe = Video2WorldInference(
        experiment_name=OFFICIAL_EXPERIMENT,
        ckpt_path=ckpt_path,
        s3_credential_path="",
        context_parallel_size=1,
    )
    print("[INIT] Model loaded successfully.")

    # ── Run generate_vid2world (identical to cosmos_predict2/inference.py line 133-143) ──
    print(f"\n[GENERATE] Starting generation...")
    print(f"  input:           {args.input_video}")
    print(f"  prompt:          {args.prompt[:80]}...")
    print(f"  resolution:      {args.resolution}")
    print(f"  guidance:        {args.guidance}")
    print(f"  steps:           {args.num_steps}")
    print(f"  cond_frames:     {args.num_latent_conditional_frames}")

    video = pipe.generate_vid2world(
        prompt=args.prompt,
        input_path=args.input_video,
        guidance=args.guidance,
        num_video_frames=93,          # official default
        num_latent_conditional_frames=args.num_latent_conditional_frames,
        resolution=args.resolution,
        seed=args.seed,
        num_steps=args.num_steps,
    )

    print(f"\n[OUTPUT] video shape: {video.shape}")
    print(f"[OUTPUT] video range: [{video.min():.4f}, {video.max():.4f}]")

    # ── Save exactly like cosmos_predict2/inference.py line 146 ───────────────
    # Official: video = (1.0 + video[0]) / 2  → save_img_or_video(video, ...)
    # We use cv2 instead of save_img_or_video (no ffmpegcv needed)
    output_path = os.path.join(args.output_dir, "output.mp4")
    save_video_cv2(video[0], output_path, fps=16)

    print(f"\n[DONE] Output saved to: {output_path}")


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/.hf_cache")
    main()
