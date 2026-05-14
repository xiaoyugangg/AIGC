#!/usr/bin/env python3
"""
Step3 动作分支推理：加载 step3_dual_stream_train 保存的 checkpoint，
用 Cosmos tokenizer 编码条件帧得到 z0 → VisualBridge → 多步 EDM 采样动作序列。

说明：
  - 采样算法见 src/dual_stream/action_sampling.py（与训练步同一套 EDM 预条件化）。
  - 动作输出在 **反归一化** 后写入 CSV；归一化统计量来自 --data_root 全集（与 AigcDataset 训练一致）。

示例：
  cd /root/autodl-tmp
  python scripts/step3_dual_stream_infer.py \\
    --step3_ckpt outputs/step3_dual_stream/final_model.pt \\
    --data_root AIGC/release/train \\
    --input_root AIGC/release/test \\
    --output_dir outputs/step3_action_infer \\
    --cosmos_ckpt nvidia/Cosmos-Predict2.5-2B \\
    --embedding_dir outputs/text_embeddings \\
    --num_sampling_steps 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
COSMOS_ROOT = ROOT / "cosmos-predict2.5-1.5.0"
sys.path.insert(0, str(COSMOS_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step3 dual-stream action inference")
    p.add_argument("--step3_ckpt", type=str, required=True, help="final_model.pt 或 checkpoint_step*.pt")
    p.add_argument(
        "--data_root",
        type=str,
        default=str(ROOT / "AIGC/release/train"),
        help="用于计算动作归一化统计量（需含多条带 action.txt 的样本）",
    )
    p.add_argument(
        "--input_root",
        type=str,
        default=None,
        help="待推理样本根目录（默认与 data_root 相同）",
    )
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument(
        "--cosmos_ckpt",
        type=str,
        default="nvidia/Cosmos-Predict2.5-2B",
        help="Cosmos 预训练权重（tokenizer.encode 需要）",
    )
    p.add_argument("--experiment", type=str, default=None)
    p.add_argument(
        "--config_file",
        type=str,
        default="cosmos_predict2/_src/predict2/configs/video2world/config.py",
    )
    p.add_argument("--skip_video_model", action="store_true", help="随机 z0，仅调试动作网络")
    p.add_argument(
        "--embedding_dir",
        type=str,
        default=None,
        help="预计算文本向量目录（与训练一致时需指向同一套 *.pt）",
    )
    p.add_argument("--num_sampling_steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_samples", type=int, default=0, help="0 表示全部")
    p.add_argument(
        "--action_row_start",
        type=int,
        default=80,
        help="写出 CSV 第一列（帧/步序号）起始值",
    )
    return p.parse_args()


def _load_training_args(ckpt: dict) -> dict:
    a = ckpt.get("args")
    if a is None:
        return {}
    if isinstance(a, dict):
        return a
    return vars(a)


def list_samples(root: Path) -> list[Path]:
    out = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if (d / "video.mp4").is_file() and (d / "instruction.txt").is_file():
            out.append(d)
    return out


def load_cond_frames_tensor(
    sample_dir: Path,
    num_cond_frames: int,
    image_size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """返回 (1, 3, T_cond, H, W) float32 [0,1]，与 AigcDataset 一致。"""
    from dual_stream.dataset import read_video_frames

    total_needed = num_cond_frames
    frames = read_video_frames(str(sample_dir / "video.mp4"), num_frames=total_needed)
    if len(frames) < total_needed:
        pad_len = total_needed - len(frames)
        frames = np.concatenate([frames, np.repeat(frames[-1:], pad_len, axis=0)], axis=0)

    cond_indices = np.linspace(0, total_needed - 1, num_cond_frames, dtype=int)
    cond_frames = frames[cond_indices]

    ft = torch.from_numpy(cond_frames).permute(3, 0, 1, 2).float() / 255.0
    Ht, Wt = image_size
    _, T, _, _ = ft.shape
    resized = torch.stack([TF.resize(ft[:, t], [Ht, Wt], antialias=True) for t in range(T)], dim=1)
    return resized.unsqueeze(0).to(device)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_root = Path(args.input_root or args.data_root)
    data_root = Path(args.data_root)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.step3_ckpt, map_location=device, weights_only=False)
    targs = _load_training_args(ckpt)

    action_dim = int(targs.get("action_dim", 26))
    visual_dim = int(targs.get("visual_dim", 256))
    text_dim = int(targs.get("text_dim", 896))
    bridge_type = targs.get("bridge_type", "simple")
    pool_out_size = tuple(int(x) for x in str(targs.get("pool_out_size", "1,2,4")).split(","))
    num_cond_frames = int(targs.get("num_cond_frames", 16))
    image_size = tuple(int(x) for x in str(targs.get("image_size", "192,320")).split(","))
    action_steps = int(targs.get("action_steps", 50))

    embed_infer = Path(args.embedding_dir).resolve() if args.embedding_dir else None
    train_emb = targs.get("embedding_dir")
    train_has_emb = bool(train_emb and Path(str(train_emb)).exists())
    infer_has_emb = embed_infer is not None and embed_infer.exists()
    # 与训练时 step3 逻辑一致：mock 或「训练时 embedding 目录存在」或「推理显式传入 embedding」均启用文本分支结构
    use_text_cond = bool(targs.get("mock_data")) or train_has_emb or infer_has_emb

    from dual_stream.visual_bridge import build_visual_bridge
    from dual_stream.action_denoiser import ActionDenoiser1DUNet
    from dual_stream.action_sampling import sample_actions_edm
    from dual_stream.dataset import AigcDataset

    stats_ds = AigcDataset(
        data_root=str(data_root),
        num_cond_frames=num_cond_frames,
        num_action_steps=action_steps,
        image_size=image_size,
    )
    action_stats = stats_ds.action_stats

    visual_bridge = build_visual_bridge(
        bridge_type=bridge_type,
        in_channels=16,
        visual_dim=visual_dim,
        pool_out_size=pool_out_size,
    ).to(device)
    action_denoiser = ActionDenoiser1DUNet(
        action_dim=action_dim,
        visual_dim=visual_dim,
        text_dim=text_dim,
        base_channels=128,
        channel_mults=(1, 2, 4),
        use_text_cond=use_text_cond,
    ).to(device)

    visual_bridge.load_state_dict(ckpt["visual_bridge_state"])
    action_denoiser.load_state_dict(ckpt["action_denoiser_state"])
    visual_bridge.eval()
    action_denoiser.eval()

    video_model = None
    if not args.skip_video_model:
        os.chdir(str(COSMOS_ROOT))
        from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint
        from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey

        exp = args.experiment or MODEL_CHECKPOINTS[ModelKey(post_trained=False)].experiment
        video_model, _ = load_model_from_checkpoint(
            experiment_name=exp,
            s3_checkpoint_dir=args.cosmos_ckpt,
            config_file=args.config_file,
            load_ema_to_reg=True,
            experiment_opts=["~data_train"],
            to_device=str(device),
        )
        video_model.eval()
        for p in video_model.parameters():
            p.requires_grad_(False)
        os.chdir(str(ROOT))

    samples = list_samples(input_root)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    for sample_dir in samples:
        name = sample_dir.name
        frames_tensor = load_cond_frames_tensor(sample_dir, num_cond_frames, image_size, device)

        if video_model is not None:
            fu = (frames_tensor * 255.0).clamp(0, 255).to(torch.uint8)
            z0 = video_model.tokenizer.encode(fu.bfloat16()).float()
        else:
            z0 = torch.randn(1, 16, 5, 24, 40, device=device)

        visual_cond = visual_bridge(z0)

        text_cond = None
        if use_text_cond and embed_infer is not None:
            emb_path = embed_infer / f"{name}.pt"
            if emb_path.is_file():
                emb = torch.load(emb_path, map_location=device, weights_only=True)
                text_cond = emb["embedding"].squeeze(0).unsqueeze(0)
            else:
                text_cond = None

        actions_norm = sample_actions_edm(
            action_denoiser,
            visual_cond,
            text_cond,
            action_steps=action_steps,
            action_dim=action_dim,
            device=device,
            num_steps=args.num_sampling_steps,
            generator=gen,
        )

        actions_np = actions_norm.squeeze(0).float().cpu().numpy()
        mean = action_stats["mean"]
        std = action_stats["std"]
        actions_denorm = actions_np * std + mean

        ref_action = sample_dir / "action.txt"
        if ref_action.is_file():
            header_line = ref_action.read_text(encoding="utf-8").splitlines()[0]
            cols = header_line.split(",")
        else:
            cols = ["Unnamed: 0"] + [f"j{i}" for i in range(actions_denorm.shape[1])]

        ncols = min(len(cols), actions_denorm.shape[1] + 1)
        cols_use = cols[:ncols]

        idx_col = np.arange(
            args.action_row_start,
            args.action_row_start + actions_denorm.shape[0],
            dtype=np.float64,
        )
        mat = np.column_stack([idx_col, actions_denorm[:, : ncols - 1]])

        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / "action_infer.txt"
        pd.DataFrame(mat, columns=cols_use).to_csv(out_csv, index=False)

        meta = {
            "sample": name,
            "step3_ckpt": args.step3_ckpt,
            "num_sampling_steps": args.num_sampling_steps,
            "action_steps": action_steps,
            "skip_video_model": args.skip_video_model,
            "use_text_cond": use_text_cond,
            "text_embedding_used": text_cond is not None,
        }
        (out_dir / "infer_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Done. Wrote {len(samples)} samples under {out_root}")


if __name__ == "__main__":
    main()
