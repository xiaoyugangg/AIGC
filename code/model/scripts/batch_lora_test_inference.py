#!/usr/bin/env python3
"""
使用 LoRA 微调后的 checkpoint，对 test 目录下全部样本做 Video2World 推理。

文本编码（重要）：
  - 训练里 text_encoder_config.compute_online=False 时，模型没有 Reason 编码器，推理会误走 get_text_embedding()
    → 去下 **google/t5-11b（~45GB）**。本脚本默认传入
    model.config.text_encoder_config.compute_online=true，与官方 Step1/2B 一致走 **Reason**，不下 T5。

预处理（决定画质是否和 Step1 一致，默认已与 Step1 对齐）：
  - **auto**（默认）：`K=num_latent_conditional_frames` 为 **1 或 2** 时用 **step1** 路径；**K≥3** 时自动改用 **training**（官方视频路径只支持 K≤2）。
  - **step1**：强制传 mp4 路径 + `read_and_process_video`（末尾条件帧 + center crop）；**仅 K∈{1,2}**。
  - **training**：自拼 tensor（开头条件帧 + 仅 resize）；**任意 K≥1**（需 `4*(K-1)+1` 帧像素，可用 `--pad_short_condition`）。

`--num_latent_conditional_frames` 即 **K**，可自行试不同值。

用法示例：
  cd /root/autodl-tmp/cosmos-predict2.5-1.5.0
  ./.venv/bin/python ../scripts/batch_lora_test_inference.py \\
    --ckpt_path .../checkpoints/iter_000000500 \\
    --test_root /root/autodl-tmp/AIGC/release/test \\
    --output_dir /root/autodl-tmp/outputs/lora_infer \\
    --num_latent_conditional_frames 5 --pad_short_condition

试 K=2（与 Step1 一致，auto 会走 step1 预处理）：
  ... --num_latent_conditional_frames 2 --guidance 3 --num_output_frames 66
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
import torchvision.transforms.functional as F

COSMOS_ROOT = Path(__file__).resolve().parent.parent / "cosmos-predict2.5-1.5.0"

# 与训练时 compute_online=False 配套：推理必须启用 Reason，否则 video2world 会调 get_text_embedding() → T5-11b
_REASON_ONLINE_OPTS = ("model.config.text_encoder_config.compute_online=true",)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch Video2World inference on test set (LoRA checkpoint)")
    p.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="训练保存的迭代目录，例如 .../checkpoints/iter_000000500（DCP）或合并的 .pt",
    )
    p.add_argument(
        "--test_root",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "AIGC/release/test"),
        help="包含子目录，每个子目录含 video.mp4 与 instruction.txt",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "outputs/lora_test_inference"),
        help="输出根目录（每个样本一个子文件夹）",
    )
    p.add_argument(
        "--experiment",
        type=str,
        default="predict2_video2world_training_2b_custom_lora",
        help="Hydra 实验名（需已注册 custom_lora）",
    )
    p.add_argument(
        "--config_file",
        type=str,
        default="cosmos_predict2/_src/predict2/configs/video2world/config.py",
    )
    p.add_argument(
        "--guidance",
        type=float,
        default=3.0,
        help="Classifier-free guidance（CFG），与训练回调采样一致时默认 3",
    )
    p.add_argument("--num_steps", type=int, default=35)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument(
        "--preprocess",
        type=str,
        choices=("auto", "step1", "training"),
        default="auto",
        help="auto=按 K 选 step1(K≤2) 或 training(K≥3)；step1/training 强制指定",
    )
    p.add_argument(
        "--num_latent_conditional_frames",
        type=int,
        default=2,
        metavar="K",
        help="latent 条件帧数 K（自行试验）。K≤2 可走 step1；K≥3 须 training 或 preprocess=auto",
    )
    p.add_argument(
        "--num_output_frames",
        type=int,
        default=66,
        help="传给 generate_vid2world 的 num_video_frames（与 Step1 --num_output_frames 一致；实际 T 仍受 state_t 约束）",
    )
    p.add_argument(
        "--pad_short_condition",
        action="store_true",
        help="条件段帧数不足时，用已有最后一帧复制补齐到 4*(K-1)+1（例如 16 帧 + K=5）",
    )
    p.add_argument(
        "--height",
        type=int,
        default=352,
        help="与训练 video_size[0] 一致",
    )
    p.add_argument(
        "--width",
        type=int,
        default=640,
        help="与训练 video_size[1] 一致",
    )
    p.add_argument(
        "--offload_text_encoder",
        action="store_true",
        help="省显存：文本编码走 CPU（较慢）",
    )
    p.add_argument(
        "--offload_tokenizer",
        action="store_true",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="仅处理前 N 个样本（0 表示全部）",
    )
    p.add_argument(
        "--allow_t5_fallback",
        action="store_true",
        help="不推荐：不开启 Reason 在线编码，会退回 T5-11b 并尝试从 Hub 下载",
    )
    return p.parse_args()


def list_test_samples(test_root: Path) -> list[Path]:
    samples = []
    for p in sorted(test_root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "video.mp4").is_file() and (p / "instruction.txt").is_file():
            samples.append(p)
    return samples


def pixel_frames_for_latent_cond(num_latent: int) -> int:
    return 4 * (num_latent - 1) + 1


def build_vid_input_first_frames_training_resize(
    video_path: str,
    num_video_frames: int,
    num_latent_conditional_frames: int,
    height: int,
    width: int,
    pad_short_condition: bool = False,
) -> torch.Tensor:
    """返回 (1, 3, T, H, W) uint8，与训练集 ResizePreprocess 一致（逐帧 resize）。"""
    from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io

    video_frames, _ = easy_io.load(video_path)
    # (T, H, W, C) uint8 / float
    vid = torch.from_numpy(video_frames).float()
    if vid.max() > 1.5:
        vid = vid / 255.0
    vid = vid.permute(0, 3, 1, 2)  # T, C, H, W

    need = pixel_frames_for_latent_cond(num_latent_conditional_frames)
    n = vid.shape[0]
    if n < need:
        if pad_short_condition and n >= 1:
            prefix = vid[:n, ...]
            last = prefix[-1:, ...]
            extracted = torch.cat([prefix, last.repeat(need - n, 1, 1, 1)], dim=0)
        else:
            raise ValueError(
                f"{video_path}: need at least {need} frames for "
                f"num_latent_conditional_frames={num_latent_conditional_frames}, got {n}. "
                f"Use smaller K (e.g. 4 needs 13 frames) or --pad_short_condition."
            )
    else:
        extracted = vid[:need, ...]  # 前 need 帧作为条件来源
    t_in = extracted.shape[0]
    c, h0, w0 = extracted.shape[1], extracted.shape[2], extracted.shape[3]
    full = torch.zeros(num_video_frames, c, h0, w0, dtype=extracted.dtype)
    full[:t_in] = extracted
    if t_in < num_video_frames:
        pad = num_video_frames - t_in
        full[t_in:] = extracted[-1:].expand(pad, -1, -1, -1)

    # 与 VideoDataset：ToTensor 后为 float，再 ResizePreprocess((H,W))
    size_hw = (height, width)
    resized = torch.stack([F.resize(full[t], size_hw, antialias=True) for t in range(full.shape[0])])
    out = (resized.clamp(0, 1) * 255.0).to(torch.uint8)
    return out.unsqueeze(0).permute(0, 2, 1, 3, 4)  # 1,3,T,H,W


def main() -> None:
    args = parse_args()
    os.chdir(str(COSMOS_ROOT))
    sys.path.insert(0, str(COSMOS_ROOT))

    from loguru import logger
    from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video
    from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference

    test_root = Path(args.test_root)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    samples = list_test_samples(test_root)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    if not samples:
        raise SystemExit(f"No samples under {test_root} (need */video.mp4 + instruction.txt)")

    resolution_str = f"{args.height},{args.width}"

    experiment_opts = list(_REASON_ONLINE_OPTS)
    if args.allow_t5_fallback:
        experiment_opts = []
        logger.warning(
            "allow_t5_fallback: 将尝试加载 T5-11b（可能从 Hugging Face 下载极大文件）。仅磁盘/调试时用。"
        )

    logger.info(f"Loading model (experiment={args.experiment}) from {args.ckpt_path}")
    if experiment_opts:
        logger.info(f"experiment_opts (Reason online): {experiment_opts}")
    inference_engine = Video2WorldInference(
        experiment_name=args.experiment,
        ckpt_path=args.ckpt_path,
        s3_credential_path="",
        context_parallel_size=1,
        config_file=args.config_file,
        experiment_opts=experiment_opts,
        offload_diffusion_model=False,
        offload_text_encoder=args.offload_text_encoder,
        offload_tokenizer=args.offload_tokenizer,
    )

    model_T = inference_engine.model.tokenizer.get_pixel_num_frames(
        inference_engine.model.config.state_t
    )
    logger.info(f"Model state_t pixel length (tokenizer) -> T={model_T}")

    if args.num_latent_conditional_frames < 1:
        raise SystemExit("--num_latent_conditional_frames (K) 须为 >=1 的正整数（本脚本为视频条件推理）")
    if args.preprocess == "step1" and args.num_latent_conditional_frames not in (1, 2):
        raise SystemExit(
            "preprocess=step1 时 K 只能是 1 或 2。要试 K≥3 请用 --preprocess training 或默认 --preprocess auto"
        )

    results: list[dict] = []

    for sample_dir in samples:
        name = sample_dir.name
        video_p = sample_dir / "video.mp4"
        instr_p = sample_dir / "instruction.txt"
        prompt = instr_p.read_text(encoding="utf-8").strip()
        sample_out = out_root / name
        sample_out.mkdir(parents=True, exist_ok=True)

        if args.preprocess == "auto":
            eff = "step1" if args.num_latent_conditional_frames in (1, 2) else "training"
        else:
            eff = args.preprocess

        logger.info(f"=== {name} === (preprocess={args.preprocess}, effective={eff}, K={args.num_latent_conditional_frames})")
        vid_tensor = output_video = video_for_save = None
        try:
            if eff == "step1":
                infer_in = str(video_p)
            else:
                vid_tensor = build_vid_input_first_frames_training_resize(
                    str(video_p),
                    num_video_frames=model_T,
                    num_latent_conditional_frames=args.num_latent_conditional_frames,
                    height=args.height,
                    width=args.width,
                    pad_short_condition=args.pad_short_condition,
                )
                infer_in = vid_tensor

            output_video = inference_engine.generate_vid2world(
                prompt=prompt,
                input_path=infer_in,
                guidance=args.guidance,
                num_video_frames=args.num_output_frames,
                num_latent_conditional_frames=args.num_latent_conditional_frames,
                resolution=resolution_str,
                seed=args.seed,
                num_steps=args.num_steps,
            )
            video_for_save = (1.0 + output_video[0]) / 2.0
            stem = sample_out / "generated_video"
            save_img_or_video(video_for_save, str(stem), fps=args.fps)

            mp4_path = str(stem) + ".mp4"
            meta = {
                "sample": name,
                "prompt": prompt,
                "checkpoint": args.ckpt_path,
                "resolution": resolution_str,
                "preprocess": args.preprocess,
                "num_latent_conditional_frames": args.num_latent_conditional_frames,
                "num_output_frames": args.num_output_frames,
                "guidance": args.guidance,
                "pad_short_condition": args.pad_short_condition,
                "num_steps": args.num_steps,
                "seed": args.seed,
                "output": mp4_path,
                "status": "ok",
            }
            (sample_out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            results.append(meta)
            logger.success(f"Saved {mp4_path}")
        except Exception as e:
            logger.exception(f"Failed {name}: {e}")
            results.append({"sample": name, "status": "error", "error": str(e)})
        finally:
            vid_tensor = output_video = video_for_save = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    (out_root / "batch_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Wrote summary: {out_root / 'batch_results.json'}")


if __name__ == "__main__":
    main()
