"""
第一步：跑通官方 Cosmos-Predict2.5-2B Pre-trained 模型推理。

用法示例：
  cd d:/ZTE/cosmos-predict2.5-1.5.0
  python ../scripts/step1_inference_test.py \
      --ckpt_path /path/to/Cosmos-Predict2.5-2B \
      --input_video ../AIGC/release/train/1_1/video.mp4 \
      --instruction_file ../AIGC/release/train/1_1/instruction.txt \
      --output_dir ../outputs/step1
"""

import argparse
import sys
import os
from pathlib import Path

# 确保官方源码在 Python 路径中
COSMOS_ROOT = Path(__file__).parent.parent / "cosmos-predict2.5-1.5.0"
sys.path.insert(0, str(COSMOS_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Cosmos Predict2.5 Inference Test (Step 1)")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="nvidia/Cosmos-Predict2.5-2B",
        help="官方预训练模型路径（本地目录或 HuggingFace ID）",
    )
    parser.add_argument(
        "--input_video",
        type=str,
        default=str(Path(__file__).parent.parent / "AIGC/release/train/1_1/video.mp4"),
        help="输入视频路径（用前几帧作为条件）",
    )
    parser.add_argument(
        "--instruction_file",
        type=str,
        default=str(Path(__file__).parent.parent / "AIGC/release/train/1_1/instruction.txt"),
        help="指令文本文件路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).parent.parent / "outputs/step1"),
        help="输出目录",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="192,320",
        help="视频分辨率 H,W（官方默认 192,320）",
    )
    parser.add_argument(
        "--num_output_frames",
        type=int,
        default=77,
        help="生成帧数（官方默认 77）",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=35,
        help="扩散步数（官方默认 35，调试时可设为 5）",
    )
    parser.add_argument(
        "--num_latent_conditional_frames",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help=(
            "条件 latent 帧数 K；仅 0/1/2 是因为官方 read_and_process_video(视频路径) 只支持 K∈{1,2}。"
            "训练里若用 K>2（如 custom_lora 的 5），需用批量脚本 --preprocess training 传自拼 tensor，或改推理管线。"
        ),
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=7.0,
        help="CFG guidance scale（0-7）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--offload_text_encoder",
        action="store_true",
        help="推理时将文本编码器卸载到 CPU（节省显存）",
    )
    parser.add_argument(
        "--offload_tokenizer",
        action="store_true",
        help="推理时将 VAE tokenizer 卸载到 CPU（节省显存）",
    )
    parser.add_argument(
        "--offload_diffusion_model",
        action="store_true",
        help="推理时将扩散模型卸载到 CPU（节省显存，但慢）",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="实验名称（覆盖默认值），用于非标准 checkpoint",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="cosmos_predict2/_src/predict2/configs/video2world/config.py",
        help="模型配置文件路径（相对于 cosmos-predict2.5-1.5.0）",
    )
    return parser.parse_args()


def read_instruction(instruction_file: str) -> str:
    """读取指令文本文件。"""
    with open(instruction_file, "r", encoding="utf-8") as f:
        return f.read().strip()


def run_inference(args):
    """运行官方 Video2World 推理。"""
    import torch
    from loguru import logger

    # 切换工作目录到官方代码根目录（官方代码中有相对路径引用）
    os.chdir(str(COSMOS_ROOT))

    from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference

    # ---- 读取指令 ----
    prompt = read_instruction(args.instruction_file)
    logger.info(f"Prompt: {prompt}")

    # ---- 确定 checkpoint 路径 ----
    ckpt_path = args.ckpt_path

    # ---- 实验配置 ----
    # Pre-trained 2B 模型对应的实验名由 config.py 中的 ModelKey 决定
    # 如果用本地路径，需要手动指定 experiment name
    if args.experiment:
        experiment_name = args.experiment
    else:
        # 尝试从官方注册表获取
        try:
            from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey
            pre_trained_key = ModelKey(post_trained=False)
            checkpoint_config = MODEL_CHECKPOINTS[pre_trained_key]
            experiment_name = checkpoint_config.experiment
            if not ckpt_path or ckpt_path == "nvidia/Cosmos-Predict2.5-2B":
                ckpt_path = checkpoint_config.s3.uri
            logger.info(f"Using pre-trained 2B experiment: {experiment_name}")
        except Exception as e:
            logger.warning(f"Could not load from registry: {e}")
            logger.warning("Please set --experiment manually for local checkpoints.")
            raise

    logger.info(f"Checkpoint path: {ckpt_path}")
    logger.info(f"Experiment: {experiment_name}")

    # custom_lora 训练使用 compute_online=False（离线 Reason 向量），推理时若不改回在线 Reason，
    # model.text_encoder 为 None，会走 get_text_embedding() → 下载 T5-11b（几十 GB）。
    experiment_opts: list[str] = []
    if experiment_name and "custom_lora" in experiment_name:
        experiment_opts.append("model.config.text_encoder_config.compute_online=true")
        logger.info(
            "custom_lora: enabling Reason text encoder online (same as official 2B); avoids T5-11b download."
        )

    # ---- 初始化推理器 ----
    logger.info("Initializing Video2WorldInference...")
    inference_engine = Video2WorldInference(
        experiment_name=experiment_name,
        ckpt_path=ckpt_path,
        s3_credential_path="",
        context_parallel_size=1,
        config_file=args.config_file,
        experiment_opts=experiment_opts,
        offload_diffusion_model=args.offload_diffusion_model,
        offload_text_encoder=args.offload_text_encoder,
        offload_tokenizer=args.offload_tokenizer,
    )
    logger.success("Model loaded successfully!")

    # ---- 打印显存使用 ----
    if torch.cuda.is_available():
        mem_gb = torch.cuda.memory_allocated() / (1024**3)
        logger.info(f"GPU memory after model load: {mem_gb:.2f} GB")

    # ---- 运行推理 ----
    logger.info(f"Running inference on: {args.input_video}")
    output_video = inference_engine.generate_vid2world(
        prompt=prompt,
        input_path=args.input_video,
        guidance=args.guidance,
        num_video_frames=args.num_output_frames,
        num_latent_conditional_frames=args.num_latent_conditional_frames,
        resolution=args.resolution,
        seed=args.seed,
        num_steps=args.num_steps,
    )

    # ---- 保存结果 ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 从官方库导入视频保存工具
    from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video

    output_path = output_dir / "generated_video"
    # 转换为 [0, 1] 范围
    print(f"DEBUG output shape: {output_video.shape}, range: [{output_video.min().item():.3f}, {output_video.max().item():.3f}]")
    # 直接用官方格式保存：(B, C, T, H, W) -> 转成 (T, H, W, C) uint8
    import numpy as np
    v = output_video[0].float()  # (C, T, H, W)
    v = v.permute(1, 2, 3, 0)   # (T, H, W, C)
    v = ((v / 2.0 + 0.5).clamp(0, 1) * 255).byte().cpu().numpy()
    import cv2
    h, w = v.shape[1], v.shape[2]
    writer = cv2.VideoWriter(str(output_path) + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), 16, (w, h))
    for frame in v:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    logger.success(f"Video saved to {output_path}.mp4")

    # ---- 保存元数据 ----
    import json
    meta = {
        "prompt": prompt,
        "input_video": args.input_video,
        "resolution": args.resolution,
        "num_output_frames": args.num_output_frames,
        "num_steps": args.num_steps,
        "guidance": args.guidance,
        "seed": args.seed,
        "num_latent_conditional_frames": args.num_latent_conditional_frames,
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info(f"Metadata saved to {output_dir / 'meta.json'}")

    return str(output_path) + ".mp4"


def batch_inference(args, sample_dirs: list[str]):
    """批量对 AIGC 数据集样本进行推理。"""
    import torch
    import json
    from loguru import logger

    os.chdir(str(COSMOS_ROOT))

    from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
    from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video
    from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey

    pre_trained_key = ModelKey(post_trained=False)
    checkpoint_config = MODEL_CHECKPOINTS[pre_trained_key]
    experiment_name = args.experiment or checkpoint_config.experiment
    ckpt_path = args.ckpt_path if args.ckpt_path != "nvidia/Cosmos-Predict2.5-2B" else checkpoint_config.s3.uri

    experiment_opts: list[str] = []
    if experiment_name and "custom_lora" in experiment_name:
        experiment_opts.append("model.config.text_encoder_config.compute_online=true")

    # 只初始化一次推理器
    inference_engine = Video2WorldInference(
        experiment_name=experiment_name,
        ckpt_path=ckpt_path,
        s3_credential_path="",
        context_parallel_size=1,
        config_file=args.config_file,
        experiment_opts=experiment_opts,
        offload_diffusion_model=args.offload_diffusion_model,
        offload_text_encoder=args.offload_text_encoder,
        offload_tokenizer=args.offload_tokenizer,
    )

    output_dir = Path(args.output_dir)
    results = []

    for sample_dir in sample_dirs:
        sample_dir = Path(sample_dir)
        video_path = sample_dir / "video.mp4"
        instruction_path = sample_dir / "instruction.txt"

        if not video_path.exists() or not instruction_path.exists():
            logger.warning(f"Skipping {sample_dir}: missing files")
            continue

        prompt = instruction_path.read_text().strip()
        sample_name = sample_dir.name
        logger.info(f"Processing {sample_name}: {prompt}")

        try:
            output_video = inference_engine.generate_vid2world(
                prompt=prompt,
                input_path=str(video_path),
                guidance=args.guidance,
                num_video_frames=args.num_output_frames,
                num_latent_conditional_frames=args.num_latent_conditional_frames,
                resolution=args.resolution,
                seed=args.seed,
                num_steps=args.num_steps,
            )

            sample_out_dir = output_dir / sample_name
            sample_out_dir.mkdir(parents=True, exist_ok=True)
            output_path = sample_out_dir / "generated_video"
            video_for_save = (1.0 + output_video[0]) / 2.0
            save_img_or_video(video_for_save, str(output_path), fps=16)
            results.append({"sample": sample_name, "output": str(output_path) + ".mp4", "status": "ok"})
            logger.success(f"Saved {sample_name}")

        except Exception as e:
            logger.error(f"Failed to process {sample_name}: {e}")
            results.append({"sample": sample_name, "status": "error", "error": str(e)})

    with open(output_dir / "batch_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    args = parse_args()
    result = run_inference(args)
    print(f"\n✅ 推理完成，输出文件：{result}")
