"""
第二步：预计算并缓存所有训练样本的文本 embedding。

官方使用 Qwen2.5-VL-7B 作为文本编码器，显存开销高达 ~14GB。
本脚本提供三种轻量替代方案，推荐使用 qwen2.5-0.5b：

  1. qwen2.5-0.5b（★ 推荐）
     - Qwen2.5 系列最小版本，0.5B 参数，2024 年发布
     - 与官方 Qwen 系列一脉相承，文本理解能力强
     - 隐藏层维度 896，float16 仅需 ~1GB 显存
     - HuggingFace: Qwen/Qwen2.5-0.5B

  2. t5-base (~250M)
     - 经典扩散模型文本编码器，稳定可靠
     - 维度 768

  3. clip
     - CLIP ViT-L/14 文本侧，SD 系列标配
     - 维度 768，最大序列长度 77

预计算后保存为 .pt 文件，训练时直接加载，避免在训练循环中运行编码器。
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="预计算文本 Embedding（第二步）")
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(Path(__file__).parent.parent / "AIGC/release/train"),
        help="AIGC 训练数据根目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).parent.parent / "outputs/text_embeddings"),
        help="保存 embedding 的目录",
    )
    parser.add_argument(
        "--encoder_type",
        type=str,
        choices=["qwen2.5-0.5b", "t5-base", "t5-small", "clip"],
        default="qwen2.5-0.5b",
        help="使用的文本编码器类型（推荐 qwen2.5-0.5b）",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="文本 token 最大长度（CLIP 限制 77，Qwen/T5 可以更长）",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="批处理大小（Qwen 可适当增大）",
    )
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


# ============================================================
# 轻量文本编码器实现
# ============================================================

class Qwen25TextEncoder(nn.Module):
    """
    基于 Qwen2.5-0.5B 的轻量文本编码器。

    Qwen2.5-0.5B 是 Qwen2.5 系列（2024年10月发布）中最小的文本模型，
    仅 0.5B 参数，隐藏层维度 896，float16 显存约 1GB。

    与官方 Cosmos 使用的 Qwen2.5-VL-7B 同源，但大幅轻量化。
    输出形状：(B, seq_len, 896)
    """

    HIDDEN_DIM = 896   # Qwen2.5-0.5B 的隐藏层维度
    MODEL_ID = "Qwen/Qwen2.5-0.5B"

    def __init__(
        self,
        model_name_or_path: str = None,
        max_length: int = 128,
        device: str = "cuda",
        use_last_n_layers: int = 1,     # 使用最后 n 层的均值（1=只用最后层）
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel

        self.max_length = max_length
        self.device = device
        self.use_last_n_layers = use_last_n_layers
        self.dtype = dtype

        _mid = model_name_or_path or self.MODEL_ID
        logger.info(f"Loading Qwen2.5-0.5B from: {_mid}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            _mid,
            trust_remote_code=True,
        )
        # 确保有 pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModel.from_pretrained(
            _mid,
            torch_dtype=dtype,
            trust_remote_code=True,
            output_hidden_states=True,
        ).to(device)
        self.model.eval()
        logger.success(f"Qwen2.5-0.5B loaded. Hidden dim: {self.HIDDEN_DIM}")

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        """
        返回 (B, seq_len, 896) embedding。
        
        使用最后 n 层 hidden states 的均值，比只用最后一层更稳定。
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        outputs = self.model(**inputs)

        if self.use_last_n_layers == 1:
            # 直接用最后一层
            emb = outputs.last_hidden_state  # (B, seq_len, 896)
        else:
            # 取最后 n 层的均值（更稳定）
            hidden_states = outputs.hidden_states  # tuple of (B, seq_len, 896)
            last_n = hidden_states[-self.use_last_n_layers:]
            emb = torch.stack(last_n, dim=0).mean(dim=0)  # (B, seq_len, 896)

        return emb.float()

    @property
    def embedding_dim(self) -> int:
        return self.HIDDEN_DIM


class LiteT5TextEncoder(nn.Module):
    """
    基于 T5-base/small 的文本编码器（备选方案）。
    T5-base: hidden_dim=768, T5-small: hidden_dim=512
    """

    def __init__(self, model_name: str = "t5-base", max_length: int = 128, device: str = "cuda"):
        super().__init__()
        from transformers import T5EncoderModel, T5Tokenizer

        self.max_length = max_length
        self.device = device

        logger.info(f"Loading {model_name}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name).to(device)
        self.encoder.eval()
        self.hidden_dim = self.encoder.config.d_model
        logger.success(f"T5 encoder loaded. Hidden dim: {self.hidden_dim}")

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        outputs = self.encoder(**inputs)
        return outputs.last_hidden_state.float()

    @property
    def embedding_dim(self) -> int:
        return self.hidden_dim


class LiteCLIPTextEncoder(nn.Module):
    """
    基于 CLIP ViT-L/14 的文本编码器（备选方案）。
    输出形状：(B, 77, 768)
    """

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14", device: str = "cuda"):
        super().__init__()
        from transformers import CLIPTextModel, CLIPTokenizer

        logger.info(f"Loading CLIP text encoder: {model_name}")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.encoder = CLIPTextModel.from_pretrained(model_name).to(device)
        self.encoder.eval()
        self.device = device
        self.hidden_dim = self.encoder.config.hidden_size
        logger.success(f"CLIP text encoder loaded. Hidden dim: {self.hidden_dim}")

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        ).to(self.device)
        outputs = self.encoder(**inputs)
        return outputs.last_hidden_state.float()

    @property
    def embedding_dim(self) -> int:
        return self.hidden_dim


def build_encoder(encoder_type: str, max_length: int, device: str, model_name_or_path=None) -> nn.Module:
    """工厂函数：根据类型构建文本编码器。"""
    if encoder_type == "qwen2.5-0.5b":
        return Qwen25TextEncoder(model_name_or_path=model_name_or_path, max_length=max_length, device=device)
    elif encoder_type in ("t5-base", "t5-small"):
        return LiteT5TextEncoder(model_name=encoder_type, max_length=max_length, device=device)
    elif encoder_type == "clip":
        return LiteCLIPTextEncoder(device=device)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# ============================================================
# 预计算主流程
# ============================================================

def precompute_embeddings(args):
    """遍历 AIGC 数据集，为每个样本预计算文本 embedding 并保存。"""
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 构建编码器
    encoder = build_encoder(args.encoder_type, args.max_length, args.device, args.model_name_or_path)

    # 保存编码器元数据
    import json
    meta = {
        "encoder_type": args.encoder_type,
        "embedding_dim": encoder.embedding_dim,
        "max_length": args.max_length,
    }
    with open(output_dir / "encoder_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Encoder meta: {meta}")

    # 收集所有样本
    sample_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    logger.info(f"Found {len(sample_dirs)} samples in {data_root}")

    # 收集所有文本（批量处理更高效）
    all_texts = []
    all_names = []
    for sample_dir in sample_dirs:
        instruction_file = sample_dir / "instruction.txt"
        if not instruction_file.exists():
            continue
        out_file = output_dir / f"{sample_dir.name}.pt"
        if out_file.exists():
            continue  # 已缓存，跳过
        all_texts.append(instruction_file.read_text().strip())
        all_names.append(sample_dir.name)

    if not all_texts:
        logger.info("All embeddings already computed!")
        return

    logger.info(f"Computing embeddings for {len(all_texts)} samples...")

    # 批量编码
    for i in tqdm(range(0, len(all_texts), args.batch_size), desc="Encoding"):
        batch_texts = all_texts[i : i + args.batch_size]
        batch_names = all_names[i : i + args.batch_size]

        embeddings = encoder.encode(batch_texts)  # (B, seq_len, dim)

        for j, (name, text) in enumerate(zip(batch_names, batch_texts)):
            torch.save(
                {
                    "text": text,
                    "embedding": embeddings[j].unsqueeze(0).cpu(),  # (1, seq_len, dim)
                    "sample": name,
                    "encoder_type": args.encoder_type,
                    "embedding_dim": encoder.embedding_dim,
                },
                output_dir / f"{name}.pt",
            )

    logger.success(
        f"Done! {len(all_texts)} embeddings saved to {output_dir}\n"
        f"  Encoder: {args.encoder_type}, dim={encoder.embedding_dim}, seq_len={args.max_length}"
    )


if __name__ == "__main__":
    args = parse_args()
    precompute_embeddings(args)
