#!/usr/bin/env python3
"""
One-shot diagnostic for text-conditioning path in Cosmos-Predict2.5.

What this checks in one run:
1) Offline embedding (.pt) sanity (shape/range/std/nan/inf).
2) Online embedding (model-native encoder) sanity for the same caption.
3) Offline vs online similarity.
4) crossattn projection weight stats from the loaded pretrained checkpoint.
5) Projection output stats for offline/online embeddings.

If projection output is near-constant (std ~ 0), this script flags checkpoint-side
conditioning collapse, which training steps alone cannot fix.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def tensor_stats(x: torch.Tensor) -> Dict[str, float | bool | list[int] | str]:
    xf = x.float()
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "mean": float(xf.mean().item()),
        "std": float(xf.std().item()),
        "min": float(xf.min().item()),
        "max": float(xf.max().item()),
        "has_nan": bool(torch.isnan(xf).any().item()),
        "has_inf": bool(torch.isinf(xf).any().item()),
    }


def flatten_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.float().reshape(-1)
    bb = b.float().reshape(-1)
    return float(F.cosine_similarity(aa, bb, dim=0).item())


def load_model_state_dict(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise RuntimeError(f"Unsupported checkpoint object type: {type(obj)}")


def find_crossattn_proj(sd: Dict[str, torch.Tensor]) -> Tuple[str, str, torch.Tensor, torch.Tensor]:
    w_key = None
    b_key = None
    for k in sd.keys():
        if k.endswith("crossattn_proj.0.weight"):
            w_key = k
            break
    if w_key is None:
        raise KeyError("Cannot find key ending with crossattn_proj.0.weight in checkpoint.")
    b_candidate = w_key[:-6] + "bias"
    if b_candidate not in sd:
        raise KeyError(f"Cannot find matching bias key: {b_candidate}")
    b_key = b_candidate
    return w_key, b_key, sd[w_key], sd[b_key]


def build_online_text_encoder(repo_root: Path):
    import sys

    sys.path.insert(0, str(repo_root / "cosmos-predict2.5-1.5.0"))
    from cosmos_predict2._src.predict2.text_encoders.text_encoder import TextEncoder, TextEncoderConfig
    import cosmos_predict2._src.imaginaire.utils.checkpoint_db as checkpoint_db

    model_id = "nvidia/Cosmos-Reason1.1-7B"
    possible_caches = [
        "/root/autodl-tmp/.hf_cache/hub",
        os.path.expanduser("~/.cache/huggingface/hub"),
        "/root/.cache/huggingface/hub",
    ]
    model_path = model_id
    for cache in possible_caches:
        cache_path = Path(cache)
        if not cache_path.exists():
            continue
        matches = list(cache_path.glob("models--nvidia--Cosmos-Reason1*/snapshots/*"))
        if matches:
            model_path = str(matches[0])
            break

    # Force TextEncoder to use local snapshot path.
    checkpoint_db.get_checkpoint_path = lambda _: model_path
    checkpoint_db.download_checkpoint = lambda *args, **kwargs: model_path

    cfg = TextEncoderConfig(
        compute_online=True,
        embedding_concat_strategy="full_concat",
        ckpt_path=model_path,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = TextEncoder(cfg, device=device)
    return encoder, device, model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=str, default="/root/autodl-tmp")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/root/autodl-tmp/datasets/custom_lora",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/root/autodl-tmp/cosmos-predict2.5/base/pre-trained/d20b7120-df3e-4911-919d-db6e08bad31c_ema_bf16.pt",
    )
    parser.add_argument("--sample_id", type=str, default="", help="e.g. 0001, 0123; empty=auto first")
    parser.add_argument(
        "--report_json",
        type=str,
        default="/root/autodl-tmp/outputs/text_condition_check_report.json",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    dataset_dir = Path(args.dataset_dir)
    metas_dir = dataset_dir / "metas"
    emb_dir = dataset_dir / "text_embeddings"
    ckpt_path = Path(args.ckpt_path)

    if args.sample_id:
        base = args.sample_id
        meta_path = metas_dir / f"{base}.txt"
        emb_path = emb_dir / f"{base}.pt"
    else:
        meta_candidates = sorted(metas_dir.glob("*.txt"))
        if not meta_candidates:
            raise FileNotFoundError(f"No meta files in {metas_dir}")
        meta_path = meta_candidates[0]
        base = meta_path.stem
        emb_path = emb_dir / f"{base}.pt"

    if not emb_path.exists():
        raise FileNotFoundError(f"Offline embedding missing: {emb_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint missing: {ckpt_path}")

    caption = meta_path.read_text(encoding="utf-8").strip()
    emb_offline = torch.load(emb_path, map_location="cpu")
    if emb_offline.ndim != 2:
        raise RuntimeError(f"Expected offline embedding shape [L, D], got {list(emb_offline.shape)}")
    emb_offline_b = emb_offline.unsqueeze(0)  # [1, L, D]

    encoder, device, reason_model_path = build_online_text_encoder(repo_root)
    with torch.no_grad():
        emb_online_b = encoder.compute_text_embeddings_online({"ai_caption": [caption]}, "ai_caption").detach().cpu()

    sd = load_model_state_dict(ckpt_path)
    w_key, b_key, w, b = find_crossattn_proj(sd)
    w_f = w.float()
    b_f = b.float()

    proj_offline = F.linear(emb_offline_b.float(), w_f, b_f)
    proj_online = F.linear(emb_online_b.float(), w_f, b_f)

    report = {
        "sample_id": base,
        "meta_path": str(meta_path),
        "embedding_path": str(emb_path),
        "checkpoint_path": str(ckpt_path),
        "reason_model_path": reason_model_path,
        "caption_preview": caption[:120],
        "offline_embedding_stats": tensor_stats(emb_offline_b),
        "online_embedding_stats": tensor_stats(emb_online_b),
        "offline_vs_online": {
            "mse": float(torch.mean((emb_offline_b.float() - emb_online_b.float()) ** 2).item()),
            "cosine": flatten_cosine(emb_offline_b, emb_online_b),
        },
        "crossattn_proj": {
            "weight_key": w_key,
            "bias_key": b_key,
            "weight_stats": tensor_stats(w_f),
            "bias_stats": tensor_stats(b_f),
        },
        "projected_offline_stats": tensor_stats(proj_offline),
        "projected_online_stats": tensor_stats(proj_online),
        "projected_offline_vs_online": {
            "mse": float(torch.mean((proj_offline - proj_online) ** 2).item()),
            "cosine": flatten_cosine(proj_offline, proj_online),
        },
    }

    # High-level diagnosis
    proj_std = report["projected_offline_stats"]["std"]
    w_std = report["crossattn_proj"]["weight_stats"]["std"]
    emb_std = report["offline_embedding_stats"]["std"]
    report["diagnosis"] = {
        "offline_embedding_looks_valid": bool(emb_std > 1e-4),
        "projection_weight_looks_collapsed": bool(w_std < 1e-8),
        "projection_output_looks_collapsed": bool(proj_std < 1e-8),
        "recommendation": (
            "Projection collapsed: prioritize fixing/loading crossattn projection weights; "
            "switching online/offline text alone will not help."
            if proj_std < 1e-8
            else "Projection not collapsed: compare offline vs online embedding mismatch and retry training."
        ),
    }

    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== One-Shot Text Conditioning Check ===")
    print(f"sample_id: {base}")
    print(f"offline std: {report['offline_embedding_stats']['std']:.6f}")
    print(f"online  std: {report['online_embedding_stats']['std']:.6f}")
    print(f"offline vs online mse: {report['offline_vs_online']['mse']:.6f}")
    print(f"crossattn_proj weight std: {report['crossattn_proj']['weight_stats']['std']:.12f}")
    print(f"projected offline std: {report['projected_offline_stats']['std']:.12f}")
    print(f"projected online  std: {report['projected_online_stats']['std']:.12f}")
    print(f"diagnosis: {report['diagnosis']['recommendation']}")
    print(f"report_json: {report_path}")
    print("\nIf you want to force native online text in training for A/B:")
    print(
        "torchrun ... -- experiment=predict2_video2world_training_2b_custom_lora "
        "model.config.text_encoder_config.compute_online=True"
    )


if __name__ == "__main__":
    main()

