"""
预计算 Reason 1.1 7B Text Embeddings (防止训练时 OOM)

用法:
  cd /root/autodl-tmp
  python scripts/get_reason_embeddings.py --dataset_dir datasets/custom_lora
"""
import argparse
import os
import torch
import numpy as np
from pathlib import Path
from loguru import logger

import sys
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "cosmos-predict2.5-1.5.0"))

from cosmos_predict2._src.reason1.tokenizer.processor import build_tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="datasets/custom_lora")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    args = parser.parse_args()

    metas_dir = os.path.join(args.dataset_dir, "metas")
    out_dir = os.path.join(args.dataset_dir, "text_embeddings")
    os.makedirs(out_dir, exist_ok=True)

    meta_files = [f for f in sorted(os.listdir(metas_dir)) if f.endswith(".txt")]
    
    logger.info(f"Loading Reason 1.1 7B Text Encoder ({args.model_name})...")
    # Using transformers AutoModelForCausalLM to get embeddings
    from transformers import AutoModelForCausalLM, AutoProcessor
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # The official repo downloads nvidia/Cosmos-Reason1.1-7B, we should use that
    model_id = "nvidia/Cosmos-Reason1.1-7B"
    
    # In offline mode, HuggingFace Hub cannot resolve the path properly without network.
    # We dynamically search for the downloaded Reason model in the cache directories.
    import glob
    possible_caches = [
        "/root/autodl-tmp/.hf_cache/hub",
        os.path.expanduser("~/.cache/huggingface/hub"),
        "/root/.cache/huggingface/hub"
    ]
    model_path = model_id
    for cache in possible_caches:
        pattern = os.path.join(cache, "models--nvidia--Cosmos-Reason1*", "snapshots", "*")
        matches = glob.glob(pattern)
        if matches:
            model_path = matches[0]
            logger.success(f"Found offline model at: {model_path}")
            break
            
    if model_path == model_id:
        logger.warning(f"Could not find offline cache for {model_id}. Attempting fallback.")
        
    logger.info(f"Using offline model path: {model_path}")
    
    # We must use the exact TextEncoder implementation from cosmos_predict2 because 
    # it applies specific chat templates, padding (512 length), and per-layer mean normalization 
    # before concatenating all 28 layers (dim=100352). Standard transformers won't match.
    from cosmos_predict2._src.predict2.text_encoders.text_encoder import TextEncoder, TextEncoderConfig
    
    config = TextEncoderConfig(
        compute_online=True,
        embedding_concat_strategy="full_concat",
        ckpt_path=model_path, # Provide the local physical path here
    )
    
    logger.info("Initializing official Cosmos TextEncoder...")
    
    # The TextEncoder internally uses get_checkpoint_path() to load the tokenizer and weights.
    # Because we are not running through the full Hydra config, the internal UUID mappings 
    # are missing and it will crash. We monkeypatch it to force it to use our offline model_path.
    import cosmos_predict2._src.imaginaire.utils.checkpoint_db as checkpoint_db
    checkpoint_db.get_checkpoint_path = lambda x: model_path
    checkpoint_db.download_checkpoint = lambda *args, **kwargs: model_path
    
    encoder = TextEncoder(config, device=device)
    
    logger.info("Starting embedding generation...")
    for filename in meta_files:
        basename = os.path.splitext(filename)[0]
        out_path = os.path.join(out_dir, f"{basename}.pt")
        
        if os.path.exists(out_path):
            continue
            
        with open(os.path.join(metas_dir, filename), "r", encoding="utf-8") as f:
            text = f.read().strip()
            
        # TextEncoder expects a data_batch dict and a key
        data_batch = {"ai_caption": [text]}
        with torch.no_grad():
            embedding = encoder.compute_text_embeddings_online(data_batch, "ai_caption")
            
        # The embedding returned is shape (1, 512, 100352)
        # Squeeze batch dim to save as (512, 100352) to match VideoDataset expectations
        embedding = embedding[0].cpu().to(torch.bfloat16)
        
        torch.save(embedding, out_path)
        logger.success(f"Saved {basename}.pt, shape: {embedding.shape}")
            


if __name__ == "__main__":
    main()
