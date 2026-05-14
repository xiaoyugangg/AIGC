"""
Prepare Custom Dataset for LoRA Fine-Tuning
===========================================
This script reformats all video/text pairs in the train directory into a dataset folder
that the official VideoDataset expects:
datasets/custom_lora/
├── metas/
│   ├── 0000.txt
│   ├── 0001.txt
│   └── ...
└── videos/
    ├── 0000.mp4
    ├── 0001.mp4
    └── ...
"""

import os
import shutil
import glob

def main():
    train_dir = "/root/autodl-tmp/AIGC/release/train"
    out_dir = "/root/autodl-tmp/datasets/custom_lora"
    
    os.makedirs(os.path.join(out_dir, "videos"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "metas"), exist_ok=True)

    # Find all subdirectories in the train directory (e.g., 1_1, 1_2, etc.)
    subdirs = sorted([d for d in glob.glob(os.path.join(train_dir, "*")) if os.path.isdir(d)])
    
    if not subdirs:
        print(f"Error: No subdirectories found in {train_dir}")
        return

    print(f"Found {len(subdirs)} subdirectories in {train_dir}")
    
    success_count = 0
    for i, subdir in enumerate(subdirs):
        src_video = os.path.join(subdir, "video.mp4")
        src_text = os.path.join(subdir, "instruction.txt")
        
        if not os.path.exists(src_video) or not os.path.exists(src_text):
            print(f"Warning: Skipping {subdir} because it is missing video.mp4 or instruction.txt")
            continue
            
        dst_video = os.path.join(out_dir, "videos", f"{i:04d}.mp4")
        dst_text = os.path.join(out_dir, "metas", f"{i:04d}.txt")
        
        shutil.copy(src_video, dst_video)
        shutil.copy(src_text, dst_text)
        success_count += 1
        
    print(f"Successfully processed {success_count} video/text pairs.")
    print(f"Dataset is ready at: {out_dir}")

if __name__ == "__main__":
    main()
