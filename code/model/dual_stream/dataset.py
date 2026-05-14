"""
AIGC 数据集加载器。

数据集结构：
  AIGC/release/train/{scene_id}/
    ├── video.mp4        (机械臂操作视频)
    ├── instruction.txt  (自然语言指令)
    ├── action.txt       (26维动作序列 CSV，~96帧)
    └── joint.txt        (关节空间数据)

本模块提供：
  1. AigcDataset — 原始帧+动作对，用于双流模型训练
  2. AigcDatasetWithEmbedding — 使用预计算的文本 embedding 版本
"""

import os
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

try:
    import mediapy
    HAS_MEDIAPY = True
except ImportError:
    HAS_MEDIAPY = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ============================================================
# 数据读取工具
# ============================================================

def read_video_frames(video_path: str, num_frames: Optional[int] = None) -> np.ndarray:
    """
    读取视频帧，返回 (T, H, W, C) uint8 数组。
    优先使用 cv2（不依赖系统 ffmpeg），其次使用 mediapy。
    """
    # 优先用 cv2（不需要系统 ffmpeg）
    if HAS_CV2:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cv2 cannot open video: {video_path}")
        frame_list = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_list.append(frame_rgb)
        cap.release()
        frames = np.stack(frame_list, axis=0)  # (T, H, W, C)
        if num_frames is not None and len(frames) > num_frames:
            indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
            frames = frames[indices]
        return frames

    # 备选：mediapy（需要系统 ffmpeg）
    if HAS_MEDIAPY:
        try:
            frames = mediapy.read_video(str(video_path))
            if num_frames is not None and len(frames) > num_frames:
                indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
                frames = frames[indices]
            return np.array(frames)
        except Exception as e:
            raise RuntimeError(
                f"mediapy failed to read video (ffmpeg may be missing): {e}\n"
                "Fix: pip install opencv-python-headless"
            ) from e

    raise RuntimeError(
        "No video reader available. Install one of:\n"
        "  pip install opencv-python-headless   (recommended, no system ffmpeg needed)\n"
        "  apt install ffmpeg && pip install mediapy"
    )



def read_action_csv(action_path: str) -> np.ndarray:
    """
    读取 action.txt（CSV 格式），返回 (T, 26) float32 数组。
    
    列：idx13~idx26（双臂 14 关节）+ 双手 12 手指位置
    """
    df = pd.read_csv(str(action_path), index_col=0)
    return df.values.astype(np.float32)  # (T, 26)


def normalize_actions(actions: np.ndarray, stats: Optional[dict] = None) -> tuple[np.ndarray, dict]:
    """
    对动作序列进行归一化（zero-mean unit-variance per dimension）。
    
    Args:
        actions: (T, 26) 原始动作值
        stats: 预计算的统计量 {'mean': ..., 'std': ...}，None 时从当前批计算
    
    Returns:
        normalized_actions: (T, 26)
        stats: {'mean': (26,), 'std': (26,)}
    """
    if stats is None:
        mean = actions.mean(axis=0)
        std = actions.std(axis=0) + 1e-8
        stats = {"mean": mean, "std": std}
    else:
        mean = stats["mean"]
        std = stats["std"]
    return (actions - mean) / std, stats


# ============================================================
# Dataset 实现
# ============================================================

class AigcDataset(Dataset):
    """
    AIGC 具身智能数据集。

    每个样本返回：
      frames: (C, T_cond, H, W) — 条件帧，uint8→float [0,1]
      text: str — 指令字符串
      actions: (T_action, 26) — 后续动作序列，归一化后

    Args:
        data_root: 数据集根目录（如 AIGC/release/train）
        num_cond_frames: 视频条件帧数（输入给视频分支）
        num_action_steps: 动作序列长度（输出给动作分支）
        frame_skip: 帧跳过（=1 表示不跳帧）
        image_size: (H, W) 目标分辨率
        action_stats: 预计算的动作归一化统计量（None 时不归一化）
        transform: 额外的图像变换
    """

    def __init__(
        self,
        data_root: str,
        num_cond_frames: int = 16,
        num_action_steps: int = 50,
        frame_skip: int = 1,
        image_size: tuple[int, int] = (192, 320),
        action_stats: Optional[dict] = None,
        transform: Optional[Callable] = None,
    ):
        self.data_root = Path(data_root)
        self.num_cond_frames = num_cond_frames
        self.num_action_steps = num_action_steps
        self.frame_skip = frame_skip
        self.image_size = image_size
        self.action_stats = action_stats
        self.transform = transform

        # 收集所有有效的样本目录
        self.samples = []
        for d in sorted(self.data_root.iterdir()):
            if not d.is_dir():
                continue
            video = d / "video.mp4"
            instruction = d / "instruction.txt"
            action = d / "action.txt"
            if video.exists() and instruction.exists() and action.exists():
                self.samples.append(d)

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found in {data_root}")

        # 计算全局动作统计量（可选）
        if action_stats is None:
            self.action_stats = self._compute_action_stats()
        else:
            self.action_stats = action_stats

    def _compute_action_stats(self) -> dict:
        """计算全数据集的动作均值和标准差（用于归一化）。"""
        all_actions = []
        for sample_dir in self.samples:
            actions = read_action_csv(str(sample_dir / "action.txt"))
            all_actions.append(actions)
        all_actions = np.concatenate(all_actions, axis=0)  # (N_total, 26)
        mean = all_actions.mean(axis=0)
        std = all_actions.std(axis=0) + 1e-8
        return {"mean": mean, "std": std}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample_dir = self.samples[idx]

        # ---- 读取视频帧 ----
        total_needed = self.num_cond_frames * self.frame_skip
        frames = read_video_frames(
            str(sample_dir / "video.mp4"),
            num_frames=total_needed,
        )  # (T_raw, H, W, C)

        # 如果视频帧数不够，重复最后一帧填充
        if len(frames) < total_needed:
            pad_len = total_needed - len(frames)
            frames = np.concatenate([frames, np.repeat(frames[-1:], pad_len, axis=0)], axis=0)

        # 均匀采样条件帧
        cond_indices = np.linspace(0, total_needed - 1, self.num_cond_frames, dtype=int)
        cond_frames = frames[cond_indices]  # (T_cond, H, W, C)

        # 转为 tensor (C, T_cond, H, W) float [0, 1]
        frames_tensor = torch.from_numpy(cond_frames).permute(3, 0, 1, 2).float() / 255.0

        # resize 到目标分辨率
        if self.image_size is not None:
            H, W = self.image_size
            # frames_tensor: (C, T, H_orig, W_orig)
            C, T, H_orig, W_orig = frames_tensor.shape
            frames_flat = frames_tensor.view(C, T, H_orig, W_orig)
            # 逐帧 resize
            resized_list = []
            for t in range(T):
                f = frames_flat[:, t, :, :]  # (C, H, W)
                f_resized = TF.resize(f, [H, W], antialias=True)
                resized_list.append(f_resized)
            frames_tensor = torch.stack(resized_list, dim=1)  # (C, T, H, W)

        if self.transform is not None:
            frames_tensor = self.transform(frames_tensor)

        # ---- 读取指令 ----
        text = (sample_dir / "instruction.txt").read_text().strip()

        # ---- 读取动作序列 ----
        actions = read_action_csv(str(sample_dir / "action.txt"))  # (T_action_raw, 26)

        # 归一化
        mean = self.action_stats["mean"]  # (26,)
        std = self.action_stats["std"]    # (26,)
        actions_norm = (actions - mean) / std  # (T_action_raw, 26)

        # 截取目标步数
        if len(actions_norm) >= self.num_action_steps:
            actions_norm = actions_norm[:self.num_action_steps]
        else:
            # 填充到 num_action_steps
            pad_len = self.num_action_steps - len(actions_norm)
            actions_norm = np.concatenate(
                [actions_norm, np.repeat(actions_norm[-1:], pad_len, axis=0)], axis=0
            )

        actions_tensor = torch.from_numpy(actions_norm).float()  # (num_action_steps, 26)

        return {
            "frames": frames_tensor,        # (C=3, T_cond, H, W) float [0,1]
            "text": text,                   # str
            "actions": actions_tensor,      # (num_action_steps, 26) normalized
            "sample_name": sample_dir.name, # str, for debugging
        }


class AigcDatasetWithEmbedding(AigcDataset):
    """
    带预计算文本 embedding 的 AIGC 数据集。
    需要先运行 step2_prepare_embeddings.py。
    """

    def __init__(self, embedding_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dir = Path(embedding_dir)

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        sample_name = item["sample_name"]

        # 加载预计算的 embedding
        emb_file = self.embedding_dir / f"{sample_name}.pt"
        if emb_file.exists():
            emb_data = torch.load(emb_file, weights_only=True)
            # embedding shape: (1, seq_len, dim)
            item["text_embedding"] = emb_data["embedding"].squeeze(0)  # (seq_len, dim)
        else:
            # 如果没有预计算，返回 None（训练时在线计算）
            item["text_embedding"] = None

        return item


# ============================================================
# 数据集工具函数
# ============================================================

def get_aigc_dataloader(
    data_root: str,
    batch_size: int = 4,
    num_workers: int = 4,
    num_cond_frames: int = 16,
    num_action_steps: int = 50,
    image_size: tuple[int, int] = (192, 320),
    embedding_dir: Optional[str] = None,
    shuffle: bool = True,
) -> DataLoader:
    """创建 AIGC 数据集的 DataLoader。"""
    if embedding_dir is not None:
        dataset = AigcDatasetWithEmbedding(
            data_root=data_root,
            embedding_dir=embedding_dir,
            num_cond_frames=num_cond_frames,
            num_action_steps=num_action_steps,
            image_size=image_size,
        )
    else:
        dataset = AigcDataset(
            data_root=data_root,
            num_cond_frames=num_cond_frames,
            num_action_steps=num_action_steps,
            image_size=image_size,
        )

    def collate_fn(batch):
        """自定义 collate，处理可能为 None 的 text_embedding。"""
        frames = torch.stack([b["frames"] for b in batch], dim=0)
        texts = [b["text"] for b in batch]
        actions = torch.stack([b["actions"] for b in batch], dim=0)
        sample_names = [b["sample_name"] for b in batch]

        result = {
            "frames": frames,
            "text": texts,
            "actions": actions,
            "sample_name": sample_names,
        }

        if "text_embedding" in batch[0] and batch[0]["text_embedding"] is not None:
            result["text_embedding"] = torch.stack([b["text_embedding"] for b in batch], dim=0)

        return result

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )


# ============================================================
# 快速测试
# ============================================================

if __name__ == "__main__":
    import sys

    data_root = str(Path(__file__).parent.parent.parent / "AIGC/release/train")
    if not Path(data_root).exists():
        print(f"Data root not found: {data_root}")
        sys.exit(1)

    print(f"Loading dataset from: {data_root}")
    dataset = AigcDataset(
        data_root=data_root,
        num_cond_frames=16,
        num_action_steps=50,
        image_size=(192, 320),
    )
    print(f"Dataset size: {len(dataset)}")
    print(f"Action stats: mean={dataset.action_stats['mean'][:5]}...")

    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"  frames: {sample['frames'].shape} {sample['frames'].dtype}")
    print(f"  text: '{sample['text']}'")
    print(f"  actions: {sample['actions'].shape} {sample['actions'].dtype}")
    print(f"  actions range: [{sample['actions'].min():.3f}, {sample['actions'].max():.3f}]")
    print("Dataset test passed ✅")
