"""
视觉桥接模块（Visual Bridge）。

功能：
  在视频扩散分支完成一步去噪后，拦截预测出的干净隐变量 z0，
  利用自适应 3D 池化技术将其在时空维度上压缩为紧凑全局特征。

架构设计：
  输入: z0 形状 (B, C, T', H', W')
    - 典型值: C=16, T'≈23, H'≈24, W'≈40 (192×320 分辨率)
  
  压缩方式:
    1. AdaptiveAvgPool3d((t_out, h_out, w_out)) → (B, C, t_out, h_out, w_out)
    2. Flatten → (B, C * t_out * h_out * w_out)
    3. 线性投影 → (B, visual_dim)
  
  输出维度:
    - visual_dim 即用户指定的"紧凑全局特征"维度
    - 论文中提到 132 维 = 需要选择合适的池化输出尺寸
    
  常见的 132 维组合示例：
    - C=16, pool_out=(1,1,1) → 16 dim (太少)
    - C=16, pool_out=(1,2,4) → 128 dim → proj to 132 ✓
    - C=16, pool_out=(2,2,4) → 256 dim → proj to 132 ✓
    - 使用线性层最终投影到所需维度
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class VisualBridge(nn.Module):
    """
    将视频扩散隐变量 z0 压缩为动作生成分支的视觉条件特征。

    Args:
        in_channels: z0 的通道数（VAE 隐空间维度，Cosmos-Predict2.5 中为 16）
        pool_out_size: 3D 池化输出尺寸 (T', H', W')
        visual_dim: 最终输出特征维度（论文中提到 132）
        use_layer_norm: 是否对输出进行 LayerNorm
    """

    def __init__(
        self,
        in_channels: int = 16,
        pool_out_size: Tuple[int, int, int] = (1, 2, 4),  # 16*1*2*4 = 128, 再线性到 visual_dim
        visual_dim: int = 132,
        use_layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.pool_out_size = pool_out_size
        self.visual_dim = visual_dim

        # 自适应 3D 平均池化
        self.pool = nn.AdaptiveAvgPool3d(pool_out_size)

        # 计算池化后的扁平维度
        t_out, h_out, w_out = pool_out_size
        flat_dim = in_channels * t_out * h_out * w_out

        # 线性投影到目标维度
        self.proj = nn.Sequential(
            nn.Linear(flat_dim, visual_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(visual_dim * 2, visual_dim),
        )

        # 可选的 LayerNorm
        self.norm = nn.LayerNorm(visual_dim) if use_layer_norm else nn.Identity()

    def forward(self, z0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z0: (B, C, T', H', W') — 视频扩散分支预测的干净隐变量

        Returns:
            visual_cond: (B, visual_dim) — 压缩后的视觉条件特征
        """
        # 3D 自适应平均池化
        pooled = self.pool(z0)  # (B, C, t_out, h_out, w_out)

        # 展平
        B = pooled.shape[0]
        flat = pooled.reshape(B, -1)  # (B, C * t_out * h_out * w_out)

        # 线性投影
        out = self.proj(flat)  # (B, visual_dim)

        # 归一化
        out = self.norm(out)  # (B, visual_dim)

        return out


class VisualBridgeWithAttention(nn.Module):
    """
    增强版视觉桥接模块，在池化前加入时序注意力以更好地聚合时空信息。

    适合在第三步（完整双流架构）中使用，第二步可先用简单版 VisualBridge。
    """

    def __init__(
        self,
        in_channels: int = 16,
        spatial_pool_size: Tuple[int, int] = (4, 7),  # H', W' 池化目标
        num_heads: int = 4,
        visual_dim: int = 132,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.visual_dim = visual_dim

        # 空间池化
        self.spatial_pool = nn.AdaptiveAvgPool2d(spatial_pool_size)  # 应用到每帧

        # 时序 self-attention（将时间序列当作 sequence）
        token_dim = in_channels * spatial_pool_size[0] * spatial_pool_size[1]
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.temporal_norm = nn.LayerNorm(token_dim)

        # 聚合后投影
        self.proj = nn.Sequential(
            nn.Linear(token_dim, visual_dim * 2),
            nn.SiLU(),
            nn.Linear(visual_dim * 2, visual_dim),
        )
        self.out_norm = nn.LayerNorm(visual_dim) if use_layer_norm else nn.Identity()

    def forward(self, z0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z0: (B, C, T', H', W')

        Returns:
            visual_cond: (B, visual_dim)
        """
        B, C, T, H, W = z0.shape

        # 空间池化: 将每帧 (C, H, W) → (C, h_p, w_p)
        z0_2d = z0.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        pooled_2d = self.spatial_pool(z0_2d)  # (B*T, C, h_p, w_p)
        h_p, w_p = pooled_2d.shape[2], pooled_2d.shape[3]
        pooled_tokens = pooled_2d.reshape(B, T, C * h_p * w_p)  # (B, T, token_dim)

        # 时序注意力
        attended, _ = self.temporal_attention(pooled_tokens, pooled_tokens, pooled_tokens)
        attended = self.temporal_norm(pooled_tokens + attended)  # residual

        # 时间维度聚合（mean pooling over T）
        aggregated = attended.mean(dim=1)  # (B, token_dim)

        # 投影
        out = self.proj(aggregated)  # (B, visual_dim)
        out = self.out_norm(out)

        return out


def build_visual_bridge(
    bridge_type: str = "simple",
    in_channels: int = 16,
    visual_dim: int = 132,
    **kwargs,
) -> nn.Module:
    """
    工厂函数：根据类型构建视觉桥接模块。

    Args:
        bridge_type: "simple" 或 "attention"
        in_channels: 输入通道数
        visual_dim: 输出特征维度
    """
    if bridge_type == "simple":
        # 计算合适的池化输出尺寸
        # 目标：in_channels * prod(pool_size) ≈ visual_dim（线性投影会调整）
        pool_out_size = kwargs.pop("pool_out_size", (1, 2, 4))
        return VisualBridge(
            in_channels=in_channels,
            pool_out_size=pool_out_size,
            visual_dim=visual_dim,
            **kwargs,
        )
    elif bridge_type == "attention":
        return VisualBridgeWithAttention(
            in_channels=in_channels,
            visual_dim=visual_dim,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown bridge type: {bridge_type}")


# ============================================================
# 形状验证工具
# ============================================================

def verify_bridge_shapes(bridge: nn.Module, device: str = "cpu"):
    """
    验证 VisualBridge 的输入输出形状是否正确。
    使用 Cosmos-Predict2.5 192×320 分辨率的典型隐变量尺寸。
    """
    print("=== VisualBridge Shape Verification ===")

    # Cosmos 2B 的 VAE tokenizer 参数：
    # spatial_compression = 8 → H/8, W/8
    # temporal_compression = 4 → (T-1)/4 + 1
    # 77 pixel frames → (77-1)/4 + 1 = 20 latent frames
    # 192×320 pixel → 24×40 latent spatial
    B = 2
    C = 16   # Cosmos VAE 隐空间通道数
    T_lat = 20
    H_lat = 24
    W_lat = 40

    z0 = torch.randn(B, C, T_lat, H_lat, W_lat, device=device)
    print(f"Input z0 shape: {z0.shape}")

    bridge = bridge.to(device)
    bridge.eval()

    with torch.no_grad():
        visual_cond = bridge(z0)

    print(f"Output visual_cond shape: {visual_cond.shape}")
    assert visual_cond.shape == (B, bridge.visual_dim), (
        f"Expected ({B}, {bridge.visual_dim}), got {visual_cond.shape}"
    )
    print(f"✅ Shape check passed: {visual_cond.shape}")
    return visual_cond


if __name__ == "__main__":
    # 测试简单版
    bridge_simple = build_visual_bridge(
        bridge_type="simple",
        in_channels=16,
        visual_dim=132,
        pool_out_size=(1, 2, 4),
    )
    verify_bridge_shapes(bridge_simple)

    n_params = sum(p.numel() for p in bridge_simple.parameters())
    print(f"Simple bridge parameters: {n_params:,}")

    # 测试注意力版
    bridge_attn = build_visual_bridge(
        bridge_type="attention",
        in_channels=16,
        visual_dim=132,
    )
    verify_bridge_shapes(bridge_attn)

    n_params = sum(p.numel() for p in bridge_attn.parameters())
    print(f"Attention bridge parameters: {n_params:,}")
