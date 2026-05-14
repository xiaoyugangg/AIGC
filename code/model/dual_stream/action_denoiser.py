"""
1D U-Net 动作去噪器（Action Denoiser）。

功能：
  作为扩散模型的去噪网络，接收：
    1. 带噪动作序列 x_t: (B, T_act, action_dim)
    2. 噪声水平 sigma/timestep: (B,)
    3. 视觉条件 visual_cond: (B, visual_dim)  ← 来自视觉桥接模块
    4. 文本条件 text_cond: (B, seq_len, text_dim)  ← 可选

  输出干净动作预测 x0: (B, T_act, action_dim)

架构：
  - 1D U-Net，包含下采样和上采样路径
  - 通过 AdaLN（自适应层归一化）注入时间步 + 视觉条件
  - 通过 Cross-Attention 注入文本条件（可选）
  - 残差连接用于稳定训练
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 时间步编码
# ============================================================

class SinusoidalPositionEmbedding(nn.Module):
    """正弦位置编码，用于时间步嵌入。"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) 时间步或 sigma 值

        Returns:
            emb: (B, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    """将 sigma 映射到固定维度的条件向量。"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.sin_emb = SinusoidalPositionEmbedding(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.SiLU(),
            nn.Linear(out_dim * 2, out_dim),
        )

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sigma: (B,) 噪声水平

        Returns:
            emb: (B, out_dim)
        """
        # log-sigma → sinusoidal
        log_sigma = sigma.log().clamp(-20, 20)
        sin_emb = self.sin_emb(log_sigma)
        return self.mlp(sin_emb)


# ============================================================
# 自适应层归一化（AdaLN）
# ============================================================

class AdaLN(nn.Module):
    """
    自适应层归一化：
    AdaLN(x, cond) = scale(cond) * LN(x) + shift(cond)
    
    cond 包含时间步 + 视觉条件的联合向量。
    """

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, channels * 2)  # scale + shift

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, channels)
            cond: (B, cond_dim)

        Returns:
            x_normed: (B, T, channels)
        """
        scale_shift = self.proj(cond)  # (B, channels*2)
        scale, shift = scale_shift.chunk(2, dim=-1)  # each (B, channels)
        scale = scale.unsqueeze(1)  # (B, 1, channels)
        shift = shift.unsqueeze(1)  # (B, 1, channels)
        return self.norm(x) * (1 + scale) + shift


# ============================================================
# 1D U-Net 基础模块
# ============================================================

class ResBlock1D(nn.Module):
    """
    1D 残差块，带 AdaLN 条件注入。
    
    结构：
      x → AdaLN → Conv1D → SiLU → Dropout → Conv1D → + (skip)
    """

    def __init__(
        self,
        channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        pad = kernel_size // 2
        self.adaLN1 = AdaLN(channels, cond_dim)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.adaLN2 = AdaLN(channels, cond_dim)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, channels, T)
            cond: (B, cond_dim)

        Returns:
            x: (B, channels, T)
        """
        residual = x

        # Block 1
        x_t = x.permute(0, 2, 1)  # (B, T, channels)
        x_t = self.adaLN1(x_t, cond)
        x = x_t.permute(0, 2, 1)  # (B, channels, T)
        x = self.act(self.conv1(x))
        x = self.dropout(x)

        # Block 2
        x_t = x.permute(0, 2, 1)
        x_t = self.adaLN2(x_t, cond)
        x = x_t.permute(0, 2, 1)
        x = self.conv2(x)

        return x + residual


class CrossAttention1D(nn.Module):
    """
    1D 交叉注意力：动作序列（query）× 文本条件（key/value）。
    用于将文本语义注入动作去噪过程。
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            kdim=context_dim,
            vdim=context_dim,
            batch_first=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(query_dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T_act, query_dim) — 动作序列特征
            context: (B, T_ctx, context_dim) — 文本编码
            context_mask: (B, T_ctx) bool mask（True=有效）

        Returns:
            x: (B, T_act, query_dim)
        """
        attn_out, _ = self.attn(
            query=x,
            key=context,
            value=context,
            key_padding_mask=(~context_mask if context_mask is not None else None),
        )
        return self.norm(x + attn_out)


class DownBlock1D(nn.Module):
    """下采样块：ResBlock + stride-2 Conv1D。"""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, num_res: int = 2):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResBlock1D(in_ch, cond_dim) for _ in range(num_res)
        ])
        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for block in self.res_blocks:
            x = block(x, cond)
        skip = x  # 保存用于 skip connection
        x = self.downsample(x)
        return x, skip


class UpBlock1D(nn.Module):
    """上采样块：ConvTranspose + channel_proj 当通道数对齐 + ResBlock。"""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, cond_dim: int, num_res: int = 2):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)
        # channel_proj 将 (out_ch + skip_ch) 压缩到 out_ch，在 ResBlock 之前执行
        # 所以所有 ResBlock 均以 out_ch 初始化
        self.channel_proj = nn.Conv1d(out_ch + skip_ch, out_ch, kernel_size=1)
        self.res_blocks = nn.ModuleList([
            ResBlock1D(out_ch, cond_dim)   # 一律使用 out_ch，因为 channel_proj 已先执行
            for _ in range(num_res)
        ])

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        x = self.upsample(x)   # (B, out_ch, T*2)

        # 对齐时间维度（处理奇数长度情况）
        if x.shape[-1] != skip.shape[-1]:
            x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))

        x = torch.cat([x, skip], dim=1)   # (B, out_ch + skip_ch, T)
        x = self.channel_proj(x)           # (B, out_ch, T)  ← 第一步：先嫹齐通道

        for block in self.res_blocks:       # ← 第二步： ResBlock 均以 out_ch 进入
            x = block(x, cond)
        return x


# ============================================================
# 主体：1D U-Net 动作去噪器
# ============================================================

class ActionDenoiser1DUNet(nn.Module):
    """
    1D U-Net 动作去噪器。
    
    输入:
      - x_t: (B, T_act, action_dim) — 带噪动作
      - sigma: (B,) — 噪声水平（扩散时间步）
      - visual_cond: (B, visual_dim) — 来自 VisualBridge
      - text_cond: (B, seq_len, text_dim) — 文本条件（可选）
    
    输出:
      - x0_pred: (B, T_act, action_dim) — 预测的干净动作
    
    Args:
        action_dim: 动作维度（AIGC 数据集为 26，仅是超参，不影响运行）
        visual_dim: 视觉条件维度（来自 VisualBridge，可任意设定）
        text_dim: 文本编码维度（Qwen2.5-0.5B=896, T5-base=768, CLIP=768）
        base_channels: U-Net 基础通道数
        channel_mults: 各层的通道倍增因子
        num_res_blocks: 每层的残差块数
        use_text_cond: 是否使用文本条件（cross-attention）
    """

    def __init__(
        self,
        action_dim: int = 26,
        visual_dim: int = 256,
        text_dim: int = 896,          # Qwen2.5-0.5B 隐藏层维度
        base_channels: int = 128,
        channel_mults: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        use_text_cond: bool = True,
        dropout: float = 0.1,
        num_attn_heads: int = 4,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.use_text_cond = use_text_cond

        # 时间步嵌入维度
        time_emb_dim = base_channels * 4

        # ---- 时间步 + 视觉条件嵌入 ----
        self.time_emb = TimestepEmbedding(base_channels, time_emb_dim)
        # 联合条件向量 = time_emb + visual_cond → 线性融合
        cond_dim = time_emb_dim
        self.visual_proj = nn.Linear(visual_dim, time_emb_dim)  # 将视觉条件投影到同维度
        self.cond_merge = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, cond_dim),
        )

        # ---- 输入投影 ----
        self.input_proj = nn.Conv1d(action_dim, base_channels, kernel_size=1)

        # ---- 编码路径 ----
        self.down_blocks = nn.ModuleList()
        in_ch = base_channels
        skip_chs = []
        for mult in channel_mults[:-1]:  # 最后一层作为 bottleneck
            out_ch = base_channels * mult
            self.down_blocks.append(DownBlock1D(in_ch, out_ch, cond_dim, num_res_blocks))
            skip_chs.append(in_ch)
            in_ch = out_ch

        # ---- Bottleneck ----
        bottleneck_ch = base_channels * channel_mults[-1]
        self.bottleneck_in = nn.Conv1d(in_ch, bottleneck_ch, kernel_size=1)
        self.bottleneck_blocks = nn.ModuleList([
            ResBlock1D(bottleneck_ch, cond_dim, dropout=dropout)
            for _ in range(num_res_blocks)
        ])
        # 可选：bottleneck 处加入文本交叉注意力
        if use_text_cond:
            self.bottleneck_text_attn = CrossAttention1D(
                query_dim=bottleneck_ch,
                context_dim=text_dim,
                num_heads=num_attn_heads,
                dropout=dropout,
            )
        else:
            self.bottleneck_text_attn = None

        # ---- 解码路径 ----
        self.up_blocks = nn.ModuleList()
        in_ch = bottleneck_ch
        for mult, skip_ch in zip(reversed(channel_mults[:-1]), reversed(skip_chs)):
            out_ch = base_channels * mult
            self.up_blocks.append(UpBlock1D(in_ch, skip_ch, out_ch, cond_dim, num_res_blocks))
            in_ch = out_ch

        # ---- 输出投影 ----
        self.output_proj = nn.Sequential(
            nn.LayerNorm(in_ch),
            nn.Conv1d(in_ch, action_dim, kernel_size=1),
        )

        # 零初始化输出层（提高训练稳定性，来自 Ho et al.）
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(
        self,
        x_t: torch.Tensor,
        sigma: torch.Tensor,
        visual_cond: torch.Tensor,
        text_cond: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x_t: (B, T_act, action_dim) — 带噪动作序列
            sigma: (B,) — 噪声水平
            visual_cond: (B, visual_dim) — 来自 VisualBridge
            text_cond: (B, seq_len, text_dim) — 文本条件（可选）
            text_mask: (B, seq_len) bool — 文本 padding mask

        Returns:
            x0_pred: (B, T_act, action_dim) — 预测的干净动作
        """
        B, T_act, _ = x_t.shape

        # ---- 构造联合条件向量 ----
        time_emb = self.time_emb(sigma)          # (B, time_emb_dim)
        vis_emb = self.visual_proj(visual_cond)  # (B, time_emb_dim)
        cond = self.cond_merge(time_emb + vis_emb)  # (B, cond_dim)

        # ---- 输入投影 ----
        x = x_t.permute(0, 2, 1)  # (B, action_dim, T_act)
        x = self.input_proj(x)    # (B, base_channels, T_act)

        # ---- 编码路径 ----
        skips = []
        for down_block in self.down_blocks:
            x, skip = down_block(x, cond)
            skips.append(skip)

        # ---- Bottleneck ----
        x = self.bottleneck_in(x)  # (B, bottleneck_ch, T')
        for block in self.bottleneck_blocks:
            x = block(x, cond)

        # 文本交叉注意力（可选）
        if self.bottleneck_text_attn is not None and text_cond is not None:
            x_t_perm = x.permute(0, 2, 1)  # (B, T', bottleneck_ch)
            x_t_perm = self.bottleneck_text_attn(x_t_perm, text_cond, text_mask)
            x = x_t_perm.permute(0, 2, 1)  # (B, bottleneck_ch, T')

        # ---- 解码路径 ----
        for up_block, skip in zip(self.up_blocks, reversed(skips)):
            x = up_block(x, skip, cond)

        # ---- 输出投影 ----
        x_norm = x.permute(0, 2, 1)  # (B, T_act', base_channels)
        # LayerNorm 在通道维度
        x_norm = self.output_proj[0](x_norm)
        x = x_norm.permute(0, 2, 1)   # (B, base_channels, T_act')
        x0_pred = self.output_proj[1](x)  # (B, action_dim, T_act')
        x0_pred = x0_pred.permute(0, 2, 1)  # (B, T_act', action_dim)

        # 对齐输出长度
        if x0_pred.shape[1] != T_act:
            x0_pred = F.interpolate(
                x0_pred.permute(0, 2, 1),
                size=T_act,
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)

        return x0_pred

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 形状验证
# ============================================================

if __name__ == "__main__":
    print("=== ActionDenoiser1DUNet Shape Test ===")

    B = 2
    T_act = 50
    action_dim = 26      # AIGC 数据集的 26 维动作
    visual_dim = 256     # 视觉条件维度（匹配默认值）
    text_dim = 896       # Qwen2.5-0.5B 的层维度
    seq_len = 128        # token 序列长度

    model = ActionDenoiser1DUNet(
        action_dim=action_dim,
        visual_dim=visual_dim,
        text_dim=text_dim,
        base_channels=128,
        channel_mults=(1, 2, 4),
        num_res_blocks=2,
        use_text_cond=True,
    )

    x_t = torch.randn(B, T_act, action_dim)
    sigma = torch.ones(B) * 1.0
    visual_cond = torch.randn(B, visual_dim)
    text_cond = torch.randn(B, seq_len, text_dim)

    model.eval()
    with torch.no_grad():
        x0_pred = model(x_t, sigma, visual_cond, text_cond)

    print(f"Input  x_t:       {x_t.shape}")
    print(f"Input  sigma:     {sigma.shape}")
    print(f"Input  visual:    {visual_cond.shape}")
    print(f"Input  text:      {text_cond.shape}")
    print(f"Output x0_pred:   {x0_pred.shape}")
    assert x0_pred.shape == (B, T_act, action_dim), f"Shape mismatch: {x0_pred.shape}"
    print(f"\u2705 Shape check passed!")
    print(f"Model parameters: {model.num_parameters:,}")
