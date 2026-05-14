"""
动作分支多步 EDM 采样（与 `step3_dual_stream_train` / `DualStreamModel.compute_action_loss` 同套预条件化）。

训练：x_t = x0 + σ·ε，网络输入为 x_t * c_in(σ)，组合得 x0_pred = c_skip·x_t + c_out·net_out。
推理：从 σ_max 到 σ_min 多步更新，每步用相同公式预测 x0，并在 VE 形式下保持噪声方向 (x - x0) / σ。
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from .action_denoiser import ActionDenoiser1DUNet


def sigma_to_precond(sigma: torch.Tensor, sigma_data: float = 0.5):
    c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
    c_out = sigma * sigma_data / (sigma**2 + sigma_data**2).sqrt()
    c_in = 1 / (sigma**2 + sigma_data**2).sqrt()
    return c_skip, c_out, c_in


def edm_predict_x0_action(
    denoiser: ActionDenoiser1DUNet,
    x_t: torch.Tensor,
    sigma: torch.Tensor,
    visual_cond: torch.Tensor,
    text_cond: Optional[torch.Tensor],
    sigma_data: float = 0.5,
) -> torch.Tensor:
    """由带噪动作 x_t 预测干净 x0，与训练循环一致。"""
    b = x_t.shape[0]
    c_skip, c_out, c_in = sigma_to_precond(sigma, sigma_data)
    c_skip = c_skip.view(b, 1, 1)
    c_out = c_out.view(b, 1, 1)
    c_in = c_in.view(b, 1, 1)
    x_t_scaled = x_t * c_in
    net_out = denoiser(
        x_t=x_t_scaled,
        sigma=sigma,
        visual_cond=visual_cond,
        text_cond=text_cond,
    )
    return c_skip * x_t + c_out * net_out


@torch.no_grad()
def sample_actions_edm(
    action_denoiser: ActionDenoiser1DUNet,
    visual_cond: torch.Tensor,
    text_cond: Optional[torch.Tensor],
    action_steps: int,
    action_dim: int,
    device: torch.device,
    num_steps: int = 30,
    generator: Optional[torch.Generator] = None,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    sigma_data: float = 0.5,
) -> torch.Tensor:
    """
    在归一化动作空间从噪声采样，返回 (B, T, D)。

    使用对数均匀 σ 序列；每步 VE 更新：x ← x0_hat + σ_next * ((x - x0_hat) / σ)。
    """
    b = visual_cond.shape[0]
    n = max(1, int(num_steps))

    log_s = torch.linspace(math.log(sigma_max), math.log(sigma_min), n + 1, device=device)
    sigmas = log_s.exp()

    x = torch.randn(
        b,
        action_steps,
        action_dim,
        device=device,
        dtype=visual_cond.dtype,
        generator=generator,
    )
    x = x * sigmas[0]

    x0_hat = x
    for i in range(n):
        sigma_cur = sigmas[i : i + 1].expand(b)
        sigma_next = sigmas[i + 1 : i + 2].expand(b)
        x0_hat = edm_predict_x0_action(
            action_denoiser, x, sigma_cur, visual_cond, text_cond, sigma_data
        )
        eps = (x - x0_hat) / sigma_cur.view(b, 1, 1).clamp_min(1e-8)
        x = x0_hat + sigma_next.view(b, 1, 1) * eps

    return x0_hat
