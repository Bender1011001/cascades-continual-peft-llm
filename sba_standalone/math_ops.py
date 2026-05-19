from __future__ import annotations

import torch


SBA_FORMULA_VARIANT = "sba-v1-standalone-valid-position-sinh-cosh-shared-denominator"


def _valid_mask(logits: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    if valid_mask is None:
        return torch.ones_like(logits, dtype=torch.bool)
    mask = valid_mask.to(device=logits.device)
    if mask.dtype != torch.bool:
        mask = mask != 0
    while mask.dim() < logits.dim():
        mask = mask.unsqueeze(1)
    return torch.broadcast_to(mask, logits.shape)


def sba_attention(logits: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Stable signed sinh-cosh attention weights.

    This implements ``sinh(S) / sum(cosh(S_valid))`` with masked positions set
    to exactly zero.  The computation is performed in float32 for stability and
    returned in the original logits dtype.
    """
    valid = _valid_mask(logits, valid_mask)
    work = logits.float()
    zeros = torch.zeros_like(work)
    floor = torch.full_like(work, torch.finfo(work.dtype).min)
    pos_logits = torch.where(valid, work, floor)
    neg_logits = torch.where(valid, -work, floor)
    max_pos = pos_logits.max(dim=-1, keepdim=True).values
    max_neg = neg_logits.max(dim=-1, keepdim=True).values
    has_valid = valid.any(dim=-1, keepdim=True)
    row_max = torch.maximum(max_pos, max_neg)
    row_max = torch.where(has_valid, row_max, torch.zeros_like(row_max))
    e_pos = torch.where(valid, torch.exp(pos_logits - row_max), zeros)
    e_neg = torch.where(valid, torch.exp(neg_logits - row_max), zeros)
    denominator = (e_pos + e_neg).sum(dim=-1, keepdim=True)
    denominator = denominator.clamp_min(torch.finfo(work.dtype).tiny)
    weights = torch.where(has_valid, (e_pos - e_neg) / denominator, zeros)
    weights = torch.where(valid, weights, zeros)
    return weights.to(dtype=logits.dtype)


def sba_value_attention(
    logits: torch.Tensor,
    values: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = sba_attention(logits, valid_mask).to(dtype=values.dtype)
    return weights, torch.matmul(weights, values)


def sba_diagnostics(weights: torch.Tensor, valid_mask: torch.Tensor | None = None) -> dict[str, float | bool]:
    valid = _valid_mask(weights, valid_mask)
    masked_weights = torch.where(valid, torch.zeros_like(weights), weights)
    row_abs_sum = weights.abs().sum(dim=-1)
    return {
        "finite": bool(torch.isfinite(weights).all().item()),
        "max_signed_l1": float(row_abs_sum.max().item()) if row_abs_sum.numel() else 0.0,
        "max_masked_leakage": float(masked_weights.abs().max().item()) if masked_weights.numel() else 0.0,
    }
