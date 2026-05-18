from __future__ import annotations

import math
from types import MethodType
from typing import Iterable

import torch
import torch.nn as nn

from cascades.math_ops import SBA_FORMULA_VARIANT, sba_value_attention


DEFAULT_SUPPORTED_ATTENTION_CLASS_NAMES = {
    "Qwen2Attention",
    "Qwen2SdpaAttention",
    "Qwen3Attention",
    "Qwen3SdpaAttention",
}


class SbaPatchError(RuntimeError):
    """Raised when an SBA patch request cannot be installed safely."""


def _module_class_name(module: nn.Module) -> str:
    return module.__class__.__name__


def collect_patch_report(model: nn.Module) -> dict[str, object]:
    names = [name for name, module in model.named_modules() if getattr(module, "sba_enabled", False)]
    return {
        "requested": bool(names),
        "formula_variant": SBA_FORMULA_VARIANT,
        "patched_module_count": len(names),
        "patched_module_names": names,
    }


def _normalize_attention_mask(attention_mask, logits: torch.Tensor) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    mask = attention_mask.to(device=logits.device)
    if mask.dtype == torch.bool:
        valid = mask
    else:
        valid = mask >= 0 if mask.dim() == logits.dim() else mask != 0
    while valid.dim() < logits.dim():
        valid = valid.unsqueeze(1)
    return torch.broadcast_to(valid, logits.shape)


def _install_sba_forward(module: nn.Module) -> None:
    original_forward = module.forward

    def sba_forward(self, hidden_states, attention_mask=None, output_attentions=False, **kwargs):
        batch, query_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        num_heads = int(getattr(self, "num_heads", 1))
        head_dim = int(getattr(self, "head_dim", query.shape[-1] // num_heads))
        query = query.view(batch, query_len, num_heads, head_dim).transpose(1, 2)
        key = key.view(batch, query_len, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch, query_len, num_heads, head_dim).transpose(1, 2)
        logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_dim)
        valid_mask = _normalize_attention_mask(attention_mask, logits)
        weights, attended = sba_value_attention(logits, value, valid_mask)
        attended = attended.transpose(1, 2).contiguous().view(batch, query_len, num_heads * head_dim)
        output = self.o_proj(attended)
        return output, weights if output_attentions else None, None

    setattr(module, "sba_original_forward", original_forward)
    module.forward = MethodType(sba_forward, module)


def apply_sba_patch(
    model: nn.Module,
    module_class_names: Iterable[str] | None = None,
) -> dict[str, object]:
    """Mark supported attention modules for SBA and reject unsupported silent fallback."""
    supported = set(module_class_names or DEFAULT_SUPPORTED_ATTENTION_CLASS_NAMES)
    patched_names: list[str] = []
    for name, module in model.named_modules():
        if _module_class_name(module) in supported:
            setattr(module, "sba_enabled", True)
            setattr(module, "sba_formula_variant", SBA_FORMULA_VARIANT)
            _install_sba_forward(module)
            patched_names.append(name)
    if not patched_names:
        raise SbaPatchError(
            "SBA patch requested but no supported attention modules were found; "
            "aborting to prevent silent baseline fallback"
        )
    return {
        "requested": True,
        "formula_variant": SBA_FORMULA_VARIANT,
        "supported_class_names": sorted(supported),
        "patched_module_count": len(patched_names),
        "patched_module_names": patched_names,
    }
