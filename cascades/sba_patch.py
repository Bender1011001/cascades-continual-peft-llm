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

    def sba_forward(self, hidden_states, *args, attention_mask=None, output_attentions=False, **kwargs):
        batch, query_len, _ = hidden_states.shape
        input_shape = hidden_states.shape[:-1]
        head_dim = int(getattr(self, "head_dim", 0))
        if head_dim <= 0:
            head_dim = int(getattr(self, "hidden_size", hidden_states.shape[-1])) // int(getattr(self, "num_heads", 1))
        query = self.q_proj(hidden_states).view(*input_shape, -1, head_dim)
        key = self.k_proj(hidden_states).view(*input_shape, -1, head_dim)
        value = self.v_proj(hidden_states).view(*input_shape, -1, head_dim)
        if hasattr(self, "q_norm"):
            query = self.q_norm(query)
        if hasattr(self, "k_norm"):
            key = self.k_norm(key)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        position_embeddings = args[0] if args and isinstance(args[0], tuple) else kwargs.get("position_embeddings")
        if position_embeddings is not None:
            cos, sin = position_embeddings
            try:
                from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
            except Exception:
                from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
        past_key_values = kwargs.get("past_key_values")
        if past_key_values is None:
            past_key_values = kwargs.get("past_key_value")
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)
        if key.shape[1] != query.shape[1]:
            repeat_factor = query.shape[1] // key.shape[1]
            key = key.repeat_interleave(repeat_factor, dim=1)
            value = value.repeat_interleave(repeat_factor, dim=1)
        if attention_mask is None and len(args) >= 2 and torch.is_tensor(args[1]):
            attention_mask = args[1]
        logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_dim)
        valid_mask = _normalize_attention_mask(attention_mask, logits)
        weights, attended = sba_value_attention(logits, value, valid_mask)
        attended = attended.transpose(1, 2).contiguous().reshape(batch, query_len, -1)
        output = self.o_proj(attended)
        if _module_class_name(self) in DEFAULT_SUPPORTED_ATTENTION_CLASS_NAMES:
            return output, weights if output_attentions else None
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
