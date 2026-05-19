from __future__ import annotations

import math
from types import MethodType
from typing import Iterable

import torch
import torch.nn as nn

from .math_ops import SBA_FORMULA_VARIANT, sba_value_attention


DEFAULT_SUPPORTED_ATTENTION_CLASS_NAMES = {
    "Qwen2Attention",
    "Qwen2SdpaAttention",
    "Qwen3Attention",
    "Qwen3SdpaAttention",
    "LlamaAttention",
    "LlamaSdpaAttention",
}


class SbaPatchError(RuntimeError):
    """Raised when the standalone SBA patch cannot be installed safely."""


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


def _normalize_attention_mask(attention_mask: torch.Tensor | None, logits: torch.Tensor) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    mask = attention_mask.to(device=logits.device)
    if mask.dtype == torch.bool:
        valid = mask
    elif mask.dim() == logits.dim():
        valid = mask >= 0
    else:
        valid = mask != 0
    while valid.dim() < logits.dim():
        valid = valid.unsqueeze(1)
    return torch.broadcast_to(valid, logits.shape)


def _repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return hidden_states
    batch, kv_heads, seq_len, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(batch, kv_heads, repeats, seq_len, head_dim)
    return expanded.reshape(batch, kv_heads * repeats, seq_len, head_dim)


def _apply_rotary(query: torch.Tensor, key: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]):
    try:
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    except Exception:
        try:
            from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
        except Exception:
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    cos, sin = position_embeddings
    return apply_rotary_pos_emb(query, key, cos, sin)


def _install_sba_forward(module: nn.Module) -> None:
    original_forward = module.forward

    def sba_forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        output_attentions: bool = False,
        **kwargs,
    ):
        if position_embeddings is None and args and isinstance(args[0], tuple):
            position_embeddings = args[0]
        if attention_mask is None:
            for arg in args:
                if torch.is_tensor(arg):
                    attention_mask = arg
                    break
        if past_key_values is None:
            past_key_values = kwargs.get("past_key_values") or kwargs.get("past_key_value")
        if position_embeddings is None:
            raise SbaPatchError("SBA patched attention requires position_embeddings from the loaded model")

        input_shape = hidden_states.shape[:-1]
        head_dim = int(getattr(self, "head_dim", 0))
        if head_dim <= 0:
            hidden_size = int(getattr(self, "hidden_size", hidden_states.shape[-1]))
            num_heads = int(getattr(self, "num_heads", 1))
            head_dim = hidden_size // num_heads
        hidden_shape = (*input_shape, -1, head_dim)

        query = self.q_proj(hidden_states).view(hidden_shape)
        key = self.k_proj(hidden_states).view(hidden_shape)
        value = self.v_proj(hidden_states).view(hidden_shape)
        if hasattr(self, "q_norm"):
            query = self.q_norm(query)
        if hasattr(self, "k_norm"):
            key = self.k_norm(key)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        query, key = _apply_rotary(query, key, position_embeddings)

        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)

        repeats = int(getattr(self, "num_key_value_groups", 0))
        if repeats <= 0:
            repeats = query.shape[1] // key.shape[1]
        key = _repeat_kv(key, repeats)
        value = _repeat_kv(value, repeats)

        scaling = float(getattr(self, "scaling", 1.0 / math.sqrt(head_dim)))
        logits = torch.matmul(query, key.transpose(2, 3)) * scaling
        valid_mask = _normalize_attention_mask(attention_mask, logits)
        weights, attended = sba_value_attention(logits, value, valid_mask)
        attended = attended.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        output = self.o_proj(attended)
        return output, weights if output_attentions else None

    setattr(module, "sba_original_forward", original_forward)
    module.forward = MethodType(sba_forward, module)


def apply_sba_patch(model: nn.Module, module_class_names: Iterable[str] | None = None) -> dict[str, object]:
    supported = set(module_class_names or DEFAULT_SUPPORTED_ATTENTION_CLASS_NAMES)
    patched_names: list[str] = []
    for name, module in model.named_modules():
        if _module_class_name(module) in supported and not getattr(module, "sba_enabled", False):
            setattr(module, "sba_enabled", True)
            setattr(module, "sba_formula_variant", SBA_FORMULA_VARIANT)
            _install_sba_forward(module)
            patched_names.append(name)
    if not patched_names:
        raise SbaPatchError("SBA requested but no supported attention modules were found")
    return {
        "requested": True,
        "formula_variant": SBA_FORMULA_VARIANT,
        "supported_class_names": sorted(supported),
        "patched_module_count": len(patched_names),
        "patched_module_names": patched_names,
    }
