import pytest
import torch
import torch.nn as nn

from cascades.config import AblationConfig
from cascades.sba_patch import SbaPatchError, apply_sba_patch, collect_patch_report


class ToyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.sba_enabled = False
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)
        self.num_heads = 1
        self.head_dim = 4
        self.hidden_size = 4

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        return self.o_proj(self.v_proj(hidden_states)), None, None


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([ToyAttention(), ToyAttention()])


def test_default_config_keeps_sba_disabled():
    cfg = AblationConfig()
    assert cfg.enable_symmetric_bipolar_attention is False
    assert cfg.sba_formula_variant == "sba-v1-valid-position-sinh-cosh-shared-denominator"


def test_apply_patch_marks_toy_attention_modules():
    model = ToyModel()
    report = apply_sba_patch(model, module_class_names={"ToyAttention"})
    assert report["requested"] is True
    assert report["patched_module_count"] == 2
    assert report["patched_module_names"] == ["layers.0", "layers.1"]
    assert model.layers[0].sba_enabled is True
    assert model.layers[1].sba_enabled is True


def test_patched_toy_attention_uses_signed_weights():
    model = ToyModel()
    apply_sba_patch(model, module_class_names={"ToyAttention"})
    hidden = torch.tensor([[[1.0, 0.0, -1.0, 0.5], [0.0, 1.0, 0.25, -0.5]]])
    output, attention_weights, _ = model.layers[0](hidden, output_attentions=True)
    assert output.shape == hidden.shape
    assert attention_weights is not None
    assert torch.isfinite(output).all().item()
    assert attention_weights.abs().sum(dim=-1).max().item() <= 1.0 + 1e-6


def test_unsupported_model_raises_instead_of_silent_fallback():
    model = nn.Linear(4, 4)
    with pytest.raises(SbaPatchError, match="no supported attention modules"):
        apply_sba_patch(model, module_class_names={"ToyAttention"})


def test_collect_patch_report_detects_enabled_modules():
    model = ToyModel()
    apply_sba_patch(model, module_class_names={"ToyAttention"})
    report = collect_patch_report(model)
    assert report["patched_module_count"] == 2
    assert report["patched_module_names"] == ["layers.0", "layers.1"]
