# Symmetric Bipolar Attention Validation

This directory contains the bounded evidence ladder for SBA v1: `sba-v1-valid-position-sinh-cosh-shared-denominator`.

## Gate order

1. CPU math gate: `cpu_math_gate.json` and `cpu_math_gate.log`.
2. Synthetic CUDA gate: `cuda_synthetic_gate.json` and `cuda_synthetic_gate.log`, only after CPU gate passes and GPU approval is explicit.
3. Tiny patched-model gate: `tiny_model_gate.json` and `tiny_model_gate.log`, only after synthetic CUDA passes and GPU approval is explicit.
4. Paired reduced training gate: `paired_reduced/control`, `paired_reduced/treatment`, and `paired_reduced/comparison.json`, only after all earlier gates pass and GPU approval is explicit.
5. Larger-suite escalation: blocked unless all prior gates pass, the result critic accepts evidence quality, and the user explicitly approves escalation.

## Non-goals

- SBA is not the default attention path.
- SBA is not a catastrophic-forgetting solution claim.
- Full current4, v10, Digital Twin, generative subset, broad settings fuzzer, and larger-suite experiments remain blocked until escalation is approved.
