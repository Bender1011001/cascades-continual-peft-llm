# SBA Validation Result Critic Packet

## Gate verdicts
- CPU math gate: valid=true; artifact=experiments\sba_validation\cpu_math_gate.json; reasons=none
- Synthetic CUDA gate: valid=false; artifact=experiments\sba_validation\cuda_synthetic_gate.json; reasons=missing artifact: experiments\sba_validation\cuda_synthetic_gate.json
- Tiny patched-model gate: valid=false; artifact=experiments\sba_validation\tiny_model_gate.json; reasons=missing artifact: experiments\sba_validation\tiny_model_gate.json
- Paired reduced comparison: valid=false; artifact=experiments\sba_validation\paired_reduced\comparison.json; reasons=missing artifact: experiments\sba_validation\paired_reduced\comparison.json

## CPU math criteria
- Formula variant: sba-v1-valid-position-sinh-cosh-shared-denominator
- Formula equivalence max diff: 2.220446049250313e-16
- Mask policy: valid_positions_only_zero_masked_weights
- Masked leakage max: 0.0
- Signed L1 max: 0.9983598138083813
- Finite outputs: true
- Finite gradients: true
- Jacobian claim: sba_jacobian_frobenius_norm <= softmax_jacobian_frobenius_norm
- Jacobian claim status: falsified
- SBA Jacobian Frobenius norm: 0.48483186078076224
- Softmax Jacobian Frobenius norm: 0.38938977465759933

## VRAM and paired deltas
- CPU peak VRAM MB: missing
- Synthetic CUDA peak VRAM MB: missing
- Tiny-model peak VRAM MB: missing
- Paired control peak VRAM MB: missing
- Paired treatment peak VRAM MB: missing
- Paired delta BWT points: missing
- Paired delta average accuracy points: missing
- Paired old-task delta gap points: missing
- Paired continue recommendation: missing

## Escalation decision guard
- All prior gates passed: false
- Larger-suite escalation allowed now: false
- Guard: Larger-suite escalation remains blocked unless every prior gate passes, the result critic accepts the evidence quality, and the user explicitly approves escalation in a later decision.
- Required next decision: route this packet to result criticism; do not claim catastrophic forgetting is solved.
- Current default: SBA remains default-off.
