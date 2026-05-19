# Standalone Symmetric Bipolar Attention

This folder isolates Symmetric Bipolar Attention as its own architecture experiment.
The code here does not import the existing project training system. It uses only
plain JSONL data files as input data and refuses model IDs that look instruct,
abliterated, heretic, or project-specific.

Default GPU comparison:

```cmd
python sba_standalone\run_gpu_compare.py --model-id Qwen/Qwen2.5-1.5B
```

The runner trains two independent LoRA adapters on the same regular base model:

1. baseline softmax attention;
2. the same base model with attention modules patched to SBA.

Results are written under `sba_standalone/results/` as per-arm `result.json`
artifacts plus a top-level `comparison.json`.
