# Qwen-Omni Raw-Manifest Pipeline

This entrypoint runs a complete Hydra YAML through Curator, using the same
pipeline-script pattern as the [audio-tagging tutorial](../tagging/).

```bash
python tutorials/audio/qwen_omni_raw_inprocess/main.py \
  --config-path /path/to/configs \
  --config-name qwen_omni_raw_inprocess_global \
  input_manifest=/data/input.jsonl \
  output_dir=/data/qwen_output \
  final_manifest=/data/qwen_output/output.jsonl
```

Only runtime I/O belongs on the command line. The selected YAML owns the
backend, worker layout, ASR adapter, model settings, model-input segmentation,
duration-aware bucketing, payload lifecycle, and performance sampling.

See the [Qwen runtime notes and prompt assets](../../../examples/audio/qwen_omni_raw_inprocess/)
for the stage flow and adapter behavior.
