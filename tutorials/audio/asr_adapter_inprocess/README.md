# YAML-Defined ASR Adapter Pipeline

This entrypoint runs a complete audio ASR pipeline declared in one Hydra YAML.
The YAML selects the reader, `ASRStage` adapter, batching policy, resources,
backend, writer, and observability settings.

From the Curator repository root:

```bash
python tutorials/audio/asr_adapter_inprocess/main.py \
  --config-path=/absolute/path/to/configs \
  --config-name=my_asr_pipeline \
  input_manifest=/data/input.jsonl \
  output_dir=/data/output \
  final_manifest=/data/output/output.jsonl
```

The command supplies only runtime I/O. All processing behavior belongs in the
selected YAML. `ASRStage.adapter_target` can select any implementation of the
shared ASR adapter contract, including `NeMoASRAdapter` or
`QwenOmniASRAdapter`.
