# Qwen-Omni Raw In-Process ASR Assets

This folder contains prompt templates used by the Qwen-Omni raw-manifest
in-process ASR adapter.

Install the runtime with `uv sync --extra audio_qwen`. The dedicated extra
keeps Qwen/vLLM dependencies out of existing `audio_cuda12` installations.

Run a complete YAML config through the pipeline entrypoint, following the same
Hydra pattern as the audio-tagging tutorial:

```bash
python tutorials/audio/qwen_omni_raw_inprocess/main.py \
  --config-path /path/to/configs \
  --config-name qwen_omni_raw_inprocess_global \
  input_manifest=/data/input.jsonl \
  output_dir=/data/qwen_output \
  final_manifest=/data/qwen_output/output.jsonl
```

The command supplies only runtime I/O. Backend, workers, model settings,
segmentation, bucketing, payload lifecycle, and performance collection belong
in the selected YAML.

The executable code path is:

```text
Pipeline
  -> ManifestReader(enable_global_bucketing=...)
  -> AudioPayloadMaterializeStage
  -> ASRStage(adapter_target=QwenOmniASRAdapter)
  -> optional DispatchBatchUnpackStage for global bucket-on
  -> PayloadReleaseStage
  -> optional GlobalSegmentAssemblerStage
  -> ManifestWriterStage
```

The adapter reads prompt text through `prompt_file`, `en_prompt_file`,
`followup_prompt_file`, or `system_prompt_file`. Curator stage behavior remains
outside the prompt files:

- graph expansion lives in `nemo_curator/pipeline/payload_lifecycle.py`;
- audio decode and payload refs live in `nemo_curator/stages/payload_lifecycle.py`;
- ASR model-input segmentation and batching live in
  `nemo_curator/stages/audio/inference/asr/stage.py`;
- atomic global owner-call rows and generic fan-out live in
  `nemo_curator/tasks/dispatch_batch.py` and
  `nemo_curator/stages/dispatch_batch.py`;
- Qwen/vLLM request construction lives in `nemo_curator/models/asr/qwen_omni.py`;
- global segment assembly lives in `nemo_curator/stages/audio/common.py`.

Prompt files may use `{language}` and `{transcript}` placeholders when the
stage supplies language or reference text columns.

Qwen3-Omni runs on the installed vLLM engine. The adapter's default prompt,
text-before-audio message order, and sampling values preserve the reference
adapter behavior. Pipelines can instead set `prompt_content_order=audio_text`,
a language-specific prompt, and `top_p` to reproduce Qwen's official ASR
recipe. The adapter returns vLLM's first output text, including output stopped
at the configured generation-length limit; it does not apply a repetition
heuristic, retry inference, or invoke another ASR model.

`ASRStage.max_inference_duration_s` is the hard per-request audio ceiling.
With bucketing disabled, `adapter_batch_size` is the item-count cap for one
adapter call. For example, `max_inference_duration_s=600` and
`adapter_batch_size=1` guarantee that every Qwen call receives one contiguous
audio segment no longer than ten minutes.
