# Qwen-Omni Raw In-Process ASR Assets

This folder contains prompt templates used by the Qwen-Omni raw-manifest
in-process ASR adapter.

Install the runtime with `uv sync --extra audio_qwen`. The dedicated extra
keeps Qwen/vLLM dependencies out of existing `audio_cuda12` installations.

The executable code path is:

```text
Pipeline
  -> ManifestReader
  -> AudioPayloadMaterializeStage
  -> ASRStage(adapter_target=QwenOmniASRAdapter)
  -> PayloadReleaseStage
  -> ManifestWriterStage
```

The adapter reads prompt text through `prompt_file`, `en_prompt_file`,
`followup_prompt_file`, or `system_prompt_file`. Curator stage behavior remains
outside the prompt files:

- graph expansion lives in `nemo_curator/pipeline/payload_lifecycle.py`;
- audio decode and payload refs live in `nemo_curator/stages/payload_lifecycle.py`;
- local/windowed ASR model-input segmentation and batching live in
  `nemo_curator/stages/audio/inference/asr/stage.py`;
- generic dispatch envelopes and fan-out live in
  `nemo_curator/tasks/dispatch_batch.py` and
  `nemo_curator/stages/dispatch_batch.py`; the local reader does not emit them;
- Qwen/vLLM request construction lives in `nemo_curator/models/asr/qwen_omni.py`.

Prompt files may use `{language}` and `{transcript}` placeholders when the
stage supplies language or reference text columns.
