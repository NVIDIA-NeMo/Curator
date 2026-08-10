# Qwen3-Omni In-Process ASR

This tutorial reads a NeMo-style audio manifest, resamples each file to a
16 kHz mono WAV, transcribes it with Qwen3-Omni through vLLM, and writes the
results to JSONL. Its four stages are `ManifestReader`, the same
`ResampleAudioStage` used by the audio tagging tutorial, the generic `ASRStage`
configured with `QwenOmniASRAdapter`, and `ManifestWriterStage`.

The tutorial uses the generic YAML runner in `nemo_curator/config/run.py`; no
tutorial-specific Python runner is needed. The YAML selects the executor
backend and, for Xenna, its execution mode; the generic runner constructs that
executor and passes it to `Pipeline.run()`.

The run also writes a compact, stage-grouped performance report to
`qwen_omni_performance.json`. With the supported Ray Data or Xenna executor,
the report keeps one invocation measurement for each successful stage batch
execution, including executions that emit no output. A fan-out execution is
recorded once rather than once per output task.

## Requirements

- x86_64 Linux with CUDA
- `ffmpeg`
- Audio files accessible from the machine running the pipeline

`ResampleAudioStage` invokes `ffmpeg`, so the input can use any format supported
by the installed `ffmpeg` build. It writes PCM 16-bit, 16 kHz, mono WAV files
under `resampled_audio_dir` before inference starts on those rows.

From the Curator repository root, install the optional Qwen stack:

```bash
uv sync --extra audio_cuda12 --extra vllm --extra cloud_filesystems
source .venv/bin/activate
```

## Run the bundled smoke input

The bundled manifest contains two short OPUS files:

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/qwen_omni_inprocess \
  --config-name pipeline \
  manifest_path=tests/fixtures/audio/tagging/sample_input.jsonl \
  output_path=/tmp/qwen_omni_output.jsonl \
  workspace_dir=/tmp/qwen_omni_workspace \
  default_language=en
```

`--config-path` is relative to `nemo_curator/config/run.py`, while manifest,
audio, prompt, and output paths are resolved from the current working
directory. Run the command from the repository root as shown.

The first run downloads `Qwen/Qwen3-Omni-30B-A3B-Instruct`. The pipeline uses
a batch size of 32 and allocates two GPUs to each ASR actor.
`gpus_per_actor: 2` is the single GPU-count setting: Curator schedules that
many GPUs, then `ASRStage` supplies the scheduled device count to the adapter
when it loads the model. The Qwen adapter uses that allocated value as
vLLM's tensor-parallel size.

## Effective defaults

The tutorial YAML makes the reference runner's effective Qwen settings
explicit:

| Setting | Tutorial value |
|---|---:|
| Executor | Ray Data |
| ASR stage batch size | `32` |
| GPUs per ASR actor / derived vLLM tensor parallelism | `2` / `2`, from `gpus_per_actor` |
| Prompt | `Transcribe the audio.` |
| Prompt content order | text, then audio |
| Concurrent vLLM sequences | `16` |
| Maximum model length | `32768` |
| Maximum generated tokens | `256` |
| Audio inputs allowed per vLLM request | `2` |

Keeping these values in the YAML makes any future drift from the reference
configuration visible in code review.

The optional Hugging Face model revision lives at `adapter_kwargs.revision`
(`model_revision` in the tutorial's top-level overrides), engine settings live
under `adapter_kwargs.vllm_kwargs`, and sampling settings live under
`adapter_kwargs.sampling_kwargs`. These are Qwen adapter settings, not generic
`ASRStage` fields. They are forwarded to the Hugging Face download, Curator's
shared vLLM construction path, and vLLM's `SamplingParams`, respectively.
Do not put `tensor_parallel_size` in `vllm_kwargs`: `gpus_per_actor` is the
single GPU-count setting and the stage derives tensor parallelism from it.

## Stage performance report

Setting `performance_report_path` on the terminal manifest writer starts one
run-scoped collector for the whole pipeline when using Ray Data or Xenna. The
public JSON is a compact summary grouped by the concrete stages in built-plan
order; it is not the collector's internal raw-record format.

The schema-version-1 report contains:

- pipeline and executor metadata, a Curator-owned run ID, optional Slurm-array
  identity, executor wall time, and the total invocation-record count;
- ordered metadata for every planned stage under `pipeline.stages`; and
- one entry per planned stage under `stage_performance`, with its stable
  `stage_id`, observed start/end window, and aligned `invocation_ids` and
  `processing_times_s` arrays.

A stage with no captured execution still appears with empty arrays and null
timestamps. Input item count, input-byte measurement, custom metrics, actor idle
time, output cardinality, and hardware telemetry remain internal or out of
scope; they are not fields in the public schema-version-1 report. See the
[complete schema and an exact 16-row example](../performance/README.md).

Override the destination like any other Hydra value:

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/qwen_omni_inprocess \
  --config-name pipeline \
  manifest_path=/data/input.jsonl \
  performance_report_path=s3://bucket/runs/qwen/performance.json
```

`ManifestWriterStage` streams the report through Curator's fsspec-aware JSON
utility, so local paths and configured remote filesystems share the same
schema. Local output uses a temporary file and atomic replacement; remote
filesystems are streamed directly and do not have a universal atomic-replace
guarantee. The driver asks the manifest writer to finalize the report after a
successful pipeline execution. A run with no captured executions has
`record_count: 0` and empty arrays in each planned stage entry; the schema does
not contain a top-level `records` list.

## Select the executor

The default `backend: ray_data` matches the reference runner. When using
Xenna, use its default streaming mode:

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/qwen_omni_inprocess \
  --config-name pipeline \
  manifest_path=/data/input.jsonl \
  output_path=/tmp/qwen_omni_output.jsonl \
  backend=xenna \
  execution_mode=streaming
```

Use Xenna batch mode only as a fallback when streaming runs out of memory for
the workload:

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/qwen_omni_inprocess \
  --config-name pipeline \
  manifest_path=/data/input.jsonl \
  output_path=/tmp/qwen_omni_output.jsonl \
  backend=xenna \
  execution_mode=batch
```

`execution_mode` applies only to Xenna and is ignored when `backend` is
`ray_data`.

## Input and output

The input is a JSONL manifest with one object per audio file. Each object must
contain `audio_filepath` and, by default, `source_lang`:

```json
{"audio_filepath": "/data/sample.wav", "source_lang": "en"}
```

`ResampleAudioStage` preserves `audio_filepath` and adds `audio_item_id`,
`resampled_audio_filepath`, and `duration`. It caches the converted file at
`${workspace_dir}/audio_resampled` by default and reuses it on later runs.

Only file paths and metadata travel between pipeline stages. `ASRStage` opens
`resampled_audio_filepath` with TorchAudio only for its current batch and
preserves the decoded sample rate while normalizing each waveform to contiguous
mono 16 kHz NumPy samples for the adapter. It never stores either the waveform
or sample rate in `task.data`, so manifest size does not cause all decoded
audio to accumulate in host RAM. The Qwen-Omni adapter validates the resulting
16 kHz contract, and the tutorial fixes `ResampleAudioStage.target_sample_rate`
at 16 kHz instead of exposing an incompatible override.

`ASRStage` adds only the configured prediction column and, when applicable,
`_skipme` or `additional_notes`. This tutorial defaults the prediction column
to `pred_text`; override it through the YAML, for example
`pred_text_key=qwen_transcript`. The writer truncates `output_path` when the
run starts and then appends one JSON object per input row.

Rows are handled differently depending on where processing fails:

- Missing, inaccessible, undecodable, or invalid audio causes
  `ResampleAudioStage` to fail with the corresponding `ffmpeg` error.
- A resampled file that cannot be reopened is logged and retained with an
  empty configured prediction column and `"_skipme": "audio_load_error"`;
  other files in the same ASR batch continue through inference.
- Adapter prompt-preparation failures, too-short audio, and empty model outputs
  remain in the output with an empty configured prediction column and
  `"_skipme": "empty_audio"`.
- A language outside `supported_language_codes` remains in the output with an
  empty configured prediction column, `"_skipme": "language_not_supported"`,
  and an explanation under `additional_notes`.
- When `supported_language_codes` is enabled, a row with neither
  `source_lang` nor `default_language` remains in the output with an empty
  prediction, `"_skipme": "language_missing"`, and `language_missing` under
  `additional_notes`.

Inspect `_skipme` and `additional_notes` before consuming a completed
manifest.

## Languages and prompts

The default uses the reference runner's inline prompt `Transcribe the audio.`
with no language-specific or system prompt.
Set `prompt_file` to use a UTF-8 prompt asset instead. Prompt text may contain
`{language}`.

Known ISO codes are converted to human-readable language names before prompt
interpolation. Unknown codes are passed through in normalized lowercase form.
The default allowlist matches the reference Qwen stage:
`en`, `zh`, `ko`, `ja`, `de`, `ru`, `it`, `fr`, `es`, `pt`, `ms`, `nl`,
`id`, `tr`, `vi`, `yue`, `ar`, and `ur`. Rows without `source_lang` are
annotated with `language_missing` unless `default_language` is explicitly set.

## Scope

This is a functional, local manifest-to-transcript example. It does not
provide recovery ASR, hallucination filtering, WER calculation,
duration-aware bucketing, sharded resumability, or benchmark reporting.
Validate output quality and row accounting on representative audio before
larger runs.
