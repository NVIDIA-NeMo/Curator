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

## Requirements

- x86_64 Linux with CUDA
- `ffmpeg`
- Audio files accessible from the machine running the pipeline

`ResampleAudioStage` invokes `ffmpeg`, so the input can use any format supported
by the installed `ffmpeg` build. It writes PCM 16-bit, 16 kHz, mono WAV files
under `resampled_audio_dir` before inference starts on those rows.

From the Curator repository root, install the optional Qwen stack:

```bash
uv sync --extra audio_vllm
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
when it loads the model. The Qwen adapter uses that stage-owned value as
vLLM's tensor-parallel size.

## Effective defaults

The tutorial YAML makes the reference runner's effective primary-Qwen settings
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

## Select the executor

The default `backend: ray_data` matches the reference runner. To use Xenna
batch mode instead, select both the backend and its execution mode:

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/qwen_omni_inprocess \
  --config-name pipeline \
  manifest_path=/data/input.jsonl \
  output_path=/tmp/qwen_omni_output.jsonl \
  backend=xenna \
  execution_mode=batch
```

To use Xenna streaming mode:

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/qwen_omni_inprocess \
  --config-name pipeline \
  manifest_path=/data/input.jsonl \
  output_path=/tmp/qwen_omni_output.jsonl \
  backend=xenna \
  execution_mode=streaming
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
`resampled_audio_filepath` with `torchaudio` only for its current batch and
passes the resulting mono NumPy samples directly to the adapter. It never
stores a waveform or sample rate in `task.data`, so manifest size does not
cause all decoded audio to accumulate in host RAM.

`ASRStage` adds only the configured prediction column and, when applicable,
`_skipme` or `additional_notes`. This tutorial defaults the prediction column
to `primary_model_prediction`, matching the reference runner; override it
through the YAML, for example `pred_text_key=qwen_transcript`. The writer
truncates `output_path` when the run starts and then appends one JSON object per
input row.

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
  empty configured prediction column and an explanation under
  `additional_notes`.

Inspect `_skipme` and `additional_notes` before consuming a completed
manifest.

## Languages and prompts

The default is single-turn transcription with the reference runner's inline prompt
`Transcribe the audio.` and no language-specific, follow-up, or system prompt.
Set `prompt_file` to use a UTF-8 prompt asset instead.

For language-aware prompting, use the bundled multilingual prompt containing
`{language}`:

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/qwen_omni_inprocess \
  --config-name pipeline \
  manifest_path=/data/input.jsonl \
  output_path=/tmp/qwen_omni_output.jsonl \
  prompt_file=examples/audio/qwen_omni_inprocess/prompts/ml_qwen3_omni_disfluency_asr.md
```

Known ISO codes are converted to human-readable language names before prompt
interpolation. Unknown codes are passed through in normalized lowercase form.
The default allowlist matches the reference Qwen stage:
`en`, `zh`, `ko`, `ja`, `de`, `ru`, `it`, `fr`, `es`, `pt`, `ms`, `nl`,
`id`, `tr`, `vi`, `yue`, `ar`, and `ur`. Rows without `source_lang` are
annotated as unsupported unless `default_language` is explicitly set.

## Reference-guided inference

For reference-guided correction, each input row must also contain a `text`
reference transcript:

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/qwen_omni_inprocess \
  --config-name pipeline \
  manifest_path=/data/input.jsonl \
  output_path=/tmp/qwen_omni_output.jsonl \
  reference_text_key=text \
  pred_text_key=revised_transcript \
  prompt_file=examples/audio/qwen_omni_inprocess/prompts/en_qwen3_omni_reference_improvement.md
```

The result is written to `revised_transcript`. A prompt containing
`{transcript}` requires a non-empty value at `reference_text_key` for every
row. A missing value makes that row fail adapter preprocessing and receive
the normal adapter skip marker.

## Scope

This is a functional, local manifest-to-transcript example. It does not
provide recovery ASR, hallucination filtering, WER calculation,
duration-aware bucketing, sharded resumability, or benchmark reporting.
Validate output quality and row accounting on representative audio before
larger runs.
