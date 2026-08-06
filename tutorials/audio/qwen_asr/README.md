# Qwen3-ASR Adapter Tutorial

This tutorial reads a NeMo-style audio manifest, normalizes each file to a
16 kHz mono WAV, transcribes it with `Qwen/Qwen3-ASR-0.6B`, and writes the
results to JSONL. The pipeline composes `ManifestReader`, the same
`ResampleAudioStage` used by the audio tagging tutorial, the generic `ASRStage`
configured with `QwenASRAdapter`, and `ManifestWriterStage`.

There is no Qwen-specific inference stage or tutorial runner. Model lifecycle,
audio loading and normalization, language routing, batching, skip handling,
and output assembly stay in the shared ASR stage; the adapter only owns the
Qwen3-ASR model call.

## Requirements

- x86_64 Linux with one CUDA GPU per ASR actor
- `ffmpeg`
- Audio files accessible from the machine running the pipeline

From the Curator repository root, install the opt-in Qwen3-ASR dependency:

```bash
uv sync --extra audio_cuda12 --extra vllm
source .venv/bin/activate
```

`ResampleAudioStage` invokes `ffmpeg`, so input audio may use any format
supported by the installed build. It writes PCM 16-bit, 16 kHz, mono WAV files
under `resampled_audio_dir` before inference.

## Run the bundled smoke input

The bundled manifest contains two short OPUS files:

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/qwen_asr \
  --config-name pipeline \
  manifest_path=tests/fixtures/audio/tagging/sample_input.jsonl \
  output_path=/tmp/qwen_asr_output.jsonl \
  workspace_dir=/tmp/qwen_asr_workspace \
  default_language=en
```

`--config-path` is relative to `nemo_curator/config/run.py`; manifest, audio,
and output paths resolve from the current working directory. Run the command
from the repository root as shown.

The first run downloads `Qwen/Qwen3-ASR-0.6B`. The pipeline allocates one GPU
to each ASR actor. `QwenASRAdapter` constructs `Qwen3ASRModel.LLM`, matching the
nkoluguri reference's vLLM backend and engine settings. The
`audio_cuda12` extra installs qwen-asr without its conflicting vLLM pin, while
the shared `vllm` extra provides the same Curator-pinned engine stack used by
Qwen-Omni.

## Effective defaults

The tutorial caps vLLM's model context at 8,192 tokens so the bundled smoke
input can run on a 12 GB GPU. This does not change the adapter's default.

| Setting | Tutorial value |
|---|---:|
| Executor | Ray Data |
| ASR stage batch size | `128` |
| GPUs per ASR actor | `1` |
| GPU memory limit | `0.7` of device memory |
| Maximum vLLM model length | `8192` tokens |
| Maximum generated tokens | `4096` |
| Qwen internal inference batch cap | `128` |
| Prediction field | `pred_text` |
| Adapter extras field | `asr_extras` |

The stage batch is passed to one adapter `transcribe_batch()` call. The adapter
forwards `max_inference_batch_size` to Qwen3-ASR as its internal cap.

## Select the executor

Ray Data is the default. To use Xenna streaming:

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/qwen_asr \
  --config-name pipeline \
  manifest_path=/data/input.jsonl \
  output_path=/tmp/qwen_asr_output.jsonl \
  backend=xenna \
  execution_mode=streaming
```

Use Xenna batch mode only when its streaming mode is unsuitable for the
workload:

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/qwen_asr \
  --config-name pipeline \
  manifest_path=/data/input.jsonl \
  output_path=/tmp/qwen_asr_output.jsonl \
  backend=xenna \
  execution_mode=batch
```

`execution_mode` applies only to Xenna and is ignored for Ray Data.

## Input and output

The input is JSONL with one object per audio file. Each object must contain
`audio_filepath` and, unless `default_language` is set, `source_lang`:

```json
{"audio_filepath": "/data/sample.wav", "source_lang": "en"}
```

`ResampleAudioStage` preserves `audio_filepath` and adds `audio_item_id`,
`resampled_audio_filepath`, and `duration`. `ASRStage` opens the resampled file
for its current batch, converts it to contiguous mono 16 kHz NumPy samples,
and passes the waveform plus language to the adapter. Waveforms are not stored
in the output task data.

`ASRStage` adds the configured prediction column and, when applicable,
`_skipme` or `additional_notes`. Non-empty adapter metadata is written as one
nested dictionary under `asr_extras`; Qwen3-ASR currently emits
`asr_extras.detected_language`. Override the fields with, for example,
`pred_text_key=qwen_transcript extras_key=qwen_metadata`, or disable metadata
output with `extras_key=null`.

Rows remain in the output when inference is skipped:

- unreadable audio uses `_skipme: audio_load_error`;
- audio shorter than the adapter minimum uses `_skipme: empty_audio`;
- unsupported languages use `_skipme: language_not_supported`;
- missing language metadata uses `_skipme: language_missing` unless
  `default_language` is configured.

The default language allowlist matches the
[official Qwen3-ASR language table](https://github.com/QwenLM/Qwen3-ASR#supported-languages):
`zh`, `en`, `yue`,
`ar`, `de`, `fr`, `es`, `pt`, `id`, `it`, `ko`, `ru`, `th`, `vi`, `ja`,
`tr`, `hi`, `ms`, `nl`, `sv`, `da`, `fi`, `pl`, `cs`, `fil`, `fa`, `el`,
`hu`, `mk`, and `ro`.

## Scope

This is a functional manifest-to-transcript example. It does not perform word
alignment, diarization, WER calculation, hallucination recovery, or benchmark
reporting. The audio tagging pipeline continues to use
`NeMoASRAlignerStage` where word timestamps are required.
