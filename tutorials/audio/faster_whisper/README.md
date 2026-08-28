# Faster-Whisper ASR Adapter Tutorial

This tutorial reads a NeMo-style audio manifest, normalizes each file to a
16 kHz mono WAV, transcribes it with Faster-Whisper Large-v3, and writes the
results to JSONL. It composes `ManifestReader`, `ResampleAudioStage`, the generic
`ASRStage` configured with `FasterWhisperASR`, and `ManifestWriterStage`.

The defaults reproduce the eligible-sample inference contract in
[`nkoluguri/integration-test`](https://github.com/nithinraok/Curator/tree/1e2d639946f28dfd289a0a9456480385d4480c28):
beam size 5, built-in VAD enabled, timestamps disabled, and one forced language
code per sample.

## Requirements

- x86_64 Linux
- `ffmpeg`
- One CUDA GPU per ASR actor for the default configuration, or a supported CPU
  for the CPU override
- Audio files accessible from the machine running the pipeline

Install the audio stack from the Curator repository root:

```bash
# GPU (default tutorial configuration)
uv sync --extra audio_cuda12

# Or CPU only
uv sync --extra audio_cpu

source .venv/bin/activate
```

`ResampleAudioStage` invokes `ffmpeg`, so input audio may use any format
supported by the installed build. It writes PCM 16-bit, 16 kHz, mono WAV files
under `resampled_audio_dir` before inference.

## Run the bundled smoke input

The bundled manifest contains two short OPUS files without language fields, so
the command supplies English as the default forced language:

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/faster_whisper \
  --config-name pipeline \
  manifest_path=tests/fixtures/audio/tagging/sample_input.jsonl \
  output_path=/tmp/faster_whisper_output.jsonl \
  workspace_dir=/tmp/faster_whisper_workspace \
  default_language=en
```

The first run downloads the configured `large-v3` checkpoint into
Faster-Whisper's local cache. Set `model_revision=<revision>` to use the same
revision for node prefetch and worker model loading.

## Effective defaults

| Setting | Tutorial value |
|---|---:|
| Model | `large-v3` |
| ASR stage batch size | `128` |
| GPUs per ASR actor | `1` |
| GPU compute type | `float16` |
| CPU compute type | `int8` |
| Beam size | `5` |
| Faster-Whisper VAD | Enabled |
| Timestamps | Disabled |
| Prediction field | `asr_prediction` |
| Adapter extras field | `asr_extras` |

The stage batch size controls how many tasks Curator sends to one
`transcribe_batch()` call. `FasterWhisperASR` then calls
`WhisperModel.transcribe()` once per eligible audio, in order. This is
sequential per-audio inference; it does not use Faster-Whisper's
`BatchedInferencePipeline` and does not turn 128 files into one native model
batch.

## Select GPU or CPU execution

The default requests one GPU per actor. To use CPU inference, install
`audio_cpu` and set the GPU request to zero:

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/faster_whisper \
  --config-name pipeline \
  manifest_path=/data/input.jsonl \
  output_path=/tmp/faster_whisper_cpu_output.jsonl \
  default_language=en \
  gpus_per_actor=0
```

GPU workers use `float16`; CPU workers use `int8` by default.

## Input and output

Each JSONL input row must contain `audio_filepath` and a supported
`source_lang`, unless `default_language` is configured:

```json
{"audio_filepath": "/data/sample.wav", "source_lang": "en"}
```

The configured allowlist contains Faster-Whisper Large-v3's 100 canonical
language codes. It also accepts the input aliases normalized by the adapter:
`fil` to `tl`, `jv` to `jw`, `iw` to `he`, `in` to `id`, `ji` to `yi`, and
`nb` to `no`. Missing or unsupported languages are filtered by `ASRStage`
before the adapter is called.

`ASRStage` writes transcription text to `asr_prediction`. It writes the forced,
normalized language code under `asr_extras.language_code`; this value is the
requested inference language, not a detected language.

The adapter intentionally discards Faster-Whisper's `TranscriptionInfo` and
does not emit detected-language confidence, duration metadata, segment
timestamps, or word timestamps. Set `extras_key=null` to omit the nested
language metadata as well.

Rows remain in the output when processing is skipped. The shared stage uses
`_skipme` and `additional_notes` to describe audio-loading, missing-language,
and unsupported-language cases.

## Select the executor

Ray Data is the default. To use Xenna streaming:

```bash
python nemo_curator/config/run.py \
  --config-path ../../tutorials/audio/faster_whisper \
  --config-name pipeline \
  manifest_path=/data/input.jsonl \
  output_path=/tmp/faster_whisper_output.jsonl \
  default_language=en \
  backend=xenna \
  execution_mode=streaming
```

Use `execution_mode=batch` for Xenna batch execution. `execution_mode` is
ignored by Ray Data.

## Scope

This is a functional manifest-to-transcript example. It does not perform native
multi-file Faster-Whisper batching, timestamp extraction, word alignment,
diarization, WER calculation, or hallucination recovery.
