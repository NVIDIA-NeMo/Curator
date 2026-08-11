# Chatterbox TTS Synthesis

Synthesise multi-speaker conversation audio from text using Chatterbox TTS with reference-voice cloning.

## Overview

This tutorial runs the `ChatterboxTTSStage` over a JSONL manifest of conversation turns and produces one WAV file per turn, with speaker voices cloned from a reference audio dataset and kept consistent within each conversation. It supports both the English-only model (`ChatterboxTTS`) and the multilingual model (`ChatterboxMultilingualTTS`, 23 languages), and caches outputs so re-runs reuse existing files.

Each cached `<hash>.wav` is written alongside a `<hash>.json` sidecar recording every setting that affected it (language, voice, sampling params, etc.). A cache hit is only trusted if the sidecar matches the current run's settings exactly, so changing any generation setting (e.g. switching `--language`) always produces fresh audio instead of silently reusing a previous, differently-configured run's file. Cache entries are written atomically, so a crash mid-write can never leave a corrupt file that looks like a valid cache hit.

### Pipeline flow

```
┌──────────────┐    ┌──────────────────┐    ┌────────────────────┐    ┌──────────────┐
│ManifestReader│───▶│ ChatterboxTTS    │───▶│ AudioToDocument    │───▶│ JsonlWriter  │
│ (turn JSONL) │    │ (synthesise WAV) │    │ (AudioTask -> doc) │    │ (manifest)   │
└──────────────┘    └──────────────────┘    └────────────────────┘    └──────────────┘
   turns input           GPU                                              enriched output
```

## Prerequisites

- Python 3.11+
- NeMo Curator installed (see [installation guide](https://docs.nvidia.com/nemo/curator/latest/admin/installation.html))
- **GPU**: Required — ~4 GB VRAM
- **System packages**: `ffmpeg`

```bash
# GPU (required for synthesis)
uv sync --extra audio_cuda12
```

`chatterbox-tts` is **not** part of the `audio_common`/`audio_cuda12` extras. It
hard-pins `transformers==5.2.0` and `torch==2.6.0`, which are incompatible with
Curator's pinned stack (`transformers>=4.56,<5.0`, `torch==2.10.0`) and would
make the audio extras unresolvable. Instead, `ChatterboxTTSStage` declares a Ray
`runtime_env` that pip-installs `chatterbox-tts` (plus `setuptools<81`, which
still provides the `pkg_resources` its watermarker needs) into an **isolated
virtualenv** at runtime, so its conflicting pins never touch the main
environment. The first run therefore provisions that environment (one-time
download of chatterbox and its deps); subsequent runs reuse Ray's cached
virtualenv.

> **Requires `--backend ray_data`.** Only the Ray Data (and Ray actor pool)
> backends honor a stage's `runtime_env`; the default `xenna` backend does not,
> so under `xenna` you must install chatterbox into the main environment
> yourself. Run this tutorial with `--backend ray_data` to use the auto-managed
> isolated environment.

If you have already provisioned chatterbox in a dedicated environment and want
to reuse it instead of the auto-managed one, disable the isolated runtime with
`ChatterboxTTSStage(...).with_(runtime_env={})`.

## Dataset

This tutorial needs two inputs you provide: a **turn manifest** and a **reference voices dataset**.

| Property | Value |
|---|---|
| **Source** | Bring your own |
| **Format** | JSONL turn manifest + reference audio directory |
| **Size** | Varies |
| **License** | N/A (your data) |
| **Auto-download** | No |

**Turn manifest** — one conversation turn per line:

```json
{"conversation_id": "conv001", "speaker": "Alice", "utterance": "Hello, how are you?"}
{"conversation_id": "conv001", "speaker": "Bob", "utterance": "I'm doing well, thanks!"}
```

**Reference voices dataset** — one of two layouts:

- `wavs/<dialog>/<speaker>.wav` (optional sibling `rttms/<dialog>/<speaker>.rttm` for silence stripping)
- MLS layout `<speaker>/<book>/<segment>.flac` (segments auto-concatenated up to `max_reference_duration`)

## Quick start

```bash
python tutorials/audio/tts/run.py --config-path . --config-name pipeline \
  input_manifest=/data/turns.jsonl \
  reference_voices_dataset=/data/reference_voices \
  output_dir=/data/tts_output
```

Expected output:

```
[TTS] conv001/Alice: 1.84s -> 8f1c.._Alice_3a9d.._b2e1.wav
[TTS] conv001/Bob: 2.07s -> 8f1c.._Bob_77c0.._d4a8.wav
Pipeline completed in 12.43s (0.21 min)
Results written to /data/tts_output/result/*.jsonl
```

## Usage

### All CLI options (`pipeline.py`)

| Argument | Default | Description |
|---|---|---|
| `--input-manifest` | *(required)* | JSONL manifest of conversation turns |
| `--reference-voices-dataset` | *(required)* | Root directory of reference audio |
| `--output-dir` | *(required)* | Root output directory for the result manifest |
| `--output-audio-dir` | `<output-dir>/audio` | Directory for generated WAV files |
| `--language` | `None` (English) | ISO 639-1 code for the multilingual model |
| `--device` | `cuda` | Torch device for inference |
| `--cache-dir` | `None` | HuggingFace cache for Chatterbox weights |
| `--sample-rate` | `24000` | Output WAV sample rate. Chatterbox always synthesises at 24000 Hz; a different value here is honored by resampling the output before writing |
| `--cfg-weight` | `0.5` | Classifier-free guidance weight |
| `--exaggeration` | `0.5` | Emotion exaggeration |
| `--temperature` | `0.8` | Sampling temperature |
| `--max-reference-duration` | `60.0` | Max seconds of reference speech to use |
| `--clean` | off | Remove `<output-dir>/result/` before running (Hydra: `clean=true`) |
| `--backend` | `xenna` | Execution backend: `xenna` or `ray_data` |

### Using custom data (argparse runner)

```bash
python tutorials/audio/tts/pipeline.py \
  --input-manifest /data/turns.jsonl \
  --reference-voices-dataset /data/reference_voices \
  --output-dir ./my_output
```

### Multilingual synthesis

```bash
python tutorials/audio/tts/run.py --config-path . --config-name pipeline \
  input_manifest=/data/turns.jsonl \
  reference_voices_dataset=/data/mls_french \
  output_dir=/data/tts_output_fr \
  language=fr
```

Supported languages: `ar`, `da`, `de`, `el`, `en`, `es`, `fi`, `fr`, `he`, `hi`, `it`, `ja`, `ko`, `ms`, `nl`, `no`, `pl`, `pt`, `ru`, `sv`, `sw`, `tr`, `zh`.

### Choosing a backend

| Backend | Description | When to use |
|---|---|---|
| `xenna` | Default. Cosmos-Xenna streaming engine with automatic worker allocation. Does **not** honor the stage `runtime_env`, so chatterbox must already be installed in the environment. | Workloads where you pre-install chatterbox yourself. |
| `ray_data` | Built on Ray Data `map_batches`. Honors the stage `runtime_env`, so chatterbox is auto-installed into an isolated virtualenv. | **Recommended for this tutorial** — enables the isolated chatterbox runtime described above. |

## Pipeline stages

### Stage 1: `ManifestReader`

Reads the JSONL turn manifest line-by-line and emits one `AudioTask` per turn (no Pandas; ~1x file-size memory).

### Stage 2: `ChatterboxTTSStage`

Loads the English or multilingual Chatterbox model, assigns each speaker a reference voice (consistent within a conversation), and synthesises audio for each turn. Output filenames are deterministic (hash of conversation/speaker/text/reference), so existing files are reused on re-runs. Runs on GPU with `batch_size=1` (turns are synthesised serially).

### Stage 3: `AudioToDocumentStage`

Converts each `AudioTask` into a document row for manifest writing.

### Stage 4: `JsonlWriter`

Writes the enriched manifest to `<output-dir>/result/*.jsonl`.

### Re-running the pipeline

`AudioToDocumentStage` carries the input manifest's path through as
`source_files`, so `JsonlWriter` names each output shard deterministically
from it instead of a random UUID. **Re-running with an unchanged manifest
does not spawn ever-more result shards** — but neither entry point cleans
`<output-dir>/result/` for you by default, so results from a *different*
prior run (e.g. a different manifest, or a bigger one) are left in place
alongside the new shards.

Both entry points share the same opt-in policy: pass `--clean`
(`pipeline.py`) or `clean=true` (`run.py`, Hydra) to remove
`<output-dir>/result/` before the run for a guaranteed-fresh set of outputs.

## Parameters and tuning

| Parameter | Range | Effect |
|---|---|---|
| `cfg_weight` | `0.0` – `1.0` | Higher tracks the reference voice/prompt more closely; lower is more free. |
| `exaggeration` | `0.25` – `1.0` | Higher increases emotional/style intensity. A `[min, max]` list randomises per conversation. |
| `temperature` | `0.5` – `1.0` | Higher increases variability of the generated speech. |
| `max_reference_duration` | `10` – `60` | More reference audio improves voice cloning but increases preprocessing time. |

## Output format

Results are written to `<output-dir>/result/*.jsonl`. Each line is the input turn enriched with:

```json
{
  "conversation_id": "string — conversation the turn belongs to",
  "speaker": "string — speaker label",
  "utterance": "string — turn text",
  "audio_filepath": "string — path to the generated WAV",
  "duration": "float — audio duration in seconds",
  "reference_voice": "string — identifier of the reference voice used"
}
```

| Field | Type | Description |
|---|---|---|
| `audio_filepath` | string | Path to the generated WAV file |
| `duration` | float | Duration of the generated audio (seconds) |
| `reference_voice` | string | Reference-voice identifier used for this speaker |

## Composability

`ChatterboxTTSStage` is a regular `AudioTask` stage, so it drops into any custom pipeline alongside other audio stages, e.g. computing duration for each generated turn:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.common import GetAudioDurationStage, ManifestReader
from nemo_curator.stages.audio.tts import ChatterboxTTSStage

pipeline = Pipeline(
    name="custom",
    stages=[
        ManifestReader(manifest_path="turns.jsonl"),
        ChatterboxTTSStage(
            output_audio_dir="out/audio",
            reference_voices_dataset="reference_voices",
        ),
        GetAudioDurationStage(),
    ],
)
```

A larger topic → conversation → TTS → forced-alignment → merge data-generation workflow is planned, but the alignment and merge stages it depends on are not part of this PR; that composed tutorial will be documented once those stages land.

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `No reference audio found` | Dataset doesn't match either expected layout | Use `wavs/<dialog>/<speaker>.wav` or MLS `<spk>/<book>/<seg>.flac` |
| `Unsupported language` | `language` is not a supported ISO 639-1 code | Use one of the codes listed above, or omit for English |
| All outputs are ~2s of silence | TTS inference failed per turn (graceful fallback) | Run with `--verbose`; check GPU memory and model download |
| No output for 2+ minutes at start | First-run model download from HuggingFace | Wait; check `~/.cache/huggingface/` for growing files |

## Citation / License

Chatterbox TTS by Resemble AI — see the [chatterbox model card / repository](https://github.com/resemble-ai/chatterbox) for model details and license terms. Reference audio and generated outputs are subject to the licenses of the data you supply.
