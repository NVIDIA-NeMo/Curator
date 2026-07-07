# Chatterbox TTS Synthesis

Synthesise multi-speaker conversation audio from text using Chatterbox TTS with reference-voice cloning.

## Overview

This tutorial runs the `ChatterboxTTSStage` over a JSONL manifest of conversation turns and produces one WAV file per turn, with speaker voices cloned from a reference audio dataset and kept consistent within each conversation. It supports both the English-only model (`ChatterboxTTS`) and the multilingual model (`ChatterboxMultilingualTTS`, 23 languages), and caches outputs deterministically so re-runs reuse existing files.

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

The `chatterbox-tts` dependency is included in the `audio_common` extra (pulled in by `audio_cuda12`).

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
| `--sample-rate` | `24000` | Output WAV sample rate |
| `--cfg-weight` | `0.5` | Classifier-free guidance weight |
| `--exaggeration` | `0.5` | Emotion exaggeration |
| `--temperature` | `0.8` | Sampling temperature |
| `--max-reference-duration` | `60.0` | Max seconds of reference speech to use |
| `--clean` | off | Remove the result directory before running |
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
| `xenna` | Default. Cosmos-Xenna streaming engine with automatic worker allocation. | Most workloads, CI/nightly benchmarks. |
| `ray_data` | Built on Ray Data `map_batches`. | Development, machines without Xenna GPU support, or Ray Data integration preferred. |

## Pipeline stages

### Stage 1: `ManifestReader`

Reads the JSONL turn manifest line-by-line and emits one `AudioTask` per turn (no Pandas; ~1x file-size memory).

### Stage 2: `ChatterboxTTSStage`

Loads the English or multilingual Chatterbox model, assigns each speaker a reference voice (consistent within a conversation), and synthesises audio for each turn. Output filenames are deterministic (hash of conversation/speaker/text/reference), so existing files are reused on re-runs. Runs on GPU with `batch_size=1` (turns are synthesised serially).

### Stage 3: `AudioToDocumentStage`

Converts each `AudioTask` into a document row for manifest writing.

### Stage 4: `JsonlWriter`

Writes the enriched manifest to `<output-dir>/result/*.jsonl`.

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

The TTS stage pairs naturally with conversation generation upstream and forced alignment downstream:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.common import ManifestReader
from nemo_curator.stages.audio.tts import ChatterboxTTSStage
from nemo_curator.stages.audio.alignment import MFAAlignmentStage

pipeline = Pipeline(
    name="custom",
    stages=[
        ManifestReader(manifest_path="turns.jsonl"),
        ChatterboxTTSStage(
            output_audio_dir="out/audio",
            reference_voices_dataset="reference_voices",
        ),
        MFAAlignmentStage(output_dir="out/alignment", text_key="utterance"),
    ],
)
```

For the full topic → conversation → TTS → alignment → merge workflow, see the [data-generation tutorial](../data-generation/).

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `No reference audio found` | Dataset doesn't match either expected layout | Use `wavs/<dialog>/<speaker>.wav` or MLS `<spk>/<book>/<seg>.flac` |
| `Unsupported language` | `language` is not a supported ISO 639-1 code | Use one of the codes listed above, or omit for English |
| All outputs are ~2s of silence | TTS inference failed per turn (graceful fallback) | Run with `--verbose`; check GPU memory and model download |
| No output for 2+ minutes at start | First-run model download from HuggingFace | Wait; check `~/.cache/huggingface/` for growing files |

## Citation / License

Chatterbox TTS by Resemble AI — see the [chatterbox model card / repository](https://github.com/resemble-ai/chatterbox) for model details and license terms. Reference audio and generated outputs are subject to the licenses of the data you supply.
