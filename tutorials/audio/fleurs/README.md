# Curating the FLEURS Dataset with NeMo Curator

Download a [FLEURS](https://huggingface.co/datasets/google/fleurs) split, run ASR, and filter by WER — all in a single pipeline.

## Overview

FLEURS contains spoken utterances across 100+ languages. This pipeline downloads a split, transcribes it with a NeMo ASR model, scores WER against the reference, computes durations, and writes a filtered JSONL manifest.

### Pipeline flow

```
┌──────────────────┐    ┌────────────────┐    ┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────┐
│ CreateInitial    │───▶│ InferenceAsr   │───▶│ GetPairwise  │───▶│ GetAudio     │───▶│ PreserveBy   │───▶│ AudioTo     │───▶│ JsonlWriter│
│ ManifestFleurs   │    │ NemoStage      │    │ WerStage     │    │ DurationStage│    │ ValueStage   │    │ DocumentStg │    │            │
└──────────────────┘    └────────────────┘    └─────────────┘    └──────────────┘    └──────────────┘    └─────────────┘    └────────────┘
  HF download + JSONL     GPU transcription    WER computation     duration calc       WER ≤ threshold     AudioTask→Doc      write JSONL
```

## Prerequisites

- Python 3.10+
- NeMo Curator installed (see [installation guide](https://docs.nvidia.com/nemo/curator/latest/admin/installation.html))
- **GPU**: Recommended for ASR inference. Minimum ~4 GB VRAM for FastConformer/Parakeet models. Pass `--gpus 0` for CPU fallback (10–50x slower).
- **System packages**: None

```bash
# GPU (recommended)
uv sync --extra audio_cuda12

# CPU only
uv sync --extra audio_cpu
```

## Dataset

| Property | Value |
|---|---|
| **Source** | [google/fleurs](https://huggingface.co/datasets/google/fleurs) on HuggingFace |
| **Format** | WAV audio + text transcriptions |
| **Size** | ~50 MB per language split (auto-downloaded) |
| **License** | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| **Auto-download** | Yes — handled by `CreateInitialManifestFleursStage` |

## Quick start

```bash
python tutorials/audio/fleurs/pipeline.py \
  --raw_data_dir ./example_audio/fleurs \
  --model_name nvidia/stt_hy_fastconformer_hybrid_large_pc \
  --lang hy_am \
  --split dev \
  --wer_threshold 75 \
  --gpus 1 \
  --clean \
  --verbose
```

## Usage

### All CLI options (`pipeline.py`)

| Argument | Default | Description |
|---|---|---|
| `--raw_data_dir` | *(required)* | Workspace directory for downloads and outputs |
| `--model_name` | `nvidia/stt_hy_fastconformer_hybrid_large_pc` | NeMo ASR model (change per language) |
| `--lang` | `hy_am` | FLEURS language code (e.g., `en_us`, `fr_fr`) |
| `--split` | `dev` | FLEURS split: `train`, `dev`, or `test` |
| `--wer_threshold` | `75.0` | Keep samples with WER ≤ this value |
| `--gpus` | `1.0` | GPUs for ASR stage (0 for CPU fallback) |
| `--backend` | `xenna` | Execution backend: `xenna` or `ray_data` |
| `--clean` | off | Remove existing `result/` directory before writing |
| `--verbose` | off | Enable DEBUG-level logging |

### Using custom data

To use a different language with the appropriate model:

```bash
python tutorials/audio/fleurs/pipeline.py \
  --raw_data_dir ./example_audio/fleurs \
  --model_name nvidia/parakeet-tdt-0.6b-v2 \
  --lang en_us \
  --split dev \
  --wer_threshold 75 \
  --gpus 1 --clean
```

FLEURS language codes follow the dataset convention (e.g., `en_us`, `fr_fr`, `hy_am`). See the [dataset card](https://huggingface.co/datasets/google/fleurs) for the full list. Use the corresponding NeMo ASR model for your target language.

### Choosing a backend

| Backend | Description | When to use |
|---|---|---|
| `xenna` | Default. Cosmos-Xenna streaming engine with automatic worker allocation. | Most workloads, CI/nightly benchmarks. |
| `ray_data` | Built on Ray Data `map_batches`. | Development, machines without Xenna GPU support. |

### YAML config + Hydra

The same pipeline can be run via `run.py` with Hydra-based configuration:

```bash
python tutorials/audio/fleurs/run.py \
  --config-path . --config-name pipeline.yaml \
  raw_data_dir=./example_audio/fleurs
```

Override values without editing the file:

```bash
python tutorials/audio/fleurs/run.py \
  --config-path . --config-name pipeline.yaml \
  raw_data_dir=./example_audio/fleurs \
  data_split=dev \
  processors.0.lang=en_us \
  processors.1.model_name=nvidia/stt_en_conformer_ctc_large \
  processors.4.target_value=50.0 \
  backend=ray_data
```

Override indices map to the `processors` list in `pipeline.yaml`:

| Override | Stage |
|---|---|
| `processors.0.lang` | FLEURS downloader language |
| `processors.1.model_name` | ASR model for inference |
| `processors.4.target_value` | WER filter threshold |
| `data_split` | Top-level variable referenced by the first stage |
| `backend` | `xenna` (default) or `ray_data` |

## Pipeline stages

### 1. `CreateInitialManifestFleursStage`

Downloads the FLEURS split from HuggingFace (if not cached) and emits one `AudioTask` per utterance with `audio_filepath` and `text`.

### 2. `InferenceAsrNemoStage`

Runs a NeMo ASR model on each audio file (GPU-accelerated). Adds `pred_text` to the task data.

### 3. `GetPairwiseWerStage`

Computes word error rate between `text` (reference) and `pred_text` (predicted). Adds `wer` as a percentage (0–100).

### 4. `GetAudioDurationStage`

Reads the audio file and computes its duration in seconds. Adds `duration`.

### 5. `PreserveByValueStage`

Keeps only tasks where `wer ≤ wer_threshold`. Tasks exceeding the threshold are dropped.

### 6. `AudioToDocumentStage` + `JsonlWriter`

Converts surviving `AudioTask` objects to `DocumentBatch` format and writes JSONL to `${raw_data_dir}/result/`.

## Parameters and tuning

### WER threshold

WER (Word Error Rate) measures transcription accuracy on a 0–100 scale: 0 = perfect match, 100 = every word wrong. Values above 100 are possible when insertions outnumber deletions.

The default `--wer_threshold 75` is **intentionally permissive** — it exists as a demonstration value, not a production recommendation. At 75%, three-quarters of the words can be wrong and the sample still passes. This ensures the tutorial produces non-empty output for any language/model combination, even poorly-matched ones.

**Recommended thresholds by use case:**

| Use case | Threshold | Rationale |
|---|---|---|
| Tutorial / demo | 75 (default) | Maximizes output for any language+model pair |
| ASR fine-tuning (high recall) | 40–60 | Keeps noisy-but-usable training data |
| ASR evaluation / benchmarking | 20–30 | Focuses on reasonably accurate transcriptions |
| Production data curation | 10–25 | High-quality data for downstream models |
| Ground-truth validation | 5–10 | Near-perfect match required |

Lower thresholds produce cleaner but smaller datasets. If output is empty, the model likely does not support the target language — try a different `--model_name`.

### Other parameters

| Parameter | Range | Effect |
|---|---|---|
| `gpus` | 0–N | 0 = CPU-only (slow). 1 = single GPU (recommended). Higher values not yet parallelized per-stage. |
| `batch_size` | 1–64+ | Increase for higher throughput on high-VRAM GPUs; decrease if OOM. Default is 4 for the download stage. |

## Output format

Results are written as JSONL to `${raw_data_dir}/result/`. Each line contains:

```json
{
  "audio_filepath": "relative/path/to/audio.wav",
  "text": "reference transcription from FLEURS",
  "pred_text": "predicted transcription from ASR model",
  "wer": 12.5,
  "duration": 4.21
}
```

| Field | Type | Description |
|---|---|---|
| `audio_filepath` | string | Relative path to the WAV file |
| `text` | string | Ground-truth transcription from the FLEURS dataset |
| `pred_text` | string | ASR model's predicted transcription |
| `wer` | float | Word Error Rate (0–100) between `text` and `pred_text` |
| `duration` | float | Audio duration in seconds |

## Performance

### Timing estimates

| Dataset | Samples | Wall-clock time | Hardware |
|---|---|---|---|
| `hy_am` dev split | ~200 | ~2–5 minutes | 1x A100 GPU |
| `en_us` dev split | ~400 | ~5–10 minutes | 1x A100 GPU |
| Any split (CPU fallback) | ~200 | ~30–60 minutes | 8-core CPU |

Most time is spent in ASR inference (Stage 2). Download, WER, and duration stages are near-instant.

### Expected filtering ratios

With default `--wer_threshold 75` (very permissive):
- Most language+model pairs: **>90% pass rate** (very few samples have WER above 75%)
- Mismatched language/model: **0–30% pass rate** (model can't transcribe the language)

With a stricter `--wer_threshold 25`:
- Well-matched model: **50–80% pass rate**
- Poorly-matched model: **<10% pass rate**

If your output is empty, the model likely does not support the target language.

## Composability

The FLEURS stages can be composed with other NeMo Curator audio stages:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.datasets.fleurs.create_initial_manifest import CreateInitialManifestFleursStage
from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
from nemo_curator.stages.audio.metrics.get_wer import GetPairwiseWerStage

pipeline = Pipeline(
    name="fleurs-custom",
    stages=[
        CreateInitialManifestFleursStage(lang="en_us", split="dev", raw_data_dir="./data"),
        InferenceAsrNemoStage(model_name="nvidia/parakeet-tdt-0.6b-v2"),
        GetPairwiseWerStage(text_key="text", pred_text_key="pred_text", wer_key="wer"),
        # Add your custom stages here (e.g., speaker diarization, additional filtering)
    ],
)
```

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `Result directory already exists` | Previous run output not cleaned | Add `--clean` flag |
| OOM during ASR inference | GPU VRAM too small for model + batch | Reduce batch size in `pipeline.py` or use a smaller model |
| `--gpus 0` but very slow | CPU inference is 10-50x slower than GPU | Use a GPU; CPU is only for testing |
| Empty output JSONL | `wer_threshold` too strict for the model+language pair | Increase `--wer_threshold` or use a better-matching ASR model |
| HuggingFace download fails | Network/auth issue | Check connectivity; some splits may need `huggingface-cli login` |
| Wrong language code | Typo in `--lang` | Consult the [FLEURS dataset card](https://huggingface.co/datasets/google/fleurs) for valid codes |
| SIGSEGV / actor crash during model load | gRPC thread-safety race | See [Known Issues](../README.md#known-issues) — set `OTEL_SDK_DISABLED=true` |

## License

FLEURS is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See the [dataset card](https://huggingface.co/datasets/google/fleurs) for full terms.
