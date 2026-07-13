# Audio Curation Tutorials

Hands-on tutorials for curating audio data with NeMo Curator.

**New to audio curation?** Start with the [Audio Getting Started Guide](https://docs.nvidia.com/nemo/curator/latest/get-started/audio.html) for setup and basic concepts.

## Platform support

Audio curation requires **x86_64 Linux**. The `audio_cpu` and `audio_cuda12` extras omit several dependencies on arm64/aarch64 (NeMo ASR, diarization, and related tooling) because upstream packages do not ship aarch64 wheels. The arm64 NeMo Curator container therefore does not include the full audio stack — use amd64 for ASR, diarization, and tagging tutorials below.

## Getting started in 5 minutes

No data download needed — run the ALM pipeline on bundled fixtures:

```bash
# Install (CPU is fine for this)
uv sync --extra audio_cpu && source .venv/bin/activate

# Run the ALM pipeline on sample data
python tutorials/audio/alm/main.py \
  --config-path . --config-name pipeline \
  manifest_path=tests/fixtures/audio/alm/sample_input.jsonl
```

Expected output in under 10 seconds:

```
PIPELINE COMPLETE
==================================================
  Output entries: 5
  [alm_data_builder] windows_created: 181
  [alm_data_overlap] output_windows (after overlap): 25
```

**With a GPU?** Try the FLEURS pipeline — it auto-downloads data and runs ASR:

```bash
uv sync --extra audio_cuda12 && source .venv/bin/activate

python tutorials/audio/fleurs/main.py \
  --config-path . --config-name pipeline \
  raw_data_dir=./example_audio/fleurs \
  lang=en_us \
  stages.1.model_name=nvidia/parakeet-tdt-0.6b-v2 \
  stages.1.resources.gpus=1
```

## Which tutorial should I use?

| I want to... | Tutorial | GPU | Data |
|---|---|---|---|
| Curate multilingual ASR data (download, transcribe, filter by WER) | [**fleurs/**](fleurs/) | Yes (~4 GB VRAM) | Auto-downloads from HuggingFace |
| Build training windows for Audio Language Models from diarized manifests | [**alm/**](alm/) | No (CPU-only) | Bundled sample fixtures |
| Label raw audio for TTS/ASR/ALM via diarization, alignment, and quality metrics | [**tagging/**](tagging/) | Yes (~8 GB VRAM) | Bring your own audio manifest |
| Evaluate speaker diarization (DER) on a benchmark dataset | [**callhome_diar/**](callhome_diar/) | Yes (~8 GB VRAM) | Requires [LDC license](https://catalog.ldc.upenn.edu/LDC97S42) |
| Filter a manifest to keep only single-speaker audio | [**single_speaker_filter/**](single_speaker_filter/) | Yes (~8 GB VRAM) | Requires a pre-existing JSONL manifest |
| Quality-filter raw audio (MOS, VAD, bandwidth, noise) | [**readspeech/**](readspeech/) | Recommended (~4 GB VRAM) | Auto-downloads DNS Challenge (4.88 GB) |

## Data availability

| Tutorial | Auto-download | Size | Notes |
|---|---|---|---|
| `fleurs/` | Yes | ~50 MB per language split | Downloads from HuggingFace `google/fleurs` |
| `alm/` | N/A | Bundled | Uses `tests/fixtures/audio/alm/sample_input.jsonl` (5 entries) |
| `tagging/` | No | Varies | Bring your own NeMo-style JSONL manifest with audio paths |
| `callhome_diar/` | No | ~1 GB | Requires LDC membership and license ([LDC97S42](https://catalog.ldc.upenn.edu/LDC97S42)) |
| `single_speaker_filter/` | No | Varies | Bring your own NeMo-style JSONL manifest |
| `readspeech/` | Yes | 4.88 GB compressed | Downloads DNS Challenge Read Speech (14,279 WAV files) |

## System dependencies

Audio pipelines require `ffmpeg` for resampling and format conversion. Install it before running any audio tutorial:

```bash
# Ubuntu / Debian
sudo apt-get install -y ffmpeg
```

| Tutorial | System packages | Pip extras |
|---|---|---|
| `fleurs/` | `ffmpeg` | `audio_cpu` or `audio_cuda12` |
| `alm/` | `ffmpeg` | `audio_cpu` |
| `tagging/` | `ffmpeg` | `audio_cuda12` |
| `callhome_diar/` | `ffmpeg`, `sox` | `audio_cuda12` |
| `single_speaker_filter/` | `ffmpeg` | `audio_cuda12` |
| `readspeech/` | `ffmpeg` | `audio_cuda12` (recommended) or `audio_cpu` |

Install pip extras from the repo root:

```bash
# GPU (recommended)
uv sync --extra audio_cuda12

# CPU only
uv sync --extra audio_cpu
```

## Troubleshooting: is my pipeline hung?

Audio pipelines can appear stuck for legitimate reasons. Before killing a run:

1. **Check logs**: Run with `--verbose` (or `level=DEBUG`) to see per-stage progress.
2. **First-run model download**: NeMo/HuggingFace models are downloaded on first use. A `FastConformer` model is ~500 MB; `Sortformer` is ~200 MB. This happens once and can take minutes on slow connections.
3. **GPU utilization**: Run `watch -n1 nvidia-smi` in another terminal. If GPU utilization is >0%, inference is running.
4. **Worker startup**: Xenna and Ray may take 10–30 seconds to allocate workers before any processing begins. This is normal.
5. **Large datasets**: Processing 10K+ files takes time. Refer to each tutorial's Performance section for expected durations.

| Symptom | Likely cause | Action |
|---|---|---|
| No output for 2+ minutes at start | Model downloading | Wait; check `~/.cache/huggingface/` or `~/.cache/nemo/` for growing files |
| GPU at 0% after startup | OOM crash or worker failure | Check logs for CUDA OOM errors; reduce batch size |
| Steady GPU usage but no log output | Processing normally, logs buffered | Wait; add `--verbose` for more frequent output |
| Process killed by OS | System OOM (CPU RAM) | Reduce number of workers or process fewer files |

## Documentation

| Category | Links |
|---|---|
| **Setup** | [Installation](https://docs.nvidia.com/nemo/curator/latest/get-started/installation.html) · [Configuration](https://docs.nvidia.com/nemo/curator/latest/get-started/configuration.html) |
| **Concepts** | [Architecture](https://docs.nvidia.com/nemo/curator/latest/about/concepts/index.html) · [Data Loading](https://docs.nvidia.com/nemo/curator/latest/about/concepts/text/data-loading-concepts.html) |
| **Advanced** | [Custom Pipelines](https://docs.nvidia.com/nemo/curator/latest/reference/index.html) · [Execution Backends](https://docs.nvidia.com/nemo/curator/latest/reference/infrastructure/execution-backends.html) · [NeMo ASR Integration](https://docs.nvidia.com/nemo/curator/latest/about/key-features.html) |

## Known Issues

### SIGSEGV in Ray StageWorker during model loading

In some environments, and under certain timing conditions, Ray workers may crash with a `SIGSEGV` during GPU model initialization. This is not a NeMo Curator code issue: it comes from a thread-safety problem in the gRPC version bundled with Ray. Any GPU pipeline (audio, text, image, or video) that loads models through Ray actors can hit the same failure.

The OpenTelemetry SDK starts a `PeriodicExportingMetricReader` background thread that periodically calls `OtlpGrpcMetricExporter::Export()` over gRPC; a `getenv()` call on that path can race with NeMo/PyTorch model initialization in another thread. **Disabling OpenTelemetry for the process** prevents Ray's OpenTelemetry background exporter from starting and removes that race. NeMo Curator does not use OpenTelemetry for its own functionality, so disabling it has no functional impact on Curator workflows.

**Container scope:** This has been observed with the `nemo-curator:26.04.rc0` image (and similar 26.04-era builds). The race was [fixed upstream in gRPC ≥ 1.60](https://github.com/grpc/grpc/pull/33508); it should stop being relevant once the bundled gRPC in the container is upgraded accordingly.

**Workaround:** Set these environment variables before running the pipeline:

```bash
export OTEL_SDK_DISABLED=true
export OTEL_METRICS_EXPORTER=none
export OTEL_TRACES_EXPORTER=none
```

## Support

[Main Docs](https://docs.nvidia.com/nemo/curator/latest/) · [API Reference](https://docs.nvidia.com/nemo/curator/latest/apidocs/index.html) · [GitHub Discussions](https://github.com/NVIDIA-NeMo/Curator/discussions)
