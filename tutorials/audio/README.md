# Audio Curation Tutorials

Hands-on tutorials for curating audio data with NeMo Curator.

**New to audio curation?** Start with the [Audio Getting Started Guide](https://docs.nvidia.com/nemo/curator/latest/get-started/audio.html) for setup and basic concepts.

## Which tutorial should I use?

| I want to... | Tutorial | GPU required | Data |
|---|---|---|---|
| Curate multilingual ASR data (download, transcribe, filter by WER) | [**fleurs/**](fleurs/) | Yes | Auto-downloads from HuggingFace |
| Build training windows for Audio Language Models from diarized manifests | [**alm/**](alm/) | No | Bundled sample fixtures |
| Evaluate speaker diarization (DER) on a benchmark dataset | [**callhome_diar/**](callhome_diar/) | Yes | Requires [LDC license](https://catalog.ldc.upenn.edu/LDC97S42) |
| Filter a manifest to keep only single-speaker audio | [**single_speaker_filter/**](single_speaker_filter/) | Yes (8 GB+ VRAM) | Requires a pre-existing JSONL manifest |
| Quality-filter raw audio (MOS, VAD, bandwidth, noise) | [**readspeech/**](readspeech/) | Recommended | Auto-downloads DNS Challenge (4.88 GB) |

## Data availability

| Tutorial | Auto-download | Size | Notes |
|---|---|---|---|
| `fleurs/` | Yes | ~50 MB per language split | Downloads from HuggingFace `google/fleurs` |
| `alm/` | N/A | Bundled | Uses `tests/fixtures/audio/alm/sample_input.jsonl` (5 entries) |
| `callhome_diar/` | No | ~1 GB | Requires LDC membership and license ([LDC97S42](https://catalog.ldc.upenn.edu/LDC97S42)) |
| `single_speaker_filter/` | No | Varies | Bring your own NeMo-style JSONL manifest |
| `readspeech/` | Yes | 4.88 GB compressed | Downloads DNS Challenge Read Speech (14,279 WAV files) |

## System dependencies

| Tutorial | System packages | Pip extras |
|---|---|---|
| `fleurs/` | None | `audio_cpu` or `audio_cuda12` |
| `alm/` | None | `audio_cpu` |
| `callhome_diar/` | `sox` | `audio_cuda12` |
| `single_speaker_filter/` | None | `audio_cuda12` |
| `readspeech/` | None | `audio_cuda12` (recommended) or `audio_cpu` |

Install pip extras from the repo root:

```bash
# GPU (recommended)
uv sync --extra audio_cuda12

# CPU only
uv sync --extra audio_cpu
```

## Documentation

| Category | Links |
|---|---|
| **Setup** | [Installation](https://docs.nvidia.com/nemo/curator/latest/get-started/installation.html) · [Configuration](https://docs.nvidia.com/nemo/curator/latest/get-started/configuration.html) |
| **Concepts** | [Architecture](https://docs.nvidia.com/nemo/curator/latest/about/concepts/index.html) · [Data Loading](https://docs.nvidia.com/nemo/curator/latest/about/concepts/text/data-loading-concepts.html) |
| **Advanced** | [Custom Pipelines](https://docs.nvidia.com/nemo/curator/latest/reference/index.html) · [Execution Backends](https://docs.nvidia.com/nemo/curator/latest/reference/infrastructure/execution-backends.html) · [NeMo ASR Integration](https://docs.nvidia.com/nemo/curator/latest/about/key-features.html) |

## Support

[Main Docs](https://docs.nvidia.com/nemo/curator/latest/) · [API Reference](https://docs.nvidia.com/nemo/curator/latest/apidocs/index.html) · [GitHub Discussions](https://github.com/NVIDIA-NeMo/Curator/discussions)
