# Audio Curation Tutorials

Hands-on tutorials for curating audio data with NeMo Curator. Complete working examples with detailed explanations.

## Quick Start

**New to audio curation?** Start with the [Audio Getting Started Guide](https://docs.nvidia.com/nemo/curator/latest/get-started/audio.html) for setup and basic concepts.

## Which Tutorial Should I Use?

| I want to... | Tutorial | Auto-downloads data? |
|---|---|---|
| Curate multilingual ASR data (download, transcribe, filter by WER) | [FLEURS Dataset](fleurs/) | Yes (HuggingFace) |
| Build training windows for Audio Language Models | [ALM Data Pipeline](alm/) | Bundled fixtures |
| Clean and filter ASR manifests at scale (hallucination, LID, regex) | [Granary v2 Postprocessing](granary_v2_postprocessing/) | No (requires input manifests) |
| Quality-filter raw audio (MOS, VAD, bandwidth) | [ReadSpeech](readspeech/) | Yes (DNS Challenge, ~4.88 GB) |
| Evaluate speaker diarization on CallHome | [CallHome Diarization](callhome_diar/) | No (requires LDC license) |
| Filter single-speaker audio from a manifest | [Single Speaker Filter](single_speaker_filter/) | No (requires pre-existing manifest) |

## Available Tutorials

| Tutorial | Description | Key Files | GPU Required |
|----------|-------------|-----------|---|
| **[FLEURS Dataset](fleurs/)** | Multilingual ASR pipeline: download → transcribe → WER filter | `pipeline.py`, `run.py`, `pipeline.yaml` | Yes (ASR model) |
| **[ALM Data Pipeline](alm/)** | Create sliding-window training data for Audio Language Models | `main.py`, `pipeline.yaml` | No |
| **[Granary v2 Postprocessing](granary_v2_postprocessing/)** | Text cleaning, hallucination detection, LID filtering for ASR manifests | `post_processsing_pipeline.py`, `common.yaml`, `en.txt` | No (CPU-only) |
| **[ReadSpeech](readspeech/)** | Audio quality filtering with UTMOS, SIGMOS, VAD, and bandwidth | `pipeline.py`, `run.py`, `pipeline.yaml` | Yes (quality models) |
| **[CallHome Diarization](callhome_diar/)** | Speaker diarization evaluation on CallHome English | `run.py` | Yes (Sortformer) |
| **[Single Speaker Filter](single_speaker_filter/)** | Filter manifest entries to keep only single-speaker segments | `run.py` | Yes (~8 GB VRAM) |

## Composability — Chaining Tutorials

These tutorials produce outputs that feed naturally into each other:

```
FLEURS/ReadSpeech (raw audio → filtered manifest)
    → Granary v2 Postprocessing (manifest → cleaned manifest)
        → ALM Data Pipeline (cleaned manifest → training windows)
```

For the full Qwen Omni transcription pipeline (tarred NeMo datasets → production manifests),
see [`examples/audio/qwen_omni_inprocess/`](../../examples/audio/qwen_omni_inprocess/) and
the [stage developer guide](../../nemo_curator/stages/audio/README.md#qwen-omni-in-process-pipeline).

## Documentation Links

| Category | Links |
|----------|-------|
| **Setup** | [Installation](https://docs.nvidia.com/nemo/curator/latest/get-started/installation.html) • [Configuration](https://docs.nvidia.com/nemo/curator/latest/get-started/configuration.html) |
| **Concepts** | [Architecture](https://docs.nvidia.com/nemo/curator/latest/about/concepts/index.html) • [Data Loading](https://docs.nvidia.com/nemo/curator/latest/about/concepts/text/data-loading-concepts.html) |
| **Advanced** | [Custom Pipelines](https://docs.nvidia.com/nemo/curator/latest/reference/index.html) • [Execution Backends](https://docs.nvidia.com/nemo/curator/latest/reference/infrastructure/execution-backends.html) • [NeMo ASR Integration](https://docs.nvidia.com/nemo/curator/latest/about/key-features.html) |

## Support

**Documentation**: [Main Docs](https://docs.nvidia.com/nemo/curator/latest/) • [API Reference](https://docs.nvidia.com/nemo/curator/latest/apidocs/index.html) • [GitHub Discussions](https://github.com/NVIDIA-NeMo/Curator/discussions)
