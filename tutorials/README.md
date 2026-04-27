# NeMo Curator Tutorials

Hands-on tutorials for curating data across all modalities with NeMo Curator. Complete working examples with detailed explanations.

## Quick Start

**New to NeMo Curator?** Start with the [Getting Started Guide](https://docs.nvidia.com/nemo/curator/latest/get-started/index.html) or try the [`quickstart.py`](quickstart.py) example to understand core concepts.

## Tutorials by Modality

| Modality | Description | Key Tutorials |
|----------|-------------|---------------|
| **[Text](text/)** | Natural language processing and curation | Deduplication, Classification, Quality Assessment, Tokenization |
| **[Image](image/)** | Computer vision and image processing | Aesthetic Classification, NSFW Detection, Deduplication |
| **[Video](video/)** | Video processing and analysis | Clipping, Frame Extraction, Filtering |
| **[Audio](audio/)** | Speech and audio data curation | FLEURS Dataset Processing |
| **[Interleaved](interleaved/)** | Multimodal (text + image) data curation | Getting Started, PDF Extraction Pipeline (Nemotron-Parse) |

## Production Recipes

Complete, production-grade pipelines built on NeMo Curator:

| Recipe | Description | Key Components |
|--------|-------------|----------------|
| [Nemotron-CC](https://github.com/NVIDIA-NeMo/Nemotron/tree/main/src/nemotron/recipes/data_curation/nemotron-cc) • [SDG tutorial (in-repo)](synthetic/nemotron_cc/) | Curate Common Crawl snapshots into an LLM-ready dataset, reproducing the [Nemotron-CC datasets](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2) | `CommonCrawlDownloadExtractStage` • Language ID & Filtering • Exact/Fuzzy/Substring Dedup • Ensemble Quality Classification (1 fasttext + 2 FineWeb classifiers) • Synthetic Data Generation (4 tasks) |

## Core Concepts Example

The [`quickstart.py`](quickstart.py) demonstrates NeMo Curator's foundational architecture:
- **Task**: Define data processing objectives
- **ProcessingStage**: Individual processing steps
- **Pipeline**: Orchestrate multiple stages

## Documentation Links

| Category | Links |
|----------|-------|
| **Getting Started** | [Installation](https://docs.nvidia.com/nemo/curator/latest/get-started/installation.html) • [Configuration](https://docs.nvidia.com/nemo/curator/latest/get-started/configuration.html) • [Core Concepts](https://docs.nvidia.com/nemo/curator/latest/about/concepts/index.html) |
| **Modality Guides** | [Text Curation](https://docs.nvidia.com/nemo/curator/latest/curate-text/index.html) • [Image Curation](https://docs.nvidia.com/nemo/curator/latest/curate-images/index.html) • [Video Curation](https://docs.nvidia.com/nemo/curator/latest/curate-video/index.html) |
| **Advanced** | [Custom Pipelines](https://docs.nvidia.com/nemo/curator/latest/reference/index.html) • [Execution Backends](https://docs.nvidia.com/nemo/curator/latest/reference/infrastructure/execution-backends.html) • [API Reference](https://docs.nvidia.com/nemo/curator/latest/apidocs/index.html) |

## Support

**Documentation**: [Main Docs](https://docs.nvidia.com/nemo/curator/latest/) • [GitHub Discussions](https://github.com/NVIDIA-NeMo/Curator/discussions)
