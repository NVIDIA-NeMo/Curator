# Text Curation Tutorials

Hands-on tutorials for curating text data with NeMo Curator. Complete working examples with detailed explanations.

## Quick Start

**New to text curation?** Start with the [Text Getting Started Guide](https://docs.nvidia.com/nemo/curator/latest/get-started/text.html) for setup and basic concepts.

## Available Tutorials

| Tutorial | Description | Files |
|----------|-------------|-------|
| **[Download & Extract](download-and-extract/)** | Data acquisition workflows | `download_extract_tutorial.ipynb` |
| **[Deduplication](deduplication/)** | Remove duplicate content | `semantic_e2e.ipynb`, `semantic_step_by_step.ipynb` |
| **[Classification](distributed-data-classification/)** | Quality assessment and categorization | `quality-classification.ipynb`, `domain-classification.ipynb`, `aegis-classification.ipynb`, `fineweb-edu-classification.ipynb` |
| **[PEFT Curation](peft-curation/)** | Instruction-tuning data preparation | `main.py`, `stages.py` |
| **[TinyStories](tinystories/)** | End-to-end processing pipeline | `main.py`, `stages.py` |
| **[Megatron Tokenizing](megatron-tokenizer/)** | Tokenization pipeline that produces Megatron-LM ready files | `main.py` |

## Documentation Links

| Category | Links |
|----------|-------|
| **Concepts** | [Processing](https://docs.nvidia.com/nemo/curator/latest/about/concepts/text/data-processing-concepts.html) • [Pipeline](https://docs.nvidia.com/nemo/curator/latest/about/concepts/text/data-curation-pipeline.html) • [Loading](https://docs.nvidia.com/nemo/curator/latest/about/concepts/text/data-loading-concepts.html) • [Generation](https://docs.nvidia.com/nemo/curator/latest/about/concepts/text/data-generation-concepts.html) |
| **Data Sources** | [Common Crawl](https://docs.nvidia.com/nemo/curator/latest/curate-text/load-data/common-crawl.html) • [Wikipedia](https://docs.nvidia.com/nemo/curator/latest/curate-text/load-data/wikipedia.html) • [ArXiv](https://docs.nvidia.com/nemo/curator/latest/curate-text/load-data/arxiv.html) • [Custom](https://docs.nvidia.com/nemo/curator/latest/curate-text/load-data/custom.html) |
| **Processing** | [Quality Assessment](https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/quality-assessment/index.html) • [Deduplication](https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/deduplication/index.html) • [Content Processing](https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/content-processing/index.html) • [PII Removal](https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/content-processing/pii.html) |
| **Advanced** | [Distributed Classification](https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/quality-assessment/distributed-classifier.html) • [Semantic Dedup](https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/deduplication/semdedup.html) • [GPU Dedup](https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/deduplication/gpudedup.html) • [Synthetic Data](https://docs.nvidia.com/nemo/curator/latest/curate-text/generate-data/pipelines/index.html) |

## Support

**Documentation**: [Main Docs](https://docs.nvidia.com/nemo/curator/latest/) • [API Reference](https://docs.nvidia.com/nemo/curator/latest/apidocs/index.html) • [GitHub Discussions](https://github.com/NVIDIA-NeMo/Curator/discussions)
