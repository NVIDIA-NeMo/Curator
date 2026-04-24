# Interleaved Multimodal Data Curation Tutorials

Hands-on tutorials for curating **interleaved multimodal data** — documents that contain text, images, and metadata interlaced in reading order — with NeMo Curator.

## Available Tutorials

| Tutorial | Description | Files |
|----------|-------------|-------|
| **[Getting Started](getting-started/)** | Load, explore, filter, and save interleaved data from MINT-1T PDF shards | `interleaved_data_quickstart.ipynb`, `interleaved_pipeline.py` |
| **[PDF Extraction Pipeline (Nemotron-Parse)](nemotron_parse_pdf/)** | Convert PDFs into structured interleaved Parquet using Nemotron-Parse v1.2 | `main.py` |

## Quick Start

**New to interleaved multimodal curation?** Start with the [Getting Started notebook](getting-started/interleaved_data_quickstart.ipynb), or run the pipeline script directly on local data:

```bash
python tutorials/interleaved/getting-started/interleaved_pipeline.py \
    --input-path /path/to/shard-0/ \
    --output-path /path/to/output/ \
    --on-materialize-error drop_row \
    --mode overwrite
```

## Documentation Links

| Category | Links |
|----------|-------|
| **Concepts** | [Core Concepts](https://docs.nvidia.com/nemo/curator/latest/about/concepts/index.html) |
| **API Reference** | [API Docs](https://docs.nvidia.com/nemo/curator/latest/apidocs/index.html) |

## Support

**Documentation**: [Main Docs](https://docs.nvidia.com/nemo/curator/latest/) • [GitHub Discussions](https://github.com/NVIDIA-NeMo/Curator/discussions)
