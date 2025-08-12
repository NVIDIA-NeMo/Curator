---
description: "Core concepts for loading and managing image datasets using WebDataset format with cloud storage support"
categories: ["concepts-architecture"]
tags: ["data-loading", "webdataset", "dali", "cloud-storage", "sharding", "gpu-accelerated"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "concept"
modality: "image-only"
---

(about-concepts-image-data-loading)=
# Data Loading Concepts (Image)

This page covers the core concepts for loading and managing image datasets in NeMo Curator.

## WebDataset Format and Directory Structure

NeMo Curator uses the [WebDataset](https://github.com/webdataset/webdataset) format for scalable, distributed image curation. A WebDataset directory contains sharded `.tar` files, each holding image-text pairs and metadata, along with corresponding `.parquet` files for tabular metadata. Optionally, `.idx` index files can be provided for fast DALI-based loading.

**Example directory structure:**

```
dataset/
├── 00000.tar
│   ├── 000000000.jpg
│   ├── 000000000.txt
│   ├── 000000000.json
│   ├── ...
├── 00001.tar
│   ├── ...
├── 00000.parquet
├── 00001.parquet
├── 00000.idx  # optional
├── 00001.idx  # optional
```

- `.tar` files: Contain images (`.jpg`), captions (`.txt`), and metadata (`.json`)
- `.parquet` files: Tabular metadata for each record
- `.idx` files: (Optional) Index files for fast DALI-based loading

Each record is identified by a unique ID (e.g., `000000031`), used as the prefix for all files belonging to that record.

## Sharding and Metadata Management

- **Sharding:** Datasets are split into multiple `.tar` files (shards) for efficient distributed processing.
- **Metadata:** Each record has a unique ID, and metadata is stored both in `.json` (per record) and `.parquet` (per shard) files. The `.parquet` files enable fast, tabular access to metadata for filtering and analysis.

## Loading from Local Disk and Cloud Storage

NeMo Curator supports loading datasets from both local disk and cloud storage (S3, GCS, Azure) using the [fsspec](https://filesystem-spec.readthedocs.io/en/latest/) library. This allows you to use the same API regardless of where your data is stored.

**Example:**
```python
from nemo_curator.datasets import ImageTextPairDataset

dataset = ImageTextPairDataset.from_webdataset(
    path="/path/to/webdataset",  # or "s3://bucket/webdataset"
    id_col="key"
)
```

## DALI Integration for High-Performance Loading

[NVIDIA DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/) is used for efficient, GPU-accelerated loading and preprocessing of images from WebDataset tar files. DALI enables:
- Fast image decoding and augmentation on GPU
- Efficient shuffling and batching
- Support for large-scale, distributed workflows

## Index Files

For large datasets, DALI can use `.idx` index files for each `.tar` to enable even faster loading. These index files are generated using DALI's `wds2idx` tool and must be placed alongside the corresponding `.tar` files.

- **How to generate:** See [DALI documentation](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/data_loading/dataloading_webdataset.html#Creating-an-index)
- **Naming:** Each index file must match its `.tar` file (e.g., `00000.tar` → `00000.idx`)
- **Usage:** Set `use_index_files=True` in your embedder or loader.

## Best Practices and Troubleshooting
- Use sharding to enable distributed and parallel processing.
- Always include `.parquet` metadata for fast access and filtering.
- For cloud storage, ensure your environment is configured with the appropriate credentials.
- Use `.idx` files for large datasets to maximize DALI performance.
- Monitor GPU memory and adjust batch size as needed.
- If you encounter loading errors, check for missing or mismatched files in your dataset structure. 