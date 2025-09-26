---
description: "Core concepts for loading and managing image datasets from tar archives with cloud storage support"
categories: ["concepts-architecture"]
tags: ["data-loading", "tar-archives", "dali", "cloud-storage", "sharding", "gpu-accelerated"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "concept"
modality: "image-only"
---

(about-concepts-image-data-loading)=

# Data Loading Concepts (Image)

This page covers the core concepts for loading and managing image datasets in NeMo Curator.

> **Input vs. Output**: This page focuses on **input** data formats for loading datasets into NeMo Curator. For information about **output** formats (including Parquet metadata files created during export), see the [Data Export Concepts](data-export-concepts.md) page.

## Input Data Format and Directory Structure

NeMo Curator loads image datasets from tar archives for scalable, distributed image curation. The `ImageReaderStage` reads only JPEG images from input `.tar` files, ignoring other content.

**Example input directory structure:**

```bash
input_dataset/
├── 00000.tar
│   ├── 000000000.jpg
│   ├── 000000000.txt
│   ├── 000000000.json
│   ├── ...
├── 00001.tar
│   ├── ...
```

**Input file types:**

- `.tar` files: Contain images (`.jpg`), captions (`.txt`), and metadata (`.json`) - only images are loaded

:::{note} While tar archives may contain captions (`.txt`) and metadata (`.json`) files, the `ImageReaderStage` only extracts JPEG images. Other file types are ignored during the loading process.
:::

Each record is identified by a unique ID (e.g., `000000031`), used as the prefix for all files belonging to that record.

## Sharding and Metadata Management

- **Sharding:** Datasets are split into multiple `.tar` files (shards) for efficient distributed processing.
- **Metadata:** Each record has a unique ID, and metadata is stored in `.json` files (per record) within the tar archives.

## Loading from Local Disk

**Example:**

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.image.io.image_reader import ImageReaderStage

# Create pipeline for loading
pipeline = Pipeline(name="image_loading")

# Partition tar files
pipeline.add_stage(FilePartitioningStage(
    file_paths="/path/to/tar_dataset",
    files_per_partition=1,
    file_extensions=[".tar"],  # Required for ImageReaderStage
))

# Load images with DALI
pipeline.add_stage(ImageReaderStage(
    task_batch_size=100,
    verbose=True,
    num_threads=8,
    num_gpus_per_worker=0.25,
))
```

## DALI Integration for High-Performance Loading

The `ImageReaderStage` uses [NVIDIA DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/) for efficient, GPU-accelerated loading and preprocessing of JPEG images from tar files. DALI enables:

- **GPU Acceleration:** Fast image decoding on GPU with automatic CPU fallback
- **Batch Processing:** Efficient batching and streaming of image data
- **Tar Archive Processing:** Built-in support for tar archive format
- **Memory Efficiency:** Streams images without loading entire datasets into memory

## Best Practices and Troubleshooting

- Use sharding to enable distributed and parallel processing.
- Watch GPU memory and adjust batch size as needed.
- If you encounter loading errors, check for missing or mismatched files in your dataset structure.
