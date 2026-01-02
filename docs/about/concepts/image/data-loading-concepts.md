---
description: "Core concepts for loading and managing image datasets from tar archives with cloud storage support"
categories: ["concepts-architecture"]
tags: ["data-loading", "tar-archives", "dali", "cloud-storage", "gpu-accelerated"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "concept"
modality: "image-only"
---

(about-concepts-image-data-loading)=

# Data Loading Concepts (Image)

This page covers the core concepts for loading and managing image datasets in NeMo Curator.

## Input Data Format and Directory Structure

NeMo Curator loads image datasets from tar archives for scalable, distributed image curation. The `ImageReaderStage` reads only JPEG images from input `.tar` files, ignoring other content.

**Example input directory structure:**

```bash
input_dataset/
├── 00000.tar          # Tar archive containing JPEG images
│   ├── 000000000.jpg
│   ├── 000000001.jpg
│   ├── 000000002.jpg
│   ├── ...
├── 00001.tar
│   ├── 000001000.jpg
│   ├── 000001001.jpg
│   ├── ...
```

**What gets loaded:**

- `.tar` files: Tar archives containing JPEG images (`.jpg`)
- Only JPEG images are extracted and processed

:::{note}
**WebDataset Format Support**: If your tar archives follow the [WebDataset format](https://github.com/webdataset/webdataset) and contain additional files (captions as `.txt`, metadata as `.json`), the `ImageReaderStage` will **only extract JPEG images**. Other file types (`.txt`, `.json`, etc.) are automatically ignored during loading.
:::

Each record is identified by a unique ID (e.g., `000000031`), used as the prefix for all files belonging to that record.

## Loading from Local Disk

**Example:**

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.image.io.image_reader import ImageReaderStage

# Create pipeline for loading
pipeline = Pipeline(name="image_loading")

# Partition tar files for parallel processing
pipeline.add_stage(FilePartitioningStage(
    file_paths="/path/to/tar_dataset",
    files_per_partition=1,         # Process one tar file per partition
    file_extensions=[".tar"],       # Only include .tar files
))

# Load JPEG images from tar files using DALI
pipeline.add_stage(ImageReaderStage(
    batch_size=100,                 # Number of images per batch
    verbose=True,
    num_threads=8,                  # Number of threads for I/O operations
    num_gpus_per_worker=0.25,       # Allocate 1/4 GPU per worker
))

# Execute the pipeline
results = pipeline.run()
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
