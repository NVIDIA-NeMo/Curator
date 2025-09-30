---
description: "Step-by-step guide to setting up and running your first image curation pipeline with NeMo Curator"
categories: ["getting-started"]
tags: ["image-curation", "installation", "quickstart", "gpu-accelerated", "embedding", "classification", "tar-archives"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "tutorial"
modality: "image-only"
---

(gs-image)=

# Get Started with Image Curation

This guide helps you set up and get started with NeMo Curator's image curation capabilities. Follow these steps to prepare your environment and run your first image curation pipeline.

## Prerequisites

To use NeMo Curator's image curation modules, ensure you meet the following requirements:

* Python 3.10 or higher
  * packaging >= 22.0
* Ubuntu 22.04/20.04
* NVIDIA GPU (required for all image modules)
  * Volta™ or higher (compute capability 7.0+)
  * CUDA 12 (or above)

:::{note}
All image curation modules require a GPU. Some text-based modules don't require a GPU, but image curation does.
:::

---

## Installation Options

You can install NeMo Curator in three ways:

::::{tab-set}

:::{tab-item} PyPI Installation

Install the image modules from PyPI:

```bash
pip install --extra-index-url https://pypi.nvidia.com nemo-curator[image_cuda12]
```

:::

:::{tab-item} Source Installation

Install the latest version directly from GitHub using uv:

```bash
git clone https://github.com/NVIDIA/NeMo-Curator.git
cd NeMo-Curator
uv sync --extra image_cuda12
```

Activate the environment and run your code:

```bash
source .venv/bin/activate
python your_script.py
```

:::{tab-item} Docker Container

You can build and run NeMo Curator in a container environment using the provided Dockerfile:

```bash
git clone https://github.com/NVIDIA/NeMo-Curator.git
cd NeMo-Curator
docker build -t nemo-curator -f docker/Dockerfile .
```

:::
::::

## Download Sample Configuration

NeMo Curator provides a working image curation example in the [Image Curation Tutorial](https://github.com/NVIDIA/NeMo-Curator/blob/main/tutorials/image/getting-started/image_curation_example.py). You can adapt this pipeline for your own datasets.

## Set Up Data Directory

Create directories to store your image datasets and models:

```bash
mkdir -p ~/nemo_curator/data/tar_archives
mkdir -p ~/nemo_curator/data/curated
mkdir -p ~/nemo_curator/models
```

For this example, you'll need:

* **Tar Archives**: JPEG images in `.tar` files (text and JSON files are ignored during loading)
* **Model Directory**: CLIP and classifier model weights (downloaded automatically on first run)

## Basic Image Curation Example

Here's a simple example to get started with NeMo Curator's image curation pipeline:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.image.io.image_reader import ImageReaderStage
from nemo_curator.stages.image.embedders.clip_embedder import ImageEmbeddingStage
from nemo_curator.stages.image.filters.aesthetic_filter import ImageAestheticFilterStage
from nemo_curator.stages.image.filters.nsfw_filter import ImageNSFWFilterStage
from nemo_curator.stages.image.io.image_writer import ImageWriterStage

# Create image curation pipeline
pipeline = Pipeline(name="image_curation", description="Basic image curation with quality filtering")

# Stage 1: Partition tar files for parallel processing
pipeline.add_stage(FilePartitioningStage(
    file_paths="~/nemo_curator/data/tar_archives",  # Path to your tar archive directory
    files_per_partition=1,
    file_extensions=[".tar"],
))

# Stage 2: Read images from tar files using DALI
pipeline.add_stage(ImageReaderStage(
    task_batch_size=100,
    verbose=True,
    num_threads=8,
    num_gpus_per_worker=0.25,
))

# Stage 3: Generate CLIP embeddings for images
pipeline.add_stage(ImageEmbeddingStage(
    model_dir="~/nemo_curator/models",  # Directory containing model weights
    model_inference_batch_size=32,
    num_gpus_per_worker=0.25,
    remove_image_data=False,
    verbose=True,
))

# Stage 4: Filter by aesthetic quality (keep images with score >= 0.5)
pipeline.add_stage(ImageAestheticFilterStage(
    model_dir="~/nemo_curator/models",
    score_threshold=0.5,
    model_inference_batch_size=32,
    num_gpus_per_worker=0.25,
    verbose=True,
))

# Stage 5: Filter NSFW content (remove images with score >= 0.5)
pipeline.add_stage(ImageNSFWFilterStage(
    model_dir="~/nemo_curator/models",
    score_threshold=0.5,
    model_inference_batch_size=32,
    num_gpus_per_worker=0.25,
    verbose=True,
))

# Stage 6: Save curated images to new tar archives
pipeline.add_stage(ImageWriterStage(
    output_dir="~/nemo_curator/data/curated",
    images_per_tar=1000,
    remove_image_data=True,
    verbose=True,
))

# Execute the pipeline
executor = XennaExecutor()
pipeline.run(executor)
```

## Expected Output

After running the pipeline, you'll have:

```text
~/nemo_curator/data/curated/
├── images-{hash}-000000.tar     # Curated images (first shard)
├── images-{hash}-000000.parquet # Metadata for corresponding tar
├── images-{hash}-000001.tar     # Curated images (second shard)
├── images-{hash}-000001.parquet # Metadata for corresponding tar
├── ...                          # Additional shards as needed
```

**Output Format Details:**

* **Tar Files**: Contain high-quality `.jpg` files that passed both aesthetic and NSFW filtering
* **Parquet Files**: Contain metadata for each corresponding tar file, including image paths, IDs, and processing scores
* **Naming Convention**: Files use hash-based prefixes (e.g., `images-a1b2c3d4e5f6-000000.tar`) for uniqueness across distributed processing
* **Scores**: Processing metadata includes `aesthetic_score` and `nsfw_score` stored in the Parquet files

## Alternative: Using the Complete Tutorial

For a more comprehensive example with data download and more configuration options, see:

```bash
# Download the complete tutorial
wget -O ~/nemo_curator/image_curation_example.py https://raw.githubusercontent.com/NVIDIA/NeMo-Curator/main/tutorials/image/getting-started/image_curation_example.py

# Run with your data
python ~/nemo_curator/image_curation_example.py \
    --input-tar-dataset-dir ~/nemo_curator/data/tar_archives \
    --output-dataset-dir ~/nemo_curator/data/curated \
    --model-dir ~/nemo_curator/models \
    --aesthetic-threshold 0.5 \
    --nsfw-threshold 0.5
```

## Next Steps

Explore the [Image Curation documentation](image-overview) for more advanced processing techniques:

* **[Tar Archive Loading](../curate-images/load-data/tar-archives.md)** - Learn about loading JPEG images from tar files
* **[CLIP Embeddings](../curate-images/process-data/embeddings/clip-embedder.md)** - Understand embedding generation
* **[Quality Filtering](../curate-images/process-data/filters/index.md)** - Advanced aesthetic and NSFW filtering
* **[Complete Tutorial](https://github.com/NVIDIA/NeMo-Curator/blob/main/tutorials/image/getting-started/image_curation_example.py)** - Full working example with data download
