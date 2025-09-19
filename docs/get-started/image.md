---
description: "Step-by-step guide to setting up and running your first image curation pipeline with NeMo Curator"
categories: ["getting-started"]
tags: ["image-curation", "installation", "quickstart", "gpu-accelerated", "embedding", "classification", "webdataset"]
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
pip install nemo-curator[image]
```

:::

:::{tab-item} Source Installation

Install the latest version directly from GitHub:

```bash
git clone https://github.com/NVIDIA/NeMo-Curator.git
cd NeMo-Curator
pip install "./NeMo-Curator[image]"
```

:::

:::{tab-item} NeMo Curator Container

NeMo Curator is available as a standalone container:

```{warning}
**Container Availability**: The standalone NeMo Curator container is currently in development. Check the [NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers) for the latest availability and container path.
```

```bash
# Pull the container 
docker pull nvcr.io/nvidia/nemo-curator:latest

# Run the container
docker run --gpus all -it --rm nvcr.io/nvidia/nemo-curator:latest
```

```{seealso}
For details on container environments and configurations, see [Container Environments](reference-infrastructure-container-environments-main).
```
:::
::::

## Download Sample Configuration

NeMo Curator provides a working image curation example in the [Image Curation Tutorial](https://github.com/NVIDIA/NeMo-Curator/blob/main/tutorials/image/getting-started/image_curation_example.py). You can adapt this pipeline for your own datasets.

## Set Up Data Directory

Create directories to store your image datasets and models:

```bash
mkdir -p ~/nemo_curator/data/webdataset
mkdir -p ~/nemo_curator/data/curated
mkdir -p ~/nemo_curator/models
```

For this example, you'll need:

* **WebDataset**: Image-text pairs in WebDataset format (`.tar` files containing `.jpg`, `.txt`, and `.json` files)
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

# Stage 1: Partition WebDataset tar files for parallel processing
pipeline.add_stage(FilePartitioningStage(
    file_paths="~/nemo_curator/data/webdataset",  # Path to your WebDataset directory
    files_per_partition=1,
    file_extensions=[".tar"],
))

# Stage 2: Read images from WebDataset tar files using DALI
pipeline.add_stage(ImageReaderStage(
    task_batch_size=100,
    verbose=True,
    num_threads=16,
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

# Stage 6: Save curated images to new WebDataset
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
├── 00000.tar              # Curated images (first shard)
├── 00001.tar              # Curated images (second shard)
├── ...                    # Additional shards as needed
```

Each output tar file contains:

* **Images**: High-quality `.jpg` files that passed both aesthetic and NSFW filtering
* **Metadata**: Corresponding `.txt` (captions) and `.json` (metadata with scores) files
* **Scores**: Each image has `aesthetic_score` and `nsfw_score` in its metadata

## Alternative: Using the Complete Tutorial

For a more comprehensive example with data download and more configuration options, see:

```bash
# Download the complete tutorial
wget -O ~/nemo_curator/image_curation_example.py https://raw.githubusercontent.com/NVIDIA/NeMo-Curator/main/tutorials/image/getting-started/image_curation_example.py

# Run with your data
python ~/nemo_curator/image_curation_example.py \
    --input-wds-dataset-dir ~/nemo_curator/data/webdataset \
    --output-dataset-dir ~/nemo_curator/data/curated \
    --model-dir ~/nemo_curator/models \
    --aesthetic-threshold 0.5 \
    --nsfw-threshold 0.5
```

## Next Steps

Explore the [Image Curation documentation](image-overview) for more advanced processing techniques:

* **[WebDataset Loading](../curate-images/load-data/webdataset.md)** - Learn about data format and loading options
* **[CLIP Embeddings](../curate-images/process-data/embeddings/clip-embedder.md)** - Understand embedding generation
* **[Quality Filtering](../curate-images/process-data/classifiers/index.md)** - Advanced aesthetic and NSFW classification
* **[Complete Tutorial](https://github.com/NVIDIA/NeMo-Curator/blob/main/tutorials/image/getting-started/image_curation_example.py)** - Full working example with data download