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

## Download Default Configuration

NeMo Curator provides example pipelines in the [Image Curation Tutorial Notebook](https://github.com/NVIDIA/NeMo-Curator/blob/main/tutorials/image-curation/image-curation.ipynb). You can adapt the code there for your own datasets.

## Set Up Data Directory

Create a directory to store your image datasets:

```bash
mkdir -p ~/nemo_curator/data
```

## Basic Image Curation Example

Here's a simple example to get started with NeMo Curator's image curation pipeline:

```python
import nemo_curator as nc
from nemo_curator.datasets import ImageTextPairDataset
from nemo_curator.image.embedders import TimmImageEmbedder
from nemo_curator.image.classifiers import AestheticClassifier

# Path to your dataset (WebDataset format)
dataset_path = "~/nemo_curator/data/mscoco/{00000..00001}.tar"  # Example pattern
id_col = "key"

# Load the dataset
image_dataset = ImageTextPairDataset.from_webdataset(dataset_path, id_col)

# Create image embeddings using a CLIP model
embedding_model = TimmImageEmbedder(
    "vit_large_patch14_clip_quickgelu_224.openai",
    pretrained=True,
    batch_size=1024,
    num_threads_per_worker=16,
    normalize_embeddings=True,
    autocast=False,
)
image_dataset = embedding_model(image_dataset)

# Annotate with aesthetic scores
aesthetic_classifier = AestheticClassifier()
image_dataset = aesthetic_classifier(image_dataset)

# Filter images with aesthetic score > 6
image_dataset.metadata["passes_aesthetic_check"] = image_dataset.metadata["aesthetic_score"] > 6
image_dataset.to_webdataset("~/nemo_curator/data/curated", filter_column="passes_aesthetic_check")
```

## Next Steps

Explore the [Image Curation documentation](image-overview).