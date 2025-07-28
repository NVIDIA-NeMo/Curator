---
description: "Step-by-step guide to setting up and running your first text curation pipeline with NeMo Curator"
categories: ["getting-started"]
tags: ["text-curation", "installation", "quickstart", "data-loading", "quality-filtering", "python-api"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "tutorial"
modality: "text-only"
---

(gs-text)=
# Get Started with Text Curation

This guide helps you set up and get started with NeMo Curator's text curation capabilities. Follow these steps to prepare your environment and run your first text curation pipeline.

## Prerequisites

To use NeMo Curator's text curation modules, ensure you meet the following requirements:

* Python 3.10 or 3.12
  * packaging >= 22.0
* Ubuntu 22.04/20.04
* NVIDIA GPU (optional for many text modules, required for GPU-accelerated operations)
  * Volta™ or higher (compute capability 7.0+)
  * CUDA 12 (or above)

---

## Installation Options

You can install NeMo Curator in three ways:

::::{tab-set}

:::{tab-item} PyPI Installation

The simplest way to install NeMo Curator:

```bash
# CPU-only text curation modules
pip install nemo-curator

# CPU + GPU text curation modules
pip install --extra-index-url https://pypi.nvidia.com nemo-curator[cuda12x]

# Text curation with bitext processing
pip install --extra-index-url https://pypi.nvidia.com nemo-curator[bitext]
```

```{note}
For other modalities (image, video) or all modules, see the [Installation Guide](../admin/installation.md).
```
:::

:::{tab-item} Source Installation

Install the latest version directly from GitHub:

```bash
git clone https://github.com/NVIDIA/NeMo-Curator.git
cd NeMo-Curator
pip install --extra-index-url https://pypi.nvidia.com ".[cuda12x]"
```

```{note}
Replace `cuda12x` with your desired extras: use `.` for CPU-only, `.[bitext]` for bitext processing, or `.[all]` for all modules.
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

NeMo Curator provides default configurations for common curation tasks. You can download a sample configuration for English text filtering:

```bash
mkdir -p ~/nemo_curator/configs
wget -O ~/nemo_curator/configs/heuristic_filter_en.yaml https://raw.githubusercontent.com/NVIDIA/NeMo-Curator/main/config/heuristic_filter_en.yaml
```

This configuration file contains a comprehensive set of heuristic filters for English text, including filters for word count, non-alphanumeric content, repeated patterns, and content quality metrics.

## Set Up Data Directory

Create a directory to store your text datasets:

```bash
mkdir -p ~/nemo_curator/data
```

## Basic Text Curation Example

Here's a simple example to get started with NeMo Curator:

```python
import nemo_curator as nc
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import WordCountFilter, NonAlphaNumericFilter
from nemo_curator.utils.distributed_utils import get_client

# Initialize a Dask client for distributed processing (CPU or GPU)
client = get_client(cluster_type="cpu")  # Use "gpu" for GPU-accelerated processing

# Load sample text data
dataset = DocumentDataset.read_json("~/nemo_curator/data/sample/*.jsonl")

# Create a simple curation pipeline
curation_pipeline = nc.Sequential([
    # Filter documents with 50-10000 words
    nc.ScoreFilter(
        WordCountFilter(min_words=50, max_words=10000),
        text_field="text",
        score_field="word_count"
    ),
    # Filter documents with excessive non-alphanumeric content  
    nc.ScoreFilter(
        NonAlphaNumericFilter(max_non_alpha_numeric_to_text_ratio=0.25),
        text_field="text",
        score_field="non_alpha_score"
    )
])

# Apply the curation pipeline
curated_dataset = curation_pipeline(dataset)

# Save the curated dataset
curated_dataset.to_json("~/nemo_curator/data/curated")
```

## Next Steps

Explore the [Text Curation documentation](text-overview) for more advanced filtering techniques, GPU acceleration options, and large-scale processing workflows.

Key areas to explore next:

- **Advanced Filtering**: Learn about the 30+ built-in filters for quality assessment
- **GPU Acceleration**: Scale your processing with RAPIDS and GPU clusters  
- **Configuration Files**: Use YAML configurations for complex filter pipelines
- **Distributed Processing**: Process datasets across multiple machines
- **Quality Classification**: Use machine learning models for document scoring