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

* Python 3.10, 3.11, or 3.12
  * packaging >= 22.0
* uv (for package management and installation)
* Ubuntu 22.04/20.04
* NVIDIA GPU (optional for most text modules, required for GPU-accelerated operations)
  * Volta™ or higher (compute capability 7.0+)
  * CUDA 12 (or above)

:::{tip}
If you don't have `uv` installed, refer to the [Installation Guide](../admin/installation.md) for setup instructions, or install it quickly with:

```bash
curl -LsSf https://astral.sh/uv/0.8.22/install.sh | sh
source $HOME/.local/bin/env
```

:::

---

## Installation Options

You can install NeMo Curator in three ways:

::::{tab-set}

:::{tab-item} PyPI Installation

The simplest way to install NeMo Curator:

```bash
uv pip install "nemo-curator[text_cuda12]"
```

```{note}
For other modalities (image, video) or all modules, see the [Installation Guide](../admin/installation.md).
```

:::

:::{tab-item} Source Installation

Install the latest version directly from GitHub:

```bash
git clone https://github.com/NVIDIA-NeMo/Curator.git
cd Curator
uv sync --extra text_cuda12 --all-groups
source .venv/bin/activate 
```

```{note}
Replace `text_cuda12` with your desired extras: use `.` for CPU-only, `.[text_cpu]` for text processing only, or `.[all]` for all modules.
```

:::

:::{tab-item} NeMo Curator Container

NeMo Curator is available as a standalone container:

```bash
# Pull the container
docker pull nvcr.io/nvidia/nemo-curator:{{ container_version }}

# Run the container
docker run --gpus all -it --rm nvcr.io/nvidia/nemo-curator:{{ container_version }}
```

```{seealso}
For details on container environments and configurations, see [Container Environments](reference-infrastructure-container-environments-main).
```

:::
::::

## Prepare Your Environment

NeMo Curator uses a pipeline-based architecture for processing text data. Before running your first pipeline, ensure you have a proper directory structure:

## Set Up Data Directory

Create a directory structure for your text datasets:

```bash
mkdir -p ~/nemo_curator/data/sample
mkdir -p ~/nemo_curator/data/curated
```

```{note}
For this example, you'll need sample JSONL files in `~/nemo_curator/data/sample/`. Each line should be a JSON object with at least `text` and `id` fields. You can create test data or refer to {ref}`Read Existing Data <text-load-data-read-existing>` and {ref}`Data Loading <text-load-data>` for information on downloading data.
```

## Basic Text Curation Example

Here's a simple example to get started with NeMo Curator's pipeline-based architecture:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.modules.score_filter import ScoreFilter
from nemo_curator.stages.text.filters import WordCountFilter, NonAlphaNumericFilter

# Create a pipeline for text curation
pipeline = Pipeline(
    name="text_curation_pipeline",
    description="Basic text quality filtering pipeline"
)

# Add stages to the pipeline
pipeline.add_stage(
    JsonlReader(
        file_paths="~/nemo_curator/data/sample/",
        files_per_partition=4,
        fields=["text", "id"]  # Only read required columns for efficiency
    )
)

# Add quality filtering stages
pipeline.add_stage(
    ScoreFilter(
        score_fn=WordCountFilter(min_words=50, max_words=100000),
        text_field="text",
        score_field="word_count"  # Optional: save scores for analysis
    )
)

pipeline.add_stage(
    ScoreFilter(
        score_fn=NonAlphaNumericFilter(max_non_alpha_numeric_to_text_ratio=0.25),
        text_field="text",
        score_field="non_alpha_score"  # Optional: save scores for analysis
    )
)

# Write the curated results
pipeline.add_stage(
    JsonlWriter("~/nemo_curator/data/curated")
)

# Execute the pipeline
results = pipeline.run()  # Uses XennaExecutor by default for distributed processing

print(f"Pipeline completed successfully! Processed {len(results) if results else 0} tasks.")
```

## Next Steps

Explore the [Text Curation documentation](text-overview) for more advanced filtering techniques, GPU acceleration options, and large-scale processing workflows.
