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
* NVIDIA GPU (optional for most text modules, required for GPU-accelerated operations)
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

# CPU + GPU text curation modules (includes RAPIDS for GPU acceleration)
pip install --extra-index-url https://pypi.nvidia.com nemo-curator[text_cuda12]

# All modules (text, image, video, audio with GPU support)
pip install --extra-index-url https://pypi.nvidia.com nemo-curator[all]
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
uv sync --extra text_cuda12 --all-groups
source .venv/bin/activate 
```

```{note}
Replace `text_cuda12` with your desired extras: use `.` for CPU-only, `.[text_cpu]` for text processing only, or `.[all]` for all modules.
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

## Prepare Your Environment

NeMo Curator uses a pipeline-based architecture for processing text data. Before running your first pipeline, ensure you have a proper directory structure:

## Set Up Data Directory

Create a directory structure for your text datasets:

```bash
mkdir -p ~/nemo_curator/data/sample
mkdir -p ~/nemo_curator/data/curated
```

```{note}
For this example, you'll need sample JSONL files in `~/nemo_curator/data/sample/`. Each line should be a JSON object with at least `text` and `id` fields. You can create test data or download sample datasets from the [NeMo Curator tutorials](../curate-text/tutorials/index.md).
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
        file_paths="~/nemo_curator/data/sample/*.jsonl",
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
