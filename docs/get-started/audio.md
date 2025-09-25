---
description: "Step-by-step guide to setting up and running your first audio curation pipeline with NeMo Curator"
categories: ["getting-started"]
tags: ["audio-curation", "installation", "quickstart", "asr-inference", "quality-filtering", "nemo-toolkit"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "tutorial"
modality: "audio-only"
---

(gs-audio)=

# Get Started with Audio Curation

This guide helps you set up and get started with NeMo Curator's audio curation capabilities. Follow these steps to prepare your environment and run your first audio curation pipeline using the FLEURS dataset.

## Prerequisites

To use NeMo Curator's audio curation modules, ensure you meet the following requirements:

* Python 3.10 or 3.12
  * packaging >= 22.0
* Ubuntu 22.04/20.04
* NVIDIA GPU (recommended for ASR inference)
  * Volta™ or higher (compute capability 7.0+)
  * CUDA 12 (or above)
* Audio processing libraries (automatically installed with audio extras)

---

## Installation Options

You can install NeMo Curator with audio support in four ways:

::::{tab-set}

:::{tab-item} uv Installation (Recommended)

The fastest and most reliable way to install NeMo Curator with audio support:

```bash
# Install uv first (if not already installed)
pip install uv

# Audio curation modules (CPU-only)
uv pip install nemo-curator[audio_cpu]

# Audio + GPU acceleration for other modalities  
uv pip install --extra-index-url https://pypi.nvidia.com nemo-curator[audio_cuda12,deduplication_cuda12]
```

```{note}
uv provides faster dependency resolution and more reliable installations. It's the same tool used by NeMo Curator developers and CI/CD systems.
```

:::

:::{tab-item} PyPI Installation

The simplest way to install NeMo Curator with audio support:

```bash
# Audio curation modules (CPU-only)
pip install nemo-curator[audio_cpu]

# Audio + GPU acceleration for other modalities
pip install --extra-index-url https://pypi.nvidia.com nemo-curator[audio_cuda12,deduplication_cuda12]
```

```{note}
The audio extras include NeMo Toolkit with ASR models. Additional audio processing libraries (soundfile, editdistance) are installed automatically as NeMo Toolkit dependencies.
```

:::

:::{tab-item} Source Installation

Install the latest version directly from GitHub:

```bash
git clone https://github.com/NVIDIA/NeMo-Curator.git
cd NeMo-Curator
uv sync --extra audio_cuda12 --all-groups
source .venv/bin/activate 
```

```{note}
Use `audio_cpu` for CPU-only audio processing, `audio_cuda12` for GPU acceleration, or `all` for all modalities.
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

# Run the container with GPU support
docker run --gpus all -it --rm nvcr.io/nvidia/nemo-curator:latest
```

```{seealso}
For details on container environments and configurations, see [Container Environments](reference-infrastructure-container-environments-main).
```

:::
::::

## Download Sample Configuration

NeMo Curator provides a sample FLEURS configuration for audio curation. You can download and customize it:

```bash
mkdir -p ~/nemo_curator/configs
wget -O ~/nemo_curator/configs/fleurs_pipeline.yaml https://raw.githubusercontent.com/NVIDIA/NeMo-Curator/main/tutorials/audio/fleurs/pipeline.yaml
```

This configuration file contains a complete audio curation pipeline for the FLEURS dataset, including ASR inference, quality assessment, and filtering.

## Set Up Data Directory

Create a directory to store your audio datasets:

```bash
mkdir -p ~/nemo_curator/audio_data
```

## Basic Audio Curation Example

Here's a simple example to get started with audio curation using the FLEURS dataset:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.datasets.fleurs.create_initial_manifest import CreateInitialManifestFleursStage
from nemo_curator.stages.audio.inference.asr_nemo import InferenceAsrNemoStage
from nemo_curator.stages.audio.metrics.get_wer import GetPairwiseWerStage
from nemo_curator.stages.audio.common import GetAudioDurationStage, PreserveByValueStage
from nemo_curator.stages.resources import Resources

# Create audio curation pipeline
pipeline = Pipeline(name="audio_curation", description="FLEURS audio curation with ASR and WER filtering")

# 1. Load FLEURS dataset (Armenian development set)
pipeline.add_stage(
    CreateInitialManifestFleursStage(
        lang="hy_am",
        split="dev", 
        raw_data_dir="~/nemo_curator/audio_data"
    ).with_(batch_size=4)
)

# 2. Perform ASR inference using NeMo model
pipeline.add_stage(
    InferenceAsrNemoStage(
        model_name="nvidia/stt_hy_fastconformer_hybrid_large_pc"
    ).with_(resources=Resources(gpus=1.0))
)

# 3. Calculate Word Error Rate (WER)
pipeline.add_stage(
    GetPairwiseWerStage(
        text_key="text",
        pred_text_key="pred_text", 
        wer_key="wer"
    )
)

# 4. Calculate audio duration
pipeline.add_stage(
    GetAudioDurationStage(
        audio_filepath_key="audio_filepath",
        duration_key="duration"
    )
)

# 5. Filter by WER threshold (keep samples with WER <= 75%)
pipeline.add_stage(
    PreserveByValueStage(
        input_value_key="wer",
        target_value=75.0,
        operator="le"  # less than or equal
    )
)

# Execute the pipeline
pipeline.run()
```

## Alternative: Configuration-Based Approach

You can also run the pipeline using the downloaded configuration:

```bash
cd ~/nemo_curator
python -m nemo_curator.examples.audio.fleurs.run \
    --config-path ~/nemo_curator/configs \
    --config-name fleurs_pipeline.yaml \
    raw_data_dir=~/nemo_curator/audio_data
```

## Expected Output

After running the pipeline, you'll have:

```text
~/nemo_curator/audio_data/
├── hy_am/                    # Armenian language data
│   ├── dev.tsv              # Transcription metadata
│   ├── dev.tar.gz          # Audio archive
│   ├── dev/                # Extracted audio files
│   └── result/             # Filtered results
│       └── *.jsonl        # High-quality audio-text pairs
```

Each output entry contains:

```json
{
    "audio_filepath": "/absolute/path/to/audio.wav",
    "text": "ground truth transcription",
    "pred_text": "asr model prediction", 
    "wer": 12.5,
    "duration": 4.2
}
```

## Next Steps

Explore the [Audio Curation documentation](audio-overview) for more advanced processing techniques and customization options.

Key areas to explore next:

* **[Custom Audio Manifests](../curate-audio/load-data/custom-manifests.md)** - Load your own audio datasets
* **[Quality Assessment](../curate-audio/process-data/quality-assessment/index.md)** - Advanced filtering and quality metrics
* **[Text Integration](../curate-audio/process-data/text-integration/index.md)** - Combine with text processing workflows  