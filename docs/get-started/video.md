---
description: "Step-by-step guide to setting up and running your first video curation pipeline with NeMo Curator"
categories: ["getting-started"]
tags: ["video-curation", "installation", "quickstart", "gpu-accelerated", "docker", "configuration"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "tutorial"
modality: "video-only"
only: not ga
---

(gs-video)=
# Get Started with Video Curation

This guide helps you set up and get started with NeMo Curator's video curation capabilities. Follow these steps to prepare your environment, install the CLI, and run your first video curation pipeline.

## Prerequisites

To use NeMo Curator's video curation modules, ensure you meet the following requirements:

* Docker
* NVIDIA GPU
  * Volta™ or higher (compute capability 7.0+)
  * CUDA 12 (or above)
  * With default settings, you will need at least 38 GB VRAM on your GPU. You can reduce VRAM required to around 21 GB by following advice given in the "Performance Considerations" sections of each pipeline.
* Access to the NGC private registry (for Docker image and CLI)
* Python 3.10 or higher

---

## Installation Options

You can install NeMo Curator for video in two main ways:

::::{tab-set}

:::{tab-item} Docker Container

The recommended way to use NeMo Curator for video is via the official Docker container:

1. Go to the [NeMo Curator for Video Processing container on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/containers/nemo-curator-video/).
2. Find the latest container tag (e.g., `nvcr.io/nvidia/nemo/nemo-curator-video:0.6.0`).
3. Pull and tag the Docker image:

```bash
docker pull nvcr.io/nvidia/nemo/nemo-curator-video:0.6.0
docker tag nvcr.io/nvidia/nemo/nemo-curator-video:0.6.0 nemo_video_curator:1.0.0
```

```{seealso}
For details on video container environments and configurations, see [Video Curator Environments](reference-infrastructure-container-environments-video).
```
:::

:::{tab-item} CLI Scripts (Early Access)

You can also install the CLI scripts locally:

1. Go to the [NeMo Curator for Video Processing CLI on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/resources/nemo-curator-video-cli).
2. Download the latest `.whl` file (e.g., `nemo_curator-0.6.0-py3-none-any.whl`).
3. Install the package:

```bash
pip install nemo_curator-0.6.0-py3-none-any.whl
```
:::
::::

## Download Default Configuration

NeMo Curator's command line interfaces (CLIs) require a configuration file to run:

* `model_download.yaml`: Defines the models to download.

To get the configuration file:

1. Go to the [NeMo Curator for Video Processing CLI page on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/resources/nemo-curator-video-cli).
2. Download the `model_download.yaml` file to `~/nemo_curator_local_workspace/`.

## Set Up Data Directory

Your data can be stored locally, on S3, or on Azure-compatible storage.

::::{tab-set}

:::{tab-item} Local Storage

- Place all your `.mp4` files under a single directory in `~/nemo_curator_local_workspace/`.
- To change the `~` prefix, set the `NEMO_CURATOR_LOCAL_WORKSPACE_PREFIX` environment variable to another directory of your choice.
:::

:::{tab-item} S3 Storage

- Add your S3 credentials to `~/.aws/credentials`:

```ini
[default]
aws_access_key_id=<key id>
aws_secret_access_key=<key>
region=<region>
```
:::

:::{tab-item} Azure Storage

- Add your Azure credentials to `~/.azure/credentials`:

```ini
[default]
azure_connection_string=<connection string>
# Or
# azure_account_name=<account name>
# azure_account_key=<account key>
```
:::
::::

## Download and Prepare Model Weights

1. Create a config file at `~/.config/nemo_curator/config.yaml` with the following content:

```yaml
huggingface:
    api_key: "<huggingface token>"
```

2. For the `InternVideo2MultiModality` model, visit [their Hugging Face page](https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4/tree/main), log in, and click "agree" if prompted.

3. Download the `model_download.yaml` file to `~/nemo_curator_local_workspace/model_download.yaml`. This config file specifies the models to download. To change the `~` prefix, set the `NEMO_CURATOR_LOCAL_WORKSPACE_PREFIX` environment variable to another directory of your choice.

4. Download the model weights to `~/nemo_curator_local_workspace/`. `--config-file` is set to `/config/model_download.yaml` because `~/nemo_curator_local_workspace/` is mounted as `/config` in the container.

```bash
video_curator launch \
    --image-name nemo_video_curator \
    --image-tag 1.0.0 -- \
        python3 -m nemo_curator.video.models.model_cli download \
        --config-file /config/model_download.yaml
```

5. If you encounter a "command not found" error, ensure your local Python bin is added to your PATH:

```bash
export PATH="$PATH:$HOME/.local/bin"
```

## Next Steps

Explore the [Video Curation documentation](video-overview).
