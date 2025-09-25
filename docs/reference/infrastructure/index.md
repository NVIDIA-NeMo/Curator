---
description: "Technical reference for NeMo Curator's infrastructure components including distributed computing, memory management, and GPU acceleration"
categories: ["reference"]
tags: ["infrastructure", "distributed", "gpu-accelerated", "memory-management", "docker", "performance"]
personas: ["admin-focused", "mle-focused", "devops-focused"]
difficulty: "reference"
content_type: "reference"
modality: "universal"
---

# Infrastructure References

This section provides technical reference documentation for NeMo Curator's infrastructure components that are used across all modalities (text, image, video). For deployment and operational configuration, see the [Admin Configuration Guide](admin-config).

---

## Infrastructure Components

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Memory Management
:link: reference-infra-memory-management
:link-type: ref
Optimize memory usage when processing large datasets
+++
{bdg-secondary}`partitioning`
{bdg-secondary}`batching`
{bdg-secondary}`monitoring`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` GPU Acceleration
:link: reference-infra-gpu-processing
:link-type: ref
Leverage NVIDIA GPUs for faster data processing
+++
{bdg-secondary}`cuda`
{bdg-secondary}`rmm`
{bdg-secondary}`performance`
:::

:::{grid-item-card} {octicon}`sync;1.5em;sd-mr-1` Resumable Processing
:link: reference-infra-resumable-processing
:link-type: ref
Continue interrupted operations across large datasets
+++
{bdg-secondary}`checkpoints`
{bdg-secondary}`recovery`
{bdg-secondary}`batching`
:::

:::{grid-item-card} {octicon}`container;1.5em;sd-mr-1` Container Environments
:link: reference-infrastructure-container-environments
:link-type: ref
Available environments and configurations in NeMo Curator containers. Includes Slurm environment variables, build arguments, and video-specific environments.
+++
{bdg-secondary}`docker`
{bdg-secondary}`conda`
{bdg-secondary}`slurm`
{bdg-secondary}`environments`
:::

::::

```{toctree}
:maxdepth: 2
:hidden:

memory-management
gpu-processing
resumable-processing
container-environments
execution-backends
```
