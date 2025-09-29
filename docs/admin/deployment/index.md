---
description: "Deploy NeMo Curator in production environments with comprehensive guides for Slurm cluster deployments"
categories: ["workflows"]
tags: ["deployment", "slurm", "production", "cluster-management", "infrastructure"]
personas: ["admin-focused", "devops-focused"]
difficulty: "intermediate"
content_type: "workflow"
modality: "universal"
---

(admin-deployment)=

# Deploy NeMo Curator

Use the following Admin guides to set up NeMo Curator in a production environment.

## Prerequisites

Before deploying NeMo Curator in a production environment, review the comprehensive requirements:

- **System**: Ubuntu 22.04/20.04, Python 3.10, 3.11, or 3.12
- **Hardware**: Multi-core CPU, 16GB+ RAM (optional: NVIDIA GPU with 16GB+ VRAM)
- **Software**: Ray (distributed computing framework), container runtime, cluster management tools
- **Infrastructure**: Shared storage, high-bandwidth networking

For detailed system, hardware, and software requirements, see [Production Deployment Requirements](admin-deployment-requirements).

---

## Deployment Options

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Slurm Deployment
:link: admin-deployment-slurm
:link-type: ref
Run NeMo Curator on Slurm clusters with shared filesystems using Singularity/Apptainer containers. Covers job scripts, Dask cluster setup, module execution, monitoring, and advanced Python-based job submission.
+++
{bdg-secondary}`Slurm`
{bdg-secondary}`Shared Filesystem`
{bdg-secondary}`Job Scripts`
{bdg-secondary}`Cluster Management`
:::

```{toctree}
:maxdepth: 4
:titlesonly:
:hidden:

Requirements <requirements>
Slurm <slurm/index.md>

```

## After Deployment

Once your infrastructure is running, you'll need to configure NeMo Curator for your specific environment. See the {doc}`Configuration Guide <../config/index>` for deployment-specific settings, environment variables, and storage credentials.
