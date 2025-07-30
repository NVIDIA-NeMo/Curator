---
description: "Deploy NeMo Curator in production environments with comprehensive guides for Kubernetes and Slurm cluster deployments"
categories: ["workflows"]
tags: ["deployment", "kubernetes", "slurm", "production", "cluster-management", "infrastructure"]
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

- **System**: Ubuntu 22.04/20.04, Python 3.10+
- **Hardware**: Multi-core CPU, 16GB+ RAM (optional: NVIDIA GPU with 16GB+ VRAM)
- **Software**: Dask, container runtime (Docker/Singularity), cluster management tools
- **Infrastructure**: Shared storage, high-bandwidth networking

For detailed system, hardware, and software requirements, see [Production Deployment Requirements](admin-deployment-requirements).

---

## Deployment Options

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Kubernetes Deployment
:link: admin-deployment-kubernetes
:link-type: ref
Deploy NeMo Curator on Kubernetes clusters using Dask Operator, GPU Operator, and PVC storage. Includes setup, storage, cluster creation, module execution, and cleanup.
+++
{bdg-secondary}`Kubernetes`
{bdg-secondary}`Dask Operator`
{bdg-secondary}`GPU`
{bdg-secondary}`PVC Storage`
{bdg-secondary}`Cluster Management`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Slurm Deployment
:link: admin-deployment-slurm
:link-type: ref
Run NeMo Curator on Slurm clusters with shared filesystems. Covers job scripts, Dask cluster setup, module execution, monitoring, and advanced Python-based job submission.
+++
{bdg-secondary}`Slurm`
{bdg-secondary}`Dask`
{bdg-secondary}`Shared Filesystem`
{bdg-secondary}`Job Scripts`
{bdg-secondary}`Cluster Management`
:::

::::

```{toctree}
:maxdepth: 4
:titlesonly:
:hidden:

Requirements <requirements>
Kubernetes <kubernetes>
Slurm <slurm/index.md>

```


## After Deployment

Once your infrastructure is running, you'll need to configure NeMo Curator for your specific environment. See the {doc}`Configuration Guide <../config/index>` for deployment-specific settings, environment variables, and storage credentials.
