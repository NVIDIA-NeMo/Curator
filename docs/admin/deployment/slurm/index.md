---
description: "Deploy NeMo Curator on Slurm-managed clusters with comprehensive guides for all modalities and multi-node configurations"
categories: ["workflows"]
tags: ["slurm", "cluster-management", "multi-node", "job-scripts", "shared-filesystem", "deployment"]
personas: ["admin-focused", "devops-focused"]
difficulty: "intermediate"
content_type: "workflow"
modality: "universal"
---

(admin-deployment-slurm)=
# Deploy NeMo Curator on Slurm

Deploy NeMo Curator on Slurm-managed clusters to scale data curation workflows across multiple nodes with shared storage. 

Slurm provides workload management and job scheduling for high-performance computing environments, enabling efficient resource allocation and queue management for large-scale data processing tasks.

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` All Modalities
:link: admin-deployment-slurm-general
:link-type: ref

How to set up and run NeMo Curator on Slurm for any modality.
+++
{bdg-secondary}`Slurm`
{bdg-secondary}`Cluster`
{bdg-secondary}`General`
:::

:::{grid-item-card} {octicon}`organization;1.5em;sd-mr-1` Multi-Node Setup Guide
:link: admin-deployment-slurm-multi-node
:link-type: ref

Advanced multi-node configurations for large-scale deployments with performance optimization and troubleshooting.
+++
{bdg-secondary}`Multi-Node`
{bdg-secondary}`Advanced`
{bdg-secondary}`Performance`
:::

:::{grid-item-card} {octicon}`image;1.5em;sd-mr-1` Image Modality Deployment & Workflow
:link: admin-deployment-slurm-image
:link-type: ref

A step-by-step Slurm pipeline for image curation, including embedding generation and semantic deduplication.
+++
{bdg-secondary}`Image`
{bdg-secondary}`Validated`
{bdg-secondary}`Scripts`
:::

:::{grid-item-card} {octicon}`typography;1.5em;sd-mr-1` Text Modality Deployment & Workflow
:link: admin-deployment-slurm-text
:link-type: ref

A step-by-step Slurm pipeline for text curation, including deduplication, classification, and PII redaction.
+++
{bdg-secondary}`Text`
{bdg-secondary}`Validated`
{bdg-secondary}`Scripts`
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

Deploy All Modalities <general.md>
Multi-Node Setup Guide <multi-node.md>
Deploy Text Modality <text.md>
Deploy Image Modality <image.md>
```
