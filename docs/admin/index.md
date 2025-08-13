---
description: "Comprehensive administration guide for NeMo Curator deployment, infrastructure management, monitoring, and scaling from development to production"
categories: ["getting-started"]
tags: ["admin", "deployment", "infrastructure", "installation", "configuration", "kubernetes", "slurm"]
personas: ["admin-focused", "devops-focused", "data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "overview"
modality: "universal"
---

(admin-overview)=
# About Setup & Deployment

The administration section provides comprehensive information for deployment, infrastructure management, monitoring, and scaling NeMo Curator. Use these resources to efficiently set up and maintain your NeMo Curator environment at any scale, from development workstations to production clusters.

---

## Installation & Configuration

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` Installation Guide
:link: admin-installation
:link-type: ref
Install NeMo Curator with system requirements, package extras, and verification steps. Covers PyPI, source, and container installation methods.
+++
{bdg-secondary}`Installation`
{bdg-secondary}`System Requirements`
{bdg-secondary}`Package Extras`
{bdg-secondary}`Verification`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuration Guide
:link: admin-config
:link-type: ref
Configure NeMo Curator for deployment environments, storage access, credentials, and environment variables for operational management.
+++
{bdg-secondary}`Deployment Config`
{bdg-secondary}`Storage & Credentials`
{bdg-secondary}`Environment Variables`
{bdg-secondary}`Operational Setup`
:::

::::

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

## Integration Options

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Spark
:link: admin-integrations-spark
:link-type: ref
Integrate NeMo Curator with Apache Spark for distributed processing
+++
{bdg-secondary}`batch-processing`
{bdg-secondary}`performance`
{bdg-secondary}`optimization`
:::
::::
