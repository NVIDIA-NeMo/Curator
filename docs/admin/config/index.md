---
description: "Configuration guide for NeMo Curator deployment environments, storage access, credentials, and operational settings"
categories: ["workflows"]
tags: ["configuration", "deployment-environments", "storage-credentials", "environment-variables", "operational-setup"]
personas: ["admin-focused", "devops-focused"]
difficulty: "intermediate"
content_type: "workflow"
modality: "universal"
---

(admin-config)=
# Configuration Guide

Configure NeMo Curator for your deployment environment including infrastructure settings, storage access, credentials, and environment variables. This section focuses on operational configuration for deployment and management.

---

## Configuration Areas

This section covers the three main areas of operational configuration for NeMo Curator deployments. Each area addresses different aspects of system setup and management, from infrastructure deployment to data access and runtime settings.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Deployment Environments
:link: admin-config-deployment-environments
:link-type: ref
Configure NeMo Curator for different deployment scenarios including Slurm, Kubernetes, and local environments.
+++
{bdg-secondary}`Slurm`
{bdg-secondary}`Kubernetes`
{bdg-secondary}`Dask Clusters`
{bdg-secondary}`GPU Settings`
:::

:::{grid-item-card} {octicon}`key;1.5em;sd-mr-1` Storage & Credentials
:link: admin-config-storage-credentials
:link-type: ref
Configure cloud storage access, API keys, and security credentials for data processing and model access.
+++
{bdg-secondary}`Cloud Storage`
{bdg-secondary}`API Keys`
{bdg-secondary}`Security`
{bdg-secondary}`File Systems`
:::

:::{grid-item-card} {octicon}`list-unordered;1.5em;sd-mr-1` Environment Variables
:link: admin-config-environment-variables
:link-type: ref
Comprehensive reference of all environment variables used by NeMo Curator across different deployment scenarios.
+++
{bdg-secondary}`Environment Variables`
{bdg-secondary}`Configuration Profiles`
{bdg-secondary}`Deployment Settings`
{bdg-secondary}`Reference`
:::

::::

---

## Module-Specific Configuration

Module-specific configuration handles processing pipeline settings for different data modalities. These configurations complement the deployment settings above and focus on algorithm parameters, model configurations, and processing behavior rather than infrastructure concerns.

For configuration of specific processing modules (deduplication, classifiers, filters), see the relevant modality sections:

::::{grid} 1 1 1 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`typography;1.5em;sd-mr-1` Text Processing
:link: text-overview
:link-type: ref
Configuration for text deduplication, classification, and filtering modules.
+++
{bdg-secondary}`Deduplication`
{bdg-secondary}`Classifiers`
{bdg-secondary}`Filters`
:::

:::{grid-item-card} {octicon}`image;1.5em;sd-mr-1` Image Processing  
:link: image-overview
:link-type: ref
Configuration for image classifiers, embedders, and filtering.
+++
{bdg-secondary}`Classifiers`
{bdg-secondary}`Embedders`
{bdg-secondary}`Filtering`
:::

::::

---

## Configuration Hierarchy

NeMo Curator follows a hierarchical configuration system where settings can be specified at multiple levels. This hierarchy ensures flexibility while maintaining clear precedence rules for resolving configuration conflicts across different deployment environments.

NeMo Curator uses the following configuration precedence (highest to lowest priority):

1. **Command-line arguments** - Direct parameter overrides
2. **Environment variables** - Runtime configuration
3. **Configuration files** - YAML/JSON configuration files
4. **Default values** - Built-in defaults

### Configuration File Locations

```{list-table} Configuration File Search Order
:header-rows: 1
:widths: 30 70

* - Location
  - Description
* - `./config/`
  - Current working directory config folder
* - `~/.nemo_curator/`
  - User-specific configuration directory
* - `/etc/nemo_curator/`
  - System-wide configuration directory
* - Package defaults
  - Built-in default configurations
```

### Example Configuration Structure

```bash
# Typical deployment configuration layout
config/
├── deployment.yaml          # Deployment-specific settings
├── storage.yaml             # Storage and credential configuration  
├── logging.yaml             # Logging configuration
└── modules/
    ├── deduplication.yaml   # Module-specific configs
    ├── classification.yaml
    └── filtering.yaml
```

---

## Quick Start Examples

These examples demonstrate common configuration patterns for different deployment scenarios. Each example includes the essential environment variables and settings needed to get NeMo Curator running in that specific environment.

::::{tab-set}

:::{tab-item} Local Development
:sync: config-local

```bash
# Set basic environment variables
export DASK_CLUSTER_TYPE="cpu"
export NEMO_CURATOR_LOG_LEVEL="INFO"
export NEMO_CURATOR_CACHE_DIR="./cache"
```
:::

:::{tab-item} Production Slurm
:sync: config-slurm

```bash
# Production Slurm environment
export DASK_CLUSTER_TYPE="gpu"
export DASK_PROTOCOL="ucx"
export RMM_WORKER_POOL_SIZE="80GiB"
export NEMO_CURATOR_LOG_LEVEL="WARNING"
```
:::

:::{tab-item} Cloud Storage
:sync: config-cloud

```bash
# AWS S3 configuration
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-west-2"
```
:::

::::

```{toctree}
:maxdepth: 2
:hidden:

deployment-environments
storage-credentials
environment-variables
``` 