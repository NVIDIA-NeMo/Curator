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

## Configuration Approach

NeMo Curator uses environment variables and YAML configuration files for deployment settings. The system supports loading configuration from YAML files using the `BaseConfig.from_yaml()` method, with environment variables providing runtime overrides.

Configuration methods (in order of common usage):

1. **Environment variables** - Runtime configuration and deployment settings
2. **YAML configuration files** - Module-specific and pipeline configurations
3. **Command-line arguments** - Script-specific parameter overrides
4. **Default values** - Built-in defaults in code

### Configuration File Usage

```{list-table} Common Configuration Patterns
:header-rows: 1
:widths: 30 70

* - Location
  - Description
* - `./config/`
  - Project-specific configuration files
* - `~/.config/nemo_curator/`
  - User-specific configuration (used in deployment scripts)
* - Module configs
  - YAML files for specific processing modules
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

These examples show common configuration patterns for different deployment scenarios. Each example includes the essential environment variables and settings needed to get NeMo Curator running in that specific environment.

::::{tab-set}

:::{tab-item} Local Development
:sync: config-local

```bash
# Set basic environment variables
export DEVICE="cpu"
export PROTOCOL="tcp"
export LOGDIR="./logs"
```

:::

:::{tab-item} Production GPU
:sync: config-gpu

```bash
# Production GPU environment
export DEVICE="gpu"
export PROTOCOL="ucx"
export RMM_WORKER_POOL_SIZE="80GiB"
export CUDF_SPILL="0"
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