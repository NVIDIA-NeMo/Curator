---
description: "Comprehensive technical reference for NeMo Curator APIs, infrastructure components, and integration tools"
categories: ["reference"]
tags: ["python-api", "infrastructure", "integrations-apis", "distributed", "gpu-accelerated"]
personas: ["mle-focused", "data-scientist-focused", "admin-focused"]
difficulty: "reference"
content_type: "reference"
modality: "universal"
---

(ref-overview)=
# References

NeMo Curator's reference documentation provides comprehensive technical details, API references, and integration information to help you maximize your NeMo Curator implementation. Use these resources to understand the technical foundation of NeMo Curator and integrate it with other tools and systems.

## API Quicklinks

Quickly access core NeMo Curator API references. Use these links to jump directly to the technical API documentation for each major module.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Classifiers API
:link: ../apidocs/classifiers/classifiers.html
:link-type: url
Core classifier base classes and interfaces.
+++
{bdg-secondary}`classification`
{bdg-secondary}`aegis`
{bdg-secondary}`prompt-task-complexity`
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Datasets API
:link: ../apidocs/datasets/datasets.html
:link-type: url
APIs for document and parallel datasets.
+++
{bdg-secondary}`doc-dataset`
{bdg-secondary}`parallel-dataset`
{bdg-secondary}`image-text-pair`
:::

:::{grid-item-card} {octicon}`git-branch;1.5em;sd-mr-1` Deduplication API
:link: ../apidocs/modules/modules.semantic_dedup.html
:link-type: url
Deduplication and semantic deduplication tools.
+++
{bdg-secondary}`semantic-dedup`
{bdg-secondary}`fuzzy-dedup`
:::

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` Download API
:link: ../apidocs/download/download.html
:link-type: url
APIs for downloading and building datasets from external sources.
+++
{bdg-secondary}`arxiv`
{bdg-secondary}`commoncrawl`
{bdg-secondary}`wikipedia`
:::

:::{grid-item-card} {octicon}`filter;1.5em;sd-mr-1` Filters API
:link: ../apidocs/filters/filters.html
:link-type: url
Filtering and quality control APIs.
+++
{bdg-secondary}`classifier-filter`
{bdg-secondary}`heuristic-filter`
:::

:::{grid-item-card} {octicon}`image;1.5em;sd-mr-1` Image API
:link: ../apidocs/image/image.html
:link-type: url
Image processing, classifiers, and embedders.
+++
{bdg-secondary}`classifiers`
{bdg-secondary}`embedders`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Modifiers API
:link: ../apidocs/modifiers/modifiers.html
:link-type: url
Text and data modification utilities.
+++
{bdg-secondary}`pii-modifier`
{bdg-secondary}`markdown-remover`
:::

:::{grid-item-card} {octicon}`shield-check;1.5em;sd-mr-1` PII API
:link: ../apidocs/pii/pii.html
:link-type: url
PII detection, recognizers, and redaction tools.
+++
{bdg-secondary}`recognizers`
{bdg-secondary}`algorithms`
:::

:::{grid-item-card} {octicon}`plug;1.5em;sd-mr-1` Services API
:link: ../apidocs/services/services.html
:link-type: url
Service clients and integrations.
+++
{bdg-secondary}`model-client`
{bdg-secondary}`openai-client`
:::

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` NeMo Run API
:link: ../apidocs/nemo_run/nemo_run.html
:link-type: url
NeMo Run integration for distributed execution.
+++
{bdg-secondary}`slurm`
{bdg-secondary}`distributed`
:::

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` Tasks API
:link: ../apidocs/tasks/tasks.html
:link-type: url
Task definitions and metrics.
+++
{bdg-secondary}`metrics`
{bdg-secondary}`downstream-task`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Utils API
:link: ../apidocs/utils/utils.html
:link-type: url
General utility functions and helpers.
+++
{bdg-secondary}`text-utils`
{bdg-secondary}`distributed-utils`
:::

::::

## Infrastructure Components

Explore the foundational infrastructure that powers NeMo Curator. Learn how to scale, optimize, and manage large data workflows efficiently.

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

::::

## Integration & Tools

Discover related tools and integrations in the NVIDIA AI ecosystem that complement NeMo Curator, enabling seamless workflows from data curation to model training and deployment.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`plug;1.5em;sd-mr-1` Related Tools
:link: reference-tools
:link-type: ref
Learn about complementary tools in the NVIDIA ecosystem
+++
{bdg-secondary}`nemo-framework`
{bdg-secondary}`triton-server`
{bdg-secondary}`tao-toolkit`
:::

::::
