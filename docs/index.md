(curator-home)=

# NeMo Curator Documentation

Welcome to the NeMo Curator documentation.

## Introduction to Curator

Learn about the Curator, how it works at a high-level, and the key features.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` About Curator
:link: about-overview
:link-type: ref
Overview of NeMo Curator and its capabilities.
+++
{bdg-secondary}`target-users` {bdg-secondary}`how-it-works`
:::

:::{grid-item-card} {octicon}`image;1.5em;sd-mr-1` Key Features
:link: about-key-features
:link-type: ref
Discover the main features of NeMo Curator for data curation.
+++
{bdg-secondary}`features` {bdg-secondary}`capabilities` {bdg-secondary}`deployments`
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Concepts
:link: about-concepts
:link-type: ref
Explore the core concepts for each modality in NeMo Curator.
+++
{bdg-secondary}`data-loading` {bdg-secondary}`data-processing` {bdg-secondary}`data-generation`
:::

::::

## Data Curation Workflows

### Workflow Modalities

Explore how you can use NeMo Curator across different content modalities.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Curate Text
:link: text-overview
:link-type: ref
Curate and prepare high-quality text datasets for LLM training.
+++
{bdg-secondary}`filtering` {bdg-secondary}`formatting` {bdg-secondary}`synthetic-data`
:::

:::{grid-item-card} {octicon}`image;1.5em;sd-mr-1` Curate Images
:link: image-overview
:link-type: ref
Curate image-text datasets with embedding, classification, and deduplication.
+++
{bdg-secondary}`embedding` {bdg-secondary}`classification` {bdg-secondary}`semantic-deduplication`
:::

::::

## Quickstart Guides

Install and run NeMo Curator for specific modalities.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Text Curation Quickstart
:link: gs-text
:link-type: ref
Quickly set up and run text curation workflows.
+++
{bdg-secondary}`setup` {bdg-secondary}`installation` {bdg-secondary}`quickstart`
:::

:::{grid-item-card} {octicon}`image;1.5em;sd-mr-1` Image Curation Quickstart
:link: gs-image
:link-type: ref
Quickly set up and run image curation workflows.
+++
{bdg-secondary}`setup` {bdg-secondary}`installation` {bdg-secondary}`quickstart`
:::

::::

## Tutorial Highlights

Check out tutorials to get a quick start on using the NeMo Curator library.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`mortar-board;1.5em;sd-mr-1` Text Beginner Tutorial
:link: gs-text
:link-type: ref
Learn the basics of text data processing with NeMo Curator.
+++
{bdg-primary}`beginner`
{bdg-secondary}`text-processing`
{bdg-secondary}`data-preparation`
:::

:::{grid-item-card} {octicon}`mortar-board;1.5em;sd-mr-1` Image Beginner Tutorial
:link: gs-image
:link-type: ref
Learn the basics of image data processing with NeMo Curator.
+++
{bdg-primary}`beginner`
{bdg-secondary}`image-processing`
{bdg-secondary}`data-curation`
:::

::::

---

::::{toctree}
:hidden:
Home <self>
::::

::::{toctree}
:hidden:
:caption: About NeMo Curator
:maxdepth: 1
about/index.md
about/key-features.md
about/concepts/index.md
about/release-notes/index.md
::::

::::{toctree}
:hidden:
:caption: Get Started
:maxdepth: 2

get-started/index.md
Text Curation Quickstart <get-started/text.md>
Image Curation Quickstart <get-started/image.md>
::::

::::{toctree}
:hidden:
:caption: Curate Text
:maxdepth: 2
curate-text/index.md
Tutorials <curate-text/tutorials/index.md>
Load Data <curate-text/load-data/index.md>
Process Data <curate-text/process-data/index.md>
Generate Data <curate-text/generate-data/index.md>
::::

::::{toctree}
:hidden:
:caption: Curate Images
:maxdepth: 2
curate-images/index.md
Tutorials <curate-images/tutorials/index.md>
Load Data <curate-images/load-data/index.md>
Process Data <curate-images/process-data/index.md>
Save and Export <curate-images/save-export.md>
::::

::::{toctree}
:hidden:
:caption: Setup & Deployment
:maxdepth: 2
admin/index.md
Install Curator <admin/installation.md>
Configure Curator <admin/config/index.md>
Deploy Curator <admin/deployment/index.md>
Integrations <admin/integrations/index.md>
::::

::::{toctree}
:hidden:
:caption: Reference
:maxdepth: 2

About References <reference/index.md>
Infrastructure <reference/infrastructure/index.md>
apidocs/index.rst
Tools <reference/related-tools.md>
::::
