---
description: "Overview of image data curation with NeMo Curator including loading, processing, classification, and export workflows"
categories: ["workflows"]
tags: ["image-curation", "webdataset", "classification", "embedding", "workflows"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "workflow"
modality: "image-only"
---

(image-overview)=
# About Image Curation

## Use Cases

## Architecture

## Introduction

Master the fundamentals of NeMo Curator and set up your text processing environment.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Concepts
:link: about-concepts-image
:link-type: ref
Learn about DocumentDataset and other core data structures for efficient text curation
+++
{bdg-secondary}`data-structures`
{bdg-secondary}`distributed`
{bdg-secondary}`architecture`
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Get Started
:link: gs-image
:link-type: ref
Learn prerequisites, setup instructions, and initial configuration for image curation
+++
{bdg-secondary}`setup`
{bdg-secondary}`configuration`
{bdg-secondary}`quickstart`
:::

::::

## Curation Tasks

### Load Data

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` WebDataset
:link: image-load-data-webdataset
:link-type: ref
Load and process sharded image-text datasets in the WebDataset format for scalable distributed curation.
+++
{bdg-secondary}`webdataset`
{bdg-secondary}`sharded`
{bdg-secondary}`distributed`
:::

::::

### Process Data

Transform and enhance your image data through classification, embeddings, and filters.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`filter;1.5em;sd-mr-1` Classifiers
:link: image-process-data-classifiers
:link-type: ref

Apply built-in classifiers such as Aesthetic and NSFW to score, filter, and curate large image datasets. These models help you assess image quality and remove or flag explicit content for downstream tasks like generative model training and quality control.
+++
{bdg-secondary}`Aesthetic` {bdg-secondary}`NSFW` {bdg-secondary}`quality filtering`

:::

:::{grid-item-card} {octicon}`pencil;1.5em;sd-mr-1` Embeddings
:link: image-process-data-embeddings
:link-type: ref

Generate image embeddings for your dataset using state-of-the-art models from the timm library or custom embedders. Embeddings enable downstream tasks such as classification, filtering, duplicate removal, and similarity search.
+++
{bdg-secondary}`timm` {bdg-secondary}`custom` {bdg-secondary}`embeddings`

:::

::::

### Save & Export

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`device-camera;1.5em;sd-mr-1` Save & Export
:link: image-save-export
:link-type: ref

Save metadata to Parquet, export filtered datasets, and reshard WebDatasets for downstream use. Learn how to efficiently store and prepare your curated image data for training or analysis.
+++
{bdg-secondary}`parquet` {bdg-secondary}`webdataset` {bdg-secondary}`resharding`

:::

::::
