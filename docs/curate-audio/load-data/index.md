---
description: "Load audio datasets from various sources including FLEURS, custom manifests, and local files"
categories: ["workflows"]
tags: ["data-loading", "audio-manifests", "fleurs", "local-files", "batch-processing"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "workflow"
modality: "audio-only"
---

# Load Audio Data

Import audio datasets from various sources into NeMo Curator's audio processing pipeline. Audio data loading supports manifest files, direct file paths, and automated dataset downloads.

## How it Works

Audio data loading in NeMo Curator centers around the `AudioBatch` data structure, which contains:

- **Audio file paths**: References to audio files (.wav, .mp3, .flac, etc.)
- **Transcriptions**: Ground truth or reference text for speech content
- **Metadata**: Duration, language, speaker information, and quality metrics

The loading process validates audio file existence and formats data for downstream ASR inference and quality assessment stages.

---

## Loading Methods

Choose the appropriate loading method based on your data source and format:

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` FLEURS Dataset
:link: fleurs-dataset
:link-type: doc
Automated download and processing of the multilingual FLEURS speech dataset
+++
{bdg-secondary}`automated`
{bdg-secondary}`multilingual`
{bdg-secondary}`102-languages`
:::

:::{grid-item-card} {octicon}`file;1.5em;sd-mr-1` Custom Manifests
:link: custom-manifests
:link-type: doc
Create and load custom audio manifests with file paths and transcriptions
+++
{bdg-secondary}`jsonl`
{bdg-secondary}`tsv`
{bdg-secondary}`custom-format`
:::

:::{grid-item-card} {octicon}`file-directory;1.5em;sd-mr-1` Local Files
:link: local-files
:link-type: doc
Load audio files directly from local directories and file systems
+++
{bdg-secondary}`local-storage`
{bdg-secondary}`batch-processing`
{bdg-secondary}`file-discovery`
:::

::::

```{toctree}
:maxdepth: 2
:titlesonly:
:hidden:

FLEURS Dataset <fleurs-dataset.md>
Custom Manifests <custom-manifests.md>
Local Files <local-files.md>
```

