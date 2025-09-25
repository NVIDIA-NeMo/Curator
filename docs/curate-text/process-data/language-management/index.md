---
description: "Handle multilingual content and language-specific processing including language identification and stop word management"
categories: ["workflows"]
tags: ["language-management", "multilingual", "fasttext", "stop-words", "language-detection"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "workflow"
modality: "text-only"
---

(text-process-data-languages)=

# Language Management

Handle multilingual content and language-specific processing requirements using NeMo Curator's tools and utilities.

NeMo Curator provides robust tools for managing multilingual text datasets through language detection, stop word management, and specialized handling for non-spaced languages. These tools are essential for creating high-quality monolingual datasets and applying language-specific processing.

## Before You Start

- The `FastTextLangId` filter (used with the `ScoreFilter` stage) requires a FastText language identification model file. Download `lid.176.bin` (or `lid.176.ftz`) from FastText: [Language identification](https://fasttext.cc/docs/en/language-identification.html).
- On a cluster, ensure the FastText model file is accessible to all workers (for example, a shared filesystem or object storage path).
- Provide newline-delimited JSON (`.jsonl`) with a `text` field, or set `text_field` in `ScoreFilter(...)`.
- For HTML extraction workflows (for example, Common Crawl), Curator uses CLD2 to provide language hints.

---

## How it Works

Language management in NeMo Curator typically follows this pattern using the Pipeline API:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.modules import ScoreFilter
from nemo_curator.stages.text.filters import FastTextLangId

# 1) Build the pipeline
pipeline = Pipeline(name="language_management")

# Read JSONL files into document batches
pipeline.add_stage(
    JsonlReader(file_paths="input_data/*.jsonl", files_per_partition=2)
)

# Identify languages and keep docs above a confidence threshold
pipeline.add_stage(
    ScoreFilter(
        FastTextLangId(model_path="/path/to/lid.176.bin", min_langid_score=0.3),
        score_field="language",
    )
)

# 2) Execute
results = pipeline.run()
```

---

## Language Processing Capabilities

- **Language detection** using FastText (176 languages) and CLD2 (used in HTML extraction pipelines)
- **Stop word management** with built-in lists and customizable thresholds
- **Special handling** for non-spaced languages (Chinese, Japanese, Thai, Korean)
- **Language-specific** text processing and quality filtering

## Available Tools

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`globe;1.5em;sd-mr-1` Language Identification
:link: language
:link-type: doc
Identify document languages and separate multilingual datasets
+++
{bdg-secondary}`fasttext`
{bdg-secondary}`176-languages`
{bdg-secondary}`detection`
{bdg-secondary}`classification`
:::

:::{grid-item-card} {octicon}`filter;1.5em;sd-mr-1` Stop Words
:link: stopwords
:link-type: doc
Manage high-frequency words to enhance text extraction and content detection
+++
{bdg-secondary}`preprocessing`
{bdg-secondary}`filtering`
{bdg-secondary}`language-specific`
{bdg-secondary}`nlp`
:::

::::

```{toctree}
:maxdepth: 4
:titlesonly:
:hidden:

Language Identification <language>
Stop Words <stopwords>
```
