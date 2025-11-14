---
description: "Domain-specific processing for code and advanced curation tasks with specialized modules"
categories: ["workflows"]
tags: ["specialized-processing", "code", "advanced"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "advanced"
content_type: "workflow"
modality: "text-only"
---

(text-process-data-specialized)=

# Specialized Processing

Domain-specific processing for code and advanced curation tasks using NeMo Curator's specialized modules.

This section covers advanced processing techniques for specific data types and use cases that require specialized handling beyond general text processing. These tools are designed for specific domains like programming content.

## How it Works

Specialized processing modules in NeMo Curator are designed for specific data types and use cases:

- **Code Processing**: Handles programming languages with syntax-aware filtering

Each specialized processor understands the unique characteristics of its target domain and applies appropriate metrics and thresholds within the broader {ref}`data processing framework <about-concepts-text-data-processing>`.

---

## Available Specialized Tools

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Code Processing
:link: code
:link-type: doc
Specialized filters for programming content and source code
+++
{bdg-secondary}`programming`
{bdg-secondary}`syntax`
{bdg-secondary}`comments`
{bdg-secondary}`languages`
:::

::::

## Usage

### Quick Examples

::::{tab-set}

:::{tab-item} Code Processing

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.modules import ScoreFilter
from nemo_curator.stages.text.filters import PythonCommentToCodeFilter, NumberOfLinesOfCodeFilter

# Filter Python code based on quality metrics
code_pipeline = Pipeline(
    name="code_processing_pipeline",
    stages=[
    ScoreFilter(
        PythonCommentToCodeFilter(
            min_comment_to_code_ratio=0.01,
            max_comment_to_code_ratio=0.8
        ),
        text_field="content",
        score_field="comment_ratio"
    ),
    ScoreFilter(
        NumberOfLinesOfCodeFilter(min_lines=5, max_lines=1000),
        text_field="content", 
        score_field="line_count"
    )
])

filtered_code = code_pipeline(code_dataset)
```

:::

::::

## When to Use Specialized Processing

- **Code datasets**: When working with programming content that needs syntax-aware filtering

## Performance Considerations

- **Code processing**: Fast heuristic-based filtering, suitable for large code repositories

```{toctree}
:maxdepth: 4
:titlesonly:
:hidden:

Code Processing <code>
```