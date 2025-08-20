---
description: "Domain-specific processing for code, bitext, synthetic data, and advanced curation tasks with specialized modules"
categories: ["workflows"]
tags: ["specialized-processing", "code", "bitext", "synthetic-data", "task-decontamination", "advanced"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "advanced"
content_type: "workflow"
modality: "text-only"
---

(text-process-data-specialized)=
# Specialized Processing

Domain-specific processing for code, bitext, synthetic data, and advanced curation tasks using NeMo Curator's specialized modules.

This section covers advanced processing techniques for specific data types and use cases that require specialized handling beyond general text processing. These tools are designed for specific domains like programming content, parallel text, AI-generated content, and benchmark contamination.

## How it Works

Specialized processing modules in NeMo Curator are designed for specific data types and use cases:

- **Code Processing**: Handles programming languages with syntax-aware filtering
- **Bitext Processing**: Manages parallel text for translation quality assessment
- **Synthetic Data Detection**: Identifies AI-generated or synthetic content
- **Task Decontamination**: Removes benchmark data from training sets

Each specialized processor understands the unique characteristics of its target domain and applies appropriate metrics and thresholds.

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

:::{grid-item-card} {octicon}`repo-forked;1.5em;sd-mr-1` Parallel Text (Bitext)
:link: bitext
:link-type: doc
Filter parallel text for translation quality and alignment
+++
{bdg-secondary}`translation`
{bdg-secondary}`bilingual`
{bdg-secondary}`quality-estimation`
{bdg-secondary}`alignment`
:::

:::{grid-item-card} {octicon}`sync;1.5em;sd-mr-1` Synthetic Data Detection
:link: synthetic
:link-type: doc
Identify AI-generated or synthetic content in datasets
+++
{bdg-secondary}`ai-detection`
{bdg-secondary}`synthetic`
{bdg-secondary}`embeddings`
{bdg-secondary}`qa-pairs`
:::

:::{grid-item-card} {octicon}`shield-check;1.5em;sd-mr-1` Task Decontamination
:link: task-decontamination
:link-type: doc
Remove downstream task data from training datasets
+++
{bdg-secondary}`benchmarks`
{bdg-secondary}`contamination`
{bdg-secondary}`evaluation`
{bdg-secondary}`deduplication`
:::

::::

## Usage

### Quick Examples

::::{tab-set}

:::{tab-item} Code Processing
```python
from nemo_curator import Sequential, ScoreFilter
from nemo_curator.filters import PythonCommentToCodeFilter, NumberOfLinesOfCodeFilter

# Filter Python code based on quality metrics
code_pipeline = Sequential([
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

:::{tab-item} Bitext Processing
```python
from nemo_curator.filters import QualityEstimationFilter

# Filter translation pairs for quality
qe_filter = QualityEstimationFilter(
    model_name="comet-qe",
    cutoff=0.5,
    mode="always_en_x",
    src_field="source",
    tgt_field="target",
    metadata_fields=["src_lang", "tgt_lang"]
)

high_quality_translations = qe_filter(parallel_dataset)
```
:::

:::{tab-item} Synthetic Detection
```python
from nemo_curator.filters.synthetic import EasinessFilter, AnswerabilityFilter

# Detect synthetic QA pairs
synthetic_pipeline = Sequential([
    ScoreFilter(
        EasinessFilter(
            base_url="https://api-endpoint",
            percentile=0.7,
            text_fields=["context", "question"]
        ),
        text_field=["context", "question"],
        score_field="easiness_score"
    ),
    ScoreFilter(
        AnswerabilityFilter(
            base_url="https://llm-endpoint",
            text_fields=["context", "question"]
        ),
        text_field=["context", "question"],
        score_field="answerability_score"
    )
])

authentic_qa = synthetic_pipeline(qa_dataset)
```
:::

:::{tab-item} Task Decontamination
```python
from nemo_curator import TaskDecontamination
from nemo_curator.tasks import Squad, TriviaQA, Winogrande

# Remove benchmark contamination
decontaminate = TaskDecontamination([
    Squad(),
    TriviaQA(), 
    Winogrande()
])

clean_dataset = decontaminate(training_dataset)
```
:::

::::

## When to Use Specialized Processing

- **Code datasets**: When working with programming content that needs syntax-aware filtering
- **Multilingual datasets**: When processing parallel text for machine translation
- **Synthetic data**: When detecting AI-generated content in training datasets  
- **Benchmark preparation**: When ensuring training data doesn't contain evaluation tasks

## Performance Considerations

- **Code processing**: Fast heuristic-based filtering, suitable for large code repositories
- **Bitext processing**: May require API calls for quality estimation, consider rate limits
- **Synthetic detection**: API-dependent, can be computationally expensive for large datasets
- **Task decontamination**: One-time preprocessing step, cache results for reuse

```{toctree}
:maxdepth: 4
:titlesonly:
:hidden:

Code Processing <code>
Parallel Text Processing <bitext>
Synthetic Data Detection <synthetic>
Task Decontamination <task-decontamination>
``` 