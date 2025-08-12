---
description: "Score and remove low-quality content using heuristics and ML classifiers with comprehensive filtering capabilities"
categories: ["workflows"]
tags: ["quality-assessment", "filtering", "heuristic", "classifier", "distributed", "scoring"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "workflow"
modality: "text-only"
---

(text-process-data-filter)=

# Quality Assessment & Filtering

Score and remove low-quality content using heuristics and ML classifiers to prepare your data for model training using Ray Curator's task-centric pipeline architecture.

Large datasets often contain documents considered to be "low quality." In this context, "low quality" data means data we don't want a downstream model to learn from, and "high quality" data is data that we do want a downstream model to learn from. The metrics that define quality can vary widely.

## How It Works

Ray Curator's filtering framework uses a task-centric architecture with these key components:

- **Tasks**: `DocumentBatch` objects containing batches of text data that flow through the pipeline
- **Stages**: Processing units that transform tasks (Score, Filter, ScoreFilter)
- **Pipelines**: Collections of stages that define the complete workflow
- **Executors**: Components that orchestrate pipeline execution on distributed systems

::::{tab-set}

:::{tab-item} ScoreFilter

The `ScoreFilter` stage combines scoring and filtering in Ray Curator. It processes `DocumentBatch` tasks through a pipeline:

```python
from ray_curator.pipeline import Pipeline
from ray_curator.backends.xenna import XennaExecutor
from ray_curator.stages.modules import ScoreFilter
from ray_curator.stages.filters import WordCountFilter
from ray_curator.stages.io.reader import JsonlReader
from ray_curator.stages.io.writer import JsonlWriter

# Create pipeline
pipeline = Pipeline(
    name="book_filtering",
    description="Filter books by word count"
)

# Add stages to pipeline
pipeline.add_stage(JsonlReader(file_paths="books_dataset/"))
pipeline.add_stage(ScoreFilter(
    filter_obj=WordCountFilter(min_words=80),
    text_field="text",
    score_field="word_count"
))
pipeline.add_stage(JsonlWriter(output_dir="long_books/"))

# Execute pipeline
executor = XennaExecutor()
results = pipeline.run(executor)
```

The `DocumentFilter` objects define two key methods:

- `score_document`: Computes a quality score for a document
- `keep_document`: Determines if a document should remain based on its score

:::

:::{tab-item} Score and Filter Stages

Ray Curator provides specialized stages for granular control:

- `Score`: A stage that adds metadata scores without filtering
  - Takes a scoring function that evaluates text and returns a score
  - Adds the score to a specified metadata field
  - Useful for analysis or multi-stage filtering pipelines
  
```python
from ray_curator.stages.modules import Score
from ray_curator.stages.filters import WordCountFilter

# Add scoring stage to pipeline
pipeline.add_stage(Score(
    score_fn=WordCountFilter().score_document,
    text_field="text",
    score_field="word_count"
))
```

- `Filter`: A stage that filters based on pre-computed metadata
  - Takes a filter function that evaluates metadata and returns True/False
  - Uses existing metadata fields (doesn't compute new scores)
  - Efficient for filtering on pre-computed metrics
  
```python
from ray_curator.stages.modules import Filter

# Add filtering stage to pipeline
pipeline.add_stage(Filter(
    filter_fn=lambda score: score >= 100,
    filter_field="word_count"
))
```

You can combine these stages in multi-step pipelines:

```python
# Multi-stage filtering pipeline
pipeline = Pipeline(name="multi_filter", description="Multi-stage quality filtering")
pipeline.add_stage(JsonlReader(file_paths="dataset/"))
pipeline.add_stage(Score(word_counter, score_field="word_count"))
pipeline.add_stage(Score(symbol_counter, score_field="symbol_ratio"))
pipeline.add_stage(Filter(lambda x: x >= 100, filter_field="word_count"))
pipeline.add_stage(Filter(lambda x: x <= 0.3, filter_field="symbol_ratio"))
pipeline.add_stage(JsonlWriter(output_dir="filtered_output/"))
```

:::

:::{tab-item} Task-Based Processing

Ray Curator processes data in batches through `DocumentBatch` tasks, providing efficient vectorized operations:

```python
from ray_curator.stages.modules import ScoreFilter
from ray_curator.stages.filters import WordCountFilter

# DocumentBatch automatically handles batch processing
class CustomFilter(DocumentFilter):
    def score_document(self, text: str) -> float:
        return len(text.split())
    
    def keep_document(self, score: float) -> bool:
        return score > 10

# Used in pipeline - batch processing is automatic
pipeline.add_stage(ScoreFilter(
    filter_obj=CustomFilter(),
    text_field="text"
))
```

Task-based processing provides performance benefits:

- **Automatic batching**: `DocumentBatch` handles many documents per task
- **Vectorized operations**: Pandas DataFrame operations within each batch
- **Resource optimization**: Tasks sized for optimal memory and compute usage
- **Fault tolerance**: Individual tasks can retry independently

:::

::::

---

## Filtering Approaches

::::{grid} 1 1 1 2
:gutter: 2

:::{grid-item-card} {octicon}`filter;1.5em;sd-mr-1` Heuristic Filtering
:link: heuristic
:link-type: doc
Filter text using configurable rules and metrics
+++
{bdg-secondary}`rules`
{bdg-secondary}`metrics`
{bdg-secondary}`fast`
:::

:::{grid-item-card} {octicon}`cpu;1.5em;sd-mr-1` Classifier Filtering
:link: classifier
:link-type: doc
Filter text using trained quality classifiers
+++
{bdg-secondary}`ml-models`
{bdg-secondary}`quality`
{bdg-secondary}`scoring`
:::

:::{grid-item-card} {octicon}`cpu;1.5em;sd-mr-1` Distributed Classification
:link: distributed-classifier
:link-type: doc
GPU-accelerated classification with pre-trained models
+++
{bdg-secondary}`gpu`
{bdg-secondary}`distributed`
{bdg-secondary}`scalable`
:::

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` Custom Filters
:link: custom
:link-type: doc
Create and combine your own custom filters
+++
{bdg-secondary}`custom`
{bdg-secondary}`flexible`
{bdg-secondary}`extensible`
:::

::::

## Usage

Quality assessment in Ray Curator uses pipeline-based workflows with the Task/Stage/Pipeline architecture. Data flows through stages as `DocumentBatch` tasks, enabling distributed processing and fault tolerance. The examples in each filtering approach section show how to create and execute filtering pipelines.

```{toctree}
:maxdepth: 4
:titlesonly:
:hidden:

Heuristic Filters <heuristic>
Classifier Filters <classifier>
Distributed Classification <distributed-classifier>
Custom Filters <custom>
```

## Best Practices

When filtering large datasets with Ray Curator, consider these performance tips:

1. **Stage ordering**: Place computationally inexpensive filters first in your pipeline
2. **Resource specification**: Configure stage resources based on computational requirements
3. **Task sizing**: Balance task size for optimal memory usage and parallel processing
4. **Pipeline composition**: Combine related operations in single stages when possible
5. **Distributed execution**: Use `XennaExecutor` with auto-scaling for TB-scale datasets

```python
# Example: Resource-aware pipeline
from ray_curator.stages.resources import Resources

# CPU-intensive filtering stage
pipeline.add_stage(ScoreFilter(
    filter_obj=ComplexFilter(),
    text_field="text"
).with_(resources=Resources(cpus=2.0)))

# GPU-accelerated classification stage  
pipeline.add_stage(ScoreFilter(
    filter_obj=MLClassifier(),
    text_field="text"
).with_(resources=Resources(gpu_memory_gb=8.0)))

# Execute with auto-scaling
executor = XennaExecutor(config={"auto_scaling": True, "max_workers": 10})
results = pipeline.run(executor)
```