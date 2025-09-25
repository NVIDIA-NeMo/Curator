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

Score and remove low-quality content using heuristics and ML classifiers to prepare your data for model training using NeMo Curator's tools and utilities.

Large datasets often contain many documents considered to be "low quality." In this context, "low quality" data simply means data we don't want a downstream model to learn from, and "high quality" data is data that we do want a downstream model to learn from. The metrics that define quality can vary widely.

## How It Works

NeMo Curator's filtering framework is built around several key components that work within the {ref}`data processing architecture <about-concepts-text-data-processing>`:

::::{tab-set}

:::{tab-item} ScoreFilter

The `ScoreFilter` is at the center of filtering in NeMo Curator. It applies a filter to a document and optionally saves the score as metadata:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.modules import ScoreFilter
from nemo_curator.stages.text.filters import WordCountFilter

# Create pipeline
pipeline = Pipeline(name="quality_filtering")

# Load dataset
reader = JsonlReader(
    file_paths="books_dataset/*.jsonl",
    fields=["text", "id"]
)
pipeline.add_stage(reader)

# Create and apply filter
filter_stage = ScoreFilter(
    score_fn=WordCountFilter(min_words=80),
    text_field="text",
    score_field="word_count",
)
pipeline.add_stage(filter_stage)

# Save filtered dataset
writer = JsonlWriter(path="long_books/")
pipeline.add_stage(writer)

# Execute pipeline (uses XennaExecutor by default)
results = pipeline.run()
```

```{note}
**Default Executor**: When you call `pipeline.run()` without specifying an executor, NeMo Curator automatically uses `XennaExecutor()` as the default. You can optionally specify a different executor by passing it as a parameter: `pipeline.run(executor=my_executor)`.
```

The filter object implements two key methods:

- `score_document`: Computes a quality score for a document
- `keep_document`: Determines if a document should be kept based on its score

:::

:::{tab-item} Filter and Score Modules

For more specific use cases, NeMo Curator provides two specialized modules:

- `Score`: A module that only adds metadata scores to records without filtering
  - Takes a scoring function that evaluates text and returns a score
  - Adds the score to a specified metadata field
  - Useful for analysis or multi-stage filtering pipelines
  
```python
# Example: Score documents without filtering
from nemo_curator.stages.text.modules import Score

scoring_step = Score(
    WordCountFilter().score_document,  # Use just the scoring part
    text_field="text",
    score_field="word_count"
)
scored_dataset = scoring_step.process(dataset)
```

- `Filter`: A module that filters based on pre-computed metadata
  - Takes a filter function that evaluates metadata and returns True/False
  - Only uses existing metadata fields (doesn't compute new scores)
  - Efficient for filtering on pre-computed metrics
  
```python
# Example: Filter using pre-computed scores
from nemo_curator.stages.text.modules import Filter

filter_step = Filter(
    lambda score: score >= 100,  # Keep documents with score >= 100
    filter_field="word_count"
)
filtered_dataset = filter_step.process(scored_dataset)
```

You can combine these modules in pipelines:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.modules import Score, Filter

pipeline = Pipeline(name="multi_stage_filtering")
pipeline.add_stage(Score(word_counter, score_field="word_count"))
pipeline.add_stage(Score(symbol_counter, score_field="symbol_ratio"))
pipeline.add_stage(Filter(lambda x: x >= 100, filter_field="word_count"))
pipeline.add_stage(Filter(lambda x: x <= 0.3, filter_field="symbol_ratio"))
```

:::

:::{tab-item} Performance Optimization

NeMo Curator's filtering framework is optimized for performance through:

```python
# Filters automatically use vectorized operations when possible
class OptimizedFilter(DocumentFilter):
    def score_document(self, text: str) -> float:
        # Individual document scoring
        return len(text.split())
    
    def keep_document(self, score: float) -> bool:
        # Individual document filtering decision
        return score >= 10
```

The framework provides built-in performance optimizations:

- Vectorized pandas operations for batch processing
- Efficient memory usage patterns
- Optimized I/O operations
- Distributed processing support

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
Implement and combine your own custom filters
+++
{bdg-secondary}`custom`
{bdg-secondary}`flexible`
{bdg-secondary}`extensible`
:::

::::

## Usage

NeMo Curator provides programmatic interfaces for document filtering through the Pipeline framework:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.modules import ScoreFilter
from nemo_curator.stages.text.filters import WordCountFilter

# Create and configure pipeline
pipeline = Pipeline(name="document_filtering")

# Add data loading
reader = JsonlReader(
    file_paths="/path/to/input/data/*.jsonl",
    fields=["text", "id"]
)
pipeline.add_stage(reader)

# Add filtering stage
filter_stage = ScoreFilter(
    score_fn=WordCountFilter(min_words=80),
    text_field="text",
    score_field="word_count"
)
pipeline.add_stage(filter_stage)

# Add output stage
writer = JsonlWriter(path="/path/to/output/filtered/")
pipeline.add_stage(writer)

# Execute pipeline (uses XennaExecutor by default)
results = pipeline.run()
```

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

When filtering large datasets, consider these performance tips:

1. **Order matters**: Place computationally inexpensive filters early in your pipeline
2. **Batch size tuning**: Adjust batch sizes based on your hardware capabilities
3. **Use vectorization**: Implement batched methods for compute-intensive filters
4. **Disk I/O**: Consider compression and chunking strategies for large datasets
5. **Distributed processing**: For TB-scale datasets, use distributed filtering with the XennaExecutor