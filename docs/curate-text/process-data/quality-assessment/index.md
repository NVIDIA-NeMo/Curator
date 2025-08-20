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

NeMo Curator's filtering framework is built around several key components:

::::{tab-set}

:::{tab-item} ScoreFilter

The `ScoreFilter` is at the center of filtering in NeMo Curator. It applies a filter to a document and optionally saves the score as metadata:

```python
import nemo_curator as nc
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.file_utils import get_all_files_paths_under
from nemo_curator.filters import WordCountFilter

# Load dataset
files = get_all_files_paths_under("books_dataset/", keep_extensions="jsonl")
books = DocumentDataset.read_json(files, add_filename=True)

# Create and apply filter
filter_step = nc.ScoreFilter(
    WordCountFilter(min_words=80),
    text_field="text",
    score_field="word_count",
)

# Get filtered dataset
long_books = filter_step(books)

# Save filtered dataset
long_books.to_json("long_books/", write_to_filename=True)
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
scoring_step = nc.Score(
    WordCountFilter().score_document,  # Use just the scoring part
    text_field="text",
    score_field="word_count"
)
scored_dataset = scoring_step(dataset)
```

- `Filter`: A module that filters based on pre-computed metadata
  - Takes a filter function that evaluates metadata and returns True/False
  - Only uses existing metadata fields (doesn't compute new scores)
  - Efficient for filtering on pre-computed metrics
  
```python
# Example: Filter using pre-computed scores
filter_step = nc.Filter(
    lambda score: score >= 100,  # Keep documents with score >= 100
    filter_field="word_count"
)
filtered_dataset = filter_step(scored_dataset)
```

You can combine these modules in pipelines:

```python
pipeline = nc.Sequential([
    nc.Score(word_counter, score_field="word_count"),
    nc.Score(symbol_counter, score_field="symbol_ratio"),
    nc.Filter(lambda x: x >= 100, filter_field="word_count"),
    nc.Filter(lambda x: x <= 0.3, filter_field="symbol_ratio")
])
```

:::

:::{tab-item} Batched Filtering

For improved performance, NeMo Curator supports batch processing using the `@batched` decorator:

```python
from nemo_curator.utils.decorators import batched
import pandas as pd

class BatchedFilter(DocumentFilter):
    @batched
    def keep_document(self, scores: pd.Series):
        # Process multiple documents in one operation
        return scores > 10
```

The batched processing can significantly improve performance on large datasets by:
- Reducing function call overhead
- Enabling vectorized operations
- Optimizing memory usage

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

NeMo Curator provides a CLI tool for document filtering that becomes available after installing the package:

```bash
filter_documents \
  --input-data-dir=/path/to/input/data \
  --filter-config-file=./config/heuristic_filter_en.yaml \
  --output-retained-document-dir=/path/to/output/high_quality \
  --output-removed-document-dir=/path/to/output/low_quality \
  --output-document-score-dir=/path/to/output/scores \
  --num-workers=4
```

For distributed processing with multiple workers:

```bash
filter_documents \
  --input-data-dir=/path/to/input/data \
  --filter-config-file=./config/heuristic_filter_en.yaml \
  --output-retained-document-dir=/path/to/output/high_quality \
  --num-workers=8 \
  --device=gpu \
  --log-dir=./logs
```

### CLI Parameters

| Parameter | Description | Required |
|-----------|-------------|----------|
| `--input-data-dir` | Directory containing input JSONL files | Yes |
| `--filter-config-file` | YAML configuration for the filter pipeline | Yes |
| `--output-retained-document-dir` | Directory for documents passing filters | Yes |
| `--output-removed-document-dir` | Directory for rejected documents | No |
| `--output-document-score-dir` | Directory for storing score metadata | No |
| `--log-dir` | Directory for storing logs | No |
| `--num-workers` | Number of Dask workers for distributed processing | No |
| `--scheduler-address` | Address of Dask scheduler for distributed processing | No |
| `--device` | Processing device: `cpu` or `gpu` (default: `cpu`) | No |
| `--input-file-type` | Input file format: `jsonl`, `parquet`, etc. (default: `jsonl`) | No |
| `--output-file-type` | Output file format: `jsonl`, `parquet`, etc. (default: `jsonl`) | No |

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
5. **Distributed processing**: For TB-scale datasets, use distributed filtering with Dask workers (`--num-workers`) or connect to an existing Dask cluster (`--scheduler-address`) 