---
description: "Filter text using rule-based metrics to identify and remove low-quality documents with configurable thresholds"
categories: ["how-to-guides"]
tags: ["heuristic-filtering", "rules", "metrics", "thresholds", "quality-control", "fast"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "how-to"
modality: "text-only"
---

(text-process-data-filter-heuristic)=
# Heuristic Filtering

Heuristic filtering uses simple, rule-based metrics to identify and filter out low-quality documents from your dataset. NVIDIA NeMo Curator provides a variety of pre-built heuristic filters that can be configured and combined to meet your specific needs.

## How It Works

Heuristic filters examine specific attributes of text documents and apply predefined thresholds to determine document quality. Unlike classifier-based filtering, heuristic filters don't require training data but rely on configurable thresholds and rules.

These filters assess quality using measurable document characteristics such as:
- Document length (word or character count)
- Punctuation ratios and patterns
- Repetitive content detection
- Language-specific patterns
- Text completeness and coherence

For details on filter structure and the filtering process, refer to {ref}`Data Processing Concepts <about-concepts-text-data-processing>`.

--- 

## Usage

::::{tab-set}

:::{tab-item} Python
```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.modules import ScoreFilter
from nemo_curator.stages.text.filters import (
    WordCountFilter,
    RepeatingTopNGramsFilter,
    PunctuationFilter
)

# Create pipeline
pipeline = Pipeline(name="heuristic_filtering")

# Load your dataset
reader = JsonlReader(
    file_paths="input_data/",
    fields=["text", "id"]
)
pipeline.add_stage(reader)

# Add filter stages
pipeline.add_stage(ScoreFilter(
    filter_obj=WordCountFilter(min_words=80),
    text_field="text",
    score_field="word_count"
))
pipeline.add_stage(ScoreFilter(
    filter_obj=PunctuationFilter(max_num_sentences_without_endmark_ratio=0.85),
    text_field="text"
))
pipeline.add_stage(ScoreFilter(
    filter_obj=RepeatingTopNGramsFilter(n=2, max_repeating_ngram_ratio=0.2),
    text_field="text"
))
pipeline.add_stage(ScoreFilter(
    filter_obj=RepeatingTopNGramsFilter(n=3, max_repeating_ngram_ratio=0.18),
    text_field="text"
))
pipeline.add_stage(ScoreFilter(
    filter_obj=RepeatingTopNGramsFilter(n=4, max_repeating_ngram_ratio=0.16),
    text_field="text"
))

# Add output stage
writer = JsonlWriter(path="high_quality_output/")
pipeline.add_stage(writer)

# Execute pipeline
results = pipeline.run()
```
:::

:::{tab-item} Configuration
```python
# Example configuration for common heuristic filters
from nemo_curator.stages.text.filters import (
    WordCountFilter,
    PunctuationFilter,
    RepeatingTopNGramsFilter,
    SymbolsToWordsFilter,
    CommonEnglishWordsFilter
)

# Define filter configurations
filters_config = [
    {
        "filter": WordCountFilter(min_words=50, max_words=10000),
        "description": "Filter by word count"
    },
    {
        "filter": PunctuationFilter(max_num_sentences_without_endmark_ratio=0.85),
        "description": "Filter by punctuation patterns"
    },
    {
        "filter": RepeatingTopNGramsFilter(n=3, max_repeating_ngram_ratio=0.18),
        "description": "Filter repetitive content"
    },
    {
        "filter": SymbolsToWordsFilter(max_symbol_to_word_ratio=0.1),
        "description": "Filter by symbol ratio"
    }
]

# Apply filters in pipeline
for config in filters_config:
    pipeline.add_stage(ScoreFilter(
        filter_obj=config["filter"],
        text_field="text"
    ))
```
:::

::::

## Available Filters

NeMo Curator includes more than 30 heuristic filters for assessing document quality. Below are the most commonly used filters with their parameters:

### Text Length Filters

| Filter | Description | Key Parameters | Default Values |
|--------|-------------|----------------|---------------|
| **WordCountFilter** | Filters by word count | `min_words`, `max_words` | min=50, max=100000 |
| **TokenCountFilter** | Filters by token count | `min_tokens`, `max_tokens` | min=0, max=∞ |
| **MeanWordLengthFilter** | Filters by average word length | `min_mean_word_length`, `max_mean_word_length` | min=3, max=10 |
| **LongWordFilter** | Filters by presence of extremely long words | `max_word_length` | 1000 |

### Repetition Detection Filters

| Filter | Description | Key Parameters | Default Values |
|--------|-------------|----------------|---------------|
| **RepeatedLinesFilter** | Detects repeated lines | `max_repeated_line_fraction` | 0.7 |
| **RepeatedParagraphsFilter** | Detects repeated paragraphs | `max_repeated_paragraphs_ratio` | 0.7 |
| **RepeatedLinesByCharFilter** | Detects repeated lines by character count | `max_repeated_lines_char_ratio` | 0.8 |
| **RepeatedParagraphsByCharFilter** | Detects repeated paragraphs by character count | `max_repeated_paragraphs_char_ratio` | 0.8 |
| **RepeatingTopNGramsFilter** | Detects excessive repetition of n-grams | `n`, `max_repeating_ngram_ratio` | n=2, ratio=0.2 |
| **RepeatingDuplicateNGramsFilter** | Detects duplicate n-grams | `n`, `max_repeating_duplicate_ngram_ratio` | n=2, ratio=0.2 |

### Character and Symbol Filters

| Filter | Description | Key Parameters | Default Values |
|--------|-------------|----------------|---------------|
| **NonAlphaNumericFilter** | Limits non-alphanumeric content | `max_non_alpha_numeric_to_text_ratio` | 0.25 |
| **SymbolsToWordsFilter** | Limits symbols in text | `max_symbol_to_word_ratio` | 0.1 |
| **NumbersFilter** | Limits numeric content | `max_number_to_text_ratio` | 0.15 |
| **UrlsFilter** | Limits URL content | `max_url_to_text_ratio` | 0.2 |
| **PunctuationFilter** | Limits sentences without proper punctuation | `max_num_sentences_without_endmark_ratio` | 0.85 |
| **WhiteSpaceFilter** | Limits excessive whitespace | `max_white_space_ratio` | 0.25 |

### Content-specific Filters

| Filter | Description | Key Parameters | Default Values |
|--------|-------------|----------------|---------------|
| **CommonEnglishWordsFilter** | Ensures text contains common words | `min_num_common_words` | 2 |
| **WordsWithoutAlphabetsFilter** | Limits words without alphabetic chars | `min_words_with_alphabets` | 0.8 |
| **BulletsFilter** | Limits bullet-point heavy content | `max_bullet_lines_ratio` | 0.9 |
| **BoilerPlateStringFilter** | Detects boilerplate text | `max_boilerplate_string_ratio`, `remove_if_at_top_or_bottom` | 0.4, True |
| **ParenthesesFilter** | Limits parentheses content | `max_parentheses_ratio` | 0.1 |

### Special Purpose Filters

| Filter | Description | Key Parameters | Default Values |
|--------|-------------|----------------|---------------|
| **PornographicUrlsFilter** | Detects URLs containing "porn" substring | None | N/A |
| **EllipsisFilter** | Limits excessive ellipses | `max_num_lines_ending_with_ellipsis_ratio` | 0.3 |
| **HistogramFilter** | Filters based on character distribution | `threshold` | 0.8 |
| **SubstringFilter** | Filters based on presence of specific substring in a position | `substring`, `position` | N/A (required) |

## Configuration

NeMo Curator pipelines can be configured using YAML files with [Hydra](https://hydra.cc/). The configuration uses `_target_` to specify class paths:

::::{tab-set}

:::{tab-item} Hydra Configuration
```yaml
# Hydra-based pipeline configuration
input_path: /path/to/input
output_path: /path/to/output
text_field: text

stages:
  - _target_: nemo_curator.stages.text.io.reader.JsonlReader
    file_paths: ${input_path}
    fields: null

  - _target_: nemo_curator.stages.text.modules.score_filter.ScoreFilter
    filter_obj:
      _target_: nemo_curator.stages.text.filters.heuristic_filter.WordCountFilter
      min_words: 50
      max_words: 100000
    text_field: ${text_field}
    score_field: word_count

  - _target_: nemo_curator.stages.text.modules.score_filter.ScoreFilter
    filter_obj:
      _target_: nemo_curator.stages.text.filters.heuristic_filter.PunctuationFilter
      max_num_sentences_without_endmark_ratio: 0.85
    text_field: ${text_field}
    score_field: null

  - _target_: nemo_curator.stages.text.io.writer.JsonlWriter
    path: ${output_path}
```

See `nemo_curator/config/text/` for complete pipeline examples.
:::

::::

For non-English texts, you may need to adjust the filter parameters based on the specific characteristics of your target language.

## Best Practices

When building filter chains, follow these best practices:

::::{tab-set}

:::{tab-item} Order for Efficiency
```python
# Efficient ordering - place fast filters first
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.modules import ScoreFilter
from nemo_curator.stages.text.filters import WordCountFilter, UrlsFilter, RepeatingTopNGramsFilter

pipeline = Pipeline(name="efficient_filtering")
# Fast filters first
pipeline.add_stage(ScoreFilter(filter_obj=WordCountFilter(min_words=50), text_field="text"))
# Medium complexity filters
pipeline.add_stage(ScoreFilter(filter_obj=UrlsFilter(), text_field="text"))
# Slow filters last
pipeline.add_stage(ScoreFilter(filter_obj=RepeatingTopNGramsFilter(), text_field="text"))
```
:::

:::{tab-item} Performance Tuning
See the [Performance Tuning](#performance-tuning) section below for executor configuration examples using Xenna or Ray backends.
:::

:::{tab-item} Precision vs. Recall
```python
# More permissive (higher recall)
lenient_filter = WordCountFilter(min_words=10, max_words=100000)

# More strict (higher precision)
strict_filter = WordCountFilter(min_words=100, max_words=10000)
```
:::

:::{tab-item} Language Considerations
```python
# Chinese text filter
from nemo_curator.stages.text.modules import ScoreFilter
from nemo_curator.stages.text.filters import SymbolsToWordsFilter

cn_filter = ScoreFilter(
    filter_obj=SymbolsToWordsFilter(max_symbol_to_word_ratio=0.15, lang="zh"),
    text_field="text"
)
```
:::

:::{tab-item} Multiple Filters
```python
# Comprehensive quality filter pipeline
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.modules import ScoreFilter
from nemo_curator.stages.text.filters import (
    WordCountFilter, 
    PunctuationFilter, 
    CommonEnglishWordsFilter, 
    RepeatingTopNGramsFilter
)

quality_pipeline = Pipeline(name="comprehensive_quality")

# Basic text quality
quality_pipeline.add_stage(ScoreFilter(
    filter_obj=WordCountFilter(min_words=50), text_field="text"
))
quality_pipeline.add_stage(ScoreFilter(
    filter_obj=PunctuationFilter(max_num_sentences_without_endmark_ratio=0.85), text_field="text"
))

# Content quality
quality_pipeline.add_stage(ScoreFilter(
    filter_obj=CommonEnglishWordsFilter(min_num_common_words=2), text_field="text"
))

# Repetition detection
quality_pipeline.add_stage(ScoreFilter(
    filter_obj=RepeatingTopNGramsFilter(n=3, max_repeating_ngram_ratio=0.18), text_field="text"
))
```
:::

::::

## Analyzing Filter Results

When tuning filter thresholds, analyze score distributions before applying filters. NeMo Curator provides two modules for this workflow:

- **`Score`**: Computes scores and adds them as columns without removing documents
- **`ScoreFilter`**: Computes scores, filters based on thresholds, and optionally retains scores in output

Use `Score` first to understand your data distribution, then apply `ScoreFilter` with tuned thresholds.

::::{tab-set}

:::{tab-item} Score Without Filtering
Use `Score` to add score columns to your data without removing any documents:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.modules import Score
from nemo_curator.stages.text.filters import WordCountFilter, RepeatingTopNGramsFilter

# Create scoring pipeline (no filtering)
pipeline = Pipeline(name="score_analysis")

# Load data
pipeline.add_stage(JsonlReader(file_paths="input_data/", fields=["text", "id"]))

# Add scores without filtering
pipeline.add_stage(Score(
    score_fn=WordCountFilter(min_words=80),
    text_field="text",
    score_field="word_count"
))
pipeline.add_stage(Score(
    score_fn=RepeatingTopNGramsFilter(n=3, max_repeating_ngram_ratio=0.18),
    text_field="text",
    score_field="ngram_ratio"
))

# Write scored data (all documents preserved)
pipeline.add_stage(JsonlWriter(path="scored_output/"))

pipeline.run()
```

Output files are written to the `scored_output/` directory with one file per input partition.
:::

:::{tab-item} Analyze Score Distribution
Load the scored output and analyze distributions to tune filter thresholds:

```python
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Load all scored output files
files = glob.glob("scored_output/*.jsonl")
scored_data = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)

# Analyze score distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Word count distribution
axes[0].hist(scored_data["word_count"], bins=50, edgecolor="black")
axes[0].axvline(x=80, color="red", linestyle="--", label="Threshold (80)")
axes[0].set_title("Word Count Distribution")
axes[0].set_xlabel("Word Count")
axes[0].legend()

# N-gram ratio distribution
axes[1].hist(scored_data["ngram_ratio"], bins=50, edgecolor="black")
axes[1].axvline(x=0.18, color="red", linestyle="--", label="Threshold (0.18)")
axes[1].set_title("3-gram Repetition Ratio")
axes[1].set_xlabel("Ratio")
axes[1].legend()

plt.tight_layout()
plt.savefig("score_distributions.png")

# Print statistics
print(f"Total documents: {len(scored_data)}")
print(f"Documents below word count threshold: {(scored_data['word_count'] < 80).sum()}")
print(f"Documents above ngram threshold: {(scored_data['ngram_ratio'] > 0.18).sum()}")
```

For large datasets, consider sampling or using Ray, Dask, or Polars for memory-efficient analysis.
:::

:::{tab-item} Apply Tuned Filters
After analyzing distributions, apply filters with your chosen thresholds:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import JsonlWriter
from nemo_curator.stages.text.modules import ScoreFilter
from nemo_curator.stages.text.filters import WordCountFilter, RepeatingTopNGramsFilter

pipeline = Pipeline(name="filtering_pipeline")
pipeline.add_stage(JsonlReader(file_paths="input_data/", fields=["text", "id"]))

# Filter with tuned thresholds (scores retained in output)
pipeline.add_stage(ScoreFilter(
    filter_obj=WordCountFilter(min_words=80),
    text_field="text",
    score_field="word_count"
))
pipeline.add_stage(ScoreFilter(
    filter_obj=RepeatingTopNGramsFilter(n=3, max_repeating_ngram_ratio=0.18),
    text_field="text",
    score_field="ngram_ratio"
))

pipeline.add_stage(JsonlWriter(path="filtered_output/"))

# Run with default XennaExecutor
pipeline.run()

# Or use Ray for distributed processing (see Performance Tuning section)
# from nemo_curator.backends.experimental.ray_data import RayDataExecutor
# pipeline.run(RayDataExecutor(ignore_head_node=True))
```
:::

::::

## Performance Tuning

For large datasets, consider these performance optimizations:

::::{tab-set}

:::{tab-item} XennaExecutor (Default)
`XennaExecutor` is the default executor, optimized for streaming workloads. You can customize its configuration or use the defaults:

```python
from nemo_curator.backends.xenna import XennaExecutor

# Custom configuration for streaming processing
executor = XennaExecutor(config={
    "execution_mode": "streaming",
    "cpu_allocation_percentage": 0.95,
    "logging_interval": 60
})
results = pipeline.run(executor)
```

If no executor is specified, `pipeline.run()` uses `XennaExecutor` with default settings.
:::

:::{tab-item} RayDataExecutor (Experimental)
`RayDataExecutor` provides distributed processing using Ray Data. It has shown performance improvements for filtering workloads compared to the default executor.

```python
from nemo_curator.backends.experimental.ray_data import RayDataExecutor

executor = RayDataExecutor(
    config={"ignore_failures": False},
    ignore_head_node=True  # Exclude head node from computation
)
results = pipeline.run(executor)
```
:::

:::{tab-item} Batch Size Optimization
```python
# Optimize pipeline stages for performance
from nemo_curator.stages.text.io.reader import JsonlReader

# Configure reader with optimal batch size
reader = JsonlReader(
    file_paths="large_dataset/*.jsonl",
    files_per_partition=4,  # Adjust based on file sizes
    fields=["text", "id"]
)
```
:::

::::

Remember that the goal of filtering is to improve the quality of your training data, not necessarily to remove as many documents as possible. Monitor your filtering results and adjust thresholds based on your specific data characteristics and downstream tasks.
