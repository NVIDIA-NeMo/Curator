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
    file_paths="input_data/*.jsonl",
    fields=["text", "id"]
)
pipeline.add_stage(reader)

# Add filter stages
pipeline.add_stage(ScoreFilter(
    score_fn=WordCountFilter(min_words=80),
    text_field="text",
    score_field="word_count"
))
pipeline.add_stage(ScoreFilter(
    score_fn=PunctuationFilter(max_num_sentences_without_endmark_ratio=0.85),
    text_field="text"
))
pipeline.add_stage(ScoreFilter(
    score_fn=RepeatingTopNGramsFilter(n=2, max_repeating_ngram_ratio=0.2),
    text_field="text"
))
pipeline.add_stage(ScoreFilter(
    score_fn=RepeatingTopNGramsFilter(n=3, max_repeating_ngram_ratio=0.18),
    text_field="text"
))
pipeline.add_stage(ScoreFilter(
    score_fn=RepeatingTopNGramsFilter(n=4, max_repeating_ngram_ratio=0.16),
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
        score_fn=config["filter"],
        text_field="text"
    ))
```
:::

::::

## Available Filters

NeMo Curator includes over 30 heuristic filters for assessing document quality. Below are the most commonly used filters with their parameters:

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
| **SubstringFilter** | Filters based on presence of specific substring in a position | `substring`, `position` | "", "any" |

## Configuration

::::{tab-set}

:::{tab-item} Example Configuration
```yaml
# Sample filter configuration (simplified)
filters:
  - name: ScoreFilter
    filter:
      name: WordCountFilter
      min_words: 50
      max_words: 100000
    text_field: text
    score_field: word_count

  - name: ScoreFilter
    filter:
      name: PunctuationFilter
      max_num_sentences_without_endmark_ratio: 0.85
    text_field: text
    score_field: punctuation_ratio

  - name: ScoreFilter
    filter:
      name: RepeatingTopNGramsFilter
      n: 2
      max_repeating_ngram_ratio: 0.18
    text_field: text
    score_field: ngram_repetition
```
:::

::::

The configuration file `config/heuristic_filter_en.yaml` contains a general-purpose set of heuristic filters that work well for English text. For non-English texts, you may need to adjust the filter parameters.

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
pipeline.add_stage(ScoreFilter(score_fn=WordCountFilter(min_words=50), text_field="text"))
# Medium complexity filters
pipeline.add_stage(ScoreFilter(score_fn=UrlsFilter(), text_field="text"))
# Slow filters last
pipeline.add_stage(ScoreFilter(score_fn=RepeatingTopNGramsFilter(), text_field="text"))
```
:::

:::{tab-item} Performance Tuning
```python
# Optimize filter performance with proper configuration

# Configure executor for better performance
executor_config = {
    "execution_mode": "streaming",
    "cpu_allocation_percentage": 0.95,
    "logging_interval": 30
}

# Use custom executor configuration when needed
executor = XennaExecutor(config=executor_config)
results = pipeline.run(executor)

# Or use default configuration
# results = pipeline.run()
```
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
    score_fn=SymbolsToWordsFilter(max_symbol_to_word_ratio=0.15, lang="zh"),
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
    score_fn=WordCountFilter(min_words=50), text_field="text"
))
quality_pipeline.add_stage(ScoreFilter(
    score_fn=PunctuationFilter(max_num_sentences_without_endmark_ratio=0.85), text_field="text"
))

# Content quality
quality_pipeline.add_stage(ScoreFilter(
    score_fn=CommonEnglishWordsFilter(min_num_common_words=2), text_field="text"
))

# Repetition detection
quality_pipeline.add_stage(ScoreFilter(
    score_fn=RepeatingTopNGramsFilter(n=3, max_repeating_ngram_ratio=0.18), text_field="text"
))
```
:::

::::

## Analyzing Filter Results

When working with non-English data or tuning your filtering pipeline, it's valuable to examine which filters are removing documents:

::::{tab-set}

:::{tab-item} Filter Analysis
```python
import pandas as pd

# Load scores from filter run
scores = pd.read_json("output/scores/scores.jsonl", lines=True)

# Analyze rejection reasons
rejection_counts = scores[scores["rejected"] == True].groupby("rejected_by").size()
print(f"Documents rejected by filter:\n{rejection_counts}")

# Analyze score distributions
import matplotlib.pyplot as plt
scores.hist(column="word_count", bins=50)
plt.title("Word Count Distribution")
plt.savefig("word_count_hist.png")
```
:::

::::

## Performance Tuning

For large datasets, consider these performance optimizations:

::::{tab-set}

:::{tab-item} Memory Efficient Processing
```python
# Process large datasets efficiently using pipeline streaming

# Configure for streaming processing
executor_config = {
    "execution_mode": "streaming",
    "cpu_allocation_percentage": 0.8,
    "logging_interval": 60
}

# Use custom configuration for large datasets
executor = XennaExecutor(config=executor_config)
results = pipeline.run(executor)

# Default configuration works for most cases
# results = pipeline.run()
```
:::

:::{tab-item} Distributed Processing
```python
# Scale processing across multiple workers

# Configure for distributed processing
executor_config = {
    "execution_mode": "streaming",
    "cpu_allocation_percentage": 0.95,
    "max_workers_per_stage": 8
}

# Use custom configuration for distributed processing
executor = XennaExecutor(config=executor_config)
results = pipeline.run(executor)

# Default configuration uses single worker
# results = pipeline.run()
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
