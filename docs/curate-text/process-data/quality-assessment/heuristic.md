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

Heuristic filtering uses simple, rule-based metrics to identify and filter out low-quality documents from your dataset. Ray Curator provides a variety of pre-built heuristic filters that you can configure and combine to meet your specific needs.

## How It Works

Heuristic filters examine specific attributes of text documents and apply predefined thresholds to determine document quality. Unlike classifier-based filtering, heuristic filters don't require training data but rely on configurable thresholds and rules.

These filters assess quality using measurable document characteristics such as:

- Document length (word or character count)
- Punctuation ratios and patterns
- Repetitive content detection
- Language-specific patterns
- Text completeness and coherence

Each heuristic filter follows a consistent structure:

```python
from ray_curator.stages.filters import DocumentFilter

class ExampleFilter(DocumentFilter):
    def __init__(self, parameter1=default1, parameter2=default2):
        super().__init__()
        self._param1 = parameter1
        self._param2 = parameter2
        self._name = "example_filter"
        
    def score_document(self, text: str) -> float:
        # Calculate and return a score between 0 and 1
        # Higher scores typically indicate lower quality
        score = compute_score(text)
        return score
        
    def keep_document(self, score: float) -> bool:
        # Return True to keep the document, False to filter it out
        return score <= self._param1
```

The filtering process typically involves:

1. Calculating a quality score for each document
2. Applying a threshold to determine whether to keep or discard the document
3. Optionally storing the score as metadata for later analysis

---

## Usage

```python
from ray_curator.pipeline import Pipeline
from ray_curator.stages.io.reader import JsonlReader
from ray_curator.stages.io.writer import JsonlWriter
from ray_curator.stages.modules import ScoreFilter
from ray_curator.stages.filters import (
    WordCountFilter,
    RepeatingTopNGramsFilter,
    PunctuationFilter
)
from ray_curator.backends.experimental.ray_data import RayDataExecutor

# Create pipeline
pipeline = Pipeline(
    name="heuristic_filtering",
    description="Filter documents using heuristic quality metrics"
)

# Add data reading stage
pipeline.add_stage(JsonlReader(
    file_paths="input_data/*.jsonl",
    files_per_partition=10
))

# Add filtering stages
pipeline.add_stage(ScoreFilter(
    filter_obj=WordCountFilter(min_words=80),
    text_field="text",
    score_field="word_count_score"
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

# Add output stage
pipeline.add_stage(JsonlWriter(
    output_dir="high_quality_output/"
))

# Execute pipeline
executor = RayDataExecutor()
results = pipeline.run(executor)
```

## Available Filters

Ray Curator includes over 30 heuristic filters for assessing document quality. Below are the most commonly used filters with their parameters:

### Text Length Filters

| Filter | Description | Key Parameters | Default Values |
|--------|-------------|----------------|---------------|
| **WordCountFilter** | Filters by word count | `min_words`, `max_words` | min=50, max=100000 |
| **TokenCountFilter** | Filters by token count | `min_tokens`, `max_tokens` | min=0, max=∞ |
| **MeanWordLengthFilter** | Filters by average word length | `min_mean_word_length`, `max_mean_word_length` | min=3, max=10 |
| **LongWordFilter** | Filters by presence of long words | `max_word_length` | 1000 |

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
| **PornographicUrlsFilter** | Detects URLs containing "porn" text | None | N/A |
| **EllipsisFilter** | Limits excessive ellipses | `max_num_lines_ending_with_ellipsis_ratio` | 0.3 |
| **HistogramFilter** | Filters based on character distribution | `threshold` | 0.8 |
| **SubstringFilter** | Filters based on presence of specific text in a position | `substring`, `position` | "", "any" |

## Advanced Usage Patterns

### Scoring Without Filtering

Use the `Score` stage to add quality metrics without filtering documents:

```python
from ray_curator.stages.modules import Score

# Add quality scores without filtering
pipeline.add_stage(Score(
    score_fn=WordCountFilter(min_words=50),
    score_field="word_count_score",
    text_field="text"
))

pipeline.add_stage(Score(
    score_fn=PunctuationFilter(),
    score_field="punctuation_score",
    text_field="text"
))
```

### Filtering on Pre-computed Scores

Use the `Filter` stage to filter based on existing metadata:

```python
from ray_curator.stages.modules import Filter

# Filter based on previously computed scores
pipeline.add_stage(Filter(
    filter_fn=lambda x: x <= 0.1,  # Custom filtering function
    filter_field="word_count_score"
))
```

### Language-Specific Filtering

```python
# Chinese text filter
pipeline.add_stage(ScoreFilter(
    filter_obj=SymbolsToWordsFilter(max_symbol_to_word_ratio=0.15, lang="zh"),
    text_field="text"
))
```

## Pipeline Execution Options

### Using Ray Data Executor (Experimental)

```python
from ray_curator.backends.experimental.ray_data import RayDataExecutor

executor = RayDataExecutor()
results = pipeline.run(executor)
```

### Using Alternative Executor

```python
from ray_curator.backends.xenna import XennaExecutor

executor = XennaExecutor()
results = pipeline.run(executor)
```

## Best Practices

When building filter pipelines, follow these best practices:

::::{tab-set}

:::{tab-item} Efficient Stage Ordering

```python
# Order stages from fastest to slowest for efficiency
pipeline.add_stage(ScoreFilter(
    filter_obj=WordCountFilter(min_words=50)  # Fast - simple counting
))

pipeline.add_stage(ScoreFilter(
    filter_obj=UrlsFilter()  # Medium - regex matching
))

pipeline.add_stage(ScoreFilter(
    filter_obj=RepeatingTopNGramsFilter()  # Slow - n-gram computation
))
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

:::{tab-item} Score Preservation

```python
# Keep scores for analysis while filtering
pipeline.add_stage(ScoreFilter(
    filter_obj=WordCountFilter(min_words=50),
    text_field="text",
    score_field="word_count_score"  # Preserve score
))
```

:::

:::{tab-item} Comprehensive Quality Pipeline

```python
# Multi-stage quality assessment
pipeline.add_stage(JsonlReader(file_paths="data/*.jsonl"))

# Basic text quality
pipeline.add_stage(ScoreFilter(
    filter_obj=WordCountFilter(min_words=50),
    text_field="text"
))

pipeline.add_stage(ScoreFilter(
    filter_obj=PunctuationFilter(max_num_sentences_without_endmark_ratio=0.85),
    text_field="text"
))

# Content quality
pipeline.add_stage(ScoreFilter(
    filter_obj=CommonEnglishWordsFilter(min_num_common_words=2),
    text_field="text"
))

# Repetition detection
pipeline.add_stage(ScoreFilter(
    filter_obj=RepeatingTopNGramsFilter(n=3, max_repeating_ngram_ratio=0.18),
    text_field="text"
))

pipeline.add_stage(JsonlWriter(output_dir="filtered_output/"))
```

:::

::::

## Performance Considerations

For large datasets, consider these optimizations:

::::{tab-set}

:::{tab-item} File Partitioning

```python
# Control how files are partitioned for parallel processing
pipeline.add_stage(JsonlReader(
    file_paths="large_dataset/*.jsonl",
    files_per_partition=5,  # Fewer files per partition for larger datasets
    blocksize="128MB"       # Or use blocksize-based partitioning
))
```

:::

:::{tab-item} Resource Management

```python
from ray_curator.stages.resources import Resources

# Configure resource requirements for compute-intensive filters
class CustomFilter(ScoreFilter):
    _resources = Resources(cpus=2.0)  # Request more CPU cores
    
    def __init__(self, filter_obj):
        super().__init__(filter_obj=filter_obj)
```

:::

::::

Remember that the goal of filtering is to improve the quality of your training data, not necessarily to remove as much content as possible. Track your filtering results and adjust thresholds based on your specific data characteristics and downstream tasks.

## Evidence and Sources

**Source**: `ray-curator/ray_curator/stages/filters/heuristic_filter.py:47-835`
**Evidence**: All heuristic filters extend the DocumentFilter abstract base class with score_document() and keep_document() methods

**Source**: `ray-curator/ray_curator/stages/modules/score_filter.py:183-279`
**Evidence**: ScoreFilter stage combines scoring and filtering, accepting DocumentFilter objects and text/score field configuration

**Source**: `ray-curator/ray_curator/stages/filters/__init__.py:15-53`
**Evidence**: Complete list of available heuristic filters exported from the filters module

**Source**: `ray-curator/ray_curator/stages/io/reader/jsonl.py:159-218`
**Evidence**: JsonlReader composite stage for reading JSONL files with file partitioning support

**Source**: `ray-curator/ray_curator/pipeline/pipeline.py:12-180`
**Evidence**: Pipeline class for composing and executing processing stages
