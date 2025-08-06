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

Each heuristic filter follows a consistent structure:

```python
class ExampleFilter(DocumentFilter):
    def __init__(self, parameter1=default1, parameter2=default2):
        super().__init__()
        self._param1 = parameter1
        self._param2 = parameter2
        self._name = "example_filter"
        
    def score_document(self, text):
        # Calculate and return a score between 0 and 1
        # Higher scores typically indicate lower quality
        score = compute_score(text)
        return score
        
    def keep_document(self, score):
        # Return True to keep the document, False to filter it out
        return score <= self._param1
```

The filtering process typically involves:
1. Calculating a quality score for each document
2. Applying a threshold to determine whether to keep or discard the document
3. Optionally storing the score as metadata for later analysis

--- 

## Usage

::::{tab-set}

:::{tab-item} Python
```python
import nemo_curator as nc
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import (
    WordCountFilter,
    RepeatingTopNGramsFilter,
    PunctuationFilter
)

# Load your dataset
dataset = DocumentDataset.read_json("input_data/*.jsonl")

# Create a filter chain using Sequential
filter_step = nc.Sequential([
    nc.ScoreFilter(
        WordCountFilter(min_words=80),
        text_field="text",
        score_field="word_count",
    ),
    nc.ScoreFilter(PunctuationFilter(max_num_sentences_without_endmark_ratio=0.85)),
    nc.ScoreFilter(RepeatingTopNGramsFilter(n=2, max_repeating_ngram_ratio=0.2)),
    nc.ScoreFilter(RepeatingTopNGramsFilter(n=3, max_repeating_ngram_ratio=0.18)),
    nc.ScoreFilter(RepeatingTopNGramsFilter(n=4, max_repeating_ngram_ratio=0.16)),
])

# Apply the filters to get the high-quality subset
high_quality_data = filter_step(dataset)

# Save the results
high_quality_data.to_json("high_quality_output/", write_to_filename=True)
```
:::

:::{tab-item} Command Line
```bash
filter_documents \
  --input-data-dir=/path/to/input/data \
  --filter-config-file=./config/heuristic_filter_en.yaml \
  --output-retained-document-dir=/path/to/output/high_quality \
  --output-removed-document-dir=/path/to/output/low_quality \
  --output-document-score-dir=/path/to/output/scores \
  --log-dir=/path/to/logs/heuristic_filter
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
# Efficient ordering
filter_chain = nc.Sequential([
    nc.ScoreFilter(WordCountFilter(min_words=50)),  # Fast
    nc.ScoreFilter(UrlsFilter()),                   # Medium
    nc.ScoreFilter(RepeatingTopNGramsFilter())      # Slow
])
```
:::

:::{tab-item} Batched Processing
```python
from nemo_curator.utils.decorators import batched

class MyCustomFilter(DocumentFilter):
    @batched
    def keep_document(self, scores):
        return scores <= self.threshold
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
cn_filter = nc.ScoreFilter(
    SymbolsToWordsFilter(max_symbol_to_word_ratio=0.15, lang="zh")
)
```
:::

:::{tab-item} Multiple Filters
```python
# Comprehensive quality filter
quality_chain = nc.Sequential([
    # Basic text quality
    nc.ScoreFilter(WordCountFilter(min_words=50)),
    nc.ScoreFilter(PunctuationFilter(max_num_sentences_without_endmark_ratio=0.85)),
    
    # Content quality
    nc.ScoreFilter(CommonEnglishWordsFilter(min_num_common_words=2)),
    
    # Repetition detection
    nc.ScoreFilter(RepeatingTopNGramsFilter(n=3, max_repeating_ngram_ratio=0.18))
])
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
# Process in chunks to reduce memory usage
for chunk in DocumentDataset.read_json_chunks("large_dataset/*.jsonl", chunk_size=10000):
    filtered_chunk = filter_step(chunk)
    filtered_chunk.to_json("output/", mode="append")
```
:::

:::{tab-item} Multi-process Filtering
```bash
# Use multiple processes with CLI
filter_documents --input-data-dir=input/ --num-proc=8 --filter-config-file=config.yaml --output-retained-document-dir=output/
```
:::

:::{tab-item} Custom Batch Sizes
```python
# Adjust batch size for specific filters
from nemo_curator.utils.decorators import batched

class CustomBatchFilter(DocumentFilter):
    @batched(batch_size=5000)  # Set custom batch size
    def keep_document(self, scores):
        return scores <= self.threshold
```
:::

::::

Remember that the goal of filtering is to improve the quality of your training data, not necessarily to remove as many documents as possible. Monitor your filtering results and adjust thresholds based on your specific data characteristics and downstream tasks.
