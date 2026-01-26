---
name: filter
description: |
  Apply heuristic filters to text datasets for quality control.
  Use when the user wants to filter text data, remove low-quality content,
  apply quality heuristics, or needs the complete filter catalog.
  Includes 33+ filters for various quality dimensions.
license: Apache-2.0
metadata:
  author: nvidia
  version: "1.0"
  modality: text
disable-model-invocation: true
---

# Text Filtering

Apply heuristic filters to text datasets for quality control.

## When to Use

- Cleaning web-scraped text data
- Removing low-quality documents
- Applying rule-based quality heuristics
- Pre-filtering before ML classification

## Quick Start

```bash
# List available filters
python scripts/list_filters.py

# List filters with descriptions
python scripts/list_filters.py --verbose

# Search for specific filters
python scripts/list_filters.py --search "word"
```

## Filter Categories

### Length Filters

| Filter | Parameter | Default | Description |
|--------|-----------|---------|-------------|
| `WordCountFilter` | `min_words`, `max_words` | 50, 100000 | Filter by word count |
| `TokenCountFilter` | `min_tokens`, `max_tokens` | varies | Filter by token count |
| `MeanWordLengthFilter` | `min_mean`, `max_mean` | 3, 10 | Filter by average word length |
| `LongWordFilter` | `max_long_word_ratio` | 0.1 | Filter by long word ratio |

### Character Composition Filters

| Filter | Parameter | Default | Description |
|--------|-----------|---------|-------------|
| `NonAlphaNumericFilter` | `max_ratio` | 0.25 | Non-alphanumeric ratio |
| `SymbolsToWordsFilter` | `max_ratio` | 0.1 | Symbol to word ratio |
| `NumbersFilter` | `max_ratio` | 0.15 | Number ratio |
| `PunctuationFilter` | `max_ratio` | 0.1 | Punctuation ratio |
| `WhiteSpaceFilter` | `max_ratio` | 0.25 | Whitespace ratio |
| `ParenthesesFilter` | `max_ratio` | 0.1 | Parentheses ratio |

### Content Filters

| Filter | Parameter | Default | Description |
|--------|-----------|---------|-------------|
| `UrlsFilter` | `max_ratio` | 0.2 | URL ratio |
| `BulletsFilter` | `max_ratio` | 0.9 | Bullet point ratio |
| `EllipsisFilter` | `max_ratio` | 0.3 | Ellipsis ratio |
| `PornographicUrlsFilter` | - | - | Adult URL detection |
| `SubstringFilter` | `substrings` | - | Substring presence |

### Repetition Filters

| Filter | Parameter | Default | Description |
|--------|-----------|---------|-------------|
| `RepeatedLinesFilter` | `max_fraction` | 0.3 | Duplicate line ratio |
| `RepeatedParagraphsFilter` | `max_ratio` | 0.3 | Duplicate paragraph ratio |
| `RepeatedLinesByCharFilter` | `max_ratio` | 0.2 | Duplicate lines (by char) |
| `RepeatedParagraphsByCharFilter` | `max_ratio` | 0.2 | Duplicate paragraphs (by char) |
| `RepeatingTopNGramsFilter` | varies | varies | Top n-gram repetition |
| `RepeatingDuplicateNGramsFilter` | varies | varies | Duplicate n-gram ratio |

### Language Filters

| Filter | Parameter | Default | Description |
|--------|-----------|---------|-------------|
| `CommonEnglishWordsFilter` | `min_num` | 2 | Common words presence |
| `WordsWithoutAlphabetsFilter` | `max_ratio` | 0.5 | Words without letters |
| `FastTextLangId` | `language` | - | Language identification |

### Code Filters

| Filter | Parameter | Default | Description |
|--------|-----------|---------|-------------|
| `AlphaFilter` | `min_ratio` | - | Alphabetic character ratio |
| `GeneralCommentToCodeFilter` | `max_ratio` | - | Comment to code ratio |
| `HTMLBoilerplateFilter` | - | - | HTML boilerplate detection |
| `NumberOfLinesOfCodeFilter` | `min`, `max` | - | Lines of code count |
| `PythonCommentToCodeFilter` | `max_ratio` | - | Python comment ratio |
| `TokenizerFertilityFilter` | `max_fertility` | - | Tokenizer efficiency |
| `XMLHeaderFilter` | - | - | XML header detection |

## Filter Presets

### Minimal (Quick Pass)

```python
filters = [
    WordCountFilter(min_words=50, max_words=100000),
    NonAlphaNumericFilter(max_non_alpha_numeric_to_text_ratio=0.25),
]
```

### Standard (Recommended)

```python
filters = [
    WordCountFilter(min_words=50, max_words=100000),
    NonAlphaNumericFilter(max_non_alpha_numeric_to_text_ratio=0.25),
    SymbolsToWordsFilter(max_symbol_to_word_ratio=0.1),
    UrlsFilter(max_url_to_text_ratio=0.2),
    MeanWordLengthFilter(min_mean_word_length=3, max_mean_word_length=10),
    RepeatedLinesFilter(max_repeated_line_fraction=0.3),
    RepeatedParagraphsFilter(max_repeated_paragraphs_ratio=0.3),
    PunctuationFilter(max_punctuation_ratio=0.1),
    CommonEnglishWordsFilter(min_num_common_words=2),
    PornographicUrlsFilter(),
]
```

### Full (Thorough Cleaning)

All 25 heuristic filters with default parameters.

### Code (Source Code)

```python
filters = [
    AlphaFilter(min_alpha_ratio=0.25),
    GeneralCommentToCodeFilter(max_comment_to_code_ratio=0.8),
    NumberOfLinesOfCodeFilter(min_lines=10, max_lines=10000),
    TokenizerFertilityFilter(max_fertility=5.0),
]
```

## Usage in Pipeline

### With ScoreFilter Module

```yaml
stages:
  - _target_: nemo_curator.stages.text.modules.ScoreFilter
    filter_obj:
      _target_: nemo_curator.stages.text.filters.WordCountFilter
      min_words: 50
      max_words: 100000
    text_field: "text"
```

### Multiple Filters

```yaml
stages:
  # Filter 1
  - _target_: nemo_curator.stages.text.modules.ScoreFilter
    filter_obj:
      _target_: nemo_curator.stages.text.filters.WordCountFilter
      min_words: 50
    text_field: "text"

  # Filter 2
  - _target_: nemo_curator.stages.text.modules.ScoreFilter
    filter_obj:
      _target_: nemo_curator.stages.text.filters.NonAlphaNumericFilter
      max_non_alpha_numeric_to_text_ratio: 0.25
    text_field: "text"

  # ... more filters
```

## Tuning Guidelines

### Dataset-Specific Adjustments

| Dataset Type | Recommended Adjustments |
|--------------|-------------------------|
| Web crawl | Use full filter set, strict thresholds |
| News articles | Relax URL filter, tighten repetition |
| Scientific papers | Relax number ratio, tighten word count |
| Social media | Reduce word count min, relax punctuation |
| Code | Use code-specific filters |

### Measuring Filter Impact

Run filters sequentially and track:

```python
original_count = len(df)
for filter_name, filtered_df in apply_filters_sequentially(df, filters):
    removed = original_count - len(filtered_df)
    print(f"{filter_name}: removed {removed} ({removed/original_count:.1%})")
    original_count = len(filtered_df)
```

Typical removal rates:
- WordCountFilter: 10-20%
- NonAlphaNumericFilter: 5-15%
- RepeatedLinesFilter: 5-10%
- Total (standard preset): 30-50%

## Related Skills

- `/curate` - Full curation workflow
- `/classify` - ML classification (use after filtering)
- `/stages` - All available stages
