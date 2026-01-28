---
name: filter
description: |
  Apply heuristic filters to text datasets for quality control.
  Use when the user wants to filter text data, remove low-quality content,
  apply quality heuristics, or needs the complete filter catalog.
  Includes 35+ filters for various quality dimensions.
license: Apache-2.0
metadata:
  author: nvidia
  version: "1.1"
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
| `TokenCountFilter` | `min_tokens`, `max_tokens` | 0, inf | Filter by token count |
| `MeanWordLengthFilter` | `min_mean_word_length`, `max_mean_word_length` | 3, 10 | Filter by average word length |
| `LongWordFilter` | `max_word_length` | 1000 | Filter documents with words exceeding max length |

### Character Composition Filters

| Filter | Parameter | Default | Description |
|--------|-----------|---------|-------------|
| `NonAlphaNumericFilter` | `max_non_alpha_numeric_to_text_ratio` | 0.25 | Non-alphanumeric ratio |
| `SymbolsToWordsFilter` | `max_symbol_to_word_ratio` | 0.1 | Symbol to word ratio |
| `NumbersFilter` | `max_number_to_text_ratio` | 0.15 | Number ratio |
| `PunctuationFilter` | `max_num_sentences_without_endmark_ratio` | 0.85 | Sentences without punctuation |
| `WhiteSpaceFilter` | `max_white_space_ratio` | 0.25 | Whitespace ratio |
| `ParenthesesFilter` | `max_parentheses_ratio` | 0.1 | Parentheses ratio |

### Content Filters

| Filter | Parameter | Default | Description |
|--------|-----------|---------|-------------|
| `UrlsFilter` | `max_url_to_text_ratio` | 0.2 | URL ratio |
| `BulletsFilter` | `max_bullet_lines_ratio` | 0.9 | Bullet point ratio |
| `EllipsisFilter` | `max_num_lines_ending_with_ellipsis_ratio` | 0.3 | Ellipsis ratio |
| `PornographicUrlsFilter` | - | - | Adult URL detection |
| `SubstringFilter` | `substring`, `position` | - | Substring presence (prefix/suffix/any) |
| `BoilerPlateStringFilter` | `max_boilerplate_string_ratio` | 0.4 | Boilerplate content ratio |
| `HistogramFilter` | `lang`, `threshold` | "en", 0.8 | Character histogram matching |

### Repetition Filters

| Filter | Parameter | Default | Description |
|--------|-----------|---------|-------------|
| `RepeatedLinesFilter` | `max_repeated_line_fraction` | 0.7 | Keep if unique/total >= threshold |
| `RepeatedParagraphsFilter` | `max_repeated_paragraphs_ratio` | 0.7 | Keep if unique/total >= threshold |
| `RepeatedLinesByCharFilter` | `max_repeated_lines_char_ratio` | 0.8 | Unique line chars ratio |
| `RepeatedParagraphsByCharFilter` | `max_repeated_paragraphs_char_ratio` | 0.8 | Unique paragraph chars ratio |
| `RepeatingTopNGramsFilter` | `n`, `max_repeating_ngram_ratio` | 2, 0.2 | Top n-gram repetition |
| `RepeatingDuplicateNGramsFilter` | `n`, `max_repeating_duplicate_ngram_ratio` | 2, 0.2 | Duplicate n-gram ratio |

### Language Filters

| Filter | Parameter | Default | Description |
|--------|-----------|---------|-------------|
| `CommonEnglishWordsFilter` | `min_num_common_words` | 2 | Common words presence |
| `WordsWithoutAlphabetsFilter` | `min_words_with_alphabets` | 0.8 | Min ratio of words with letters |
| `FastTextLangId` | `model_path`, `min_langid_score` | -, 0.3 | Language identification |
| `FastTextQualityFilter` | `model_path`, `label`, `alpha` | -, "__label__hq", 3 | FastText quality score |

### Code Filters

| Filter | Parameter | Default | Description |
|--------|-----------|---------|-------------|
| `AlphaFilter` | `min_alpha_ratio` | 0.25 | Alphabetic character ratio |
| `GeneralCommentToCodeFilter` | `language`, `min_comment_to_code_ratio`, `max_comment_to_code_ratio` | -, 0.01, 0.85 | Comment to code ratio |
| `HTMLBoilerplateFilter` | `min_lang_content_ratio`, `min_lang_content_num_chars` | 0.2, 100 | HTML boilerplate detection |
| `NumberOfLinesOfCodeFilter` | `min_lines`, `max_lines` | 10, 20000 | Lines of code count |
| `PythonCommentToCodeFilter` | `min_comment_to_code_ratio`, `max_comment_to_code_ratio` | 0.01, 0.85 | Python comment ratio |
| `TokenizerFertilityFilter` | `path_to_tokenizer`, `min_char_to_token_ratio` | -, 2.5 | Tokenizer efficiency |
| `XMLHeaderFilter` | `char_prefix_search_length` | 100 | XML header detection |
| `PerExtensionFilter` | `lang`, `extension`, `metadata_file` | -, -, "code_meta.csv" | Extension-specific filters |

## Filter Presets

### Minimal (Quick Pass)

```python
from nemo_curator.stages.text.filters import (
    WordCountFilter,
    NonAlphaNumericFilter,
)

filters = [
    WordCountFilter(min_words=50, max_words=100000),
    NonAlphaNumericFilter(max_non_alpha_numeric_to_text_ratio=0.25),
]
```

### Standard (Recommended)

```python
from nemo_curator.stages.text.filters import (
    WordCountFilter,
    NonAlphaNumericFilter,
    SymbolsToWordsFilter,
    UrlsFilter,
    MeanWordLengthFilter,
    RepeatedLinesFilter,
    RepeatedParagraphsFilter,
    PunctuationFilter,
    CommonEnglishWordsFilter,
    PornographicUrlsFilter,
)

filters = [
    WordCountFilter(min_words=50, max_words=100000),
    NonAlphaNumericFilter(max_non_alpha_numeric_to_text_ratio=0.25),
    SymbolsToWordsFilter(max_symbol_to_word_ratio=0.1),
    UrlsFilter(max_url_to_text_ratio=0.2),
    MeanWordLengthFilter(min_mean_word_length=3, max_mean_word_length=10),
    RepeatedLinesFilter(max_repeated_line_fraction=0.7),
    RepeatedParagraphsFilter(max_repeated_paragraphs_ratio=0.7),
    PunctuationFilter(max_num_sentences_without_endmark_ratio=0.85),
    CommonEnglishWordsFilter(min_num_common_words=2),
    PornographicUrlsFilter(),
]
```

### Full (Thorough Cleaning)

All 35 heuristic filters with default parameters.

### Code (Source Code)

```python
from nemo_curator.stages.text.filters import (
    AlphaFilter,
    GeneralCommentToCodeFilter,
    NumberOfLinesOfCodeFilter,
    TokenizerFertilityFilter,
)

filters = [
    AlphaFilter(min_alpha_ratio=0.25),
    GeneralCommentToCodeFilter(
        language="text/x-python",  # MIME type required
        max_comment_to_code_ratio=0.85,
    ),
    NumberOfLinesOfCodeFilter(min_lines=10, max_lines=20000),
    TokenizerFertilityFilter(
        path_to_tokenizer="/path/to/tokenizer.model",  # Required
        min_char_to_token_ratio=2.5,
    ),
]
```

## Usage in Pipeline

### With ScoreFilter Module

The `ScoreFilter` module computes a score for each document and filters based on the `DocumentFilter`'s criteria.

```yaml
stages:
  - _target_: nemo_curator.stages.text.modules.ScoreFilter
    filter_obj:
      _target_: nemo_curator.stages.text.filters.WordCountFilter
      min_words: 50
      max_words: 100000
    text_field: "text"
    score_field: "word_count"  # Optional: save score for analysis
```

### Multiple Filters

```yaml
stages:
  # Filter 1: Word count
  - _target_: nemo_curator.stages.text.modules.ScoreFilter
    filter_obj:
      _target_: nemo_curator.stages.text.filters.WordCountFilter
      min_words: 50
      max_words: 100000
    text_field: "text"

  # Filter 2: Non-alphanumeric ratio
  - _target_: nemo_curator.stages.text.modules.ScoreFilter
    filter_obj:
      _target_: nemo_curator.stages.text.filters.NonAlphaNumericFilter
      max_non_alpha_numeric_to_text_ratio: 0.25
    text_field: "text"

  # Filter 3: Repeated lines
  - _target_: nemo_curator.stages.text.modules.ScoreFilter
    filter_obj:
      _target_: nemo_curator.stages.text.filters.RepeatedLinesFilter
      max_repeated_line_fraction: 0.7
    text_field: "text"
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
