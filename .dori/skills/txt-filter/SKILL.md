---
name: txt-filter
description: Filter text documents using heuristic rules (word count, URLs, repetition, punctuation). CPU-only, no GPU required. Entry point for text curation.
license: Apache-2.0
metadata:
  author: nvidia
  version: "1.0"
  modality: text
  gpu-required: "false"
  parent-skill: text
---

# Text Filtering Skill

Filter text documents using NeMo Curator's heuristic filters. CPU-only - great starting point for text curation.

## When This Skill Applies

- User wants to clean/filter text data
- User mentions: "filter", "clean", "remove low quality", "heuristic", "rule-based"
- User has JSONL/Parquet text data
- User doesn't have GPU (or wants fast CPU-based filtering)

## Skill Workflow

### Step 1: Understand the Goal

Ask the user:
1. What's your end goal? (LLM training, RAG, search index)
2. What quality issues do you see? (short docs, spam, duplicates)
3. Do you have a sample I can analyze?

### Step 2: Analyze User Data

**Use the active analysis tool** to get data-driven recommendations:

```bash
python skills/shared/scripts/analyze_data.py --input /path/to/data.jsonl --sample 100 --json
```

This will analyze the data and output specific filter recommendations based on:
- Word count distribution (% short/long docs)
- URL density analysis
- Language distribution
- Repetition scores

**Pattern reference** (if analyzing manually):

| Pattern | Indicator | Recommended Filter |
|---------|-----------|-------------------|
| Very short documents | < 50 words | `WordCountFilter(min_words=50)` |
| Very long documents | > 100k words | `WordCountFilter(max_words=100000)` |
| High non-alphanumeric | > 25% symbols | `NonAlphaNumericFilter(max_non_alpha_numeric_to_text_ratio=0.25)` |
| URL-heavy content | > 20% URLs | `UrlsFilter(max_url_to_text_ratio=0.2)` |
| Repeated lines | Same lines repeat | `RepeatedLinesFilter(max_repeated_line_fraction=0.7)` |
| Missing punctuation | No sentence endings | `PunctuationFilter(max_num_sentences_without_endmark_ratio=0.85)` |

### Step 3: Recommend Filters

Based on analysis, recommend specific filters with tuned thresholds:

```
Based on your data sample:
- 15% of docs are under 30 words → WordCountFilter(min_words=50)
- Some docs are mostly URLs → UrlsFilter(max_url_to_text_ratio=0.15)
- No repetition issues → skip RepeatedLinesFilter
```

> **Note**: Filter parameters use descriptive names like `max_url_to_text_ratio` rather than generic `max_ratio`. Always check the specific parameter name for each filter.

### Step 4: Validate Before Generating

**Always validate the pipeline before generating code:**

```bash
python skills/shared/scripts/validate_pipeline.py \
  --stages "WordCountFilter,UrlsFilter,RepeatedLinesFilter" --json
```

This confirms:
- ✅ Type flow is valid (all DocumentBatch → DocumentBatch)
- ✅ No GPU required (CPU-only filters)
- ✅ No credentials needed

### Step 5: Generate Pipeline Code

```python
# Text Filtering Pipeline (CPU Only)
import json

from nemo_curator.stages.text.filters import (
    WordCountFilter,
    NonAlphaNumericFilter,
    UrlsFilter,
    RepeatedLinesFilter,
)

FILTERS = [
    WordCountFilter(min_words=50, max_words=100000),
    NonAlphaNumericFilter(max_non_alpha_numeric_to_text_ratio=0.25),
    UrlsFilter(max_url_to_text_ratio=0.2),
    RepeatedLinesFilter(max_repeated_line_fraction=0.7),  # Note: keeps docs with >= 70% unique lines
]

def run_filtering(input_path: str, output_path: str, text_field: str = "text"):
    with open(input_path) as f:
        docs = [json.loads(line) for line in f]
    
    print(f"Loaded {len(docs)} documents")
    
    for filter_obj in FILTERS:
        before = len(docs)
        docs = [
            doc for doc in docs
            if filter_obj.keep_document(filter_obj.score_document(doc[text_field]))
        ]
        print(f"  {type(filter_obj).__name__}: {before} -> {len(docs)}")
    
    with open(output_path, "w") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")
    
    print(f"Output: {len(docs)} documents")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-field", default="text")
    args = parser.parse_args()
    run_filtering(args.input, args.output, args.text_field)
```

## Available Filters

### Content Length
| Filter | Parameters | Default | Description |
|--------|------------|---------|-------------|
| `WordCountFilter` | `min_words`, `max_words`, `lang` | 50, 100000, "en" | Filter by word count |

### Content Quality
| Filter | Parameters | Default | Description |
|--------|------------|---------|-------------|
| `NonAlphaNumericFilter` | `max_non_alpha_numeric_to_text_ratio` | 0.25 | Remove symbol-heavy docs |
| `SymbolsToWordsFilter` | `max_symbol_to_word_ratio`, `lang` | 0.1, "en" | Symbols vs words ratio |
| `MeanWordLengthFilter` | `min_mean_word_length`, `max_mean_word_length`, `lang` | 3, 10, "en" | Filter word salad |
| `CommonEnglishWordsFilter` | `min_num_common_words`, `stop_at_false` | 2, True | Ensure English content |

### Repetition
| Filter | Parameters | Default | Description |
|--------|------------|---------|-------------|
| `RepeatedLinesFilter` | `max_repeated_line_fraction` | 0.7 | Remove boilerplate (keeps if ≥70% unique) |
| `RepeatedParagraphsFilter` | `max_repeated_paragraphs_ratio` | 0.7 | Remove repeated paragraphs |
| `RepeatingTopNGramsFilter` | `n`, `max_repeating_ngram_ratio`, `lang` | 2, 0.2, "en" | Remove docs with dominant n-grams |
| `RepeatingDuplicateNGramsFilter` | `n`, `max_repeating_duplicate_ngram_ratio`, `lang` | 2, 0.2, "en" | Remove docs with duplicate n-grams |

### Structure
| Filter | Parameters | Default | Description |
|--------|------------|---------|-------------|
| `PunctuationFilter` | `max_num_sentences_without_endmark_ratio` | 0.85 | Ensure proper punctuation |
| `EllipsisFilter` | `max_num_lines_ending_with_ellipsis_ratio` | 0.3 | Remove ellipsis-heavy docs |
| `BulletsFilter` | `max_bullet_lines_ratio` | 0.9 | Remove bullet-list-only docs |

### Web Content
| Filter | Parameters | Default | Description |
|--------|------------|---------|-------------|
| `UrlsFilter` | `max_url_to_text_ratio` | 0.2 | Remove URL-heavy docs |
| `WhiteSpaceFilter` | `max_white_space_ratio` | 0.25 | Remove whitespace-heavy docs |

## Execution

```bash
# Native
python filter_pipeline.py --input data.jsonl --output filtered.jsonl

# Docker (no --gpus needed)
docker run --rm -v $(pwd):/data nvcr.io/nvidia/nemo-curator:latest \
    python /data/filter_pipeline.py --input /data/data.jsonl --output /data/filtered.jsonl
```
