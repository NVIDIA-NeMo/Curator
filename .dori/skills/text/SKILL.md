---
name: text
description: Curate text/document data with filtering, classification, and deduplication. Use when the user wants to process text, clean documents, or build text datasets. Entry point for all text curation.
license: Apache-2.0
metadata:
  author: nvidia
  version: "1.0"
  modality: text
  gpu-required: "varies"
---

# Text Curation Skill

Main entry point for text/document data curation. Routes to specialized sub-skills or orchestrates full pipelines.

## When This Skill Applies

- User wants to process text or document data
- User mentions: "text", "documents", "corpus", "articles", "web data"
- User wants a complete text curation pipeline
- User is unsure which specific text operation they need

## Sub-Skills Available

| Skill | Command | GPU | Purpose |
|-------|---------|-----|---------|
| **Filter** | `/txt-filter` | No | Heuristic filtering (word count, URLs, repetition) |
| **Classify** | `/txt-classify` | Yes | ML classification (quality, domain, safety) |
| **Dedup** | `/txt-dedup` | Yes | Fuzzy deduplication (MinHash + LSH) |

## Quick Start: YAML CLI (No Code Required)

For standard text filtering, use the built-in YAML configs instead of writing Python:

```bash
# Run full English heuristic filtering (20+ filters)
python -m nemo_curator.config.run \
  --config-path ./text \
  --config-name heuristic_filter_english_pipeline.yaml \
  input_path=/data/input \
  output_path=/data/output
```

With Docker:

```bash
docker run --rm --shm-size=8g \
  -v $(pwd):/data \
  nvcr.io/nvidia/nemo-curator:25.09 \
  python -m nemo_curator.config.run \
    --config-path /opt/NeMo-Curator/nemo_curator/config/text \
    --config-name heuristic_filter_english_pipeline.yaml \
    input_path=/data/input \
    output_path=/data/output
```

### Available Text YAML Configs

| Config | Purpose |
|--------|---------|
| `heuristic_filter_english_pipeline.yaml` | Full English filtering (20+ filters) |
| `heuristic_filter_non_english_pipeline.yaml` | Non-English text filtering |
| `exact_deduplication_pipeline.yaml` | Hash-based exact dedup |
| `fuzzy_deduplication_pipeline.yaml` | MinHash fuzzy dedup |
| `fasttext_filter_pipeline.yaml` | Language detection + filtering |
| `code_filter_pipeline.yaml` | Code-specific filtering |

**When to use YAML CLI**: Standard filtering without custom logic
**When to use Python**: Custom filters, complex conditionals, integration needs

## Skill Workflow (Chain-of-Thought)

### Step 1: Reason About the Goal

Before recommending anything, think through:

```
THOUGHT: What is the user trying to accomplish?
┌──────────────────────────────────────────────────────────┐
│ Goal?      LLM training / RAG / search / analysis        │
│ Data size? MB (small) / GB (medium) / TB (large)         │
│ Quality?   Strict (keep 20%) / balanced / permissive     │
│ GPU?       Available / CPU-only                          │
└──────────────────────────────────────────────────────────┘
```

### Step 2: Route Based on Reasoning

| If... | Then... | Why |
|-------|---------|-----|
| No GPU available | `/txt-filter` (YAML) | Heuristic filters are CPU-only |
| Need ML quality scoring | `/txt-classify` | Requires GPU for inference |
| Suspect duplicates | `/txt-dedup` | GPU recommended for large data |
| Full LLM training prep | Orchestrate all stages | Filter → Classify → Dedup |
| User is learning | Start with YAML CLI | Simplest path to results |

### Step 3: Analyze Before Recommending

**Don't guess. Let the data tell you:**

```bash
python skills/shared/scripts/analyze_data.py --input /path/to/data.jsonl --sample 100 --json
```

```
OBSERVE: Analysis results
         - 28% docs under 50 words → WordCountFilter(min_words=50)
         - 12% URL-heavy → UrlsFilter(max_url_to_text_ratio=0.2)
         - 5% repeated lines → RepeatedLinesFilter needed
         
THOUGHT: Parameters should match these findings, not defaults
```

### Step 4: Build Pipeline

For a complete pipeline, combine stages:

```python
# Full Text Curation Pipeline
# GPU Required: Yes (for classification)
# 
# Note: FuzzyDeduplicationWorkflow is a separate workflow, not a Pipeline stage.
# Run filtering/classification first, then dedup as a separate step.

import torch
print(f"GPU available: {torch.cuda.is_available()}")

# === Stage 1: Heuristic Filtering (CPU) ===
from nemo_curator.stages.text.filters import (
    WordCountFilter,
    NonAlphaNumericFilter,
    UrlsFilter,
    RepeatedLinesFilter,
)

filters = [
    WordCountFilter(min_words=50, max_words=100000),
    NonAlphaNumericFilter(max_non_alpha_numeric_to_text_ratio=0.3),
    UrlsFilter(max_url_to_text_ratio=0.2),
    RepeatedLinesFilter(max_repeated_line_fraction=0.7),  # Keeps docs with ≥70% unique lines
]

# === Stage 2: Quality Classification (GPU) ===
from nemo_curator.stages.text.classifiers import QualityClassifier

classifier = QualityClassifier(
    text_field="text",
    model_inference_batch_size=256,
    filter_by=["High", "Medium"],  # Keep High and Medium quality only
)

# === Build Pipeline (filtering + classification) ===
from nemo_curator.pipeline import Pipeline

pipeline = Pipeline(
    name="text_curation",
    stages=[*filters, classifier],
)

# Run filtering and classification
results = pipeline.run()
print(f"After filtering/classification: {len(results)} tasks")

# === Stage 3: Fuzzy Deduplication (Separate Workflow) ===
# Deduplication is run as a separate workflow on the filtered output
from nemo_curator.stages.deduplication.fuzzy import FuzzyDeduplicationWorkflow

dedup_workflow = FuzzyDeduplicationWorkflow(
    input_path="./filtered_output",      # Output from previous pipeline
    output_path="./dedup_output",         # Where to write duplicate IDs
    cache_path="./dedup_cache",           # Intermediate files
    input_filetype="jsonl",
    text_field="text",
    char_ngrams=24,
    num_bands=20,
    minhashes_per_band=13,                # ~80% similarity threshold
    perform_removal=False,                # Note: removal not yet implemented
)

dedup_result = dedup_workflow.run()
print(f"Duplicates identified: {dedup_result.metadata.get('num_duplicates', 0)}")
```

### Step 5: Iterative Refinement

After initial run, help user tune:

1. **Too aggressive filtering?** Lower thresholds
2. **Too many duplicates removed?** Raise Jaccard threshold
3. **Quality scores too strict?** Lower min_score

## Pipeline Templates

### Template 1: Quick Clean (CPU Only)

For users without GPU or small datasets:

```python
from nemo_curator.stages.text.filters import (
    WordCountFilter,
    NonAlphaNumericFilter,
    RepeatedLinesFilter,
)
from nemo_curator.pipeline import Pipeline

pipeline = Pipeline(
    name="quick_clean",
    stages=[
        WordCountFilter(min_words=50),
        NonAlphaNumericFilter(max_non_alpha_numeric_to_text_ratio=0.3),
        RepeatedLinesFilter(max_repeated_line_fraction=0.7),
    ],
)
```

### Template 2: Quality Focus (GPU)

For high-quality training data:

```python
from nemo_curator.stages.text.filters import WordCountFilter
from nemo_curator.stages.text.classifiers import (
    QualityClassifier,
    FineWebEduClassifier,
)
from nemo_curator.pipeline import Pipeline

pipeline = Pipeline(
    name="quality_focus",
    stages=[
        WordCountFilter(min_words=100),
        QualityClassifier(filter_by=["High"]),  # Keep only High quality
        FineWebEduClassifier(filter_by=["high_quality"]),  # Educational content
    ],
)
```

### Template 3: Dedup Focus (GPU)

For large datasets with duplicates:

```python
from nemo_curator.stages.text.filters import WordCountFilter
from nemo_curator.stages.deduplication.fuzzy import FuzzyDeduplicationWorkflow
from nemo_curator.pipeline import Pipeline

# Step 1: Filter first
filter_pipeline = Pipeline(
    name="pre_dedup_filter",
    stages=[WordCountFilter(min_words=50)],
)
filter_pipeline.run()  # Outputs to filtered files

# Step 2: Run deduplication workflow (separate from Pipeline)
dedup_workflow = FuzzyDeduplicationWorkflow(
    input_path="./filtered_data",
    output_path="./dedup_output",
    cache_path="./dedup_cache",
    input_filetype="jsonl",
    num_bands=20,
    minhashes_per_band=13,  # ~80% similarity threshold
    perform_removal=False,   # Removal not yet implemented
)
dedup_workflow.run()
```

### Template 4: Full Pipeline (GPU)

Complete curation workflow:

```python
from nemo_curator.stages.text.filters import (
    WordCountFilter,
    NonAlphaNumericFilter,
    UrlsFilter,
    RepeatedLinesFilter,
    PunctuationFilter,
)
from nemo_curator.stages.text.classifiers import QualityClassifier, AegisClassifier
from nemo_curator.stages.deduplication.fuzzy import FuzzyDeduplicationWorkflow
from nemo_curator.pipeline import Pipeline

# Step 1: Filter + Classify Pipeline
pipeline = Pipeline(
    name="full_curation",
    stages=[
        # Heuristic filters (CPU)
        WordCountFilter(min_words=50, max_words=100000),
        NonAlphaNumericFilter(max_non_alpha_numeric_to_text_ratio=0.3),
        UrlsFilter(max_url_to_text_ratio=0.2),
        RepeatedLinesFilter(max_repeated_line_fraction=0.7),
        PunctuationFilter(max_num_sentences_without_endmark_ratio=0.85),
        
        # Quality classification (GPU)
        QualityClassifier(filter_by=["High", "Medium"]),
        
        # Safety classification (GPU) - requires HF token for LlamaGuard access
        AegisClassifier(
            hf_token="YOUR_HF_TOKEN",  # Or set HF_TOKEN env var
            filter_by=["safe"],
        ),
    ],
)
pipeline.run()

# Step 2: Deduplication (separate workflow)
dedup_workflow = FuzzyDeduplicationWorkflow(
    input_path="./classified_output",
    output_path="./final_output",
    cache_path="./dedup_cache",
    input_filetype="jsonl",
    num_bands=20,
    minhashes_per_band=13,
    perform_removal=False,  # Note: removal not yet implemented
)
dedup_workflow.run()
```

## Example Interactions

### Example 1: New user, unsure where to start
```
User: I want to clean up some text data for training

Agent: I can help you curate text data. A few questions:

1. How big is your dataset? (rough estimate)
2. Do you have a GPU available?
3. What's your main concern - quality, duplicates, or both?

User: About 10GB of JSONL, I have an RTX 3090, mainly worried about quality

Agent: For 10GB with quality focus on RTX 3090:

Recommended pipeline:
1. Heuristic filters (CPU) - fast pre-filtering
2. Quality classifier (GPU) - ML-based scoring
3. Optional: dedup if you suspect duplicates

Let me analyze a sample of your data first:
[runs analyze_jsonl.py]

Based on the analysis, I recommend...
```

### Example 2: Specific sub-task
```
User: I just want to remove duplicates from my data

Agent: For deduplication, I'll hand off to the specialized skill.

→ Routing to /txt-dedup

[The txt-dedup skill takes over]
```

### Example 3: No GPU available
```
User: I want to filter my documents but I don't have a GPU

Agent: No problem! Heuristic filtering is CPU-only and very effective.

Available CPU filters:
- Word count (remove too short/long)
- URL ratio (remove URL-heavy docs)
- Repeated lines (remove boilerplate)
- Non-alphanumeric (remove garbled text)
- Punctuation (remove non-prose)

→ Routing to /txt-filter

[The txt-filter skill takes over]
```

## Routing Logic (Decision Tree)

When this skill is invoked, reason through:

```
THOUGHT: What specific operation does the user need?

┌─────────────────────────────────────────────────────────┐
│ User mentions "filter", "heuristic", "clean", "remove"? │
│ OR user has no GPU?                                     │
│   → Route to /txt-filter (CPU heuristics)               │
├─────────────────────────────────────────────────────────┤
│ User mentions "classify", "quality", "score", "rank"?   │
│   → Route to /txt-classify (GPU required)               │
├─────────────────────────────────────────────────────────┤
│ User mentions "dedup", "duplicate", "similar"?          │
│   → Route to /txt-dedup (GPU recommended)               │
├─────────────────────────────────────────────────────────┤
│ User wants "full pipeline" OR is unsure?                │
│   → Stay in /text, orchestrate complete workflow        │
│   → Analyze data first to determine what's needed       │
└─────────────────────────────────────────────────────────┘
```

### Verification Before Routing

- [ ] Did I ask about GPU availability?
- [ ] Did I ask about data size?
- [ ] Did I offer to analyze a sample?
- [ ] Did I explain YAML vs Python options?
