---
name: dedup-fuzzy
description: |
  Perform fuzzy deduplication on text datasets using MinHash + LSH algorithm.
  Use when the user wants to remove near-duplicate documents, mentions fuzzy
  matching, MinHash, LSH, similarity-based deduplication, or has large text
  datasets with potential duplicates. Supports datasets from 1GB to multi-TB scale.
license: Apache-2.0
metadata:
  author: nvidia
  version: "1.0"
  modality: text
  gpu-required: true
  nemo-curator-version: ">=0.5.0"
compatibility: Requires Python 3.10+, Ray cluster, GPU with 16GB+ memory
disable-model-invocation: true
---

# Fuzzy Deduplication

Remove near-duplicate documents from text datasets using MinHash signatures and
Locality Sensitive Hashing (LSH).

## When to Use

- Large text datasets (>1GB) with potential duplicates
- Web-scraped or aggregated data sources
- Common Crawl, news articles, or similar content
- Need similarity-based matching (~80% threshold by default)
- Preparing training data for LLMs

## Quick Start

### 1. Generate Configuration

```bash
python scripts/generate_fuzzy_config.py \
  --input-path /data/text \
  --output-path /data/deduped \
  --cache-path /data/cache \
  --output-file fuzzy_dedup.yaml
```

### 2. Estimate Resources (Optional)

```bash
python scripts/estimate_resources.py \
  --input-path /data/text \
  --input-format parquet
```

### 3. Execute Pipeline

```bash
python -m nemo_curator.config.run \
  --config-path=. \
  --config-name=fuzzy_dedup \
  input_path=/data/text \
  output_path=/data/deduped \
  cache_path=/data/cache
```

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                  FuzzyDeduplicationWorkflow                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. MinHashStage                                                │
│     └── Generate 260 hash signatures per document               │
│                                                                  │
│  2. LSHStage                                                    │
│     └── Group similar documents into buckets (20 bands)         │
│                                                                  │
│  3. BucketsToEdgesStage                                         │
│     └── Convert buckets to document pair edges                  │
│                                                                  │
│  4. ConnectedComponentsStage                                    │
│     └── Find clusters of duplicates via graph traversal         │
│                                                                  │
│  5. IdentifyDuplicatesStage                                     │
│     └── Mark documents for removal within each cluster          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Important**: `FuzzyDeduplicationWorkflow` is a `WorkflowBase` class, not a `ProcessingStage`. 
It orchestrates multiple internal pipelines.

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `char_ngrams` | 24 | 20-50 | Shingle size for hashing |
| `num_bands` | 20 | 5-50 | Number of LSH bands |
| `minhashes_per_band` | 13 | 5-25 | Hashes per band |
| `bands_per_iteration` | 5 | 1-num_bands | Memory control |
| `use_64_bit_hash` | false | - | For very large datasets |

### Similarity Threshold

The similarity threshold is approximately:

```
threshold ≈ (1/num_bands)^(1/minhashes_per_band)
```

With defaults (20 bands, 13 hashes): `(1/20)^(1/13) ≈ 0.80` (80% similarity)

### Dataset Size Recommendations

| Dataset Size | char_ngrams | bands_per_iteration | 64-bit |
|--------------|-------------|---------------------|--------|
| < 100GB | 24 | 5 (default) | No |
| 100-500GB | 24 | 3 | No |
| 500GB-1TB | 24-30 | 1-2 | Consider |
| > 1TB | 30 | 1 | Yes |

See [references/FUZZY_DEDUP_PARAMS.md](references/FUZZY_DEDUP_PARAMS.md) for detailed tuning.

## Output Files

The workflow produces:

```
{output_path}/
├── duplicate_ids.parquet     # Document IDs to remove
└── fuzzy_id_generator.json   # ID mapping for removal
```

To actually remove duplicates, run a second pass:

```python
# Read duplicate IDs
duplicates = pd.read_parquet(f"{output_path}/duplicate_ids.parquet")

# Filter original dataset
original = pd.read_parquet(input_path)
deduped = original[~original["id"].isin(duplicates["id"])]
```

## Common Issues

### Out of Memory (OOM)

**Symptom**: Process killed during LSH or Connected Components stage

**Solution**: Reduce `bands_per_iteration`:

```bash
python scripts/generate_fuzzy_config.py \
  --bands-per-iteration 1 \
  ...
```

### High False Positive Rate

**Symptom**: Too many unique documents marked as duplicates

**Solution**: Increase `char_ngrams`:

```bash
python scripts/generate_fuzzy_config.py \
  --char-ngrams 30 \
  ...
```

### Missing Near-Duplicates

**Symptom**: Known duplicates not detected

**Solution**: Decrease `num_bands` to lower threshold:

```bash
python scripts/generate_fuzzy_config.py \
  --num-bands 15 \
  ...
```

### Slow Performance

**Causes**:
1. CPU-bound: Increase cluster size
2. IO-bound: Use local SSD storage
3. Shuffle-bound: Increase `bands_per_iteration` if memory allows

## Script Reference

### generate_fuzzy_config.py

Generate Hydra YAML configuration:

```bash
python scripts/generate_fuzzy_config.py \
  --input-path PATH \
  --output-path PATH \
  --cache-path PATH \
  [--input-filetype parquet|jsonl] \
  [--text-field text] \
  [--char-ngrams 24] \
  [--num-bands 20] \
  [--minhashes-per-band 13] \
  [--bands-per-iteration 5] \
  [--use-64-bit-hash] \
  [--output-file FILE]
```

### estimate_resources.py

Estimate memory and GPU requirements:

```bash
python scripts/estimate_resources.py \
  --input-path PATH \
  [--input-format parquet|jsonl] \
  [--num-bands 20] \
  [--minhashes-per-band 13]
```

## Related Skills

- `/dedup-exact` - Hash-based exact duplicate removal (faster, lower memory)
- `/dedup-semantic` - Embedding-based semantic deduplication (catches paraphrases)
- `/curate` - Full curation workflow including deduplication
- `/filter` - Apply heuristic filters before deduplication
