---
name: txt-dedup
description: Remove near-duplicate documents using MinHash and LSH. GPU recommended for large datasets.
license: Apache-2.0
metadata:
  author: nvidia
  version: "1.0"
  modality: text
  gpu-required: "recommended"
  parent-skill: text
---

# Text Deduplication Skill

Remove near-duplicate documents using MinHash + LSH. GPU recommended for large datasets.

## When This Skill Applies

- User wants to remove duplicates or near-duplicates
- User mentions: "dedup", "deduplicate", "duplicates", "similar documents"
- User has large-scale text data with potential redundancy

## Resource Requirements

| Documents | Recommended Setup |
|-----------|-------------------|
| < 100K | Single GPU, 32GB RAM |
| 100K - 10M | Single GPU, 64GB+ RAM |
| 10M - 100M | Multi-GPU node, 256GB+ RAM |
| > 100M | Multi-node GPU cluster |

## Skill Workflow

### Step 1: Understand the Data Scale

Ask the user:
1. How many documents? (thousands, millions, billions)
2. Total data size? (MB, GB, TB)
3. GPU cluster or single machine?

### Step 2: Explain the Algorithm

MinHash + LSH finds near-duplicates efficiently:
1. **Shingling**: Convert docs to character n-grams
2. **MinHash**: Create compact signatures
3. **LSH Banding**: Group similar signatures
4. **Jaccard Similarity**: Compare candidates

### Step 3: Recommend Parameters

| Use Case | num_bands | minhashes_per_band | ~Threshold |
|----------|-----------|--------------------| -----------|
| Exact duplicates | 10 | 26 | ~0.95 |
| Near-duplicates (strict) | 14 | 18 | ~0.85 |
| Near-duplicates (standard) | 20 | 13 | ~0.80 |
| Aggressive dedup | 25 | 10 | ~0.70 |

**Parameters:**
- `char_ngrams=24`: Shingle size (20-30 typical)
- `use_64_bit_hash=False`: True for >1B documents
- `bands_per_iteration=5`: Lower = less memory, slower

### Step 4: Generate Pipeline Code

```python
# Fuzzy Deduplication Pipeline
# GPU Recommended: Yes
# 
# This workflow identifies duplicate document IDs but does NOT remove them automatically.
# The output includes duplicate IDs that you can use for post-processing removal.

import os
from pathlib import Path

try:
    import torch
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: No GPU. Deduplication will be slow.")
except ImportError:
    pass

from nemo_curator.stages.deduplication.fuzzy import FuzzyDeduplicationWorkflow

CONFIG = {
    "char_ngrams": 24,
    "num_bands": 20,
    "minhashes_per_band": 13,
    "use_64_bit_hash": False,
    "bands_per_iteration": 5,
    "seed": 42,
}

def run_deduplication(input_path: str, output_path: str, cache_path: str, text_field: str = "text"):
    input_filetype = "parquet" if input_path.endswith(".parquet") else "jsonl"
    
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Cache: {cache_path}")
    
    threshold = (1 / CONFIG["num_bands"]) ** (1 / CONFIG["minhashes_per_band"])
    print(f"Similarity threshold: {threshold:.2%}")
    
    # Note: perform_removal is NOT yet implemented in NeMo Curator.
    # This workflow identifies duplicates and outputs their IDs.
    # You must implement removal as a post-processing step.
    workflow = FuzzyDeduplicationWorkflow(
        input_path=input_path,
        output_path=output_path,
        cache_path=cache_path,
        input_filetype=input_filetype,
        text_field=text_field,
        **CONFIG,
        perform_removal=False,  # Only False is supported currently
    )
    
    result = workflow.run()
    
    # Check results
    num_duplicates = result.metadata.get("num_duplicates", 0)
    print(f"Duplicates identified: {num_duplicates}")
    print(f"Duplicate IDs written to: {output_path}")
    print(f"ID mapping written to: {result.metadata.get('id_generator_path', 'N/A')}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--text-field", default="text")
    args = parser.parse_args()
    run_deduplication(args.input, args.output, args.cache, args.text_field)
```

### Step 5: Post-Processing Removal (Manual)

Since `perform_removal` is not yet implemented, remove duplicates manually:

```python
import json

def remove_duplicates(original_path: str, duplicate_ids_path: str, output_path: str):
    """Remove documents whose IDs are in the duplicate list."""
    # Load duplicate IDs (output from deduplication workflow)
    with open(duplicate_ids_path) as f:
        duplicate_ids = set(json.loads(line)["id"] for line in f)
    
    # Filter original data
    kept = 0
    removed = 0
    with open(original_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            doc = json.loads(line)
            if doc.get("id") not in duplicate_ids:
                fout.write(line)
                kept += 1
            else:
                removed += 1
    
    print(f"Kept: {kept}, Removed: {removed}")
```

## Threshold Math

Probability two docs with Jaccard similarity `s` are marked as duplicates:
```
P(candidate) = 1 - (1 - s^minhashes_per_band)^num_bands
```

## Execution

```bash
docker run --gpus all --rm -v $(pwd):/data nvcr.io/nvidia/nemo-curator:latest \
    python /data/dedup_pipeline.py \
    --input /data/input.jsonl \
    --output /data/deduplicated.jsonl \
    --cache /data/cache
```

## Interpreting Results

| Data Source | Typical Duplicate Rate |
|-------------|------------------------|
| Common Crawl | 30-50% |
| Web scrapes | 40-60% |
| Curated datasets | <10% |

**Too aggressive?** Increase `minhashes_per_band` or decrease `num_bands`
**Missing duplicates?** Decrease `minhashes_per_band` or increase `num_bands`
