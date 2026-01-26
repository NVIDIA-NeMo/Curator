# Fuzzy Deduplication Parameter Tuning Guide

Detailed guidance for tuning MinHash + LSH parameters in `FuzzyDeduplicationWorkflow`.

## Parameter Overview

| Parameter | Purpose | Trade-off |
|-----------|---------|-----------|
| `char_ngrams` | Shingle size | Higher = more precise, fewer false positives |
| `num_bands` | LSH bands | More = finer threshold control |
| `minhashes_per_band` | Hashes per band | More = stricter matching |
| `bands_per_iteration` | Memory control | Lower = less memory, slower |
| `use_64_bit_hash` | Hash size | True for >1B documents |

---

## Similarity Threshold

The probability of two documents being placed in the same bucket is:

```
P(same_bucket) = 1 - (1 - s^minhashes_per_band)^num_bands
```

Where `s` is the Jaccard similarity.

### Quick Threshold Lookup

| num_bands | minhashes_per_band | Threshold |
|-----------|-------------------|-----------|
| 10 | 13 | ~0.70 (70%) |
| 15 | 13 | ~0.75 (75%) |
| 20 | 13 | ~0.80 (80%) ← default |
| 25 | 13 | ~0.83 (83%) |
| 30 | 13 | ~0.85 (85%) |
| 20 | 10 | ~0.74 (74%) |
| 20 | 15 | ~0.83 (83%) |

### Adjusting Threshold

**To be more aggressive (find more duplicates)**:
- Decrease `num_bands` (e.g., 15 instead of 20)
- Decrease `minhashes_per_band` (e.g., 10 instead of 13)

**To be more conservative (fewer false positives)**:
- Increase `num_bands` (e.g., 25 instead of 20)
- Increase `minhashes_per_band` (e.g., 15 instead of 13)

---

## char_ngrams (Shingle Size)

The `char_ngrams` parameter controls how documents are broken into shingles for hashing.

### How It Works

```
Document: "The quick brown fox"
char_ngrams=5: ["The q", "he qu", "e qui", "quic", ...]
char_ngrams=10: ["The quick ", "he quick b", "e quick br", ...]
```

### Recommendations

| Scenario | char_ngrams | Rationale |
|----------|-------------|-----------|
| Default | 24 | Balanced precision/recall |
| High false positives | 30-40 | More context per shingle |
| Missing duplicates | 20-22 | Less context, more matches |
| Short documents (<500 chars) | 15-20 | Enough shingles per doc |
| Long documents (>10K chars) | 25-35 | Better discrimination |

### Common Issues

**Too Low (< 20)**:
- High false positive rate
- Unrelated documents match on common phrases
- Symptom: >30% of corpus marked as duplicates

**Too High (> 40)**:
- Misses near-duplicates
- Only catches very similar documents
- Symptom: Known duplicates not detected

---

## Memory Optimization: bands_per_iteration

The `bands_per_iteration` parameter controls how many LSH bands are processed in a single shuffle operation.

### Memory Impact

```
shuffle_memory ≈ dataset_memory × (bands_per_iteration / num_bands)
```

With `num_bands=20`:
- `bands_per_iteration=20`: Full shuffle (highest memory)
- `bands_per_iteration=5`: 4 shuffles, 1/4 memory each
- `bands_per_iteration=1`: 20 shuffles, 1/20 memory each

### Time Impact

```
runtime ≈ base_runtime × (num_bands / bands_per_iteration) × shuffle_overhead
```

Lower `bands_per_iteration` means more shuffle operations, but each is smaller.

### Recommendations by Dataset Size

| Dataset Size | bands_per_iteration | Memory per Shuffle | Shuffles |
|--------------|---------------------|-------------------|----------|
| < 50GB | 10-20 | High | 1-2 |
| 50-100GB | 5 | Medium | 4 |
| 100-250GB | 3-4 | Medium-Low | 5-7 |
| 250-500GB | 2 | Low | 10 |
| 500GB-1TB | 1 | Minimal | 20 |
| > 1TB | 1 | Minimal | 20 |

### OOM Recovery

If you encounter Out-of-Memory errors:

1. **First**: Reduce `bands_per_iteration` by half
2. **Second**: If still OOM, reduce to 1
3. **Third**: Increase cluster memory or nodes
4. **Fourth**: Consider partitioning the dataset

---

## 64-bit Hash

The `use_64_bit_hash` parameter switches from 32-bit to 64-bit MinHash signatures.

### When to Use

| Condition | use_64_bit_hash |
|-----------|-----------------|
| < 100M documents | false |
| 100M - 1B documents | Consider |
| > 1B documents | true |

### Trade-offs

| Aspect | 32-bit | 64-bit |
|--------|--------|--------|
| Memory per doc | Lower | 2x higher |
| Collision probability | Higher | Much lower |
| Suitable scale | < 1B docs | > 1B docs |

---

## Dataset-Specific Tuning

### Web Crawl Data (Common Crawl)

```yaml
char_ngrams: 24
num_bands: 20
minhashes_per_band: 13
bands_per_iteration: 3  # For 100-500GB
```

Web data has many near-duplicates from templates. Default settings work well.

### News Articles

```yaml
char_ngrams: 30
num_bands: 20
minhashes_per_band: 15
```

Higher precision to avoid matching different articles about same topic.

### Code Repositories

```yaml
char_ngrams: 40
num_bands: 25
minhashes_per_band: 13
```

Very high precision needed; code has many similar patterns.

### Scientific Papers

```yaml
char_ngrams: 25
num_bands: 15
minhashes_per_band: 13
```

Lower threshold to catch preprints vs published versions.

### Social Media Posts

```yaml
char_ngrams: 20
num_bands: 15
minhashes_per_band: 10
```

Short documents need smaller shingles and lower threshold.

---

## Validation

After running deduplication, validate the results:

### Check Duplicate Rate

```python
import pandas as pd

original_count = len(pd.read_parquet(input_path))
duplicate_count = len(pd.read_parquet(f"{output_path}/duplicate_ids.parquet"))
duplicate_rate = duplicate_count / original_count

print(f"Duplicate rate: {duplicate_rate:.2%}")
```

Expected rates:
- Web crawl: 20-50% duplicates
- Curated dataset: 5-15% duplicates
- High-quality corpus: < 5% duplicates

### Spot Check Duplicates

Manually review a sample of detected duplicates:

```python
# Sample duplicate pairs and verify they are actually similar
```

If many false positives, increase `char_ngrams`.
If many missed duplicates, decrease `num_bands`.

---

## Performance Profiling

### Bottleneck Identification

| Stage | Symptom | Solution |
|-------|---------|----------|
| MinHash | High CPU, low GPU | Normal, CPU-bound |
| LSH | High memory | Reduce bands_per_iteration |
| Connected Components | Long runtime | More workers |
| Shuffle | Network saturated | Better interconnect |

### Monitoring Commands

```bash
# Ray dashboard
ray dashboard

# GPU utilization
nvidia-smi -l 1

# Memory usage
htop
```
