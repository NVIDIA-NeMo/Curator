# Interleaved IO Format Benchmark Results

**Branch:** `feat/interleaved-io-readers-writers`
**Date:** 2026-03-04
**Dataset:** MINT-1T PDF CC-2024-18 (10 GB, 90 tar shards, 75,843 interleaved rows, 7,489 samples)
**Hardware:** 64 CPUs, no GPU
**Executor:** Xenna (streaming mode)

## Summary

All benchmarks run without filter, `files_per_partition=1`, `per_image_fields=["image_metadata"]`.

### Without Materialization

| # | Reader | Writer | Rows | Time (s) | Output MB | Rows/s |
|---|--------|--------|------|----------|-----------|--------|
| 1 | WDS | Parquet | 75,843 | 20.6 | 66.7 | 3,677 |
| 2 | WDS | WebDataset | 7,489* | 18.8 | 125.3 | 399* |
| 3 | WDS | Lance (fragments) | 75,843 | 21.8 | 135.0 | 3,478 |
| 4 | Parquet | Parquet | 75,843 | 20.0 | 66.7 | 3,790 |
| 5 | Parquet | WebDataset | 7,489* | 20.8 | 125.3 | 360* |
| 6 | Parquet | Lance (fragments) | 75,843 | 21.6 | 135.0 | 3,511 |

*WebDataset metrics count samples (JSON members), not individual rows.

### With Materialization (binary images fetched from source tar shards)

| # | Reader | Writer | Rows | Time (s) | Output MB | Rows/s |
|---|--------|--------|------|----------|-----------|--------|
| 7 | WDS | Parquet | 75,843 | 61.9 | 6,171.8 | 1,226 |
| 8 | WDS | WebDataset | 7,489* | 63.4 | 6,574.9 | 118* |
| 9 | WDS | Lance (fragments) | 75,843 | 56.7 | 6,523.4 | 1,337 |

## Key Findings

### Writer Performance (no materialization)

- **Parquet is the most space-efficient**: 66.7 MB (1.8x smaller than WDS, 2.0x smaller than Lance)
- **All writers have comparable e2e time** (~20s) because the reader dominates
- **Lance fragment commit is fast**: 90 fragments committed into 1 dataset in <1s

### Writer Performance (with materialization)

- **Lance is the fastest with materialization**: 56.7s vs 61.9s (Parquet) and 63.4s (WDS)
- **Materialization dominates**: adds ~40s to every pipeline (fetching images from tar shards)
- **Output sizes are similar** when images are embedded: 6.0-6.5 GB

### Reader Performance

- Parquet reader and WDS reader have similar e2e times on this dataset (~20s)
- WDS reader parses tar archives; Parquet reader does native pyarrow reads

### Space Efficiency (no materialization)

| Writer | Output MB | KB/Row |
|--------|-----------|--------|
| Parquet | 66.7 | 0.90 |
| WebDataset | 125.3 | 1.70 |
| Lance (fragments) | 135.0 | 1.82 |

## Cross-Format Consistency

Verified on first shard (763 rows, 81 samples):

| Comparison | sample_ids | modality | text_content | passthrough cols |
|------------|-----------|----------|-------------|------------------|
| Parquet vs Lance | MATCH | MATCH | MATCH (sorted) | MATCH (sorted) |
| Parquet vs WDS | MATCH | MATCH (text+meta) | MATCH | MATCH |

- **WDS without materialization loses image rows** (expected: no binary content = no tar member)
- **Lance has a known pylance 2.0.1 parallel scanner bug**: `ds.to_table()` panics on
  multi-fragment datasets (>1 fragment) due to an assertion failure in `arrow-array`'s
  `PrimitiveArray`/`StringArray` buffer validation. Sequential per-fragment reads
  (`frag.to_table()`) work correctly for all 90 fragments (verified: 75,843 rows, 0
  position ordering errors). This is a lance library read bug, not a data correctness issue.

## Comparison with PR #1559

| Pipeline | This PR | PR #1559 | Delta |
|----------|---------|----------|-------|
| WDS->Parquet | 20.6s | 22.1s | 7% faster |
| WDS->WDS | 18.8s | 22.7s | 17% faster |
| WDS->Lance | 21.8s | 23.3s | 6% faster |
| WDS->Parquet MAT | 61.9s | 64.8s | 4% faster |
| WDS->Lance MAT | 56.7s | 70.6s | 20% faster |

This PR uses `lance.fragment.write_fragments()` (distributed fragment API) instead of
`lance.write_dataset()` per task, producing a single unified dataset instead of 90 separate ones.
