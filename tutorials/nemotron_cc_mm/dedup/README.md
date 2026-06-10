# Text deduplication for the Nemotron-CC-MM pipeline

CLIs that run **exact** and **fuzzy** doc-level dedup over the
InterleavedParquet output of the extract / text-filter stages. Both mirror the
official Nemotron-CC reference recipe
([`step_2a-exact_dedup.py` + `step_2b-fuzzy_dedup.py`](https://github.com/NVIDIA-NeMo/Nemotron/tree/main/src/nemotron/recipes/data/curation/nemotron-cc))
but adapt the input/output for our multi-row-per-doc format and replace
Curator's `TextDuplicatesRemovalWorkflow` with `multiprocessing.Pool` removal
(~10× faster at single-node, 9.8 K-task scale — see
`dedup_perf_results.md` in the project memory).

## Files

| File | Purpose |
|---|---|
| `exact_dedup.py`  | `ExactDeduplicationWorkflow` (cuDF MD5 hash + RapidsMPF shuffle). Run FIRST. |
| `fuzzy_dedup.py`  | `FuzzyDeduplicationWorkflow` (MinHash → LSH → BucketsToEdges → ConnectedComponents). Run SECOND. |
| `substring/`      | Substring dedup wrapper (Google Research deduplicate-text-datasets, Rust suffix arrays). Run THIRD. |
| `aggregate_all.py`| **Multi-batch orchestrator** — mp.Pool(N) over (seg, batch) units. Use to pre-build the global aggregate before identify on full-snapshot data. |
| `remove_all.py`   | **Multi-batch orchestrator** — mp.Pool(N) over all `seg_NN/batch_M/idx_NNNNN/` shards to filter duplicates. Preserves layout. |
| `_shared.py`      | Shared aggregate / resolve / remove phases imported by both `exact_dedup.py` and `fuzzy_dedup.py`. |

## Recommended order

```
Stage 2 (text filter)                                       <-- 1.6 TB, ~617M docs
        │
        ▼  aggregate_all.py    ~10 min on 1 CPU node (60 workers)
cache/aggregate/*.parquet                                   <-- 882 GB, 564M docs (-9% null-text)
        │
        ▼  exact_dedup.py --skip-aggregate --skip-remove ~12 min on 1× H100
duplicate_sample_ids.json (exact)                           <-- 170M IDs, 30% doc drop
        │
        ▼  remove_all.py       ~20-30 min on 1 CPU node
stage2a_exact_dedup/output/                                 <-- 394M docs survive
        │
        ▼  aggregate_all.py (against exact output)  ~7 min
        ▼  fuzzy_dedup.py --skip-aggregate --skip-remove ~20-40 min on 1× H100
duplicate_sample_ids.json (fuzzy)
        │
        ▼  remove_all.py       ~10-20 min on 1 CPU node
stage2b_fuzzy_dedup/output/                                 <-- post-fuzzy docs
        │
        ▼  substring/run_substring_smoke.sh (validate first)
        ▼  full substring run (high-mem CPU node)
stage2c_substring_dedup/output/                             <-- final dedup output
        │
        ▼
Stage 3 (image acquire) — only download survivors
```

Running exact before fuzzy shrinks the input to MinHash/LSH proportionally,
saving significant GPU time. Running fuzzy before substring shrinks the
suffix-array RAM/disk footprint by the prior drop.

## Pipeline phases (shared)

Both scripts implement the same 4-phase structure:

1. **Aggregate** (CPU) — InterleavedParquet → `aggregate/aggregate.parquet`
   with one row per doc `(sample_id, text)`. By default `text` is the per-doc
   Resiliparse text from the metadata row (`text-source=metadata`).
2. **Identify** (GPU) — Curator's workflow over `aggregate.parquet`. Writes
   integer duplicate IDs + an id-generator JSON under `--cache/.../`.
3. **Resolve** (CPU) — translate integer IDs back to original `sample_id`
   strings, write `cache/duplicate_sample_ids.json`.
4. **Remove** (CPU) — per-shard `mp.Pool` filter, preserving the
   `idx_NNNNN/<hash>.parquet` layout under `--output`.

Each phase is independently skippable via `--skip-aggregate /
--skip-identify / --skip-remove`, so partial reruns are cheap.

## Hardware

| Phase | exact | fuzzy | Why |
|---|---|---|---|
| Aggregate | CPU | CPU | Just pyarrow projection + filter |
| Identify  | **GPU** | **GPU** | cuDF hash (exact); cuDF MinHash + LSH + CC (fuzzy) |
| Resolve   | CPU | CPU | Build dict from JSON + parquet read |
| Remove    | CPU (≤16 workers) | CPU (≤16 workers) | pyarrow filter + write |

Submit on a single GPU node (1×H100 has been enough through 1-segment scale,
~4.5M docs); upgrade to multi-GPU only if MinHash/CC overruns single-GPU
memory on the full snapshot.

## **CRITICAL**: start Ray externally before running

Curator's `create_id_generator_actor()` calls `ray.shutdown()` after creating
the detached actor. If Ray was driver-started, the cluster (and the actor)
die — subsequent stages re-init Ray and the actor lookup fails with
`Failed to look up actor with name 'curator_deduplication_id_generator'`.

**Fix**: pre-start Ray + `export RAY_ADDRESS=auto` before invoking the script.

```bash
ray stop --force 2>/dev/null
ray start --head --num-cpus=16 --num-gpus=1
export RAY_ADDRESS=auto
```

## Example invocations

### Smoke test (1 batch)

```bash
# 1. Allocate GPU node + start Ray
srun -A nemotron_n4_pre -p batch --time=02:00:00 \
     --gres=gpu:1 --cpus-per-task=16 --mem=180G \
     --container-image=$USER_DIR/sqsh/curator_2604.sqsh \
     --container-mounts=/scratch:/scratch,/home:/home --pty bash

ray stop --force 2>/dev/null
ray start --head --num-cpus=16 --num-gpus=1
export RAY_ADDRESS=auto
cd $USER_DIR/codebase/Curator && pip install --no-deps -e . > /dev/null 2>&1
cd tutorials/nemotron_cc_mm

# 2. Exact dedup on one batch (~200 idx dirs)
python dedup/exact_dedup.py \
    --input  $STAGE2_ROOT/seg_00/batch_0 \
    --output $STAGE2A_ROOT/seg_00/batch_0 \
    --cache  $CACHE_ROOT/exact/seg_00/batch_0

# 3. Fuzzy dedup on the exact-deduped output
python dedup/fuzzy_dedup.py \
    --input  $STAGE2A_ROOT/seg_00/batch_0 \
    --output $STAGE2B_ROOT/seg_00/batch_0 \
    --cache  $CACHE_ROOT/fuzzy/seg_00/batch_0
```

### Default params

| Param | exact default | fuzzy default | Notes |
|---|---|---|---|
| `--text-source` | metadata | metadata | Resiliparse per-doc text from metadata row |
| `--input-blocksize` | 256MiB | 1GiB | Identification read blocksize |
| `--workers` | 16 | 16 | mp.Pool size for removal phase |
| `--char-ngrams` | — | 24 | FineWeb default |
| `--num-bands` | — | 20 | FineWeb default → Jaccard ≈ 0.80 |
| `--minhashes-per-band` | — | 13 | FineWeb default |
| `--identification-batchsize` | 12 | — | Blocks per identify batch (exact only) |
| `--rmm-pool-size` | auto | — | ~90% of free GPU mem |
| `--spill-memory-limit` | auto | — | ~80% of RMM pool |

## Output layout

```
$CACHE_ROOT/exact/<run>/
├── aggregate/aggregate.parquet               ← 1 row per doc (sample_id, text)
├── exact_identify/
│   ├── exact_id_generator.json
│   └── ExactDuplicateIds/*.parquet           ← int IDs (col: _curator_dedup_id)
└── duplicate_sample_ids.json                 ← resolved sample_id list

$CACHE_ROOT/fuzzy/<run>/
├── aggregate/aggregate.parquet               ← can reuse from exact pass
├── minhash_cache/                            ← MinHash sigs, LSH buckets, CC edges
├── duplicates/
│   ├── fuzzy_id_generator.json
│   └── FuzzyDuplicateIds/part.0.parquet      ← int IDs (col: _curator_dedup_id)
└── duplicate_sample_ids.json                 ← resolved sample_id list

$OUTPUT_ROOT/exact/<run>/                     ← filtered shards (same layout as input)
$OUTPUT_ROOT/fuzzy/<run>/
└── idx_NNNNN/<same_name>.parquet
```

## Reference

* Nemotron-CC official recipes:
  https://github.com/NVIDIA-NeMo/Nemotron/tree/main/src/nemotron/recipes/data/curation/nemotron-cc
* Curator workflow source:
  `nemo_curator/stages/deduplication/exact/workflow.py`
  `nemo_curator/stages/deduplication/fuzzy/workflow.py`
* Perf measurements: `dedup_perf_results.md` in project memory
