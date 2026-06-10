"""Exact text dedup over InterleavedParquet outputs.

Mirrors the Nemotron-CC reference recipe step 2a
(``NVIDIA-NeMo/Nemotron:src/nemotron/recipes/data/curation/nemotron-cc/step_2a-exact_dedup.py``)
but adapted for our InterleavedParquet input format and wrapped with a
single-node ``multiprocessing.Pool`` removal phase (empirically ~10× faster
than Curator's ``TextDuplicatesRemovalWorkflow`` at single-node scale; see
``dedup_perf_results.md``).

Use this BEFORE ``fuzzy_dedup.py`` to cheaply strip byte-identical
duplicates — exact dedup is GPU-hash-based (one cuDF group-by) and runs in a
fraction of fuzzy's wall-clock. Typical 5–15% doc drop, shrinking the input
to the more expensive MinHash/LSH pass.

Pipeline phases (shared with ``fuzzy_dedup.py``)
------------------------------------------------
1. **Aggregate**:  read all shards → project (sample_id, text) → aggregate.parquet.
2. **Identify**:   ``ExactDeduplicationWorkflow`` →
                   hash text column (xxhash via cuDF) → integer duplicate IDs.
3. **Resolve**:    translate integer IDs → original sample_ids.
4. **Remove**:     per-shard filter via ``multiprocessing.Pool(--workers)``,
                   write to ``--output``, preserving the ``idx_NNNNN/`` layout.

Each phase can be skipped by ``--skip-aggregate / --skip-identify /
--skip-remove`` so partial reruns are cheap (intermediates live under ``--cache``).

Hardware
--------
Phases 1, 3, 4 are CPU-only. Phase 2 requires a GPU (Curator's
ExactDeduplicationWorkflow uses cuDF). Submit on a single GPU node — Curator
handles intra-node parallelism via Ray.

Example
-------
    # Run exact dedup first
    python exact_dedup.py \\
        --input  /scratch/.../stage2/cc_main_2025_26 \\
        --output /scratch/.../stage2a_exact_dedup \\
        --cache  /scratch/.../stage2a_exact_dedup_cache

    # Then fuzzy on the exact-deduped output
    python fuzzy_dedup.py \\
        --input  /scratch/.../stage2a_exact_dedup \\
        --output /scratch/.../stage2b_fuzzy_dedup \\
        --cache  /scratch/.../stage2b_fuzzy_dedup_cache

See also: ``fuzzy_dedup.py`` for the MinHash+LSH near-duplicate pass.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

# Allow running as a standalone script: `python dedup/exact_dedup.py ...`
sys.path.insert(0, str(Path(__file__).parent))
from _shared import (  # noqa: E402
    AGGREGATE_FILE,
    aggregate_doc_text,
    TEXT_SOURCES,
    find_shards,
    remove_duplicates,
    resolve_duplicate_sample_ids,
)


# Subdir + filename conventions mirror step_2a-exact_dedup.py
EXACT_DEDUP_IDS_SUBDIR = "ExactDuplicateIds"
ID_GENERATOR_FILENAME = "exact_id_generator.json"


def run_exact_identify(
    aggregate_path: Path,
    cache_dir: Path,
    *,
    input_blocksize: str,
    identification_batchsize: int,
    rmm_pool_size,
    spill_memory_limit,
) -> None:
    """Run ExactDeduplicationWorkflow on the aggregate.parquet. Requires GPU.

    Writes ``<cache_dir>/ExactDuplicateIds/*.parquet`` and
    ``<cache_dir>/exact_id_generator.json``.
    """
    # Imported here so the script can do --skip-identify on a CPU node.
    from nemo_curator.stages.deduplication.exact.workflow import (
        ExactDeduplicationWorkflow,
    )

    cache_dir.mkdir(parents=True, exist_ok=True)

    wf = ExactDeduplicationWorkflow(
        input_path=str(aggregate_path.parent),
        output_path=str(cache_dir),
        input_filetype="parquet",
        text_field="text",
        input_blocksize=input_blocksize,
        identification_batchsize=identification_batchsize,
        assign_id=True,
        perform_removal=False,  # we run our own mp.Pool removal phase
        rmm_pool_size=rmm_pool_size,
        spill_memory_limit=spill_memory_limit,
    )
    print(f"[identify] ExactDeduplicationWorkflow starting (blocksize={input_blocksize}, batchsize={identification_batchsize})")
    print(f"[identify]   input    = {aggregate_path}")
    print(f"[identify]   output   = {cache_dir}")
    t0 = time.time()
    result = wf.run()
    elapsed = time.time() - t0
    num_dup = result.metadata.get("num_duplicates", 0)
    print(f"[identify] done in {elapsed:.1f}s — found {num_dup:,} exact duplicate IDs")


def _parse_memory_arg(value: str):
    """Parse a memory arg like '8GB' / 'auto' / 'none' (matches step_2a-exact_dedup.py)."""
    if value.lower() == "none":
        return None
    if value.lower() == "auto":
        return "auto"
    return int(value)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path, help="Directory with idx_*/*.parquet shards")
    ap.add_argument("--output", type=Path, help="Where to write filtered shards (skip with --skip-remove)")
    ap.add_argument("--cache", required=True, type=Path, help="Intermediates: aggregate.parquet + ExactDuplicateIds")
    ap.add_argument("--text-source", choices=list(TEXT_SOURCES), default="metadata",
                    help="Dedup on the per-doc Resiliparse text (metadata) or concat of all text rows.")

    # Identification (GPU) tuning — defaults mirror step_2a-exact_dedup.py
    ap.add_argument("--input-blocksize", default="256MiB",
                    help="ExactDeduplicationWorkflow read blocksize (e.g. 256MiB, 2GiB).")
    ap.add_argument("--identification-batchsize", type=int, default=12,
                    help="Number of input blocks to process per identification batch.")
    ap.add_argument("--rmm-pool-size", type=_parse_memory_arg, default="auto",
                    help="RMM GPU memory pool size in bytes, 'auto' (~90%% free GPU), or 'none'.")
    ap.add_argument("--spill-memory-limit", type=_parse_memory_arg, default="auto",
                    help="Device memory limit before spilling to host, 'auto' (~80%% of RMM pool), or 'none'.")

    # Phase skips
    ap.add_argument("--skip-aggregate", action="store_true", help="Reuse aggregate.parquet under --cache.")
    ap.add_argument("--skip-identify", action="store_true", help="Reuse duplicate IDs under --cache/duplicates/.")
    ap.add_argument("--skip-remove", action="store_true", help="Stop after emitting duplicate_sample_ids.json.")
    ap.add_argument("--clear-cache", action="store_true", help="Wipe --cache contents before running.")

    # Removal
    ap.add_argument("--workers", type=int, default=16,
                    help="Worker processes for the removal phase (default 16; set 1 for in-process).")

    args = ap.parse_args()

    args.input = args.input.resolve()
    args.cache = args.cache.resolve()
    if args.output:
        args.output = args.output.resolve()

    if args.clear_cache and args.cache.exists():
        print(f"[init] wiping {args.cache}")
        shutil.rmtree(args.cache)
    args.cache.mkdir(parents=True, exist_ok=True)

    shards = find_shards(args.input)
    print(f"[init] {len(shards)} input shard(s) under {args.input}")

    aggregate_dir = args.cache / "aggregate"
    aggregate_path = aggregate_dir / AGGREGATE_FILE
    # ExactDeduplicationWorkflow writes both ExactDuplicateIds/ and the id_generator
    # JSON directly under output_path. Point it at a dedicated subdir to keep
    # cache contents organised.
    identify_out = args.cache / "exact_identify"
    duplicates_dir = identify_out / EXACT_DEDUP_IDS_SUBDIR
    id_generator_path = identify_out / ID_GENERATOR_FILENAME
    dup_sample_ids_path = args.cache / "duplicate_sample_ids.json"

    # ---- Phase 1: aggregate -------------------------------------------------
    if not args.skip_aggregate:
        print(f"[aggregate] text_source={args.text_source}")
        t0 = time.time()
        n_docs, n_dropped = aggregate_doc_text(shards, aggregate_path, args.text_source)
        print(f"[aggregate] wrote {n_docs:,} docs to {aggregate_path} "
              f"({n_dropped:,} dropped null-text) in {time.time() - t0:.1f}s")
        if n_docs == 0:
            print("[aggregate] ERROR: no docs after aggregation", file=sys.stderr)
            return 2
    else:
        existing = sorted(aggregate_dir.glob("*.parquet"))
        print(f"[aggregate] SKIPPED — reusing {len(existing)} parquet file(s) under {aggregate_dir}")
        if not existing:
            print(f"[aggregate] ERROR: no *.parquet found under {aggregate_dir}", file=sys.stderr)
            return 2

    # ---- Phase 2: exact identify --------------------------------------------
    if not args.skip_identify:
        run_exact_identify(
            aggregate_path=aggregate_path,
            cache_dir=identify_out,
            input_blocksize=args.input_blocksize,
            identification_batchsize=args.identification_batchsize,
            rmm_pool_size=args.rmm_pool_size,
            spill_memory_limit=args.spill_memory_limit,
        )
    else:
        print(f"[identify] SKIPPED — reusing {identify_out}")

    # ---- Phase 3: resolve int IDs -------------------------------------------
    if not id_generator_path.exists():
        print(f"[resolve] ERROR: {id_generator_path} not found — did the workflow finish?", file=sys.stderr)
        return 2
    dup_sample_ids = resolve_duplicate_sample_ids(
        aggregate_path, duplicates_dir, id_generator_path,
    )
    with open(dup_sample_ids_path, "w") as f:
        json.dump(sorted(dup_sample_ids), f)
    print(f"[resolve] wrote {len(dup_sample_ids):,} duplicate sample_ids → {dup_sample_ids_path}")

    # ---- Phase 4: remove ----------------------------------------------------
    if args.skip_remove:
        print("[remove] SKIPPED — duplicate IDs available at", dup_sample_ids_path)
        return 0
    if args.output is None:
        print("[remove] ERROR: --output not provided (or pass --skip-remove)", file=sys.stderr)
        return 2

    print(f"[remove] filtering shards (n_workers={args.workers}) → {args.output}")
    remove_duplicates(shards, args.output, dup_sample_ids, n_workers=args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
