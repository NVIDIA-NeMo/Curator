"""Fuzzy text dedup over InterleavedParquet outputs.

Mirrors the Nemotron-CC reference recipe step 2b
(``NVIDIA-NeMo/Nemotron:src/nemotron/recipes/data/curation/nemotron-cc/step_2b-fuzzy_dedup.py``)
but adapted for our InterleavedParquet input format and wrapped with a
single-node ``multiprocessing.Pool`` removal phase (empirically ~10× faster
than Curator's ``TextDuplicatesRemovalWorkflow`` at single-node scale; see
``dedup_perf_results.md``).

Pipeline phases (shared with ``exact_dedup.py``)
------------------------------------------------
1. **Aggregate**:  read all shards → project (sample_id, text) → aggregate.parquet.
2. **Identify**:   ``FuzzyDeduplicationWorkflow`` →
                   MinHash → LSH → BucketsToEdges → ConnectedComponents →
                   integer duplicate IDs.
3. **Resolve**:    translate integer IDs → original sample_ids.
4. **Remove**:     per-shard filter via ``multiprocessing.Pool(--workers)``,
                   write to ``--output``, preserving the ``idx_NNNNN/`` layout.

Each phase can be skipped by ``--skip-aggregate / --skip-identify /
--skip-remove`` so partial reruns are cheap (intermediates live under ``--cache``).

Hardware
--------
Phases 1, 3, 4 are CPU-only. Phase 2 requires a GPU (Curator's MinHash uses
cuDF). Submit on a single GPU node — Curator handles intra-node parallelism
via Ray.

Example
-------
    python fuzzy_dedup.py \\
        --input  /scratch/.../stage2/cc_main_2025_26 \\
        --output /scratch/.../stage2b_fuzzy_dedup \\
        --cache  /scratch/.../stage2b_fuzzy_dedup_cache \\
        --text-source metadata \\
        --char-ngrams 24 --num-bands 20 --minhashes-per-band 13

See also: ``exact_dedup.py`` for the byte-identical pre-pass.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

# Allow running as a standalone script: `python dedup/fuzzy_dedup.py ...`
sys.path.insert(0, str(Path(__file__).parent))
from _shared import (  # noqa: E402
    AGGREGATE_FILE,
    aggregate_doc_text,
    TEXT_SOURCES,
    find_shards,
    remove_duplicates,
    resolve_duplicate_sample_ids,
)


def run_fuzzy_identify(
    aggregate_path: Path,
    cache_dir: Path,
    duplicates_dir: Path,
    *,
    char_ngrams: int,
    num_bands: int,
    minhashes_per_band: int,
    input_blocksize: str,
) -> None:
    """Run FuzzyDeduplicationWorkflow on the aggregate.parquet. Requires GPU."""
    # Imported here so the script can do --skip-identify on a CPU node.
    from nemo_curator.stages.deduplication.fuzzy.workflow import (
        FuzzyDeduplicationWorkflow,
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    duplicates_dir.mkdir(parents=True, exist_ok=True)

    wf = FuzzyDeduplicationWorkflow(
        cache_path=str(cache_dir),
        output_path=str(duplicates_dir),
        input_path=str(aggregate_path.parent),
        input_filetype="parquet",
        input_blocksize=input_blocksize,
        text_field="text",
        perform_removal=False,
        char_ngrams=char_ngrams,
        num_bands=num_bands,
        minhashes_per_band=minhashes_per_band,
    )
    print(f"[identify] FuzzyDeduplicationWorkflow starting (B={num_bands}, R={minhashes_per_band}, ngrams={char_ngrams})")
    print(f"[identify]   input    = {aggregate_path}")
    print(f"[identify]   cache    = {cache_dir}")
    print(f"[identify]   dup-ids  = {duplicates_dir}")
    t0 = time.time()
    wf.run()
    print(f"[identify] done in {time.time() - t0:.1f}s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path, help="Directory with idx_*/*.parquet shards")
    ap.add_argument("--output", type=Path, help="Where to write filtered shards (skip with --skip-remove)")
    ap.add_argument("--cache", required=True, type=Path, help="Intermediates: aggregate.parquet + MinHash/LSH/CC")
    ap.add_argument("--text-source", choices=list(TEXT_SOURCES), default="metadata",
                    help="Dedup on the per-doc Resiliparse text (metadata) or concat of all text rows.")
    ap.add_argument("--char-ngrams", type=int, default=24,
                    help="Char n-gram size for MinHash. FineWeb default.")
    ap.add_argument("--num-bands", type=int, default=20,
                    help="LSH bands. FineWeb default → Jaccard ≈ 0.80.")
    ap.add_argument("--minhashes-per-band", type=int, default=13,
                    help="MinHashes per LSH band. FineWeb default.")
    ap.add_argument("--input-blocksize", default="1GiB",
                    help="FuzzyDeduplicationWorkflow read blocksize.")
    ap.add_argument("--skip-aggregate", action="store_true", help="Reuse aggregate.parquet under --cache.")
    ap.add_argument("--skip-identify", action="store_true", help="Reuse duplicate IDs under --cache/duplicates/.")
    ap.add_argument("--skip-remove", action="store_true", help="Stop after emitting duplicate_sample_ids.json.")
    ap.add_argument("--clear-cache", action="store_true", help="Wipe --cache contents before running.")
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
    minhash_cache = args.cache / "minhash_cache"
    duplicates_dir = args.cache / "duplicates"
    id_generator_path = duplicates_dir / "fuzzy_id_generator.json"
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

    # ---- Phase 2: fuzzy identify --------------------------------------------
    if not args.skip_identify:
        run_fuzzy_identify(
            aggregate_path=aggregate_path,
            cache_dir=minhash_cache,
            duplicates_dir=duplicates_dir,
            char_ngrams=args.char_ngrams,
            num_bands=args.num_bands,
            minhashes_per_band=args.minhashes_per_band,
            input_blocksize=args.input_blocksize,
        )
    else:
        print(f"[identify] SKIPPED — reusing {duplicates_dir}")

    # ---- Phase 3: resolve int IDs -------------------------------------------
    if not id_generator_path.exists():
        print(f"[resolve] ERROR: {id_generator_path} not found — did the workflow finish?", file=sys.stderr)
        return 2
    dup_sample_ids = resolve_duplicate_sample_ids(aggregate_path, duplicates_dir, id_generator_path)
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
