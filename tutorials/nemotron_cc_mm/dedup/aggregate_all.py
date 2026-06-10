"""Aggregate all (seg, batch) groups of a Stage 2 root in parallel on one CPU node.

For cross-batch exact/fuzzy dedup we need a SINGLE logical aggregate over the
whole snapshot so the GPU identify phase can hash and shuffle across batch
boundaries. This script discovers ``<input-root>/seg_NN/batch_M/`` directories
and runs ``aggregate_doc_text`` (from ``_shared.py``) in parallel via
``multiprocessing.Pool``, writing each batch's output to a uniquely-named
parquet file under ``<output-dir>/``.

The downstream identify (``exact_dedup.py --skip-aggregate --skip-remove``)
can then point its ``--cache`` at the parent directory and Curator's
``FilePartitioningStage`` will treat all per-batch parquets as one dataset.

Example
-------
    python aggregate_all.py \\
        --input-root /scratch/.../stage2/cc_main_2025_26 \\
        --output-dir /scratch/.../stage2a_exact_dedup/cache/aggregate \\
        --workers 60
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

# Allow running as a standalone script: `python dedup/aggregate_all.py ...`
sys.path.insert(0, str(Path(__file__).parent))
from _shared import TEXT_SOURCES, aggregate_doc_text  # noqa: E402


def _aggregate_one_batch(args: tuple[str, str, str]) -> tuple[str, int, int, float]:
    """Worker: aggregate one (seg, batch) directory's idx_* shards.

    Returns (batch_dir_str, n_docs, n_dropped, elapsed_s).
    """
    batch_dir_str, out_path_str, text_source = args
    batch_dir = Path(batch_dir_str)
    out_path = Path(out_path_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    shards = sorted(p for p in batch_dir.glob("idx_*") if p.is_dir())
    if not shards:
        return (batch_dir_str, 0, 0, 0.0)

    t0 = time.time()
    n_docs, n_dropped = aggregate_doc_text(shards, out_path, text_source)
    return (batch_dir_str, n_docs, n_dropped, time.time() - t0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-root", required=True, type=Path,
                    help="Stage 2 root with seg_NN/batch_M/ layout (e.g. .../stage2/cc_main_2025_26)")
    ap.add_argument("--output-dir", required=True, type=Path,
                    help="Where to write seg_NN_batch_M.parquet files (one per batch)")
    ap.add_argument("--text-source", choices=list(TEXT_SOURCES), default="metadata",
                    help="Per-doc text source; matches exact_dedup/fuzzy_dedup default.")
    ap.add_argument("--workers", type=int, default=60,
                    help="mp.Pool worker count. Set to ~num cores on the node.")
    args = ap.parse_args()

    args.input_root = args.input_root.resolve()
    args.output_dir = args.output_dir.resolve()

    # Discover all (seg, batch) directories under input_root
    batches: list[tuple[str, str, str]] = []
    for seg_dir in sorted(args.input_root.glob("seg_*")):
        if not seg_dir.is_dir():
            continue
        for batch_dir in sorted(seg_dir.glob("batch_*")):
            if not batch_dir.is_dir():
                continue
            out_path = args.output_dir / f"{seg_dir.name}_{batch_dir.name}.parquet"
            batches.append((str(batch_dir), str(out_path), args.text_source))

    print("=== aggregate_all ===", flush=True)
    print(f"  input root:  {args.input_root}")
    print(f"  output dir:  {args.output_dir}")
    print(f"  workers:     {args.workers}")
    print(f"  batches:     {len(batches)}")
    print(f"  text source: {args.text_source}")
    print(flush=True)

    if not batches:
        print("ERROR: no seg_*/batch_* dirs found under input-root", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_done = 0
    total_docs = 0
    total_dropped = 0

    ctx = mp.get_context("fork")
    with ctx.Pool(processes=args.workers) as pool:
        for result in pool.imap_unordered(_aggregate_one_batch, batches, chunksize=1):
            batch_dir_str, n_docs, n_dropped, elapsed = result
            n_done += 1
            total_docs += n_docs
            total_dropped += n_dropped
            now = time.time()
            bd = Path(batch_dir_str)
            print(
                f"[{n_done:>3}/{len(batches)}] {bd.parent.name}/{bd.name} "
                f"docs={n_docs:>7,} dropped={n_dropped:>6,} ({elapsed:.1f}s)  "
                f"total_elapsed={now-t0:.0f}s rate={n_done/(now-t0):.2f}/s",
                flush=True,
            )

    total_elapsed = time.time() - t0
    print()
    print(f"=== DONE in {total_elapsed/60:.1f} min ===")
    print(f"  batches processed:    {n_done}")
    print(f"  total docs aggregated:{total_docs:,}")
    print(f"  total dropped (null): {total_dropped:,}")
    print(f"  output files:         {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
