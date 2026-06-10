"""Remove duplicate sample_ids from all (seg, batch) groups in parallel on one CPU node.

Mirrors ``aggregate_all.py`` for Phase 4. Reads
``duplicate_sample_ids.json`` (produced by Phase 3 of ``exact_dedup.py``)
and walks every ``<input-root>/seg_NN/batch_M/idx_NNNNN/*.parquet`` shard,
filtering out duplicate rows and writing to
``<output-root>/seg_NN/batch_M/idx_NNNNN/<same_name>.parquet``.

Uses ``multiprocessing.Pool`` with ``fork`` start method so the duplicate
sample_id ``pa.Array`` is shared via copy-on-write rather than pickled per
task.

Example
-------
    python remove_all.py \\
        --input-root  /scratch/.../stage2/cc_main_2025_26 \\
        --output-root /scratch/.../stage2a_exact_dedup/output \\
        --dup-ids-json /scratch/.../stage2a_exact_dedup/cache/duplicate_sample_ids.json \\
        --workers 60
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# Sorted numpy byte array of duplicate sample_ids, shared via fork COW.
# Rationale for not using Python set: CPython refcounts live inside each
# string object, so set lookups touch and dirty the refcount page → CoW
# explodes per-worker private memory under fork. numpy arrays have no
# per-element refcounts → pages stay shared. With 170M UUIDs × 36 bytes
# = ~6 GB, this lives once in physical memory regardless of worker count.
# Membership test = vectorized numpy.searchsorted (binary search).
_WORKER_DUP_SORTED: "np.ndarray | None" = None
_UUID_BYTES = 36  # sample_ids in stage 1/2 output are 36-char UUIDs


def _filter_one_idx(args: tuple[str, str]) -> tuple[int, int]:
    """Filter all parquets under one idx_NNNNN directory.

    args = (input_idx_dir, output_idx_dir). Returns (rows_in, rows_out).
    """
    in_idx = Path(args[0])
    out_idx = Path(args[1])
    out_idx.mkdir(parents=True, exist_ok=True)
    dup_sorted = _WORKER_DUP_SORTED
    rows_in = 0
    rows_out = 0
    for f in sorted(in_idx.glob("*.parquet")):
        t = pq.read_table(f)
        rows_in += t.num_rows
        if dup_sorted is not None and len(dup_sorted) > 0:
            # Vectorized binary search: convert column to fixed-width byte array,
            # searchsorted into the sorted dup_sorted, check for exact match.
            sids_np = np.array(t.column("sample_id").to_pylist(), dtype=f"|S{_UUID_BYTES}")
            idx = np.searchsorted(dup_sorted, sids_np)
            in_bounds = idx < len(dup_sorted)
            # Where idx is in bounds, check equality; otherwise force False
            safe_idx = np.where(in_bounds, idx, 0)
            found = in_bounds & (dup_sorted[safe_idx] == sids_np)
            keep = pa.array(~found, type=pa.bool_())
            t = t.filter(keep)
        rows_out += t.num_rows
        pq.write_table(t, out_idx / f.name)
    return rows_in, rows_out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-root", required=True, type=Path,
                    help="Stage 2 root with seg_NN/batch_M/idx_NNNNN/ layout")
    ap.add_argument("--output-root", required=True, type=Path,
                    help="Where to write filtered shards (mirrors input layout)")
    ap.add_argument("--dup-ids-json", required=True, type=Path,
                    help="Path to duplicate_sample_ids.json from Phase 3")
    ap.add_argument("--workers", type=int, default=60,
                    help="mp.Pool worker count (default 60)")
    args = ap.parse_args()

    args.input_root = args.input_root.resolve()
    args.output_root = args.output_root.resolve()
    args.dup_ids_json = args.dup_ids_json.resolve()

    print("=== remove_all ===", flush=True)
    print(f"  input root  : {args.input_root}")
    print(f"  output root : {args.output_root}")
    print(f"  dup-ids JSON: {args.dup_ids_json}")
    print(f"  workers     : {args.workers}")

    # Load duplicate sample_ids (already a sorted list per exact_dedup.py:json.dump(sorted(...)))
    t_load = time.time()
    with open(args.dup_ids_json) as f:
        dup_ids = json.load(f)
    if not isinstance(dup_ids, list):
        print("ERROR: --dup-ids-json must contain a JSON list of strings", file=sys.stderr)
        return 2
    print(f"  loaded {len(dup_ids):,} duplicate sample_ids in {time.time()-t_load:.1f}s")

    # Build a sorted numpy fixed-width byte array in the parent. workers
    # inherit it via COW (one physical copy). Python set would CoW-explode
    # because each CPython string carries a refcount that gets dirtied on
    # every lookup. numpy bytes have no refcounts → pages stay shared.
    print(f"  building shared sorted numpy dup array in parent ...")
    t_build = time.time()
    # Sanity-check the UUID width assumption on a small sample
    bad = [s for s in dup_ids[:1000] if len(s) != _UUID_BYTES]
    if bad:
        print(f"  WARNING: sample_ids are not all {_UUID_BYTES} chars; first bad: {bad[:3]}", file=sys.stderr)
    arr = np.array(dup_ids, dtype=f"|S{_UUID_BYTES}")
    arr.sort()
    global _WORKER_DUP_SORTED
    _WORKER_DUP_SORTED = arr
    # Free the Python list
    del dup_ids, arr
    import gc; gc.collect()
    print(f"  built sorted numpy array ({len(_WORKER_DUP_SORTED):,} elements, "
          f"{_WORKER_DUP_SORTED.nbytes/1024**3:.1f} GiB) in {time.time()-t_build:.1f}s")

    # Discover all (seg, batch, idx) work items
    work_items: list[tuple[str, str]] = []
    for seg_dir in sorted(args.input_root.glob("seg_*")):
        if not seg_dir.is_dir():
            continue
        for batch_dir in sorted(seg_dir.glob("batch_*")):
            if not batch_dir.is_dir():
                continue
            for idx_dir in sorted(batch_dir.glob("idx_*")):
                if not idx_dir.is_dir():
                    continue
                rel = idx_dir.relative_to(args.input_root)
                out_idx = args.output_root / rel
                work_items.append((str(idx_dir), str(out_idx)))
    print(f"  work items  : {len(work_items):,} idx_* dirs", flush=True)

    if not work_items:
        print("ERROR: no idx_* shards found under input-root", file=sys.stderr)
        return 2

    args.output_root.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rows_in_total = 0
    rows_out_total = 0
    n_done = 0
    print_every = max(1, len(work_items) // 50)

    # fork start method — workers inherit _WORKER_DUP_ARR via COW from the parent
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=args.workers) as pool:
        for rows_in, rows_out in pool.imap_unordered(_filter_one_idx, work_items, chunksize=4):
            rows_in_total += rows_in
            rows_out_total += rows_out
            n_done += 1
            if n_done % print_every == 0 or n_done == len(work_items):
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 1e-3)
                print(
                    f"[{n_done:>6}/{len(work_items)}] elapsed={elapsed:.0f}s "
                    f"rate={rate:.1f} idx/s  rows_in={rows_in_total:,} rows_out={rows_out_total:,}",
                    flush=True,
                )

    elapsed = time.time() - t0
    print()
    print(f"=== DONE in {elapsed/60:.1f} min ===")
    drop_pct = (rows_in_total - rows_out_total) / max(rows_in_total, 1) * 100
    print(f"  rows in :  {rows_in_total:,}")
    print(f"  rows out:  {rows_out_total:,}")
    print(f"  dropped :  {rows_in_total - rows_out_total:,}  ({drop_pct:.2f}%)")
    print(f"  output  :  {args.output_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
