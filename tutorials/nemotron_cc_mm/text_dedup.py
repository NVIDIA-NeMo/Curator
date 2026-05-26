"""Standalone fuzzy text dedup over InterleavedParquet outputs.

Input: a directory of shards (`idx_NNNNN/*.parquet`) produced by the
`extract` or `text_filter` preset.  Each shard is many rows-per-doc;
this script reduces to one row per doc (using the per-doc metadata row's
``text_content``, which is the Resiliparse-extracted full doc text by default),
runs NeMo Curator's GPU-accelerated fuzzy MinHash + LSH + connected-components
pipeline, and emits the duplicate ``sample_id``s.  Optionally writes filtered
output shards with duplicate docs removed.

Pipeline phases
---------------
1. **Aggregate**:    read all shards → project (sample_id, text) → single Parquet.
2. **Dedup**:        FuzzyDeduplicationWorkflow → duplicate integer IDs.
3. **Resolve**:      translate integer IDs → original sample_ids.
4. **Remove**:       per-shard filter via ``multiprocessing.Pool(--workers)``,
                     write to ``--output``, preserving the ``idx_NNNNN/`` layout
                     so downstream chained-pipeline stages still work.

Each phase can be skipped by `--skip-aggregate / --skip-dedup / --skip-remove`
so partial reruns are cheap (intermediates live under `--cache`).

Hardware
--------
Phases 1, 3, 4 are CPU-only.  Phase 2 requires a GPU (Curator's MinHash uses
cuDF).  Submit this script on a single GPU node — Curator handles its own
intra-node parallelism via Ray.

Example
-------
    python text_dedup.py \\
        --input  /scratch/.../out/1segment_chain/02_text \\
        --output /scratch/.../out/1segment_chain/02b_text_dedup \\
        --cache  /scratch/.../out/1segment_chain/02b_text_dedup_cache \\
        --text-source metadata \\
        --char-ngrams 24 --num-bands 20 --minhashes-per-band 13
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


# ---------- Phase 1: aggregate ------------------------------------------------

AGGREGATE_FILE = "aggregate.parquet"


def find_shards(input_dir: Path) -> list[Path]:
    """Return sorted idx_* directories under input_dir.  Tolerates a flat
    Parquet directory too (treats it as a single shard)."""
    shard_dirs = sorted(p for p in input_dir.glob("idx_*") if p.is_dir())
    if shard_dirs:
        return shard_dirs
    if any(input_dir.glob("*.parquet")):
        return [input_dir]
    msg = f"No idx_NNNNN/*.parquet or flat *.parquet found under {input_dir}"
    raise FileNotFoundError(msg)


def aggregate_metadata_rows(
    shards: list[Path],
    out_path: Path,
    text_source: str,
) -> tuple[int, int]:
    """Reduce many-rows-per-doc shards to one-row-per-doc Parquet.

    text_source=metadata: take ``text_content`` from each doc's metadata row
        (position == -1).  This is the Resiliparse full-doc text when the
        extract preset has ``resiliparse_text: true`` (default).
    text_source=concat: concatenate ``text_content`` across all text rows
        (modality == "text") per sample_id.  Slower; higher recall.

    Returns (num_docs, num_dropped_null_text).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer: pq.ParquetWriter | None = None
    n_docs = 0
    n_dropped_null = 0
    schema = pa.schema([
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
    ])

    for shard_dir in shards:
        parquet_files = sorted(shard_dir.glob("*.parquet"))
        if not parquet_files:
            continue

        if text_source == "metadata":
            # cheap path: read only metadata rows (one per doc) with pushdown
            cols = ["sample_id", "position", "modality", "text_content"]
            tables = []
            for f in parquet_files:
                t = pq.read_table(f, columns=cols)
                mask = pc.equal(t["modality"], "metadata")
                tables.append(t.filter(mask).select(["sample_id", "text_content"]))
            tbl = pa.concat_tables(tables)
        else:
            # concat path: group all text rows by sample_id, join paragraphs
            cols = ["sample_id", "position", "modality", "text_content"]
            tables = []
            for f in parquet_files:
                t = pq.read_table(f, columns=cols)
                mask = pc.is_in(t["modality"], value_set=pa.array(["metadata", "text"]))
                tables.append(t.filter(mask))
            tbl = pa.concat_tables(tables)
            # sort by (sample_id, position) so concat reflects DOM order
            tbl = tbl.sort_by([("sample_id", "ascending"), ("position", "ascending")])
            # group_by + list-agg in pyarrow gives us the joined text
            agg = tbl.group_by("sample_id").aggregate([("text_content", "list")])
            joined = pa.array(
                ["\n".join([x for x in chunk if x]) for chunk in agg["text_content_list"].to_pylist()],
                type=pa.string(),
            )
            tbl = pa.Table.from_arrays([agg["sample_id"], joined], names=["sample_id", "text_content"])

        # drop null-text rows (Resiliparse failed, or all text rows were null)
        before = tbl.num_rows
        tbl = tbl.filter(pc.and_(pc.is_valid(tbl["text_content"]), pc.greater(pc.utf8_length(tbl["text_content"]), 0)))
        n_dropped_null += before - tbl.num_rows

        # rename text_content → text and ensure schema match
        tbl = tbl.rename_columns(["sample_id", "text"]).cast(schema)
        if writer is None:
            writer = pq.ParquetWriter(out_path, schema)
        writer.write_table(tbl)
        n_docs += tbl.num_rows

    if writer is not None:
        writer.close()
    return n_docs, n_dropped_null


# ---------- Phase 2: fuzzy dedup ----------------------------------------------

def run_fuzzy_dedup(
    aggregate_path: Path,
    cache_dir: Path,
    duplicates_dir: Path,
    *,
    char_ngrams: int,
    num_bands: int,
    minhashes_per_band: int,
    input_blocksize: str,
) -> None:
    """Run FuzzyDeduplicationWorkflow.  Requires GPU."""
    # Imports here so the script can do --skip-dedup on a CPU node.
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
    print(f"[dedup] FuzzyDeduplicationWorkflow starting (B={num_bands}, R={minhashes_per_band}, ngrams={char_ngrams})")
    print(f"[dedup]   input    = {aggregate_path}")
    print(f"[dedup]   cache    = {cache_dir}")
    print(f"[dedup]   dup-ids  = {duplicates_dir}")
    t0 = time.time()
    wf.run()
    print(f"[dedup] done in {time.time() - t0:.1f}s")


# ---------- Phase 3: resolve int IDs → sample_ids -----------------------------

def resolve_duplicate_sample_ids(
    aggregate_path: Path,
    duplicates_dir: Path,
    id_generator_path: Path,
) -> set[str]:
    """Map Curator's integer dup IDs back to original ``sample_id``s.

    Curator's IdGenerator assigns each input file a contiguous integer range
    ``[min_id, max_id]``.  When the aggregate is one file (our case), all
    integer dup IDs are row indices in that file → sample_id = row's sample_id.
    Multi-file aggregates work the same way via per-file offsets in the
    id_generator JSON.
    """
    with open(id_generator_path) as f:
        id_gen = json.load(f)

    # id_gen schema: {"batch_registry": {hash: (min_id, max_id)}, ...}
    # we built the aggregate as a single file → there's exactly one entry,
    # but support multi-file for forward-compat.
    file_ranges: list[tuple[int, int, Path]] = []
    # In Curator the registry maps batch_hash → (min, max); the hash is over
    # filenames.  But there's a simpler invariant we can rely on: integer IDs
    # are assigned in input-file order.  So we re-scan the aggregate file in
    # order and use a flat (row → sample_id) lookup.
    del file_ranges  # noqa: PLW0128

    # Read aggregate sample_ids in order → flat array.  Accept either a
    # single aggregate.parquet (legacy) or multiple sorted chunk files
    # (when aggregate is split to fit GPU memory).  FilePartitioningStage
    # enumerates files in sorted order, so id_generator assigns IDs in the
    # same sorted-name order.
    agg_files = sorted(aggregate_path.parent.glob("*.parquet"))
    if not agg_files:
        msg = f"No aggregate parquet files under {aggregate_path.parent}"
        raise FileNotFoundError(msg)
    sample_ids: list[str] = []
    for f in agg_files:
        sample_ids.extend(pq.read_table(f, columns=["sample_id"])["sample_id"].to_pylist())
    n = len(sample_ids)
    print(f"[resolve] aggregate spans {len(agg_files)} file(s), {n:,} docs total")

    # Read all duplicate int IDs.  Curator writes them under
    # ``duplicates/FuzzyDuplicateIds/*.parquet`` (nested subdir), but for
    # forward-compat we recurse.
    dup_files = sorted(duplicates_dir.rglob("*.parquet"))
    # filter out the id_generator.json's sibling files (none today, but be safe)
    dup_files = [f for f in dup_files if "id_generator" not in f.name.lower()]
    if not dup_files:
        print(f"[resolve] WARNING: no duplicate parquet files under {duplicates_dir} (zero duplicates?)")
        return set()

    dup_int_ids: list[int] = []
    for f in dup_files:
        t = pq.read_table(f)
        # Curator's CURATOR_DEDUP_ID_STR = "_curator_dedup_id"
        if "_curator_dedup_id" in t.schema.names:
            dup_int_ids.extend(t["_curator_dedup_id"].to_pylist())
        elif "id" in t.schema.names:
            dup_int_ids.extend(t["id"].to_pylist())
        else:
            msg = f"Don't recognize ID column in {f}: columns={t.schema.names}"
            raise RuntimeError(msg)
    print(f"[resolve] {len(dup_int_ids):,} duplicate int IDs read")

    out: set[str] = set()
    for int_id in dup_int_ids:
        if 0 <= int_id < n:
            out.add(sample_ids[int_id])
        else:
            msg = f"int_id {int_id} out of aggregate range [0, {n}); id_generator mismatch?"
            raise RuntimeError(msg)
    print(f"[resolve] {len(out):,} distinct duplicate sample_ids")
    return out


# ---------- Phase 4: remove from input shards --------------------------------

# Pool ``fork`` on Linux gives workers copy-on-write access to module globals,
# so we keep the dup-id Arrow array there to avoid per-task pickling.
_WORKER_DUP_ARR: pa.Array | None = None


def _init_worker(dup_ids_sorted: list[str]) -> None:
    """Pool initializer: build the pa.Array once per worker process."""
    global _WORKER_DUP_ARR
    _WORKER_DUP_ARR = pa.array(dup_ids_sorted, type=pa.string())


def _filter_one_shard(args: tuple[str, str]) -> tuple[int, int]:
    """Worker entry: filter one shard's parquet files into the output dir.

    args = (shard_dir_str, output_shard_dir_str).  Returns (rows_in, rows_out).
    """
    shard_dir = Path(args[0])
    out_shard = Path(args[1])
    out_shard.mkdir(parents=True, exist_ok=True)
    dup_arr = _WORKER_DUP_ARR  # local ref for speed
    rows_in = 0
    rows_out = 0
    for f in sorted(shard_dir.glob("*.parquet")):
        t = pq.read_table(f)
        rows_in += t.num_rows
        if dup_arr is not None and len(dup_arr) > 0:
            keep_mask = pc.invert(pc.is_in(t["sample_id"], value_set=dup_arr))
            t = t.filter(keep_mask)
        rows_out += t.num_rows
        pq.write_table(t, out_shard / f.name)
    return rows_in, rows_out


def remove_duplicates(
    input_shards: list[Path],
    output_dir: Path,
    duplicate_sample_ids: set[str],
    *,
    n_workers: int = 16,
) -> tuple[int, int]:
    """Per-shard removal across N worker processes.

    Preserves input shard layout (writes ``output_dir/idx_NNNNN/<same_name>.parquet``).
    ``n_workers=1`` runs in-process; higher values use ``multiprocessing.Pool``
    with the ``fork`` start method so the dup-id list is shared via
    copy-on-write rather than pickled per task.
    """
    import time
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if n_workers <= 1:
        # In-process — no Pool overhead.
        dup_arr = pa.array(sorted(duplicate_sample_ids), type=pa.string())
        rows_in = 0
        rows_out = 0
        for shard_dir in input_shards:
            out_shard = output_dir / shard_dir.name if shard_dir.name.startswith("idx_") else output_dir
            out_shard.mkdir(parents=True, exist_ok=True)
            for f in sorted(shard_dir.glob("*.parquet")):
                t = pq.read_table(f)
                rows_in += t.num_rows
                if duplicate_sample_ids:
                    keep_mask = pc.invert(pc.is_in(t["sample_id"], value_set=dup_arr))
                    t = t.filter(keep_mask)
                rows_out += t.num_rows
                pq.write_table(t, out_shard / f.name)
        elapsed = time.time() - t0
        print(f"[remove n_workers=1] {rows_in:,} rows in → {rows_out:,} rows out "
              f"({(rows_in - rows_out) / max(rows_in, 1):.1%} dropped) in {elapsed:.1f}s")
        return rows_in, rows_out

    # Multiprocessing pool path.
    import multiprocessing as mp

    dup_sorted = sorted(duplicate_sample_ids)
    work_items = []
    for shard_dir in input_shards:
        out_shard_name = shard_dir.name if shard_dir.name.startswith("idx_") else ""
        out_shard = output_dir / out_shard_name if out_shard_name else output_dir
        work_items.append((str(shard_dir), str(out_shard)))

    rows_in_total = 0
    rows_out_total = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=n_workers, initializer=_init_worker, initargs=(dup_sorted,)) as pool:
        done = 0
        for rows_in, rows_out in pool.imap_unordered(_filter_one_shard, work_items, chunksize=4):
            rows_in_total += rows_in
            rows_out_total += rows_out
            done += 1
            if done % max(1, len(work_items) // 20) == 0 or done == len(work_items):
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-3)
                print(f"[remove n_workers={n_workers}] {done}/{len(work_items)} shards "
                      f"({elapsed:.0f}s elapsed, {rate:.1f} shards/s)")

    elapsed = time.time() - t0
    print(f"[remove n_workers={n_workers}] {rows_in_total:,} rows in → "
          f"{rows_out_total:,} rows out "
          f"({(rows_in_total - rows_out_total) / max(rows_in_total, 1):.1%} dropped) "
          f"in {elapsed:.1f}s total")
    return rows_in_total, rows_out_total


# ---------- driver ------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path, help="Directory with idx_*/*.parquet shards")
    ap.add_argument("--output", type=Path, help="Where to write filtered shards (skip with --skip-remove)")
    ap.add_argument("--cache", required=True, type=Path, help="Intermediates: aggregate.parquet + MinHash/LSH/CC")
    ap.add_argument("--text-source", choices=["metadata", "concat"], default="metadata",
                    help="Dedup on the per-doc Resiliparse text (metadata) or concat of all text rows.")
    ap.add_argument("--char-ngrams", type=int, default=24)
    ap.add_argument("--num-bands", type=int, default=20)
    ap.add_argument("--minhashes-per-band", type=int, default=13)
    ap.add_argument("--input-blocksize", default="1GiB",
                    help="FuzzyDeduplicationWorkflow read blocksize.")
    ap.add_argument("--skip-aggregate", action="store_true", help="Reuse aggregate.parquet under --cache.")
    ap.add_argument("--skip-dedup", action="store_true", help="Reuse duplicate IDs under --cache/duplicates/.")
    ap.add_argument("--skip-remove", action="store_true", help="Stop after emitting duplicate_sample_ids.json.")
    ap.add_argument("--clear-cache", action="store_true", help="Wipe --cache contents before running.")
    ap.add_argument("--workers", type=int, default=16,
                    help="Number of worker processes for the removal phase "
                         "(default 16; set 1 for in-process / no Pool overhead).")
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

    # Curator's FuzzyDeduplicationWorkflow uses {cache_path: minhashes etc, output_path: duplicates}.
    # We split cache_path → ./cache_minhash/, output_path → ./duplicates/, aggregate → ./aggregate/.
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
        n_docs, n_dropped = aggregate_metadata_rows(shards, aggregate_path, args.text_source)
        print(f"[aggregate] wrote {n_docs:,} docs to {aggregate_path} "
              f"({n_dropped:,} dropped null-text) in {time.time() - t0:.1f}s")
        if n_docs == 0:
            print("[aggregate] ERROR: no docs after aggregation", file=sys.stderr)
            return 2
    else:
        # Accept either the single aggregate.parquet OR any *.parquet files
        # under the aggregate dir (we may have manually split the aggregate
        # into smaller chunks to fit GPU memory).
        existing = sorted(aggregate_dir.glob("*.parquet"))
        print(f"[aggregate] SKIPPED — reusing {len(existing)} parquet file(s) under {aggregate_dir}")
        if not existing:
            print(f"[aggregate] ERROR: no *.parquet found under {aggregate_dir}", file=sys.stderr)
            return 2

    # ---- Phase 2: fuzzy dedup ----------------------------------------------
    if not args.skip_dedup:
        run_fuzzy_dedup(
            aggregate_path=aggregate_path,
            cache_dir=minhash_cache,
            duplicates_dir=duplicates_dir,
            char_ngrams=args.char_ngrams,
            num_bands=args.num_bands,
            minhashes_per_band=args.minhashes_per_band,
            input_blocksize=args.input_blocksize,
        )
    else:
        print(f"[dedup] SKIPPED — reusing {duplicates_dir}")

    # ---- Phase 3: resolve int IDs ------------------------------------------
    if not id_generator_path.exists():
        print(f"[resolve] ERROR: {id_generator_path} not found — did the workflow finish?", file=sys.stderr)
        return 2
    dup_sample_ids = resolve_duplicate_sample_ids(aggregate_path, duplicates_dir, id_generator_path)
    with open(dup_sample_ids_path, "w") as f:
        json.dump(sorted(dup_sample_ids), f)
    print(f"[resolve] wrote {len(dup_sample_ids):,} duplicate sample_ids → {dup_sample_ids_path}")

    # ---- Phase 4: remove ---------------------------------------------------
    if args.skip_remove:
        print("[remove] SKIPPED — duplicate IDs available at",
              dup_sample_ids_path)
        return 0
    if args.output is None:
        print("[remove] ERROR: --output not provided (or pass --skip-remove)", file=sys.stderr)
        return 2

    print(f"[remove] filtering shards (n_workers={args.workers}) → {args.output}")
    remove_duplicates(shards, args.output, dup_sample_ids, n_workers=args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
