"""Shared phases for the InterleavedParquet dedup scripts (exact + fuzzy).

The Nemotron-CC reference recipes (NVIDIA-NeMo/Nemotron:
``src/nemotron/recipes/data/curation/nemotron-cc/step_2[ab]-*_dedup.py``)
consume one-row-per-doc parquet/jsonl with a ``text`` field. Our pipeline
emits InterleavedParquet (many rows per doc: metadata + text + image), so we
add an aggregate phase that projects each doc's metadata row down to
``(sample_id, text)`` before handing off to Curator's deduplication workflows.

Both exact and fuzzy dedup share the surrounding phases — only the
identification step differs:

  1. AGGREGATE  (CPU)   InterleavedParquet → aggregate/*.parquet   [shared]
  2. IDENTIFY   (GPU)   Curator ExactDeduplicationWorkflow         [exact only]
                    OR  Curator FuzzyDeduplicationWorkflow         [fuzzy only]
  3. RESOLVE    (CPU)   int dup IDs → original sample_ids          [shared]
  4. REMOVE     (CPU)   mp.Pool filter → output shards             [shared]

Phase 4 uses ``multiprocessing.Pool`` rather than Curator's
``TextDuplicatesRemovalWorkflow`` because the latter was empirically ~10×
slower at single-node, 9.8 K-task scale (Curator's RayActorPoolExecutor runs
stages sequentially and per-task Ray + plasma overhead dwarfs the actual
per-file work). See ``dedup_perf_results.md`` in the project memory for the
measurement.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


AGGREGATE_FILE = "aggregate.parquet"


# ---------- Phase 1: aggregate ------------------------------------------------

def find_shards(input_dir: Path) -> list[Path]:
    """Return sorted idx_* directories under input_dir.

    Tolerates a flat Parquet directory too (treats it as a single shard) — useful
    for quick smoke tests against a single batch.
    """
    shard_dirs = sorted(p for p in input_dir.glob("idx_*") if p.is_dir())
    if shard_dirs:
        return shard_dirs
    if any(input_dir.glob("*.parquet")):
        return [input_dir]
    msg = f"No idx_NNNNN/*.parquet or flat *.parquet found under {input_dir}"
    raise FileNotFoundError(msg)


TEXT_SOURCES = ("metadata", "text_rows", "interleaved", "concat")


def aggregate_doc_text(
    shards: list[Path],
    out_path: Path,
    text_source: str,
) -> tuple[int, int]:
    """Reduce many-rows-per-doc InterleavedParquet shards to one-row-per-doc.

    Output schema: (sample_id, text). One row per doc that survives null-text
    filtering. Used as input to the cuDF ExactDedup / FuzzyDedup workflows.

    text_source values:
      "metadata"  — take ``text_content`` from each doc's metadata row
                    (modality == "metadata"). This is the Resiliparse-extracted
                    full-doc text from Stage 1. ~2.6 KB / doc mean; ~9% null.
                    Catches whole-doc Resiliparse-clean duplicates.
      "text_rows" — concatenate ``text_content`` across all rows where
                    modality == "text", sorted by position. This is the
                    magic-html (or trafilatura fallback) paragraphs — exactly
                    the interleaved text the downstream multimodal model sees.
      "interleaved" — text rows AND image URLs (from source_ref.url) interleaved
                    by position. Each image becomes one token "IMG:<url>". So
                    two docs are exact-equal iff they have identical paragraphs
                    AND identical image URLs in the same positions. Useful for
                    multimodal dedup where image identity matters.
      "concat"    — metadata + text rows concatenated (legacy; has redundancy).

    Returns (num_docs_written, num_dropped_null_text).
    """
    if text_source not in TEXT_SOURCES:
        msg = f"text_source must be one of {TEXT_SOURCES}, got {text_source!r}"
        raise ValueError(msg)

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

        cols = ["sample_id", "position", "modality", "text_content"]

        if text_source == "metadata":
            # Cheap path: just the metadata row (one per doc, already aggregated).
            tables = []
            for f in parquet_files:
                t = pq.read_table(f, columns=cols)
                mask = pc.equal(t["modality"], "metadata")
                tables.append(t.filter(mask).select(["sample_id", "text_content"]))
            tbl = pa.concat_tables(tables)

        elif text_source == "text_rows":
            # Magic-html/trafilatura paragraphs: group by sample_id, sort by
            # position, join with newline. This is what the multimodal model
            # sees interleaved with image rows (minus the image rows themselves).
            tables = []
            for f in parquet_files:
                t = pq.read_table(f, columns=cols)
                mask = pc.equal(t["modality"], "text")
                tables.append(t.filter(mask).select(["sample_id", "position", "text_content"]))
            tbl = pa.concat_tables(tables)
            tbl = tbl.sort_by([("sample_id", "ascending"), ("position", "ascending")])
            agg = tbl.group_by("sample_id").aggregate([("text_content", "list")])
            joined = pa.array(
                ["\n".join([x for x in chunk if x]) for chunk in agg["text_content_list"].to_pylist()],
                type=pa.string(),
            )
            tbl = pa.Table.from_arrays([agg["sample_id"], joined], names=["sample_id", "text_content"])

        elif text_source == "interleaved":
            # Text rows + image URLs interleaved by position. Each image row
            # contributes one "IMG:<url>" token. Two docs are exact-equal iff
            # they have identical paragraphs AND identical image URLs in the
            # same positions. JSON parsing is done in Python since pyarrow
            # can't natively decode JSON cells.
            cols_il = ["sample_id", "position", "modality", "text_content", "source_ref"]
            tables = []
            for f in parquet_files:
                t = pq.read_table(f, columns=cols_il)
                mask = pc.is_in(t["modality"], value_set=pa.array(["text", "image"]))
                tables.append(t.filter(mask))
            tbl = pa.concat_tables(tables)
            tbl = tbl.sort_by([("sample_id", "ascending"), ("position", "ascending")])

            sids = tbl["sample_id"].to_pylist()
            mods = tbl["modality"].to_pylist()
            texts = tbl["text_content"].to_pylist()
            refs = tbl["source_ref"].to_pylist()

            doc_parts: dict[str, list[str]] = {}
            for sid, mod, txt, ref in zip(sids, mods, texts, refs):
                if mod == "text":
                    if txt:
                        doc_parts.setdefault(sid, []).append(txt)
                elif mod == "image" and ref:
                    url = ""
                    try:
                        d = json.loads(ref)
                        url = d.get("url") or d.get("src") or ""
                    except (ValueError, TypeError):
                        pass
                    if url:
                        doc_parts.setdefault(sid, []).append("IMG:" + url)

            out_sids = list(doc_parts.keys())
            out_texts = ["\n".join(doc_parts[s]) for s in out_sids]
            tbl = pa.Table.from_arrays(
                [pa.array(out_sids, type=pa.string()), pa.array(out_texts, type=pa.string())],
                names=["sample_id", "text_content"],
            )

        else:  # "concat" — legacy; metadata + text rows.
            tables = []
            for f in parquet_files:
                t = pq.read_table(f, columns=cols)
                mask = pc.is_in(t["modality"], value_set=pa.array(["metadata", "text"]))
                tables.append(t.filter(mask))
            tbl = pa.concat_tables(tables)
            tbl = tbl.sort_by([("sample_id", "ascending"), ("position", "ascending")])
            agg = tbl.group_by("sample_id").aggregate([("text_content", "list")])
            joined = pa.array(
                ["\n".join([x for x in chunk if x]) for chunk in agg["text_content_list"].to_pylist()],
                type=pa.string(),
            )
            tbl = pa.Table.from_arrays([agg["sample_id"], joined], names=["sample_id", "text_content"])

        before = tbl.num_rows
        tbl = tbl.filter(pc.and_(pc.is_valid(tbl["text_content"]), pc.greater(pc.utf8_length(tbl["text_content"]), 0)))
        n_dropped_null += before - tbl.num_rows

        tbl = tbl.rename_columns(["sample_id", "text"]).cast(schema)
        if writer is None:
            writer = pq.ParquetWriter(out_path, schema)
        writer.write_table(tbl)
        n_docs += tbl.num_rows

    if writer is not None:
        writer.close()
    return n_docs, n_dropped_null


# Back-compat alias — older scripts imported the old name.
aggregate_metadata_rows = aggregate_doc_text


# ---------- Phase 3: resolve int IDs → sample_ids -----------------------------

def resolve_duplicate_sample_ids(
    aggregate_path: Path,
    duplicates_dir: Path,
    id_generator_path: Path,
    *,
    id_column: str = "_curator_dedup_id",
) -> set[str]:
    """Map Curator's integer dup IDs back to original ``sample_id`` strings.

    Curator's IdGenerator assigns each batch a contiguous int ID range. A
    batch is identified by ``uuid5(NAMESPACE_URL, ";".join(file_paths))`` so
    we can match each aggregate file to its assigned range and read sample_ids
    in id-range order. Multi-file batches are not yet supported (set
    ``--input-blocksize`` smaller than any aggregate file to force 1 file per
    batch).

    aggregate_path can be either the single aggregate.parquet (legacy 1-file
    case) or any path whose parent is the directory of aggregate parquet files.

    id_column: column name holding integer dup IDs. Defaults to Curator's
        ``_curator_dedup_id`` (matches both exact + fuzzy workflows).
    """
    import uuid

    with open(id_generator_path) as f:
        id_gen = json.load(f)
    registry = id_gen["batch_registry"]  # {batch_hash: [min_id, max_id]}
    next_id = id_gen["next_id"]

    agg_dir = aggregate_path.parent
    agg_files = sorted(str(p) for p in agg_dir.glob("*.parquet"))
    if not agg_files:
        msg = f"No aggregate parquet files under {agg_dir}"
        raise FileNotFoundError(msg)
    print(f"[resolve] {len(agg_files)} aggregate file(s) under {agg_dir}")
    print(f"[resolve] id_generator: next_id={next_id:,}, {len(registry)} batch(es) in registry")

    # Match each aggregate file to a single-file batch via uuid5 hash
    def hash_single(filepath: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, filepath))

    file_to_range: dict[str, tuple[int, int]] = {}
    for f in agg_files:
        h = hash_single(f)
        rng = registry.get(h)
        if rng is not None:
            file_to_range[f] = (rng[0], rng[1])

    if len(file_to_range) != len(agg_files):
        unmatched = [f for f in agg_files if hash_single(f) not in registry][:3]
        msg = (
            f"Only matched {len(file_to_range)}/{len(agg_files)} aggregate files via single-file hash. "
            f"Registry has {len(registry)} batches. First unmatched files: {unmatched}. "
            f"This means Curator grouped files into multi-file batches. "
            f"Re-run identify with --input-blocksize smaller than min(aggregate_file_size) "
            f"to force one file per batch."
        )
        raise NotImplementedError(msg)
    print(f"[resolve] every aggregate file matched a single-file batch hash ✓")

    # Sort files by min_id (the order in which IDs were assigned)
    files_sorted = sorted(file_to_range.items(), key=lambda kv: kv[1][0])

    # Verify contiguity: first range starts at 0, every range starts where the previous ended + 1
    expected_min = 0
    for f, (min_id, max_id) in files_sorted:
        if min_id != expected_min:
            msg = f"Gap in id ranges: expected min={expected_min}, got {min_id} for {f}"
            raise RuntimeError(msg)
        expected_min = max_id + 1
    if expected_min != next_id:
        msg = f"Last range ended at {expected_min - 1} but id_generator next_id={next_id}"
        raise RuntimeError(msg)

    # Read all sample_ids in id-range order → flat list, index = global int_id
    print(f"[resolve] reading sample_ids from {len(files_sorted)} aggregate files (id-range order)...")
    sample_ids: list[str] = []
    for f, (min_id, max_id) in files_sorted:
        sids = pq.read_table(f, columns=["sample_id"])["sample_id"].to_pylist()
        expected = max_id - min_id + 1
        if len(sids) != expected:
            msg = f"{f}: read {len(sids)} sample_ids but range covers {expected}"
            raise RuntimeError(msg)
        sample_ids.extend(sids)
    n = len(sample_ids)
    print(f"[resolve] loaded {n:,} sample_ids total")

    # Read all duplicate int IDs
    dup_files = sorted(duplicates_dir.rglob("*.parquet"))
    dup_files = [f for f in dup_files if "id_generator" not in f.name.lower()]
    if not dup_files:
        print(f"[resolve] WARNING: no duplicate parquet files under {duplicates_dir} (zero duplicates?)")
        return set()

    dup_int_ids: list[int] = []
    for f in dup_files:
        t = pq.read_table(f)
        if id_column in t.schema.names:
            dup_int_ids.extend(t[id_column].to_pylist())
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

# Pool `fork` on Linux gives workers copy-on-write access to module globals,
# so we keep the dup-id Arrow array there to avoid per-task pickling.
_WORKER_DUP_ARR: pa.Array | None = None


def _init_worker(dup_ids_sorted: list[str]) -> None:
    """Pool initializer: build the pa.Array once per worker process."""
    global _WORKER_DUP_ARR
    _WORKER_DUP_ARR = pa.array(dup_ids_sorted, type=pa.string())


def _filter_one_shard(args: tuple[str, str]) -> tuple[int, int]:
    """Worker entry: filter one shard's parquet files into the output dir.

    args = (shard_dir_str, output_shard_dir_str). Returns (rows_in, rows_out).
    """
    shard_dir = Path(args[0])
    out_shard = Path(args[1])
    out_shard.mkdir(parents=True, exist_ok=True)
    dup_arr = _WORKER_DUP_ARR
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

    Validated ~6.5× speedup vs single-process at 1-segment scale
    (multiproc-16: ~4.6 min for 1000 shards). See ``dedup_perf_results.md``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if n_workers <= 1:
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
