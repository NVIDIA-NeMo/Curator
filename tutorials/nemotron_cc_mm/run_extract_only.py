"""Stage-1 extract-only pipeline (one Ray cluster, multiple WARCs).

Stripped-down sibling of ``run_warc_pipeline.py`` that runs only the
extract stages (no text filters, no image stages, no manifest/markers).

Pipeline:
    1. FilePartitioningStage          (one task per input WARC)
    2. DocumentIterateExtractStage    (streams WARC records via
                                      CommonCrawlWarcIterator)
    3. WarcDocumentToInterleavedStage (HTML → InterleavedBatch with the
                                      configured extractor)
    4. InterleavedParquetWriterStage  (writes parquet to --output-path)

Usage:
    # Single WARC
    python run_extract_only.py \\
        --input-path s3://crawl-data/.../00000.warc.gz \\
        --output-path /scratch/.../out_one_warc

    # Many WARCs (comma-separated) — one Ray cluster handles them all
    python run_extract_only.py \\
        --input-path s3://...00000.warc.gz,s3://...00001.warc.gz,... \\
        --output-path /scratch/.../out_many

    # All WARCs from a directory / S3 prefix
    python run_extract_only.py \\
        --input-path s3://crawl-data/CC-MAIN-2025-26/segments/.../warc/ \\
        --output-path /scratch/.../out_segment

Optional concurrency override (avoid Ray Data's default autoscaling
which under- or over-shoots on a single big node):
    --force-workers N            pin to exactly N actors
    --concurrency-min M --max X  pass (M, X) autoscale range to Ray Data
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import ray

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.interleaved.io.writers import InterleavedParquetWriterStage
from nemo_curator.stages.nemotron_cc_mm.extraction import WarcDocumentToInterleavedStage
from nemo_curator.stages.text.download.base.iterator import DocumentIterateExtractStage
from nemo_curator.stages.text.download.common_crawl.warc_iterator import CommonCrawlWarcIterator


# ---------- helpers ----------------------------------------------------------

def _parse_input_paths(raw: str) -> str | list[str]:
    """Accept either a single path, a comma-separated list, or a directory.

    Returns whatever FilePartitioningStage's ``file_paths`` accepts.
    """
    if "," in raw:
        return [p.strip() for p in raw.split(",") if p.strip()]
    return raw


def _resolve_storage_options(input_path: str | list[str]) -> dict:
    """Honor AWS_PROFILE for s3:// inputs (matches run_warc_pipeline.py)."""
    sample = input_path[0] if isinstance(input_path, list) else input_path
    if sample.startswith("s3://"):
        return {"profile": os.environ.get("AWS_PROFILE", "cc")}
    return {}


def _install_concurrency_override(stage_name_substr: str, value) -> None:
    """Monkey-patch Curator's adapter so the matching stage uses ``value``
    (int for fixed count, tuple ``(min, max)`` for autoscale range)
    instead of the default autoscaled range.
    """
    from nemo_curator.backends.ray_data import adapter as adapter_mod

    if not hasattr(adapter_mod, "_calc_orig"):
        adapter_mod._calc_orig = adapter_mod.calculate_concurrency_for_actors_for_stage

    def patched(stage, ignore_head_node=False):
        if stage_name_substr in getattr(stage, "name", ""):
            print(f"[concurrency-patch] {stage.name}: forced concurrency = {value}", flush=True)
            return value
        return adapter_mod._calc_orig(stage, ignore_head_node)

    adapter_mod.calculate_concurrency_for_actors_for_stage = patched


# ---------- pipeline ---------------------------------------------------------

def build_pipeline(args: argparse.Namespace) -> Pipeline:
    """Construct the 4-stage extract-only pipeline."""
    input_paths = _parse_input_paths(args.input_path)
    storage_options = _resolve_storage_options(input_paths)

    pipe = Pipeline(
        name="extract_only",
        description="WARC → InterleavedParquet (extract stage only)",
    )

    # 1. partition input files (one Ray Data task per file by default)
    pipe.add_stage(
        FilePartitioningStage(
            file_paths=input_paths,
            files_per_partition=args.files_per_partition,
            file_extensions=[".gz", ".warc"],
            storage_options=storage_options,
        )
    )

    # 2. iterate WARC records (S3-aware via CommonCrawlWarcIterator)
    pipe.add_stage(
        DocumentIterateExtractStage(
            iterator=CommonCrawlWarcIterator(storage_options=storage_options),
            record_limit=args.record_limit,
            add_filename_column=True,
        )
    )

    # 3. HTML → InterleavedBatch rows
    pipe.add_stage(
        WarcDocumentToInterleavedStage(
            extractor=args.extractor,
            min_text_chars=args.min_text_chars,
            max_text_chars=args.max_text_chars,
            resiliparse_text=args.resiliparse_text,
            max_batch_bytes=args.max_batch_bytes,
        )
    )

    # 4. write parquet
    pipe.add_stage(
        InterleavedParquetWriterStage(
            path=args.output_path,
            mode=args.mode,
        )
    )

    return pipe


# ---------- driver -----------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # I/O
    p.add_argument("--input-path", required=True,
                   help="WARC path: single file, comma-separated list, OR directory / S3 prefix.")
    p.add_argument("--output-path", required=True, type=Path,
                   help="Where to write parquet output.")
    p.add_argument("--mode", default="overwrite",
                   choices=["ignore", "overwrite", "append", "error"])
    p.add_argument("--files-per-partition", type=int, default=1,
                   help="WARCs per Ray Data partition (1 = one actor task per WARC).")
    p.add_argument("--record-limit", type=int, default=None,
                   help="Cap records read per WARC (smoke testing).")

    # Extractor config (matches run_warc_pipeline.py + extract_packed.yaml defaults)
    p.add_argument("--extractor", default="hybrid",
                   choices=["naive", "magic_html", "hybrid"])
    p.add_argument("--min-text-chars", type=int, default=1)
    p.add_argument("--max-text-chars", type=int, default=50_000)
    p.add_argument("--resiliparse-text", action=argparse.BooleanOptionalAction, default=True,
                   help="Also run Resiliparse on each WARC HTML; populates the "
                        "metadata row's text_content (used by downstream dedup).")
    p.add_argument("--max-batch-bytes", type=int, default=64 * 1024 * 1024,
                   help="Per-Arrow-batch byte cap.  Smaller = lower peak memory "
                        "but more output parquets.  Default 64 MiB.")

    # Ray cluster sizing
    p.add_argument("--object-store-gb", type=int, default=32,
                   help="Ray plasma store size for the single local cluster.")
    p.add_argument("--ray-tmp-dir", default=None,
                   help="Override Ray's _temp_dir (default: /tmp/ray_extract_<pid>).")

    # Concurrency control for the actor (extract) stage
    cc = p.add_mutually_exclusive_group()
    cc.add_argument("--force-workers", type=int, default=None,
                    help="Pin extract stage to exactly N actors (int).")
    cc.add_argument("--concurrency-min", type=int, default=None,
                    help="Lower bound for actor autoscale (use with --concurrency-max).")
    p.add_argument("--concurrency-max", type=int, default=None,
                   help="Upper bound for actor autoscale (use with --concurrency-min).")

    args = p.parse_args()

    if (args.concurrency_min is None) != (args.concurrency_max is None):
        p.error("--concurrency-min and --concurrency-max must be supplied together")

    args.output_path = args.output_path.resolve()
    args.output_path.mkdir(parents=True, exist_ok=True)

    # Install concurrency patch BEFORE build_pipeline so adapter sees it.
    if args.force_workers is not None:
        _install_concurrency_override("warc_to_interleaved_extract", args.force_workers)
    elif args.concurrency_min is not None:
        _install_concurrency_override(
            "warc_to_interleaved_extract",
            (args.concurrency_min, args.concurrency_max),
        )

    # Pretty-print configuration
    parsed_input = _parse_input_paths(args.input_path)
    n_inputs = len(parsed_input) if isinstance(parsed_input, list) else 1
    print(f"[init] inputs:           {n_inputs} path(s)")
    if isinstance(parsed_input, list):
        print(f"        first: {parsed_input[0]}")
        if n_inputs > 1:
            print(f"        last:  {parsed_input[-1]}")
    else:
        print(f"        {parsed_input}")
    print(f"[init] output:           {args.output_path}")
    print(f"[init] extractor:        {args.extractor}  resiliparse_text={args.resiliparse_text}")
    print(f"[init] max_batch_bytes:  {args.max_batch_bytes // (1024*1024)} MiB")
    print(f"[init] files_per_part:   {args.files_per_partition}")
    print(f"[init] object_store:     {args.object_store_gb} GB")
    if args.force_workers is not None:
        print(f"[init] concurrency:      fixed {args.force_workers} actors")
    elif args.concurrency_min is not None:
        print(f"[init] concurrency:      range ({args.concurrency_min}, {args.concurrency_max})")
    else:
        print(f"[init] concurrency:      Curator default (1, N_CPUS)")
    sys.stdout.flush()

    # Start one local Ray cluster
    ray_tmp = args.ray_tmp_dir or f"/tmp/ray_extract_{os.getpid()}"
    print(f"[init] ray.init(local) — tmp={ray_tmp}")
    sys.stdout.flush()
    ray.init(
        address="local",
        _temp_dir=ray_tmp,
        ignore_reinit_error=True,
        object_store_memory=args.object_store_gb * 1024 ** 3,
    )

    # Build + run
    pipe = build_pipeline(args)
    print(f"[init] pipeline built ({len(pipe.stages)} stages)")
    for s in pipe.stages:
        print(f"        {s.name}")
    sys.stdout.flush()

    t0 = time.time()
    print(f"[run]  starting pipeline at {time.strftime('%H:%M:%S')}")
    sys.stdout.flush()
    try:
        pipe.run(executor=RayDataExecutor())
    finally:
        elapsed = time.time() - t0
        print(f"[run]  finished in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        try:
            import subprocess
            sz = subprocess.check_output(["du", "-sh", str(args.output_path)], text=True).strip()
            n_pq = len(list(args.output_path.glob("*.parquet")))
            print(f"[done] output: {sz}, {n_pq} parquet files")
        except Exception as e:  # noqa: BLE001
            print(f"[done] could not summarize output: {e}")
        sys.stdout.flush()
        ray.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
