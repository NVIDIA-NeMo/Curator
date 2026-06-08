"""Stage-1 extract-only pipeline (one Ray cluster, multiple WARCs).

Stripped-down sibling of ``run_warc_pipeline.py`` that runs only the
extract stages (no text filters, no image stages, no manifest/markers).

Pipeline:
    1. FilePartitioningStage          (one task per input WARC)
    2. WarcStreamingExtractStage      (single-pass: iterate WARC records +
                                      extract → InterleavedBatch, no
                                      intermediate DataFrame)
    3. InterleavedParquetWriterStage  (writes parquet to --output-path)

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
import re
import sys
import time
import uuid
from pathlib import Path

import ray

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.interleaved.io.writers import InterleavedParquetWriterStage
from nemo_curator.stages.nemotron_cc_mm.warc_streaming_extract import WarcStreamingExtractStage


# ---------- idx_NNNNN subdir writer ------------------------------------------

_WARC_INDEX_RE = re.compile(r"-(\d{5})\.warc(?:\.gz)?$", re.IGNORECASE)


def _idx_from_source_files(source_files) -> str | None:
    """Parse ``idx_NNNNN`` from a WARC URL/path list.  Returns None if no
    5-digit warc index can be extracted."""
    if not source_files:
        return None
    sample = source_files[0] if isinstance(source_files, (list, tuple)) else source_files
    m = _WARC_INDEX_RE.search(str(sample))
    return f"idx_{m.group(1)}" if m else None


class IdxSubdirParquetWriter(InterleavedParquetWriterStage):
    """Write each batch to ``<output>/idx_NNNNN/<hash>.parquet`` where
    NNNNN is parsed from the source WARC filename.  Matches the layout
    produced by the prior ``submit_array.sh`` per-WARC array runs.
    """

    def process(self, task):
        import nemo_curator.stages.text.io.writer.utils as writer_utils
        from nemo_curator.tasks import FileGroupTask
        from nemo_curator.utils.client_utils import is_remote_url

        source_files = task._metadata.get("source_files")
        idx = _idx_from_source_files(source_files)

        # Compute filename the same way the base class does
        if source_files:
            filename = writer_utils.get_deterministic_hash(source_files, task.task_id)
        else:
            filename = uuid.uuid4().hex

        if idx is not None:
            idx_dir = self.fs.sep.join([self._fs_path, idx])
            # Ensure idx subdir exists (local fs auto-creates; for s3 this is a no-op)
            try:
                self.fs.makedirs(idx_dir, exist_ok=True)
            except (OSError, AttributeError):
                pass
            file_path = self.fs.sep.join([idx_dir, f"{filename}.{self.file_extension}"])
        else:
            file_path = self.fs.sep.join([self._fs_path, f"{filename}.{self.file_extension}"])

        file_path_with_protocol = (
            self.fs.unstrip_protocol(file_path) if is_remote_url(self.path) else file_path
        )
        self.write_data(task, file_path_with_protocol)
        return FileGroupTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=[file_path_with_protocol],
            _metadata={**task._metadata, "format": self.file_extension},
            _stage_perf=task._stage_perf,
        )


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

    Cumulative across calls — each invocation registers a (substr → value)
    mapping; the patched function checks the mapping each time.
    """
    from nemo_curator.backends.ray_data import adapter as adapter_mod

    if not hasattr(adapter_mod, "_calc_orig"):
        adapter_mod._calc_orig = adapter_mod.calculate_concurrency_for_actors_for_stage
        adapter_mod._concurrency_overrides = {}

        def patched(stage, ignore_head_node=False):
            stage_name = getattr(stage, "name", "") or ""
            for substr, override_value in adapter_mod._concurrency_overrides.items():
                if substr in stage_name:
                    print(
                        f"[concurrency-patch] {stage_name}: forced concurrency = {override_value}",
                        flush=True,
                    )
                    return override_value
            return adapter_mod._calc_orig(stage, ignore_head_node)

        adapter_mod.calculate_concurrency_for_actors_for_stage = patched

    adapter_mod._concurrency_overrides[stage_name_substr] = value


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

    # 2. streaming WARC → InterleavedBatch (one pass, never accumulates the
    # raw HTML of all records into an intermediate DataFrame — replaces the
    # old iterator + extractor stages).
    pipe.add_stage(
        WarcStreamingExtractStage(
            extractor=args.extractor,
            min_text_chars=args.min_text_chars,
            max_text_chars=args.max_text_chars,
            resiliparse_text=args.resiliparse_text,
            max_batch_bytes=args.max_batch_bytes,
            record_limit=args.record_limit,
            storage_options=storage_options,
        )
    )

    # 3. write parquet — materialize_on_write=False so the writer doesn't
    # try to fetch image binaries (no image_acquire stage in extract-only).
    # IdxSubdirParquetWriter writes to <output>/idx_NNNNN/<hash>.parquet
    # matching the prior submit_array.sh layout.
    # For s3:// output, pass storage_options so fsspec uses the right profile
    # / endpoint (the curator profile, not the cc profile used for input).
    _writer_kwargs = {}
    if str(args.output_path).startswith("s3://"):
        _writer_kwargs["write_kwargs"] = {
            "storage_options": {
                "profile": os.environ.get("OUTPUT_AWS_PROFILE", "curator"),
                "endpoint_url": os.environ.get(
                    "OUTPUT_AWS_ENDPOINT_URL", "https://pdx.s8k.io",
                ),
            },
        }
    pipe.add_stage(
        IdxSubdirParquetWriter(
            path=str(args.output_path),
            mode=args.mode,
            materialize_on_write=False,
            **_writer_kwargs,
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
    p.add_argument("--extractor", default="magic_traf",
                   choices=["magic_html", "magic_traf"])
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
    p.add_argument("--target-max-block-size-mib", type=int, default=32,
                   help="Ray Data DataContext.target_max_block_size in MiB.  Smaller "
                        "shrinks the per-task ResourceBudget reservation, letting more "
                        "in-flight tasks dispatch before backpressure fires.  Default "
                        "32 (vs Ray's built-in 128).  Set to 128 to disable the override.")

    # Concurrency control for the actor (extract) stage
    cc = p.add_mutually_exclusive_group()
    cc.add_argument("--force-workers", type=int, default=None,
                    help="Pin extract stage to exactly N actors (int).")
    cc.add_argument("--concurrency-min", type=int, default=None,
                    help="Lower bound for actor autoscale (use with --concurrency-max).")
    p.add_argument("--concurrency-max", type=int, default=None,
                   help="Upper bound for actor autoscale (use with --concurrency-min).")

    p.add_argument("--executor", choices=["raydata", "xenna"], default="raydata",
                   help="Which backend executor to use.  raydata = streaming actor pool "
                        "(default); xenna = NVIDIA cosmos_xenna pipeline executor.")

    args = p.parse_args()

    if (args.concurrency_min is None) != (args.concurrency_max is None):
        p.error("--concurrency-min and --concurrency-max must be supplied together")

    # Allow s3:// (or other fsspec-style) output paths — the writer uses
    # fsspec under the hood and can write remote directly.  For local paths
    # we resolve and mkdir as before.
    _out_str = str(args.output_path)
    if "://" in _out_str:
        args.output_path = _out_str.rstrip("/")
    else:
        args.output_path = args.output_path.resolve()
        args.output_path.mkdir(parents=True, exist_ok=True)

    # Install concurrency patch BEFORE build_pipeline so adapter sees it.
    if args.force_workers is not None:
        _install_concurrency_override("warc_streaming_extract", args.force_workers)
    elif args.concurrency_min is not None:
        _install_concurrency_override(
            "warc_streaming_extract",
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

    # Shrink Ray Data's per-task block-size reservation.  Default is 128 MiB,
    # which combined with N actors × 2-deep pipelining easily exhausts the
    # 16 GiB /dev/shm cap and triggers ResourceBudget backpressure that holds
    # in-flight task count well below the actor count.  Lowering this lets
    # all N × 2 tasks dispatch from the start.
    from ray.data import DataContext
    DataContext.get_current().target_max_block_size = (
        args.target_max_block_size_mib * 1024 * 1024
    )
    print(
        f"[init] target_max_block_size: {args.target_max_block_size_mib} MiB "
        f"(Ray Data ResourceBudget reservation = N_actors × 2 × this)"
    )
    sys.stdout.flush()

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
        if args.executor == "xenna":
            from nemo_curator.backends.xenna import XennaExecutor
            executor = XennaExecutor()
            print("[run]  executor: XennaExecutor")
        else:
            executor = RayDataExecutor()
            print("[run]  executor: RayDataExecutor")
        sys.stdout.flush()
        pipe.run(executor=executor)
    finally:
        elapsed = time.time() - t0
        print(f"[run]  finished in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        try:
            if "://" in str(args.output_path):
                print(f"[done] output: {args.output_path} (remote — skipping du/glob)")
            else:
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
