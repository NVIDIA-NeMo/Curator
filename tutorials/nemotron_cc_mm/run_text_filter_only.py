"""Stage 2 (text-filter-only) pipeline — slim analogue of run_extract_only.py.

Reads Stage 1 parquet outputs (one or more idx_NNNNN directories), runs the
12 heuristic doc-level filters configured by --preset, writes the filtered
parquets to ``<output>/idx_NNNNN/<hash>.parquet`` (matching the Stage 1
layout so Stage 3/4 can chain off it).

Pipeline:
    1. InterleavedParquetReaderStage     (reads parquet shards)
    2. 12× heuristic filter stages  (configurable via preset YAML)
    3. IdxSubdirParquetWriter       (writes survivors to per-idx subdir)

Usage:
    # Single idx dir
    python run_text_filter_only.py \\
        --preset text_filter_multimodal \\
        --input-path  /.../stage1/seg_00/batch_0/idx_00000 \\
        --output-path /.../stage2/seg_00/batch_0

    # Multiple idx dirs (comma-separated) — one Ray cluster handles them all
    python run_text_filter_only.py \\
        --preset text_filter_multimodal \\
        --input-path  /.../idx_00000,/.../idx_00001,/.../idx_00002 \\
        --output-path /.../stage2_out

Concurrency knobs mirror run_extract_only.py:
    --force-workers N            pin filter stage to N actors
    --concurrency-min M --max X  pass (M, X) autoscale range
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import uuid
import yaml
from pathlib import Path

import ray

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.interleaved.io.reader import InterleavedParquetReader
from nemo_curator.stages.interleaved.io.writers import InterleavedParquetWriterStage
from nemo_curator.stages.nemotron_cc_mm import (
    InterleavedAlphabeticWordRatioFilterStage,
    InterleavedBadWordsFilterStage,
    InterleavedBulletLineRatioFilterStage,
    InterleavedContinuousLineBreaksFilterStage,
    InterleavedDuplicateLineRatioFilterStage,
    InterleavedEllipsisLineRatioFilterStage,
    InterleavedFastTextLangIDAnnotatorStage,
    InterleavedFastTextLangIDFilterStage,
    InterleavedLoremIpsumFilterStage,
    InterleavedMeanWordLengthFilterStage,
    InterleavedNGramRepetitionFilterStage,
    InterleavedStopwordCountFilterStage,
    InterleavedSymbolToWordRatioFilterStage,
    InterleavedTopWordFractionFilterStage,
    InterleavedURLSubstringNSFWFilterStage,
    InterleavedWordCountFilterStage,
)
from nemo_curator.stages.nemotron_cc_mm.lang_id import DEFAULT_LID_PATH


# ---------- idx_NNNNN subdir writer (parquet→parquet variant) ---------------

_IDX_RE = re.compile(r"idx_(\d{5})")


def _idx_from_source_files(source_files) -> str | None:
    """Pull ``idx_NNNNN`` from a parquet path or its parent dir.

    Stage 1 wrote each WARC's parquets under ``.../idx_NNNNN/<hash>.parquet``;
    here we walk up the path and pull the first ``idx_NNNNN`` we find.
    """
    if not source_files:
        return None
    sample = source_files[0] if isinstance(source_files, (list, tuple)) else source_files
    m = _IDX_RE.search(str(sample))
    return f"idx_{m.group(1)}" if m else None


class IdxSubdirParquetWriter(InterleavedParquetWriterStage):
    """Write each batch to ``<output>/idx_NNNNN/<hash>.parquet`` where the
    idx is parsed from the input parquet path (set during Stage 1)."""

    def process(self, task):
        import nemo_curator.stages.text.io.writer.utils as writer_utils
        from nemo_curator.tasks import FileGroupTask
        from nemo_curator.utils.client_utils import is_remote_url

        source_files = task._metadata.get("source_files")
        idx = _idx_from_source_files(source_files)

        if source_files:
            filename = writer_utils.get_deterministic_hash(source_files, task.task_id)
        else:
            filename = uuid.uuid4().hex

        if idx is not None:
            idx_dir = self.fs.sep.join([self._fs_path, idx])
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


# ---------- helpers ---------------------------------------------------------

def _parse_input_paths(raw: str) -> str | list[str]:
    if "," in raw:
        return [p.strip() for p in raw.split(",") if p.strip()]
    return raw


def _install_concurrency_override(stage_name_substr: str, value) -> None:
    """Force a specific stage's actor concurrency."""
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


# ---------- preset loading --------------------------------------------------

PRESETS_DIR = Path(__file__).resolve().parent / "presets"


def _load_preset(name_or_path: str) -> dict:
    """Load a preset YAML dict (only the text-filter knobs are used here)."""
    p = Path(name_or_path)
    if not p.exists():
        p = PRESETS_DIR / f"{name_or_path}.yaml"
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------- pipeline build --------------------------------------------------

def build_pipeline(args: argparse.Namespace) -> Pipeline:
    pipe = Pipeline(name="text_filter_only")

    # 1. Parquet reader
    pipe.add_stage(
        InterleavedParquetReader(
            file_paths=args.input_path,
            files_per_partition=args.files_per_partition,
            max_batch_bytes=args.max_batch_bytes,
        )
    )

    # 2. Heuristic filters (order matches run_warc_pipeline.py — cheap first)
    ts = args.text_source
    if args.url_substr_nsfw:
        pipe.add_stage(InterleavedURLSubstringNSFWFilterStage())
    if args.lorem_ipsum:
        pipe.add_stage(InterleavedLoremIpsumFilterStage(text_source=ts))
    if args.word_count:
        pipe.add_stage(InterleavedWordCountFilterStage(
            min_words=args.word_count_min,
            max_words=args.word_count_max,
            text_source=ts,
        ))
    if args.mean_word_length:
        pipe.add_stage(InterleavedMeanWordLengthFilterStage(
            min_len=args.mean_word_length_min,
            max_len=args.mean_word_length_max,
            text_source=ts,
        ))
    if args.symbol_ratio:
        pipe.add_stage(InterleavedSymbolToWordRatioFilterStage(
            max_ratio=args.symbol_ratio_max,
            text_source=ts,
        ))
    if args.stopword_count:
        pipe.add_stage(InterleavedStopwordCountFilterStage(
            min_distinct=args.stopword_count_min,
            text_source=ts,
        ))
    if args.ngram_repetition:
        pipe.add_stage(InterleavedNGramRepetitionFilterStage(text_source=ts))
    if args.alpha_ratio:
        pipe.add_stage(InterleavedAlphabeticWordRatioFilterStage(
            min_ratio=args.alpha_ratio_min,
            text_source=ts,
        ))
    if args.ellipsis_line:
        pipe.add_stage(InterleavedEllipsisLineRatioFilterStage(
            max_ratio=args.ellipsis_line_max,
            text_source=ts,
        ))
    if args.bullet_line:
        pipe.add_stage(InterleavedBulletLineRatioFilterStage(
            max_ratio=args.bullet_line_max,
            text_source=ts,
        ))
    if args.dup_line:
        pipe.add_stage(InterleavedDuplicateLineRatioFilterStage(
            max_ratio=args.dup_line_max,
            text_source=ts,
        ))
    if args.top_word:
        pipe.add_stage(InterleavedTopWordFractionFilterStage(
            max_ratio=args.top_word_max,
            text_source=ts,
        ))
    if args.continuous_line_breaks:
        pipe.add_stage(InterleavedContinuousLineBreaksFilterStage(
            max_ratio=args.continuous_line_breaks_max,
            text_source=ts,
        ))
    if args.bad_words and args.bad_words_path:
        pipe.add_stage(InterleavedBadWordsFilterStage(
            wordlist_path=args.bad_words_path,
            text_source=ts,
        ))
    # lang_id LAST among text filters (most expensive)
    if args.lang_id:
        pipe.add_stage(InterleavedFastTextLangIDFilterStage(
            model_path=args.lang_id_model,
            target_lang=args.lang_id_target,
            min_score=args.lang_id_min_score,
            text_source=ts,
        ))
    # lang_id_annotate — detect but don't drop; writes detected_lang +
    # lang_score into each metadata row's source_ref JSON.
    if args.lang_id_annotate:
        pipe.add_stage(InterleavedFastTextLangIDAnnotatorStage(
            model_path=args.lang_id_model,
            text_source=ts,
        ))

    # 3. Writer
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


# ---------- CLI -------------------------------------------------------------

def _add_filter_flag(parser, name: str, default: bool, help_text: str) -> None:
    dest = name.replace("-", "_")
    parser.add_argument(f"--{name}", dest=dest, action="store_true", default=default,
                        help=help_text + " (default: " + ("on" if default else "off") + ")")
    parser.add_argument(f"--no-{name}", dest=dest, action="store_false",
                        help=f"Disable {name}.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Preset (loaded first; CLI overrides win)
    p.add_argument("--preset", default="text_filter_multimodal",
                   help="Preset name (in ./presets/) or YAML path.  "
                        "Default: text_filter_multimodal.")

    # I/O
    p.add_argument("--input-path", required=True,
                   help="Comma-separated parquet file(s) or directory of parquets.")
    p.add_argument("--output-path", required=True, type=Path,
                   help="Output directory.  idx_NNNNN/ subdirs created beneath.")
    p.add_argument("--mode", default="overwrite",
                   choices=["overwrite", "append", "ignore", "error"])
    p.add_argument("--files-per-partition", type=int, default=1)
    p.add_argument("--max-batch-bytes", type=int, default=128 * 1024 * 1024)

    # Text source (where filters get the aggregated doc text from)
    p.add_argument("--text-source", default="text_rows",
                   choices=["text_rows", "metadata"],
                   help="Where doc-level filters get their aggregated text. "
                        "'text_rows' (default) = concat magic-html paragraph "
                        "rows. 'metadata' = use the Resiliparse full-doc text "
                        "stored in the metadata row's text_content.")

    # Filter toggles (all default-on except bad_words / lang_id, matching preset)
    _add_filter_flag(p, "url-substr-nsfw", True, "Drop NSFW-substring URLs")
    _add_filter_flag(p, "lorem-ipsum", True, "Drop lorem-ipsum docs")
    _add_filter_flag(p, "word-count", True, "Drop by word count")
    _add_filter_flag(p, "mean-word-length", True, "Drop by mean word length")
    _add_filter_flag(p, "symbol-ratio", True, "Drop by symbol ratio")
    _add_filter_flag(p, "stopword-count", True, "Drop by stopword count")
    _add_filter_flag(p, "ngram-repetition", True, "Drop repetitive n-grams")
    _add_filter_flag(p, "alpha-ratio", True, "Drop low-alpha-ratio docs")
    _add_filter_flag(p, "ellipsis-line", True, "Drop ellipsis-heavy docs")
    _add_filter_flag(p, "bullet-line", True, "Drop bullet-heavy docs")
    _add_filter_flag(p, "dup-line", True, "Drop high-duplicate-line docs")
    _add_filter_flag(p, "top-word", True, "Drop top-word-dominated docs")
    _add_filter_flag(p, "continuous-line-breaks", True, "Drop continuous-line-break docs")
    _add_filter_flag(p, "bad-words", False, "Apply LDNOOBW (needs --bad-words-path)")
    _add_filter_flag(p, "lang-id", False, "FastText lang-id filter (drops non-target)")
    _add_filter_flag(p, "lang-id-annotate", True, "FastText lang-id annotator (tags only, no drop)")

    # Thresholds
    p.add_argument("--word-count-min", type=int, default=20)
    p.add_argument("--word-count-max", type=int, default=100000)
    p.add_argument("--mean-word-length-min", type=float, default=3.0)
    p.add_argument("--mean-word-length-max", type=float, default=10.0)
    p.add_argument("--symbol-ratio-max", type=float, default=0.1)
    p.add_argument("--stopword-count-min", type=int, default=2)
    p.add_argument("--alpha-ratio-min", type=float, default=0.8)
    p.add_argument("--ellipsis-line-max", type=float, default=0.3)
    p.add_argument("--bullet-line-max", type=float, default=0.9)
    p.add_argument("--dup-line-max", type=float, default=0.3)
    p.add_argument("--top-word-max", type=float, default=0.30)
    p.add_argument("--continuous-line-breaks-max", type=float, default=0.05)
    p.add_argument("--bad-words-path", default="")
    p.add_argument("--lang-id-target", default="en")
    p.add_argument("--lang-id-min-score", type=float, default=0.65)
    p.add_argument("--lang-id-model", default=str(DEFAULT_LID_PATH))

    # Ray sizing
    p.add_argument("--object-store-gb", type=int, default=32)
    p.add_argument("--ray-tmp-dir", default=None)
    p.add_argument("--target-max-block-size-mib", type=int, default=32)

    # Concurrency
    cc = p.add_mutually_exclusive_group()
    cc.add_argument("--force-workers", type=int, default=None)
    cc.add_argument("--concurrency-min", type=int, default=None)
    p.add_argument("--concurrency-max", type=int, default=None)

    # First pass: parse --preset, apply YAML defaults, then re-parse so CLI wins
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--preset", default="text_filter_multimodal")
    pre_args, _ = pre.parse_known_args()
    preset = _load_preset(pre_args.preset)

    # Apply preset defaults to argparse
    valid_keys = {a.dest for a in p._actions if a.dest != "help"}
    for k, v in preset.items():
        if k in valid_keys:
            p.set_defaults(**{k: v})

    args = p.parse_args()
    if (args.concurrency_min is None) != (args.concurrency_max is None):
        p.error("--concurrency-min and --concurrency-max must be supplied together")

    # Allow s3:// output paths
    _out_str = str(args.output_path)
    if "://" in _out_str:
        args.output_path = _out_str.rstrip("/")
    else:
        args.output_path = args.output_path.resolve()
        args.output_path.mkdir(parents=True, exist_ok=True)

    # Pretty-print
    parsed_input = _parse_input_paths(args.input_path)
    n_inputs = len(parsed_input) if isinstance(parsed_input, list) else 1
    print(f"[init] preset:           {pre_args.preset}")
    print(f"[init] inputs:           {n_inputs} path(s)")
    if isinstance(parsed_input, list):
        print(f"        first: {parsed_input[0]}")
        if n_inputs > 1:
            print(f"        last:  {parsed_input[-1]}")
    else:
        print(f"        {parsed_input}")
    print(f"[init] output:           {args.output_path}")
    print(f"[init] word_count_min:   {args.word_count_min}")
    print(f"[init] lang_id:          {args.lang_id}")
    print(f"[init] max_batch_bytes:  {args.max_batch_bytes // (1024*1024)} MiB")
    print(f"[init] object_store:     {args.object_store_gb} GB")
    if args.force_workers is not None:
        print(f"[init] concurrency:      fixed {args.force_workers} actors")
    elif args.concurrency_min is not None:
        print(f"[init] concurrency:      range ({args.concurrency_min}, {args.concurrency_max})")
    sys.stdout.flush()

    # Ray
    ray_tmp = args.ray_tmp_dir or f"/tmp/ray_text_filter_{os.getpid()}"
    print(f"[init] ray.init(local) — tmp={ray_tmp}")
    sys.stdout.flush()
    ray.init(
        address="local",
        _temp_dir=ray_tmp,
        ignore_reinit_error=True,
        object_store_memory=args.object_store_gb * 1024 ** 3,
    )

    from ray.data import DataContext
    DataContext.get_current().target_max_block_size = (
        args.target_max_block_size_mib * 1024 * 1024
    )
    print(f"[init] target_max_block_size: {args.target_max_block_size_mib} MiB")
    sys.stdout.flush()

    # Install concurrency patch on the heaviest filter (ngram_repetition or
    # word_count) so user can pin a stage; substring match catches any filter.
    if args.force_workers is not None:
        _install_concurrency_override("filter", args.force_workers)
    elif args.concurrency_min is not None:
        _install_concurrency_override(
            "filter", (args.concurrency_min, args.concurrency_max),
        )

    # Normalize input_path
    args.input_path = parsed_input

    pipe = build_pipeline(args)
    print(f"[init] pipeline built ({len(pipe.stages)} stages)")
    for s in pipe.stages:
        print(f"        {s.name}")
    sys.stdout.flush()

    t0 = time.time()
    print(f"[run]  starting pipeline at {time.strftime('%H:%M:%S')}")
    sys.stdout.flush()
    try:
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
                sz = subprocess.check_output(
                    ["du", "-sh", str(args.output_path)], text=True,
                ).strip()
                n_pq = len(list(Path(args.output_path).rglob("*.parquet")))
                print(f"[done] output: {sz}, {n_pq} parquet files")
        except Exception as e:  # noqa: BLE001
            print(f"[done] could not summarize output: {e}")
        sys.stdout.flush()
        ray.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
