"""End-to-end Curator pipeline: WARC → InterleavedBatch → filtered → Parquet.

Stages:
    1. ``FilePartitioningStage``            (_EmptyTask → FileGroupTask)
    2. ``DocumentIterateExtractStage``      (FileGroupTask → DocumentBatch)
       Uses Curator's ``CommonCrawlWarcIterator``.
    3. ``WarcDocumentToInterleavedStage``   (DocumentBatch → InterleavedBatch)
       Our DOM walker; preserves image positions.
    4. Stage-3 text filters                 (InterleavedBatch → InterleavedBatch)
       Each runs as its own stage so individual filters can be toggled
       and their drop rates measured independently.
    5. ``InterleavedParquetWriterStage``    (InterleavedBatch → FileGroupTask)

Usage::

    python Curator/tutorials/nemotron_cc_mm/run_warc_pipeline.py \\
        --input-path /home/aot/codebase/nemotron_cc_mm/data/warc/ \\
        --output-path /home/aot/codebase/nemotron_cc_mm/data/out/ \\
        --record-limit 200 \\
        --mode overwrite
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

from run_manifest import utc_iso, write_run_manifest
from shard import Shard

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.interleaved.io.writers import InterleavedParquetWriterStage
from nemo_curator.stages.interleaved.stages import InterleavedAspectRatioFilterStage
from nemo_curator.stages.nemotron_cc_mm import (
    InterleavedAestheticFilter,
    InterleavedAlphabeticWordRatioFilterStage,
    InterleavedBadWordsFilterStage,
    InterleavedBulletLineRatioFilterStage,
    InterleavedContinuousLineBreaksFilterStage,
    InterleavedDuplicateLineRatioFilterStage,
    InterleavedEllipsisLineRatioFilterStage,
    InterleavedFastTextLangIDFilterStage,
    InterleavedGeometryFilter,
    InterleavedImageCountFilter,
    InterleavedLoremIpsumFilterStage,
    InterleavedMeanWordLengthFilterStage,
    InterleavedNGramRepetitionFilterStage,
    InterleavedNSFWFilter,
    InterleavedPIIRedactorStage,
    InterleavedStopwordCountFilterStage,
    InterleavedSymbolToWordRatioFilterStage,
    InterleavedTopWordFractionFilterStage,
    InterleavedURLSubstringNSFWFilterStage,
    InterleavedWordCountFilterStage,
    ParallelImageDownloader,
    WarcDocumentToInterleavedStage,
)
from nemo_curator.stages.nemotron_cc_mm.lang_id import DEFAULT_LID_PATH
from nemo_curator.stages.nemotron_cc_mm.nsfw_filter import (
    DEFAULT_MODEL_DIR as DEFAULT_NSFW_MODEL_DIR,
)
from nemo_curator.stages.text.download.base.iterator import DocumentIterateExtractStage
from nemo_curator.stages.text.download.common_crawl.warc_iterator import (
    CommonCrawlWarcIterator,
)


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    pipe = Pipeline(
        name="warc_to_interleaved",
        description="CC WARC → interleaved rows → Stage-3 filters → Parquet",
    )

    pipe.add_stage(
        FilePartitioningStage(
            file_paths=args.input_path,
            files_per_partition=args.files_per_partition,
            file_extensions=[".gz", ".warc"],
        )
    )

    pipe.add_stage(
        DocumentIterateExtractStage(
            iterator=CommonCrawlWarcIterator(),
            record_limit=args.record_limit,
            add_filename_column=True,
        )
    )

    pipe.add_stage(
        WarcDocumentToInterleavedStage(
            extractor=args.extractor,
            min_text_chars=args.min_text_chars,
            resiliparse_text=args.resiliparse_text,
            max_batch_bytes=args.max_batch_bytes,
            max_text_chars=args.max_text_chars,
        )
    )

    # ---------------- Stage 3 (toggleable per filter) ----------------

    # URL-substring NSFW first — cheapest, doesn't even touch text.
    if args.url_substr_nsfw:
        pipe.add_stage(InterleavedURLSubstringNSFWFilterStage())

    # Cheap text heuristics.
    if args.lorem_ipsum:
        pipe.add_stage(InterleavedLoremIpsumFilterStage())
    if args.word_count:
        pipe.add_stage(InterleavedWordCountFilterStage(
            min_words=args.word_count_min,
            max_words=args.word_count_max,
        ))
    if args.mean_word_length:
        pipe.add_stage(InterleavedMeanWordLengthFilterStage(
            min_len=args.mean_word_length_min,
            max_len=args.mean_word_length_max,
        ))
    if args.symbol_ratio:
        pipe.add_stage(InterleavedSymbolToWordRatioFilterStage(
            max_ratio=args.symbol_ratio_max,
        ))
    if args.stopword_count:
        pipe.add_stage(InterleavedStopwordCountFilterStage(
            min_distinct=args.stopword_count_min,
        ))
    if args.ngram_repetition:
        pipe.add_stage(InterleavedNGramRepetitionFilterStage())

    # Gopher / OmniCorpus finishing filters (cheap, all on aggregated text).
    if args.alpha_ratio:
        pipe.add_stage(InterleavedAlphabeticWordRatioFilterStage(
            min_ratio=args.alpha_ratio_min,
        ))
    if args.ellipsis_line:
        pipe.add_stage(InterleavedEllipsisLineRatioFilterStage(
            max_ratio=args.ellipsis_line_max,
        ))
    if args.bullet_line:
        pipe.add_stage(InterleavedBulletLineRatioFilterStage(
            max_ratio=args.bullet_line_max,
        ))
    if args.dup_line:
        pipe.add_stage(InterleavedDuplicateLineRatioFilterStage(
            max_ratio=args.dup_line_max,
        ))
    if args.top_word:
        pipe.add_stage(InterleavedTopWordFractionFilterStage(
            max_ratio=args.top_word_max,
        ))
    if args.continuous_line_breaks:
        pipe.add_stage(InterleavedContinuousLineBreaksFilterStage(
            max_ratio=args.continuous_line_breaks_max,
        ))
    if args.bad_words and args.bad_words_path:
        pipe.add_stage(InterleavedBadWordsFilterStage(
            wordlist_path=args.bad_words_path,
        ))

    # FastText lang-ID last among text filters — it's the most expensive.
    if args.lang_id:
        pipe.add_stage(InterleavedFastTextLangIDFilterStage(
            model_path=args.lang_id_model,
            target_lang=args.lang_id_target,
            min_score=args.lang_id_min_score,
        ))

    # ---------------- Stage 5 (image acquire + filter) ----------------
    if args.image_download:
        pipe.add_stage(ParallelImageDownloader(
            concurrency=args.image_concurrency,
            timeout_s=args.image_timeout_s,
            max_retries=args.image_retries,
            max_bytes=args.image_max_bytes,
            url_dedup=args.image_url_dedup,
        ))
    if args.image_geometry:
        pipe.add_stage(InterleavedGeometryFilter(
            min_width=args.image_min_dim,
            min_height=args.image_min_dim,
            max_width=args.image_max_dim,
            max_height=args.image_max_dim,
        ))
    if args.image_aspect_ratio:
        pipe.add_stage(InterleavedAspectRatioFilterStage(
            min_aspect_ratio=args.image_aspect_min,
            max_aspect_ratio=args.image_aspect_max,
            drop_invalid_rows=True,
        ))
    if args.image_nsfw:
        pipe.add_stage(InterleavedNSFWFilter(
            model_dir=args.image_nsfw_model_dir,
            max_nsfw_score=args.image_nsfw_max_score,
        ))
    if args.image_aesthetic:
        pipe.add_stage(InterleavedAestheticFilter(
            model_dir=args.image_aesthetic_model_dir,
            min_aesthetic_score=args.image_aesthetic_min_score,
        ))
    if args.image_count:
        pipe.add_stage(InterleavedImageCountFilter(
            min_images=args.image_count_min,
            max_images=args.image_count_max,
        ))

    # Stage 7 — Detailed filter + safety (PII redaction; more to come).
    if args.pii_redact:
        pipe.add_stage(InterleavedPIIRedactorStage(
            redact_email=args.pii_email,
            redact_phone=args.pii_phone,
            redact_ipv4=args.pii_ipv4,
            redact_ssn=args.pii_ssn,
            redact_alt_text=args.pii_alt_text,
        ))

    pipe.add_stage(
        InterleavedParquetWriterStage(
            path=args.output_path,
            mode=args.mode,
            materialize_on_write=False,
            on_materialize_error="warn",
        )
    )

    return pipe




def main(args: argparse.Namespace) -> None:
    # Short-circuit if this shard already completed in a previous submission.
    # Driven by env vars (CURATOR_SHARD_INDEX, CURATOR_NUM_SHARDS); when not
    # running under SLURM these default to (0, 1) so the marker check works
    # consistently for single-WARC dev runs too.
    #
    # Markers live at *array-root level* (one dir up from the per-WARC
    # output_path), so they collect under <root>/_SUCCESS/shard_NNNNN.json
    # rather than being scattered inside per-shard dirs.
    try:
        idx, num_shards = Shard.env()
    except ValueError:
        idx, num_shards = 0, 1
    marker_root = Path(args.output_path).parent
    if Shard.has_marker(marker_root, idx):
        print(
            f"[shard] {idx}/{num_shards} already has marker at "
            f"{Shard.marker_path(marker_root, idx)} — skipping",
            file=sys.stderr,
        )
        return

    started = utc_iso()

    # Start Ray with address="local" so we always start a brand-new local
    # cluster instead of attaching to /tmp/ray/ray_current_cluster (which
    # collides when SLURM packs multiple tasks on one node).
    #
    # _temp_dir must NOT live under args.output_path because the writer's
    # --mode overwrite rmtree-s that directory mid-pipeline, which would
    # wipe Ray's active session dir.  Use a per-task path under /tmp
    # (compute-node-local) keyed on SLURM_JOB_ID + shard so siblings on
    # the same node don't collide.
    ray_tmp = (
        f"/tmp/ray_ccmm_{os.environ.get('SLURM_JOB_ID', os.getpid())}"
        f"_{idx:05d}"
    )
    ray_ctx = None
    try:
        import ray
        if not ray.is_initialized():
            ray_ctx = ray.init(
                address="local",
                _temp_dir=ray_tmp,
                ignore_reinit_error=True,
            )
            # RayClient short-circuits its own ``ray start --head`` subprocess
            # only when RAY_ADDRESS is set in the env, so advertise our cluster.
            gcs = getattr(ray_ctx, "address_info", {}).get("gcs_address")
            if gcs:
                os.environ["RAY_ADDRESS"] = gcs
    except Exception as e:  # noqa: BLE001
        print(f"[shard] WARNING — explicit ray.init failed ({e}); falling back to RayClient", file=sys.stderr)

    ray_client = RayClient()
    ray_client.start()

    # Tolerate occasional transient block errors (network blips during image
    # download, etc.) instead of aborting the whole pipeline.
    try:
        import ray.data
        ray.data.DataContext.get_current().max_errored_blocks = 100
    except Exception:  # noqa: BLE001
        pass  # best-effort; older Ray versions may not expose this

    pipeline_ok = False
    try:
        pipeline = build_pipeline(args)
        print(pipeline.describe())
        pipeline.run(executor=RayDataExecutor())
        pipeline_ok = True
    finally:
        ray_client.stop()
        finished = utc_iso()
        write_run_manifest(args, started, finished)
        if pipeline_ok:
            Shard.write_marker(
                marker_root, idx, num_shards,
                payload={
                    "preset": args.preset,
                    "input_path": args.input_path,
                    "started_utc": started,
                    "finished_utc": finished,
                },
            )


def _add_filter_flag(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    """Add a ``--<name>/--no-<name>`` boolean pair (default given)."""
    parser.add_argument(
        f"--{name}", dest=name.replace("-", "_"),
        action=argparse.BooleanOptionalAction, default=default,
        help=help_text,
    )


PRESETS_DIR = Path(__file__).resolve().parent / "presets"


def _resolve_preset(name_or_path: str) -> Path:
    """Accept either a preset name (``omnicorpus``) or an explicit file path."""
    p = Path(name_or_path).expanduser()
    if p.is_file():
        return p
    candidate = PRESETS_DIR / f"{name_or_path}.yaml"
    if candidate.is_file():
        return candidate
    available = sorted(p.stem for p in PRESETS_DIR.glob("*.yaml"))
    raise SystemExit(
        f"Preset {name_or_path!r} not found.  Pass a path or one of: "
        f"{', '.join(available)}"
    )


def _load_preset_namespace(preset: str | None) -> argparse.Namespace:
    """Return a Namespace pre-populated from the preset YAML (if any)."""
    ns = argparse.Namespace()
    if not preset:
        return ns
    preset_path = _resolve_preset(preset)
    with open(preset_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # YAML keys map 1:1 to argparse `dest` (underscore form).  Strip the few
    # description-only keys.
    for k, v in data.items():
        if k in {"name", "description"}:
            continue
        setattr(ns, k, v)
    print(f"[preset] loaded {preset_path}  ({len(vars(ns))} overrides)",
          file=sys.stderr)
    return ns


if __name__ == "__main__":
    # First pass: only --preset, so we can preload the namespace before the
    # main parser applies its defaults.  Precedence: CLI > preset > defaults.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--preset", default=None)
    pre_args, _ = pre.parse_known_args()
    preset_ns = _load_preset_namespace(pre_args.preset)

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--preset", default=None,
                        help="Preset name (in ./presets/) or path to a YAML file. "
                             "Loaded BEFORE CLI defaults; CLI flags still win.")

    # ---- I/O ----
    parser.add_argument("--input-path", required=True,
                        help="WARC file or directory of .warc.gz files")
    parser.add_argument("--output-path", required=True,
                        help="Output directory for Parquet shards")
    parser.add_argument("--record-limit", type=int, default=None,
                        help="Cap records read per WARC (smoke testing)")
    parser.add_argument("--files-per-partition", type=int, default=1)
    parser.add_argument("--mode", default="overwrite",
                        choices=["ignore", "overwrite", "append", "error"])
    parser.add_argument("--log-path", default=None,
                        help="Path to the run's log file.  If provided, "
                             "_run.json gets a 'funnel' field aggregated "
                             "from per-stage docs/rows/tokens lines in this log.")

    # ---- Stage 2 (extraction) ----
    parser.add_argument("--extractor", default="naive",
                        choices=["naive", "magic_html", "hybrid"],
                        help="HTML → interleaved-rows extractor implementation")
    parser.add_argument("--min-text-chars", type=int, default=1,
                        help="Drop text rows shorter than N chars (at extraction)")
    parser.add_argument("--max-text-chars", type=int, default=50_000,
                        help="Cap each row's text_content at N chars at extraction "
                             "time.  Bounds downstream numpy fixed-width allocations "
                             "(an ArrowDtype Series upcast to <U{max_len}> blows up "
                             "memory on pathological docs).  0 disables.  Default 50K.")
    parser.add_argument("--max-batch-bytes", type=int, default=256 * 1024 * 1024,
                        help="Chunk each WARC's InterleavedBatch into sub-batches no "
                             "larger than this many bytes (Arrow nbytes) — keeps all "
                             "rows of a sample together.  Smaller = lower peak memory "
                             "for image/GPU stages.  Default 256 MiB.")
    _add_filter_flag(parser, "resiliparse-text", True,
                     "Also run Curator's Resiliparse text extractor and stuff "
                     "its joined output into the doc's metadata-row text_content")

    # ---- Stage 3 (text filters) ----
    _add_filter_flag(parser, "url-substr-nsfw", True, "URL-substring NSFW filter")
    _add_filter_flag(parser, "lorem-ipsum", True, "Drop docs containing 'lorem ipsum'")
    _add_filter_flag(parser, "word-count", True, "Gopher word-count bounds")
    parser.add_argument("--word-count-min", type=int, default=50)
    parser.add_argument("--word-count-max", type=int, default=100_000)
    _add_filter_flag(parser, "mean-word-length", True, "Gopher mean word length bounds")
    parser.add_argument("--mean-word-length-min", type=float, default=3.0)
    parser.add_argument("--mean-word-length-max", type=float, default=10.0)
    _add_filter_flag(parser, "symbol-ratio", True, "Gopher symbol-to-word ratio cap")
    parser.add_argument("--symbol-ratio-max", type=float, default=0.1)
    _add_filter_flag(parser, "stopword-count", True, "Gopher English stopword count")
    parser.add_argument("--stopword-count-min", type=int, default=2)
    _add_filter_flag(parser, "ngram-repetition", True, "Gopher n-gram repetition cap")
    _add_filter_flag(parser, "alpha-ratio", True,
                     "Gopher alphabetic-word ratio (>= min_ratio)")
    parser.add_argument("--alpha-ratio-min", type=float, default=0.8)
    _add_filter_flag(parser, "ellipsis-line", True,
                     "Gopher ellipsis-line ratio (<= max_ratio)")
    parser.add_argument("--ellipsis-line-max", type=float, default=0.3)
    _add_filter_flag(parser, "bullet-line", True,
                     "Gopher bullet-line ratio (<= max_ratio)")
    parser.add_argument("--bullet-line-max", type=float, default=0.9)
    _add_filter_flag(parser, "dup-line", True,
                     "MassiveText duplicate-line ratio (<= max_ratio)")
    parser.add_argument("--dup-line-max", type=float, default=0.3)
    _add_filter_flag(parser, "top-word", True,
                     "OmniCorpus top single-word fraction (<= max_ratio)")
    parser.add_argument("--top-word-max", type=float, default=0.30)
    _add_filter_flag(parser, "continuous-line-breaks", True,
                     "OmniCorpus continuous-line-breaks ratio (<= max_ratio); "
                     "no-op with the current whitespace-normalizing extractor")
    parser.add_argument("--continuous-line-breaks-max", type=float, default=0.05)
    _add_filter_flag(parser, "bad-words", True,
                     "Bad-words wordlist filter (off unless --bad-words-path is set)")
    parser.add_argument("--bad-words-path", default="",
                        help="Path to UTF-8 wordlist (e.g. LDNOOBW); empty = disabled")
    _add_filter_flag(parser, "lang-id", True, "FastText English lang-ID filter")
    parser.add_argument("--lang-id-model", default=str(DEFAULT_LID_PATH))
    parser.add_argument("--lang-id-target", default="en")
    parser.add_argument("--lang-id-min-score", type=float, default=0.65)

    # ---- Stage 5 (image acquire + filter) ----
    _add_filter_flag(parser, "image-download", True, "Parallel image downloader (fills binary_content)")
    parser.add_argument("--image-concurrency", type=int, default=500)
    parser.add_argument("--image-timeout-s", type=float, default=20.0)
    parser.add_argument("--image-retries", type=int, default=1)
    parser.add_argument("--image-max-bytes", type=int, default=20 * 1024 * 1024)
    _add_filter_flag(parser, "image-url-dedup", True,
                     "Within-batch URL dedup in downloader (OmniCorpus Bloom-style)")
    _add_filter_flag(parser, "image-geometry", True, "Geometry filter (drop too-small/too-large)")
    parser.add_argument("--image-min-dim", type=int, default=150)
    parser.add_argument("--image-max-dim", type=int, default=20_000)
    _add_filter_flag(parser, "image-aspect-ratio", True, "Aspect-ratio filter (Curator class)")
    parser.add_argument("--image-aspect-min", type=float, default=0.5)
    parser.add_argument("--image-aspect-max", type=float, default=2.0)
    _add_filter_flag(parser, "image-nsfw", True, "LAION-NSFW image filter")
    parser.add_argument("--image-nsfw-model-dir", default=DEFAULT_NSFW_MODEL_DIR)
    parser.add_argument("--image-nsfw-max-score", type=float, default=0.8)
    _add_filter_flag(parser, "image-aesthetic", True,
                     "LAION aesthetic-score filter (OmniCorpus < 3.7 cutoff)")
    parser.add_argument("--image-aesthetic-model-dir", default=DEFAULT_NSFW_MODEL_DIR,
                        help="Reuses the same cache dir as NSFW (CLIP weights are shared)")
    parser.add_argument("--image-aesthetic-min-score", type=float, default=3.7)
    _add_filter_flag(parser, "image-count", True,
                     "Drop docs whose surviving image-row count is out of bounds")
    parser.add_argument("--image-count-min", type=int, default=1)
    parser.add_argument("--image-count-max", type=int, default=30)

    # ---- Stage 7 (detailed filter + safety) ----
    _add_filter_flag(parser, "pii-redact", True,
                     "Redact PII patterns (email, phone, IP, SSN) in text rows")
    _add_filter_flag(parser, "pii-email", True, "Redact email addresses")
    _add_filter_flag(parser, "pii-phone", True, "Redact US-format phone numbers")
    _add_filter_flag(parser, "pii-ipv4",  True, "Redact dotted-quad IPv4 addresses")
    _add_filter_flag(parser, "pii-ssn",   True, "Redact US-format SSNs")
    _add_filter_flag(parser, "pii-alt-text", False,
                     "Also redact PII inside image alt-text (off by default)")

    args = parser.parse_args(namespace=preset_ns)
    main(args)
