"""
Audio Data Filtration Pipeline - explicit stage-by-stage composition.

Adds each stage individually to the pipeline using pipeline.add_stage(),
just like the FLEURS tutorial. No CompositeStage. Each stage is visible
to the executor as a separate unit with its own resource allocation.

Pipeline:
    MonoConversion -> VAD -> BandFilter -> NISQA -> SIGMOS ->
    SegmentConcat -> SpeakerSeparation ->
    VAD_Speaker -> BandFilter_Speaker -> NISQA_Speaker -> SIGMOS_Speaker ->
    TimestampMapper

Usage:
    python pipeline.py --raw_data_dir ./audio --enable-vad --enable-band-filter \
        --enable-nisqa --enable-sigmos --enable-speaker-separation
"""

import argparse
import glob
import json
import os
import shutil
import sys

from loguru import logger

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio import (
    MonoConversionStage,
    VADSegmentationStage,
    BandFilterStage,
    NISQAFilterStage,
    SIGMOSFilterStage,
    SpeakerSeparationStage,
    SegmentConcatenationStage,
    TimestampMapperStage,
)
from nemo_curator.stages.audio.configs import (
    VADConfig, BandFilterConfig, NISQAConfig, SIGMOSConfig, SpeakerSeparationConfig,
)
from nemo_curator.stages.audio.advance_pipelines.Audio_data_filter.config import (
    SUPPORTED_AUDIO_FORMATS,
)
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioBatch


def create_pipeline(args: argparse.Namespace) -> Pipeline:
    """Build audio curation pipeline by adding each stage explicitly."""
    pipeline = Pipeline(
        name="audio_data_filter",
        description="Audio curation: Mono -> VAD -> Filters -> SpeakerSep -> Filters -> TimestampMapper",
    )

    gpu_res = Resources(cpus=1.0, gpus=args.gpus)
    cpu_res = Resources(cpus=1.0)
    band_res = Resources(cpus=4.0)

    # 1. Mono conversion
    pipeline.add_stage(MonoConversionStage(
        output_sample_rate=args.sample_rate,
        strict_sample_rate=False,
    ))

    # 2. VAD (batch mode: 1 task -> 1 task with N segment items)
    if args.enable_vad:
        pipeline.add_stage(
            VADSegmentationStage(
                config=VADConfig(
                    min_duration_sec=args.vad_min_duration,
                    max_duration_sec=args.vad_max_duration,
                    threshold=args.vad_threshold,
                ),
                mode="batch",
            ).with_(resources=gpu_res)
        )

    # 3. Band filter (CPU-only, sklearn)
    if args.enable_band_filter:
        pipeline.add_stage(
            BandFilterStage(
                config=BandFilterConfig(band_value=args.band_value),
            ).with_(resources=band_res)
        )

    # 4. NISQA
    if args.enable_nisqa:
        pipeline.add_stage(
            NISQAFilterStage(
                config=NISQAConfig(
                    mos_threshold=args.nisqa_mos_threshold,
                    noi_threshold=args.nisqa_noi_threshold,
                ),
            ).with_(resources=gpu_res)
        )

    # 5. SIGMOS
    if args.enable_sigmos:
        pipeline.add_stage(
            SIGMOSFilterStage(
                config=SIGMOSConfig(
                    noise_threshold=args.sigmos_noise_threshold,
                    ovrl_threshold=args.sigmos_ovrl_threshold,
                ),
            ).with_(resources=gpu_res)
        )

    if args.enable_speaker_separation:
        # 6. Concatenation (CPU)
        pipeline.add_stage(SegmentConcatenationStage(silence_duration_sec=0.5))

        # 7. Speaker separation (GPU, fan-out)
        pipeline.add_stage(
            SpeakerSeparationStage(
                config=SpeakerSeparationConfig(
                    exclude_overlaps=args.speaker_exclude_overlaps,
                    min_duration=args.speaker_min_duration,
                ),
            ).with_(resources=gpu_res)
        )

        # 8-11. Per-speaker: VAD + filters (with distinct names)
        if args.enable_vad:
            pipeline.add_stage(
                VADSegmentationStage(
                    config=VADConfig(
                        min_duration_sec=args.vad_min_duration,
                        max_duration_sec=args.vad_max_duration,
                        threshold=args.vad_threshold,
                    ),
                    mode="batch",
                    name="VAD_Speaker",
                ).with_(resources=gpu_res)
            )

        if args.enable_band_filter:
            pipeline.add_stage(
                BandFilterStage(
                    config=BandFilterConfig(band_value=args.band_value),
                    name="BandFilter_Speaker",
                ).with_(resources=band_res)
            )

        if args.enable_nisqa:
            pipeline.add_stage(
                NISQAFilterStage(
                    config=NISQAConfig(
                        mos_threshold=args.nisqa_mos_threshold,
                        noi_threshold=args.nisqa_noi_threshold,
                    ),
                    name="NISQA_Speaker",
                ).with_(resources=gpu_res)
            )

        if args.enable_sigmos:
            pipeline.add_stage(
                SIGMOSFilterStage(
                    config=SIGMOSConfig(
                        noise_threshold=args.sigmos_noise_threshold,
                        ovrl_threshold=args.sigmos_ovrl_threshold,
                    ),
                    name="SIGMOS_Speaker",
                ).with_(resources=gpu_res)
            )

    # 12. Timestamp mapper
    pipeline.add_stage(TimestampMapperStage())

    return pipeline


def load_audio_tasks(input_dir: str, recursive: bool = False,
                     formats: tuple = SUPPORTED_AUDIO_FORMATS) -> list:
    audio_files = []
    for ext in formats:
        ext = ext if ext.startswith('.') else f'.{ext}'
        if recursive:
            audio_files.extend(glob.glob(os.path.join(input_dir, "**", f"*{ext}"), recursive=True))
        else:
            audio_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
    audio_files = sorted(set(audio_files))
    if not audio_files:
        logger.warning(f"No audio files found in {input_dir}")
        return []
    logger.info(f"Found {len(audio_files)} audio files")
    return [AudioBatch(data={"audio_filepath": f}, task_id=f"audio_{i:05d}",
                       dataset_name="audio_filter")
            for i, f in enumerate(audio_files)]


def save_results(results: list, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.jsonl")
    with open(manifest_path, "w") as f:
        for entry in results:
            clean = {}
            for k, v in entry.items():
                if hasattr(v, "item"):
                    clean[k] = v.item()
                elif isinstance(v, (int, float, str, bool, type(None))):
                    clean[k] = v
                else:
                    clean[k] = str(v)
            f.write(json.dumps(clean) + "\n")
    logger.info(f"Saved {len(results)} results to {manifest_path}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Audio Data Filtration Pipeline")

    parser.add_argument("--raw_data_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--gpus", type=float, default=1.0)
    parser.add_argument("--sample_rate", type=int, default=48000)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--local", action="store_true",
                        help="Run locally without Ray (fastest for <10 files)")

    parser.add_argument("--enable-vad", action="store_true")
    parser.add_argument("--vad-min-duration", type=float, default=2.0)
    parser.add_argument("--vad-max-duration", type=float, default=60.0)
    parser.add_argument("--vad-threshold", type=float, default=0.5)

    parser.add_argument("--enable-band-filter", action="store_true")
    parser.add_argument("--band-value", choices=["full_band", "narrow_band"], default="full_band")

    parser.add_argument("--enable-nisqa", action="store_true")
    parser.add_argument("--nisqa-mos-threshold", type=float, default=4.5)
    parser.add_argument("--nisqa-noi-threshold", type=float, default=4.3)

    parser.add_argument("--enable-sigmos", action="store_true")
    parser.add_argument("--sigmos-noise-threshold", type=float, default=4.0)
    parser.add_argument("--sigmos-ovrl-threshold", type=float, default=3.5)

    parser.add_argument("--enable-speaker-separation", action="store_true")
    parser.add_argument("--speaker-exclude-overlaps", action="store_true", default=True)
    parser.add_argument("--speaker-min-duration", type=float, default=0.8)

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.raw_data_dir, "result")

    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    if args.clean and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Audio Data Filtration Pipeline (explicit stages)")
    logger.info("=" * 60)

    pipeline = create_pipeline(args)
    logger.info(pipeline.describe())

    tasks = load_audio_tasks(args.raw_data_dir, args.recursive)
    if not tasks:
        logger.error("No audio files to process")
        sys.exit(1)

    try:
        from datetime import datetime

        start_time = datetime.now()

        if args.local:
            logger.info("Running in LOCAL mode (no Ray, sequential)")
            stages = pipeline.stages
            for s in stages:
                s.setup()
            current = list(tasks)
            for stage in stages:
                next_tasks = []
                for t in current:
                    if t is None or len(t.data) == 0:
                        continue
                    result = stage.process(t)
                    if result is None:
                        continue
                    if isinstance(result, list):
                        next_tasks.extend(r for r in result if r and len(r.data) > 0)
                    elif len(result.data) > 0:
                        next_tasks.append(result)
                logger.info(f"  [{stage.name}] {len(current)} -> {len(next_tasks)} tasks")
                current = next_tasks
                if not current:
                    break
            for s in stages:
                if hasattr(s, 'teardown'):
                    s.teardown()
            result_tasks = current
        else:
            from nemo_curator.backends.xenna import XennaExecutor
            executor = XennaExecutor(config={"execution_mode": "batch"})
            result_tasks = pipeline.run(executor, initial_tasks=tasks)

        duration = (datetime.now() - start_time).total_seconds()

        all_results = []
        for task in result_tasks:
            if task is not None and hasattr(task, 'data'):
                if isinstance(task.data, list):
                    all_results.extend(task.data)
                else:
                    all_results.append(task.data)

        logger.info(f"Completed in {duration:.2f}s, {len(all_results)} segments")

        if all_results:
            save_results(all_results, args.output_dir)
            sample = all_results[0]
            logger.info(f"Sample: {sample.get('original_file', 'N/A')} "
                        f"[{sample.get('original_start_ms', '?')}-{sample.get('original_end_ms', '?')}ms] "
                        f"speaker={sample.get('speaker_id', 'N/A')}")
        else:
            logger.warning("No segments passed filters")

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)

    logger.info("Done!")


if __name__ == "__main__":
    main()
