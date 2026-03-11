"""
Hydra-based runner for Audio Data Filtration Pipeline.

Builds the pipeline by adding each stage explicitly (like FLEURS),
then runs via Xenna executor.

Usage:
    python run.py --config-path . --config-name pipeline.yaml \
        raw_data_dir=/path/to/audio/files
"""

import glob
import json
import os
from typing import Tuple

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nemo_curator.backends.xenna import XennaExecutor
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


def create_pipeline(cfg: DictConfig) -> Pipeline:
    """Build pipeline by adding each stage explicitly."""
    pipeline = Pipeline(
        name="audio_data_filter",
        description="Audio curation pipeline (explicit stages)",
    )

    gpu_res = Resources(cpus=1.0, gpus=cfg.gpus)
    band_res = Resources(cpus=4.0)

    pipeline.add_stage(MonoConversionStage(
        output_sample_rate=cfg.sample_rate, strict_sample_rate=False))

    if cfg.enable_vad:
        pipeline.add_stage(VADSegmentationStage(
            config=VADConfig(min_duration_sec=cfg.vad_min_duration_sec,
                             max_duration_sec=cfg.vad_max_duration_sec,
                             threshold=cfg.vad_threshold),
            mode="batch").with_(resources=gpu_res))

    if cfg.enable_band_filter:
        pipeline.add_stage(BandFilterStage(
            config=BandFilterConfig(band_value=cfg.band_value),
        ).with_(resources=band_res))

    if cfg.enable_nisqa:
        pipeline.add_stage(NISQAFilterStage(
            config=NISQAConfig(mos_threshold=cfg.nisqa_mos_threshold,
                               noi_threshold=cfg.nisqa_noi_threshold),
        ).with_(resources=gpu_res))

    if cfg.enable_sigmos:
        pipeline.add_stage(SIGMOSFilterStage(
            config=SIGMOSConfig(noise_threshold=cfg.sigmos_noise_threshold,
                                ovrl_threshold=cfg.sigmos_ovrl_threshold),
        ).with_(resources=gpu_res))

    if cfg.enable_speaker_separation:
        pipeline.add_stage(SegmentConcatenationStage(silence_duration_sec=0.5))

        pipeline.add_stage(SpeakerSeparationStage(
            config=SpeakerSeparationConfig(
                exclude_overlaps=cfg.speaker_exclude_overlaps,
                min_duration=cfg.speaker_min_duration),
        ).with_(resources=gpu_res))

        if cfg.enable_vad:
            pipeline.add_stage(VADSegmentationStage(
                config=VADConfig(min_duration_sec=cfg.vad_min_duration_sec,
                                 max_duration_sec=cfg.vad_max_duration_sec,
                                 threshold=cfg.vad_threshold),
                mode="batch", name="VAD_Speaker").with_(resources=gpu_res))

        if cfg.enable_band_filter:
            pipeline.add_stage(BandFilterStage(
                config=BandFilterConfig(band_value=cfg.band_value),
                name="BandFilter_Speaker").with_(resources=band_res))

        if cfg.enable_nisqa:
            pipeline.add_stage(NISQAFilterStage(
                config=NISQAConfig(mos_threshold=cfg.nisqa_mos_threshold,
                                   noi_threshold=cfg.nisqa_noi_threshold),
                name="NISQA_Speaker").with_(resources=gpu_res))

        if cfg.enable_sigmos:
            pipeline.add_stage(SIGMOSFilterStage(
                config=SIGMOSConfig(noise_threshold=cfg.sigmos_noise_threshold,
                                    ovrl_threshold=cfg.sigmos_ovrl_threshold),
                name="SIGMOS_Speaker").with_(resources=gpu_res))

    pipeline.add_stage(TimestampMapperStage())

    return pipeline


def load_audio_tasks(raw_data_dir: str,
                     formats: Tuple[str, ...] = SUPPORTED_AUDIO_FORMATS) -> list:
    audio_files = []
    for ext in formats:
        ext = ext if ext.startswith('.') else f'.{ext}'
        audio_files.extend(glob.glob(os.path.join(raw_data_dir, f"*{ext}")))
        audio_files.extend(glob.glob(os.path.join(raw_data_dir, "**", f"*{ext}"), recursive=True))
    audio_files = sorted(set(audio_files))
    if not audio_files:
        logger.warning(f"No audio files found in {raw_data_dir}")
        return []
    logger.info(f"Found {len(audio_files)} audio files")
    return [AudioBatch(data={"audio_filepath": f}, task_id=f"audio_{i:05d}",
                       dataset_name="audio_filter")
            for i, f in enumerate(audio_files)]


def save_results(results: list, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.jsonl")
    entries = []
    for r in results:
        if r and hasattr(r, "data"):
            if isinstance(r.data, list):
                entries.extend(r.data)
            elif isinstance(r.data, dict):
                entries.append(r.data)
    with open(manifest_path, "w") as f:
        for entry in entries:
            clean = {}
            for k, v in entry.items():
                if hasattr(v, "item"):
                    clean[k] = v.item()
                elif isinstance(v, (int, float, str, bool, type(None))):
                    clean[k] = v
                else:
                    clean[k] = str(v)
            f.write(json.dumps(clean) + "\n")
    logger.info(f"Saved {len(entries)} segments to {manifest_path}")


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    raw_data_dir = cfg.get("raw_data_dir")
    if not raw_data_dir or not os.path.isdir(raw_data_dir):
        logger.error(f"raw_data_dir not found: {raw_data_dir}")
        return

    tasks = load_audio_tasks(raw_data_dir)
    if not tasks:
        return

    pipeline = create_pipeline(cfg)
    logger.info(pipeline.describe())

    execution_mode = cfg.get("execution_mode", "batch")
    executor = XennaExecutor(config={"execution_mode": execution_mode})
    logger.info(f"Starting pipeline with {len(tasks)} files (mode={execution_mode})...")
    results = pipeline.run(executor, initial_tasks=tasks)

    output_dir = cfg.get("output_dir", os.path.join(raw_data_dir, "result"))
    if results:
        save_results(results, output_dir)
    else:
        logger.warning("No results")

    logger.info("Pipeline completed!")


if __name__ == "__main__":
    main()
