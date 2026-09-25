# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ASR data processing pipeline runner for NeMo Curator.

This YAML-driven runner is intended for downloaded ASR datasets that already
include transcripts. It runs ingestion, transcript normalization, transcript
statistics, and split-aware manifest writing stages.

Usage:
    python tutorials/audio/asr_data_pipeline/main.py \\
        --config-path ../../../configs \\
        --config-name indicvoices \\
        raw_data_dir=/data/asr/IndicVoices/raw \\
        output_dir=/data/asr/IndicVoices/curated \\
        'langs=[gu]'
"""

import importlib
import traceback

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nemo_curator.config.run import create_pipeline_from_yaml
from nemo_curator.stages.audio.asr.normalization.stats import TranscriptStatsStage
from nemo_curator.tasks.utils import TaskPerfUtils

_EXECUTOR_FACTORIES = {
    "xenna": "nemo_curator.backends.xenna:XennaExecutor",
    "ray_data": "nemo_curator.backends.ray_data:RayDataExecutor",
}


def _create_executor(backend: str, config: dict) -> object:
    module_path, class_name = _EXECUTOR_FACTORIES[backend].rsplit(":", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)(config=config)


def _validate_backend(backend: str) -> None:
    if backend not in _EXECUTOR_FACTORIES:
        msg = f"Unknown backend '{backend}'. Choose from: {list(_EXECUTOR_FACTORIES)}"
        raise ValueError(msg)


def _resolve_stats_summary_path(cfg: DictConfig) -> str | None:
    path = OmegaConf.select(cfg, "stats_summary_path")
    if path:
        return str(path)
    for stage_cfg in cfg.get("stages", []):
        path = stage_cfg.get("output_summary_path")
        if path:
            return str(path)
    return None


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    """Run an ASR data processing pipeline using Hydra configuration."""
    try:
        pipeline = create_pipeline_from_yaml(cfg)

        logger.info(pipeline.describe())
        logger.info("\n" + "=" * 50 + "\n")

        backend = cfg.get("backend", "xenna")
        _validate_backend(backend)
        logger.info(f"Using backend: {backend}")
        mode = cfg.get("execution_mode", "streaming")
        config = {"execution_mode": mode}
        executor = _create_executor(backend, config=config)

        logger.info("Starting ASR data processing pipeline...")
        results = pipeline.run(executor)
    except Exception:
        logger.error("ASR data pipeline failed with full chained traceback:\n{}", traceback.format_exc())
        raise

    num_tasks = len(results) if results else 0

    logger.info("\n" + "=" * 50)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 50)
    logger.info(f"  Tasks processed: {num_tasks}")
    if "final_manifest" in cfg:
        logger.info(f"  Output manifest: {cfg.final_manifest}")
    elif "output_dir" in cfg:
        logger.info(f"  Output directory: {cfg.output_dir}")

    TranscriptStatsStage.log_summary_from_path(_resolve_stats_summary_path(cfg))

    stage_metrics = TaskPerfUtils.collect_stage_metrics(results)
    for stage_name, metrics in stage_metrics.items():
        logger.info(f"  [{stage_name}]")
        logger.info(
            f"    process_time: mean={metrics['process_time'].mean():.4f}s, total={metrics['process_time'].sum():.2f}s"
        )
        logger.info(f"    items_processed: {metrics['num_items_processed'].sum():.0f}")


if __name__ == "__main__":
    main()
