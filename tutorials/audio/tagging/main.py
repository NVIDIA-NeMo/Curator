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
Audio Tagging Pipeline for NeMo Curator.

Processes raw audio data through diarization, ASR alignment, text
normalization, quality metrics, and segment preparation to produce
labelled training manifests for TTS or ASR.

The pipeline is YAML-driven via Hydra and supports both TTS and ASR
modalities by switching the configuration file.

Usage:
    # TTS pipeline with bundled sample data (from Curator repo root)
    python tutorials/audio/tagging/main.py \\
        --config-path . \\
        --config-name tts_pipeline \\
        input_manifest=tests/fixtures/audio/tagging/sample_input.jsonl \\
        final_manifest=/tmp/tts_output.jsonl \\
        hf_token=<your_hf_token>

    # Override backend
    python tutorials/audio/tagging/main.py \\
        --config-path . \\
        --config-name tts_pipeline \\
        input_manifest=tests/fixtures/audio/tagging/sample_input.jsonl \\
        final_manifest=/tmp/tts_output.jsonl \\
        hf_token=<your_hf_token> \\
        backend=ray_data

    # Override parameters
    python tutorials/audio/tagging/main.py \\
        --config-path . \\
        --config-name tts_pipeline \\
        input_manifest=tests/fixtures/audio/tagging/sample_input.jsonl \\
        final_manifest=/tmp/output.jsonl \\
        hf_token=<your_hf_token> \\
        max_segment_length=30 \\
        stages.4.batch_size=16
"""

import importlib
import json
import traceback
from pathlib import Path
from typing import Any

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nemo_curator.config.run import create_pipeline_from_yaml
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


def _load_stats_summary(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    summary_path = Path(path)
    if not summary_path.exists():
        logger.warning(f"Stats summary path does not exist: {summary_path}")
        return None
    with summary_path.open(encoding="utf-8") as f:
        return json.load(f)


def _log_stats_summary(summary: dict[str, Any] | None, path: str | None) -> None:
    if not summary:
        return
    logger.info("  ASR transcript stats summary:")
    if path:
        logger.info(f"    Summary JSON: {path}")
    logger.info(
        "    transcripts: "
        f"total={summary.get('total_transcripts', 0)} "
        f"valid={summary.get('valid_transcripts', 0)} "
        f"invalid={summary.get('invalid_transcripts', 0)} "
        f"dropped={summary.get('dropped_invalid', 0)} "
        f"emitted={summary.get('emitted_transcripts', 0)}"
    )
    logger.info(
        "    hours: "
        f"total={float(summary.get('total_duration_hours', 0.0)):.2f} "
        f"valid={float(summary.get('valid_duration_hours', 0.0)):.2f} "
        f"invalid={float(summary.get('invalid_duration_hours', 0.0)):.2f}"
    )
    logger.info(
        "    chars: "
        f"total={summary.get('total_chars', 0)} "
        f"unique_known={summary.get('unique_known_chars', 0)} "
        f"unique_unknown={summary.get('unique_unknown_chars', 0)}"
    )
    logger.info(f"    split_counts: {summary.get('split_counts', {})}")
    logger.info(f"    split_hours: {summary.get('split_hours', {})}")


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    """Run audio tagging pipeline using Hydra configuration."""
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

        logger.info("Starting audio tagging pipeline...")
        results = pipeline.run(executor)
    except Exception:
        logger.error("Audio pipeline failed with full chained traceback:\n{}", traceback.format_exc())
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

    stats_summary_path = _resolve_stats_summary_path(cfg)
    _log_stats_summary(_load_stats_summary(stats_summary_path), stats_summary_path)

    stage_metrics = TaskPerfUtils.collect_stage_metrics(results)
    for stage_name, metrics in stage_metrics.items():
        logger.info(f"  [{stage_name}]")
        logger.info(
            f"    process_time: mean={metrics['process_time'].mean():.4f}s, total={metrics['process_time'].sum():.2f}s"
        )
        logger.info(f"    items_processed: {metrics['num_items_processed'].sum():.0f}")


if __name__ == "__main__":
    main()
