# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Run the executor-backed generation and keyed conversation-merge phases."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import hydra
from loguru import logger

if TYPE_CHECKING:
    from omegaconf import DictConfig

from nemo_curator.config.run import (
    create_executor_from_yaml,
    create_pipeline_from_yaml,
    create_ray_client_from_yaml,
)
from nemo_curator.tasks import group_tasks_by_data_key


def _run_phase(cfg: DictConfig, *, initial_tasks: list[Any] | None = None) -> list[Any]:
    """Build and execute one declarative pipeline phase through its executor."""
    pipeline = create_pipeline_from_yaml(cfg)
    executor = create_executor_from_yaml(cfg)
    results = pipeline.run(executor=executor, initial_tasks=initial_tasks)
    return results or []


@hydra.main(version_base=None, config_name="pipeline")
def main(cfg: DictConfig) -> None:
    """Generate turn audio, then merge explicitly grouped conversations."""
    ray_client = create_ray_client_from_yaml(cfg)
    ray_client.start()
    try:
        phase1_results = _run_phase(cfg.phase1)
        if not phase1_results:
            logger.warning("Generation phase produced no turns; nothing to merge.")
            return

        groups = group_tasks_by_data_key(phase1_results, "conversation_id")
        logger.info(f"Merging {len(groups)} complete conversations from {len(phase1_results)} generated turns.")
        phase2_results = _run_phase(cfg.phase2, initial_tasks=groups)
        logger.info(f"Pipeline complete: wrote {len(phase2_results)} merged conversations.")
    finally:
        ray_client.stop()


if __name__ == "__main__":
    main()
