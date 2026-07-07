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
Multi-Speaker Conversation Data Generation Pipeline for NeMo Curator.

Generates synthetic multi-speaker conversation audio from topic prompts
through four phases: LLM conversation generation, TTS synthesis, MFA
forced alignment, and SDP-style conversation merging.

The pipeline runs in two execution phases:

**Phase 1** (GPU): ManifestReader -> vLLMInference -> ChatterboxTTSStage
    -> MFAAlignmentStage.  Each stage produces one ``AudioTask`` per
    conversation turn, processed via the standard Curator executor.

**Phase 2** (CPU): MergeConversationSDPStage -> ManifestWriterStage.
    The merge stage requires all turns of a single conversation in one
    batch.  This runner groups phase-1 output by ``conversation_id``
    before calling ``process_batch`` per conversation.

Usage:
    python tutorials/audio/data-generation/main.py \\
        --config-path . \\
        --config-name pipeline \\
        input_manifest=topics/topics_dialog.jsonl \\
        output_dir=/data/output \\
        reference_voices_dataset=/data/voices \\
        prompt_file=prompts/dialog_prompt.yaml

    # Override LLM model
    python tutorials/audio/data-generation/main.py \\
        --config-path . \\
        --config-name pipeline \\
        input_manifest=topics/topics_dialog.jsonl \\
        output_dir=/data/output \\
        reference_voices_dataset=/data/voices \\
        prompt_file=prompts/dialog_prompt.yaml \\
        llm_model=Qwen/Qwen2.5-7B-Instruct-1M
"""

from __future__ import annotations

import importlib
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nemo_curator.config.run import _instantiate_stage
from nemo_curator.pipeline import Pipeline

if TYPE_CHECKING:
    from nemo_curator.tasks import AudioTask

_EXECUTOR_FACTORIES = {
    "xenna": "nemo_curator.backends.xenna:XennaExecutor",
    "ray_data": "nemo_curator.backends.ray_data:RayDataExecutor",
}

_PHASE1_STAGE_COUNT = 4


def _create_executor(backend: str, config: dict | None = None) -> object:
    module_path, class_name = _EXECUTOR_FACTORIES[backend].rsplit(":", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    if config:
        return cls(config=config)
    return cls()


def _group_tasks_by_conversation(
    tasks: list[AudioTask],
) -> dict[str, list[AudioTask]]:
    """Group tasks by ``conversation_id`` for the merge stage."""
    groups: dict[str, list[AudioTask]] = defaultdict(list)
    for task in tasks:
        conv_id = task.data.get("conversation_id", "unknown")
        groups[conv_id].append(task)
    return dict(groups)


def _run_phase2(
    cfg: DictConfig,
    phase1_results: list[AudioTask],
) -> list[AudioTask]:
    """Run merge + manifest-writer on phase-1 output.

    Groups tasks by ``conversation_id`` and calls
    ``MergeConversationSDPStage.process_batch`` per conversation, then
    writes the merged manifest.
    """
    stage_cfgs = list(cfg.stages)
    merge_cfg = stage_cfgs[_PHASE1_STAGE_COUNT]
    writer_cfg = stage_cfgs[_PHASE1_STAGE_COUNT + 1]

    merge_stage = _instantiate_stage(merge_cfg)
    writer_stage = _instantiate_stage(writer_cfg)

    merge_stage.setup()
    writer_stage.setup()

    conversation_groups = _group_tasks_by_conversation(phase1_results)
    logger.info(f"Phase 2: merging {len(conversation_groups)} conversations from {len(phase1_results)} turns")

    merged_tasks: list[AudioTask] = []
    for conv_id, turns in conversation_groups.items():
        logger.info(f"  Merging conversation {conv_id} ({len(turns)} turns)")
        try:
            result = merge_stage.process_batch(turns)
            merged_tasks.extend(result)
        except (OSError, RuntimeError, ValueError):
            logger.exception(f"Failed to merge conversation {conv_id}")

    logger.info(f"Phase 2: writing {len(merged_tasks)} merged conversations")
    written_tasks: list[AudioTask] = []
    for task in merged_tasks:
        result = writer_stage.process(task)
        written_tasks.append(result)

    merge_stage.teardown()
    writer_stage.teardown()

    return written_tasks


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the two-phase conversation data generation pipeline."""
    logger.info("Hydra config:\n" + OmegaConf.to_yaml(cfg))

    stage_cfgs = list(cfg.stages)
    total_stages = len(stage_cfgs)
    if total_stages < _PHASE1_STAGE_COUNT + 2:
        msg = f"Expected at least {_PHASE1_STAGE_COUNT + 2} stages (phase1 + merge + writer), got {total_stages}"
        raise ValueError(msg)

    # --- Phase 1: LLM -> TTS -> MFA (standard pipeline) ---
    phase1_pipeline = Pipeline(
        name="phase1_generation",
        description="LLM conversation generation, TTS synthesis, and MFA alignment",
    )
    for stage_cfg in stage_cfgs[:_PHASE1_STAGE_COUNT]:
        stage = _instantiate_stage(stage_cfg)
        phase1_pipeline.add_stage(stage)

    logger.info(phase1_pipeline.describe())
    logger.info("\n" + "=" * 60)

    backend = cfg.get("backend", "xenna")
    if backend not in _EXECUTOR_FACTORIES:
        msg = f"Unknown backend '{backend}'. Choose from: {list(_EXECUTOR_FACTORIES)}"
        raise ValueError(msg)
    logger.info(f"Using backend: {backend}")

    mode = cfg.get("execution_mode", "streaming")
    executor = _create_executor(backend, config={"execution_mode": mode})

    logger.info("Phase 1: Starting LLM -> TTS -> MFA pipeline...")
    phase1_results = phase1_pipeline.run(executor)

    if not phase1_results:
        logger.warning("Phase 1 produced no results. Nothing to merge.")
        return

    logger.info(f"Phase 1 complete: {len(phase1_results)} tasks produced")
    logger.info("=" * 60)

    # --- Phase 2: Group by conversation -> Merge -> Write ---
    logger.info("Phase 2: Starting conversation merge...")
    final_results = _run_phase2(cfg, phase1_results)

    # --- Summary ---
    output_path = OmegaConf.select(cfg, "stages")[-1].get("output_path", "output/merged_manifest.jsonl")
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Conversations merged: {len(final_results)}")
    logger.info(f"  Output manifest: {output_path}")

    output_file = Path(output_path)
    if output_file.exists():
        with open(output_file) as f:
            line_count = sum(1 for _ in f)
        logger.info(f"  Manifest entries: {line_count}")


if __name__ == "__main__":
    main()
