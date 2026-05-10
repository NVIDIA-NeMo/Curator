# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os
from typing import Any

from loguru import logger

from nemo_curator.backends.base import BaseExecutor
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.tasks import Task


class Pipeline:
    """User-facing pipeline definition for composing processing stages."""

    def __init__(
        self,
        name: str,
        description: str | None = None,
        stages: list[ProcessingStage] | None = None,
        config: dict[str, Any] | None = None,
    ):
        """Initialize a new pipeline.

        Args:
            name (str): Name of the pipeline
            description (str, optional): Pipeline Description. Defaults to None.
            stages (list[ProcessingStage], optional): List of stages to add to the pipeline. Defaults to None.
            config (dict[str, Any], optional): Pipeline configuration that is valid across all executors. Defaults to None.
        """
        self.name = name
        self.description = description
        self.stages: list[ProcessingStage] = stages or []
        self.config = config or {}

    def add_stage(self, stage: ProcessingStage) -> "Pipeline":
        """Add a stage to the pipeline.

        Args:
            stage (ProcessingStage): Processing stage to add

        Returns:
            Pipeline: Self (Pipeline) for method chaining
        """
        if not isinstance(stage, ProcessingStage):
            msg = f"Stage must be a ProcessingStage, got {type(stage)}"
            raise TypeError(msg)

        self.stages.append(stage)
        logger.info(f"Added stage '{stage.name}' to pipeline '{self.name}'")
        return self

    def build(self) -> None:
        """Build an execution plan from the pipeline.

        Raises:
            ValueError: If the pipeline has no stages
        """
        logger.info(f"Planning pipeline: {self.name}")

        # 1. Validate pipeline has stages
        if not self.stages:
            msg = f"Pipeline '{self.name}' has no stages"
            raise ValueError(msg)

        # 2. Decompose composite stages into execution stages
        execution_stages, decomposition_info = self._decompose_stages(self.stages)

        self.stages = execution_stages
        self.decomposition_info = decomposition_info

    def _decompose_stages(
        self, stages: list[ProcessingStage | CompositeStage]
    ) -> tuple[list[ProcessingStage], dict[str, list[str]]]:
        """Decompose composite stages into execution stages.

        Args:
            stages (list[ProcessingStage  |  CompositeStage]): List of stages that may include composite stages

        Raises:
            TypeError: If a composite stage is decomposed into another composite stage

        Returns:
            tuple[list[ProcessingStage], dict[str, list[str]]]: Tuple of (execution stages, decomposition info dict)
        """
        execution_stages = []
        decomposition_info = {}

        for stage in stages:
            # Get the decomposed stages (returns [self] for regular stages)
            sub_stages = stage.decompose_and_apply_with() if isinstance(stage, CompositeStage) else [stage]

            if len(sub_stages) > 1:
                # This was a composite stage
                logger.info(f"Decomposing composite stage: {stage.name}")

                # Validate that decomposed stages are not composite
                for sub_stage in sub_stages:
                    if isinstance(sub_stage, CompositeStage) and len(sub_stage.decompose()) > 1:
                        msg = (
                            f"Composite stage '{stage.name}' decomposed into another "
                            f"composite stage '{sub_stage.name}'. Nested composition "
                            "is not supported."
                        )
                        raise TypeError(msg)

                execution_stages.extend(sub_stages)
                decomposition_info[stage.name] = [s.name for s in sub_stages]
                logger.info(f"Expanded '{stage.name}' into {len(sub_stages)} execution stages")
            else:
                # Regular stage, add as-is
                execution_stages.append(stage)

        return execution_stages, decomposition_info

    def __repr__(self) -> str:
        """String representation of the pipeline."""
        stage_info = ", ".join([f"{s.name}({s.__class__.__name__})" for s in self.stages])
        return f"Pipeline(name='{self.name}', stages=[{stage_info}])"

    def describe(self) -> str:
        """Get a detailed description of the pipeline stages and their requirements."""
        lines = [
            f"Pipeline: {self.name}",
            f"Description: {self.description or 'No description provided'}",
            f"Stages: {len(self.stages)}",
            "",
        ]

        for i, stage in enumerate(self.stages):
            lines.append(f"Stage {i + 1}: {stage.name}")

            try:
                required_attrs, required_cols = stage.inputs()
                output_attrs, output_cols = stage.outputs()

                lines.append(f"  Resources: {stage.resources.cpus} CPUs")
                if stage.resources.requires_gpu:
                    lines.append(f"    GPU Memory: {stage.resources.gpu_memory_gb} GB ({stage.resources.gpus} GPUs)")

                lines.append(f"  Batch size: {stage.batch_size}")

                # Input requirements
                if required_attrs or required_cols:
                    lines.append("  Inputs:")
                    if required_attrs:
                        lines.append(f"    Required attributes: {', '.join(required_attrs)}")
                    if required_cols:
                        lines.append(f"    Required columns: {', '.join(required_cols)}")

                # Output specification
                if output_attrs or output_cols:
                    lines.append("  Outputs:")
                    if output_attrs:
                        lines.append(f"    Output attributes: {', '.join(output_attrs)}")
                    if output_cols:
                        lines.append(f"    Output columns: {', '.join(output_cols)}")

            except Exception as e:  # noqa: BLE001
                lines.append(f"  Error getting stage info: {e}")

        lines.append("")

        return "\n".join(lines)

    def run(
        self,
        executor: BaseExecutor | None = None,
        initial_tasks: list[Task] | None = None,
        checkpoint_path: str | None = None,
    ) -> list[Task] | None:
        """Run the pipeline.

        Args:
            executor (BaseExecutor): Executor to use
            initial_tasks (list[Task], optional): Initial tasks to start the pipeline with. Defaults to None.
            checkpoint_path (str, optional): Directory for resumability checkpoints (LMDB).
                If set, completed source partitions are skipped on subsequent runs.
                Relative paths are resolved to absolute on the driver. Defaults to None.

                .. warning::
                    Checkpointing and resumability are **experimental**. Support is currently
                    limited to pipelines without fan-in stages (stages that merge multiple
                    upstream tasks into one). Pipelines containing fan-in stages may produce
                    incorrect resume behavior.

        Returns:
            list[Task] | None: List of tasks
        """
        self.build()

        if executor is None:
            from nemo_curator.backends.xenna import XennaExecutor

            executor = XennaExecutor()

        from nemo_curator.core.serve import is_inference_server_active

        if is_inference_server_active():
            gpu_stages = [s for s in self.stages if s.resources.requires_gpu]
            if gpu_stages:
                names = ", ".join(s.name for s in gpu_stages)
                from nemo_curator.backends.xenna import XennaExecutor

                if isinstance(executor, XennaExecutor):
                    msg = (
                        f"Cannot run XennaExecutor with GPU stages [{names}] while Ray Serve is active. "
                        "Xenna manages GPU assignment independently of Ray's resource scheduler, "
                        "which causes GPU contention with served models. "
                        "Use RayDataExecutor instead."
                    )
                    raise RuntimeError(msg)
                logger.info(
                    f"Ray Serve is active and pipeline has GPU stages: [{names}]. "
                    "The executor will schedule GPU stages on GPUs not held by Serve."
                )

        if checkpoint_path:
            logger.warning(
                "Checkpointing and resumability are experimental. Support is currently limited to "
                "pipelines without fan-in stages (stages that merge multiple upstream tasks into one). "
                "Pipelines containing fan-in stages may produce incorrect resume behavior."
            )
            if "://" not in checkpoint_path:
                checkpoint_path = os.path.abspath(checkpoint_path)
            stages = self._with_checkpoint_stages(self.stages, checkpoint_path)
        else:
            self._clear_checkpoint_attrs(self.stages)
            stages = self.stages

        return executor.execute(stages, initial_tasks)

    def _with_checkpoint_stages(self, stages: list[ProcessingStage], checkpoint_path: str) -> list[ProcessingStage]:
        """Return a new stage list with checkpoint filter/recorder stages injected.

        Does NOT mutate ``self.stages``.  The filter is inserted after the first
        ``is_source_stage()`` stage only; the recorder is appended at the end.
        """
        from nemo_curator.stages.checkpoint import _CheckpointFilterStage, _CheckpointRecorderStage

        source_indices = [i for i, s in enumerate(stages) if s.is_source_stage()]
        if not source_indices:
            msg = (
                "Resumability is enabled (checkpoint_path is set) but no source stage found in the "
                "pipeline. Mark a source stage with is_source_stage() = True and ensure it sets "
                "_metadata['resumability_key'] on each output task."
            )
            raise ValueError(msg)

        result: list[ProcessingStage] = []
        first_source_idx = source_indices[0]
        filter_inserted = False

        for i, stage in enumerate(stages):
            stage._checkpoint_path = checkpoint_path
            result.append(stage)
            if i == first_source_idx and not filter_inserted:
                result.append(_CheckpointFilterStage(checkpoint_path))
                filter_inserted = True

        result.append(_CheckpointRecorderStage(checkpoint_path))
        return result

    def _clear_checkpoint_attrs(self, stages: list[ProcessingStage]) -> None:
        """Remove checkpoint attrs stamped by a previous run (prevents stale state)."""
        for stage in stages:
            stage.__dict__.pop("_checkpoint_path", None)
