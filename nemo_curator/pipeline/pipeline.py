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

from pathlib import Path
from typing import Any

from loguru import logger

from nemo_curator.backends.base import BaseExecutor
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.tasks import EmptyTask, Task


def assign_root_task_ids(initial_tasks: list[Task]) -> list[Task]:
    """Assign root ``task_id``s to user-provided initial tasks.

    Every task in a run descends from the implicit root ``"0"`` (the id of
    :class:`EmptyTask`). User-provided initial tasks are its direct
    children, so they get ``"0_0"``, ``"0_1"``, … ``EmptyTask`` instances
    are skipped (already ``"0"``). All downstream ``task_id`` assignment
    happens in ``BaseStageAdapter``.

    NOTE: we deliberately use the positional index here, NOT
    ``get_deterministic_id()``, even for content-bearing tasks like
    ``FileGroupTask``. The source stage is the single place content-based
    ids are assigned (to its outputs); hashing here too would put the
    content hash at two levels of the id path (``"0_<hashA>_<hashB>"``).
    Passing initial tasks directly is rare; if you need reorder-stable
    source ids, let a source stage emit them.
    """
    for i, task in enumerate(initial_tasks):
        if isinstance(task, EmptyTask):
            continue
        task._set_task_id("0", i)
    return initial_tasks


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

        # 3. Source / sink defaults: at most one stage may be explicitly
        # marked; if none, the first stage is the source and the last is
        # the sink. The source flag activates content-based ids in the
        # default ``process_batch``; the sink flag tells the resumability
        # counters that a sink consumes its outputs (see
        # ``BaseStageAdapter._apply_resumability_counters``).
        self._assign_source_sink_roles()

    def _assign_source_sink_roles(self) -> None:
        explicit_sources = [s for s in self.stages if s.is_source_stage]
        if len(explicit_sources) > 1:
            names = [s.name for s in explicit_sources]
            msg = f"Pipeline has multiple source stages marked: {names}. At most one is supported."
            raise ValueError(msg)
        if not explicit_sources:
            self.stages[0].is_source_stage = True

        explicit_sinks = [s for s in self.stages if s.is_sink_stage]
        if len(explicit_sinks) > 1:
            names = [s.name for s in explicit_sinks]
            msg = f"Pipeline has multiple sink stages marked: {names}. At most one is supported."
            raise ValueError(msg)
        if not explicit_sinks:
            self.stages[-1].is_sink_stage = True

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

    def run(  # noqa: C901, PLR0912
        self,
        executor: BaseExecutor | None = None,
        initial_tasks: list[Task] | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> list[Task] | None:
        """Run the pipeline.

        Args:
            executor (BaseExecutor): Executor to use
            initial_tasks (list[Task], optional): Initial tasks to start the pipeline with. Defaults to None.
            checkpoint_path (str | Path, optional): Resumability directory. Must
                be a LOCAL filesystem path (the LMDB state is written locally),
                not a remote/cloud URI. When set, completed source partitions are
                tracked (in a ``.nemo_curator_metadata`` subdir) and skipped on
                rerun. Multiple runs (e.g. a SLURM array) may share the directory
                — each writes its own LMDB file, so there is no contention.

        Returns:
            list[Task] | None: List of tasks
        """
        self.build()

        if checkpoint_path is not None:
            non_resumable = [s.name for s in self.stages if not s.is_resumable]
            if non_resumable:
                msg = (
                    f"checkpoint_path was set, but these stages are not marked resumable: "
                    f"{non_resumable}. Set is_resumable=True on a stage only once you've "
                    f"confirmed its input→output mapping is resumability-safe."
                )
                raise ValueError(msg)
            checkpoint_path = Path(checkpoint_path).absolute()
            checkpoint_path.mkdir(parents=True, exist_ok=True)

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

        if initial_tasks:
            assign_root_task_ids(initial_tasks)

        from nemo_curator.backends.failed_task_markers import (
            configure_slurm_array_failed_task_manifest_dir,
            failed_task_manifest_exists,
        )
        from nemo_curator.backends.slurm_array import (
            SlurmArrayConfig,
            build_slurm_array_completion_manifest,
            is_slurm_array_driver_process,
        )

        slurm_array = SlurmArrayConfig.from_env()
        completion_manifest = None
        if slurm_array is not None:
            is_driver = is_slurm_array_driver_process()
            if checkpoint_path is not None:
                configure_slurm_array_failed_task_manifest_dir(checkpoint_path, slurm_array.shard_index)
            completion_manifest = build_slurm_array_completion_manifest(
                checkpoint_path=checkpoint_path if is_driver else None,
                shard_index=slurm_array.shard_index,
                total_shards=slurm_array.total_shards,
                minimum_shard_index=slurm_array.minimum_shard_index,
            )

        if checkpoint_path is None:
            result = executor.execute(self.stages, initial_tasks)
        else:
            result = self._run_with_resumability(executor, initial_tasks, checkpoint_path)

        if completion_manifest is not None:
            if failed_task_manifest_exists():
                logger.warning(
                    "Pipeline completed without raising, but a FailedTask manifest exists. "
                    "The shard remains incomplete and will be selected for retry."
                )
            else:
                manifest_file = completion_manifest.mark_completed()
                logger.info(f"Wrote Slurm array completion manifest to {manifest_file}")

        return result

    def _run_with_resumability(
        self,
        executor: BaseExecutor,
        initial_tasks: list[Task] | None,
        checkpoint_path: Path,
    ) -> list[Task] | None:
        """Run with resumability around a pre-existing Ray cluster (e.g. one
        started by ``RayClient``).

        We briefly connect with ``with ray.init()`` to spawn the detached
        checkpoint actor, then disconnect *before* ``executor.execute`` so the
        executor's own ``ray.init`` runs un-nested — a nested
        ``ray.init(runtime_env=...)`` is silently dropped, so the executor's env
        vars wouldn't propagate otherwise. The detached actor lives in the
        cluster across the executor's separate Ray session; a final
        ``with ray.init()`` closes and kills it. The cluster must pre-exist: had
        we started it, the first ``with``-exit shutdown would tear it down and
        take the actor with it.
        """
        import os

        import ray

        from nemo_curator.utils.resumability_actor import create_resumability_actor, shutdown_resumability_actor

        if not os.environ.get("RAY_ADDRESS"):
            msg = (
                "Resumability (checkpoint_path) requires a Ray cluster started before pipeline.run() — "
                "start one with RayClient().start() (or the SLURM Ray client). Without a pre-existing "
                "cluster the checkpoint actor would be torn down with this run's Ray session."
            )
            raise RuntimeError(msg)

        with ray.init(ignore_reinit_error=True):
            create_resumability_actor(str(checkpoint_path))
        try:
            return executor.execute(self.stages, initial_tasks)
        finally:
            with ray.init(ignore_reinit_error=True):
                shutdown_resumability_actor()
