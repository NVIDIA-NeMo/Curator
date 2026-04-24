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

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.io import AudioToDocumentStage
from nemo_curator.tasks.audio_task import ensure_checkpoint_shard_id, ensure_sample_key

from .io_utils import normalize_for_json, write_json_atomic
from .store import SampleCheckpointRecord, StageCheckpointStore, fingerprint_stage

if TYPE_CHECKING:
    from nemo_curator.backends.base import BaseExecutor
    from nemo_curator.stages.base import ProcessingStage
    from nemo_curator.tasks import AudioTask
class AudioCheckpointRunner:
    """Run an audio pipeline stage-by-stage with checkpointing."""

    def __init__(
        self,
        *,
        pipeline: Pipeline,
        checkpoint_dir: str,
        executor: BaseExecutor | None = None,
        ignore_failed: bool = False,
    ) -> None:
        self.pipeline = pipeline
        self.checkpoint_dir = Path(checkpoint_dir)
        self.executor = executor
        self.ignore_failed = ignore_failed

    def run(self) -> list[Any] | None:
        self.pipeline.build()
        self._write_pipeline_metadata(self.pipeline.stages)
        stages = self._prepare_stages(self.pipeline.stages)
        audio_stages, tail_stages = self._split_stages(stages)

        current_tasks: list[AudioTask] | None = None
        for stage_index, stage in enumerate(audio_stages):
            store = StageCheckpointStore(
                checkpoint_dir=self.checkpoint_dir,
                stage_index=stage_index,
                stage_name=stage._name,
                config_fingerprint=fingerprint_stage(stage),
            )

            if store.is_complete():
                logger.info(f"Stage {stage_index} ({stage._name}) already checkpointed, loading outputs")
                current_tasks = store.load_output_tasks()
                continue

            try:
                if current_tasks is None:
                    output_tasks = self._run_single_stage(stage, None)
                    store.write_stage_result(input_tasks=None, output_tasks=output_tasks)
                else:
                    output_tasks, failed_records = self._run_audio_stage(stage, current_tasks)
                    store.write_stage_result(
                        input_tasks=current_tasks,
                        output_tasks=output_tasks,
                        failed_records=failed_records,
                    )
            except Exception as error:
                store.mark_failed(error)
                raise

            current_tasks = store.load_output_tasks() if self._link_stages_via_io() else output_tasks

        if tail_stages:
            tail_pipeline = Pipeline(name=f"{self.pipeline.name}_finalize", stages=tail_stages)
            return tail_pipeline.run(executor=self.executor, initial_tasks=current_tasks)
        return current_tasks

    def _prepare_stages(self, stages: list[ProcessingStage]) -> list[ProcessingStage]:
        return list(stages)

    def _split_stages(self, stages: list[ProcessingStage]) -> tuple[list[ProcessingStage], list[ProcessingStage]]:
        for index, stage in enumerate(stages):
            if isinstance(stage, AudioToDocumentStage):
                return stages[:index], stages[index:]
        return stages, []

    def _run_audio_stage(
        self, stage: ProcessingStage, input_tasks: list[AudioTask]
    ) -> tuple[list[AudioTask], list[SampleCheckpointRecord]]:
        retry_tasks = deepcopy(input_tasks) if self.ignore_failed else None
        try:
            return self._run_single_stage(stage, input_tasks), []
        except Exception as error:
            if not self.ignore_failed:
                raise

            self._cleanup_retry_artifacts(input_tasks)
            logger.warning(
                f"Stage {stage._name} failed for a batch of {len(input_tasks)} tasks, retrying one-by-one: {error}"
            )
            outputs: list[AudioTask] = []
            failed_records: list[SampleCheckpointRecord] = []
            for task in retry_tasks or input_tasks:
                stage_outputs, failed_record = self._run_single_task_with_retry(stage, task)
                outputs.extend(stage_outputs)
                if failed_record is not None:
                    failed_records.append(
                        failed_record
                    )
            return outputs, failed_records

    def _run_single_task_with_retry(
        self,
        stage: ProcessingStage,
        task: AudioTask,
    ) -> tuple[list[AudioTask], SampleCheckpointRecord | None]:
        try:
            return self._run_single_stage(stage, [task]), None
        except Exception as task_error:  # noqa: BLE001
            self._cleanup_retry_artifacts([task])
            return [], SampleCheckpointRecord(
                sample_key=ensure_sample_key(task),
                status="failed_retriable",
                checkpoint_shard_id=ensure_checkpoint_shard_id(task),
                task=None,
                error_type=type(task_error).__name__,
                error_message=str(task_error),
            )

    def _run_single_stage(self, stage: ProcessingStage, initial_tasks: list[AudioTask] | None) -> list[AudioTask]:
        stage_pipeline = Pipeline(name=f"checkpoint_stage_{stage._name}", stages=[stage])
        results = stage_pipeline.run(executor=self.executor, initial_tasks=initial_tasks)
        if results is None:
            return []
        return results

    def _write_pipeline_metadata(self, stages: list[ProcessingStage]) -> None:
        payload = {
            "pipeline_name": self.pipeline.name,
            "description": self.pipeline.description,
            "config": normalize_for_json(self.pipeline.config),
            "ignore_failed": self.ignore_failed,
            "link_stages_via_io": self._link_stages_via_io(),
            "stages": [stage._name for stage in stages],
        }
        write_json_atomic(self.checkpoint_dir / "pipeline.json", payload)

    def _link_stages_via_io(self) -> bool:
        return bool(self.pipeline.config.get("link_stages_via_io", False))

    def _cleanup_retry_artifacts(self, tasks: list[AudioTask]) -> None:
        for task in tasks:
            temp_path = task.data.get("_temporary_audio_path")
            if not temp_path:
                continue
            path = Path(temp_path)
            try:
                if path.exists():
                    path.unlink()
            except OSError as error:
                logger.warning(f"Failed to cleanup retry artifact {path}: {error}")
