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

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from loguru import logger

from nemo_curator.backends.base import BaseExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.io import AudioToDocumentStage, MaterializeTarredAudioStage
from nemo_curator.tasks import AudioTask
from nemo_curator.tasks.audio_task import ensure_sample_key

from .store import SampleCheckpointRecord, StageCheckpointStore, fingerprint_stage


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as temp:
        json.dump(payload, temp, indent=2, sort_keys=True)
        temp.flush()
        temp_path = Path(temp.name)
    temp_path.replace(path)


class AudioCheckpointRunner:
    """Run an audio pipeline stage-by-stage with checkpointing."""

    def __init__(
        self,
        *,
        pipeline: Pipeline,
        checkpoint_dir: str,
        executor: BaseExecutor | None = None,
        ignore_failed: bool = False,
        materialization_dir: str | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.checkpoint_dir = Path(checkpoint_dir)
        self.executor = executor
        self.ignore_failed = ignore_failed
        self.materialization_dir = materialization_dir

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

    def _prepare_stages(self, stages: list[Any]) -> list[Any]:  # noqa: ANN401
        prepared = list(stages)
        materialization_dir = self._effective_materialization_dir()
        if materialization_dir is None:
            return prepared

        for stage in prepared:
            if isinstance(stage, MaterializeTarredAudioStage) and stage.materialization_dir is None:
                stage.materialization_dir = materialization_dir
        return prepared

    def _split_stages(self, stages: list[Any]) -> tuple[list[Any], list[Any]]:  # noqa: ANN401
        for index, stage in enumerate(stages):
            if isinstance(stage, AudioToDocumentStage):
                return stages[:index], stages[index:]
        return stages, []

    def _run_audio_stage(
        self, stage: Any, input_tasks: list[AudioTask]  # noqa: ANN401
    ) -> tuple[list[AudioTask], list[SampleCheckpointRecord]]:
        try:
            return self._run_single_stage(stage, input_tasks), []
        except Exception as error:
            if not self.ignore_failed:
                raise

            logger.warning(
                f"Stage {stage._name} failed for a batch of {len(input_tasks)} tasks, retrying one-by-one: {error}"
            )
            outputs: list[AudioTask] = []
            failed_records: list[SampleCheckpointRecord] = []
            for task in input_tasks:
                try:
                    outputs.extend(self._run_single_stage(stage, [task]))
                except Exception as task_error:
                    failed_records.append(
                        SampleCheckpointRecord(
                            sample_key=ensure_sample_key(task),
                            status="failed_retriable",
                            task=None,
                            error_type=type(task_error).__name__,
                            error_message=str(task_error),
                        )
                    )
            return outputs, failed_records

    def _run_single_stage(self, stage: Any, initial_tasks: list[AudioTask] | None) -> list[AudioTask]:  # noqa: ANN401
        stage_pipeline = Pipeline(name=f"checkpoint_stage_{stage._name}", stages=[stage])
        results = stage_pipeline.run(executor=self.executor, initial_tasks=initial_tasks)
        if results is None:
            return []
        return results

    def _write_pipeline_metadata(self, stages: list[Any]) -> None:  # noqa: ANN401
        payload = {
            "pipeline_name": self.pipeline.name,
            "description": self.pipeline.description,
            "config": self.pipeline.config,
            "ignore_failed": self.ignore_failed,
            "link_stages_via_io": self._link_stages_via_io(),
            "materialization_dir": self._effective_materialization_dir(),
            "stages": [stage._name for stage in stages],
        }
        _write_json_atomic(self.checkpoint_dir / "pipeline.json", payload)

    def _link_stages_via_io(self) -> bool:
        return bool(self.pipeline.config.get("link_stages_via_io", False))

    def _effective_materialization_dir(self) -> str | None:
        if self.materialization_dir is not None:
            return self.materialization_dir
        if self._link_stages_via_io():
            return str(self.checkpoint_dir / "artifacts" / "materialized_audio")
        return None
