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

from unittest.mock import Mock, patch

from ray.data import ActorPoolStrategy

from nemo_curator.backends.experimental.ray_data.adapter import RayDataStageAdapter
from nemo_curator.backends.experimental.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch


class ActorStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    name = "actor_stage"

    def ray_stage_spec(self) -> dict[str, bool]:
        return {RayStageSpecKeys.IS_ACTOR_STAGE: True}

    def process(self, task: DocumentBatch) -> DocumentBatch:
        return task


class TaskStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    name = "task_stage"

    def process(self, task: DocumentBatch) -> DocumentBatch:
        return task


class TestRayDataStageAdapter:
    @patch("nemo_curator.backends.experimental.ray_data.adapter.create_actor_from_stage", return_value=Mock())
    @patch("nemo_curator.backends.experimental.ray_data.adapter.calculate_concurrency_for_actors_for_stage")
    def test_actor_stages_use_compute_strategy(self, mock_calculate_concurrency: Mock, mock_create_actor: Mock):
        dataset = Mock()
        dataset.map_batches.return_value = dataset
        mock_calculate_concurrency.return_value = (1, 4)
        assert mock_create_actor is not None

        adapter = RayDataStageAdapter(ActorStage())
        adapter.process_dataset(dataset)

        compute = dataset.map_batches.call_args.kwargs["compute"]
        assert isinstance(compute, ActorPoolStrategy)
        assert compute.min_size == 1
        assert compute.max_size == 4
        assert "concurrency" not in dataset.map_batches.call_args.kwargs

    @patch("nemo_curator.backends.experimental.ray_data.adapter.create_task_from_stage", return_value=Mock())
    def test_task_stages_do_not_pass_deprecated_concurrency(self, mock_create_task: Mock):
        dataset = Mock()
        dataset.map_batches.return_value = dataset
        assert mock_create_task is not None

        adapter = RayDataStageAdapter(TaskStage())
        adapter.process_dataset(dataset)

        assert "concurrency" not in dataset.map_batches.call_args.kwargs
        assert "compute" not in dataset.map_batches.call_args.kwargs
