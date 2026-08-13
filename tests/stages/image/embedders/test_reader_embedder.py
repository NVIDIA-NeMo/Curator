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

from unittest.mock import Mock

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.image.embedders.reader_embedder import ImageReaderEmbeddingStage
from nemo_curator.tasks import FileGroupTask, ImageBatch, ImageObject


def test_stage_configures_reader_and_embedder() -> None:
    stage = ImageReaderEmbeddingStage(
        model_dir="models",
        dali_batch_size=100,
        reader_num_threads=4,
        num_gpus_per_worker=0.25,
        model_inference_batch_size=500,
        remove_image_data=True,
    )

    assert stage.name == "image_reader_embedding"
    assert stage.resources.gpus == 0.25
    assert stage.batch_size == 1
    assert stage.is_fanout_stage()
    assert stage._reader.dali_batch_size == 100
    assert stage._reader.num_threads == 4
    assert stage._embedder.model_inference_batch_size == 500
    assert stage._embedder.remove_image_data


def test_setup_forwards_to_embedder() -> None:
    stage = ImageReaderEmbeddingStage(model_dir="models")
    stage._embedder.setup_on_node = Mock()
    stage._embedder.setup = Mock()
    node_info = Mock(spec=NodeInfo)
    worker_metadata = Mock(spec=WorkerMetadata)

    stage.setup_on_node(node_info, worker_metadata)
    stage.setup(worker_metadata)

    stage._embedder.setup_on_node.assert_called_once_with(node_info, worker_metadata)
    stage._embedder.setup.assert_called_once_with(worker_metadata)


def test_process_decodes_and_embeds_one_shard_without_transport_boundary() -> None:
    stage = ImageReaderEmbeddingStage(model_dir="models")
    task = FileGroupTask(dataset_name="images", data=["shard-000.tar"])
    batches = [
        ImageBatch(dataset_name="images", data=[ImageObject(), ImageObject()]),
        ImageBatch(dataset_name="images", data=[ImageObject()]),
    ]
    stage._reader.process = Mock(return_value=batches)
    stage._embedder.process_batch = Mock(return_value=batches)

    result = stage.process(task)

    assert result is batches
    stage._reader.process.assert_called_once_with(task)
    stage._embedder.process_batch.assert_called_once_with(batches)
    metrics = stage._consume_custom_metrics()
    assert metrics["num_images_processed"] == 3
    assert metrics["reader_process_time"] >= 0
    assert metrics["embedding_process_time"] >= 0
