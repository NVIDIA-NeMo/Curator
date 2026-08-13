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

from dataclasses import dataclass, field

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.image.embedders.clip_embedder import ImageEmbeddingStage
from nemo_curator.stages.image.io.image_reader import ImageReaderStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import FileGroupTask, ImageBatch


@dataclass
class ImageReaderEmbeddingStage(ProcessingStage[FileGroupTask, ImageBatch]):
    """Decode WebDataset images and embed them within the same GPU actor.

    Keeping decoded pixels local to the actor avoids serializing large image
    arrays through an executor transport between the reader and embedding
    stages. Original encoded bytes remain attached for lossless output writing.
    """

    model_dir: str = None
    dali_batch_size: int = 100
    reader_num_threads: int = 4
    num_gpus_per_worker: float = 0.25
    model_inference_batch_size: int = 500
    remove_image_data: bool = True
    verbose: bool = False
    batch_size: int = 1
    name: str = "image_reader_embedding"
    _reader: ImageReaderStage = field(init=False, repr=False)
    _embedder: ImageEmbeddingStage = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.resources = Resources(gpus=self.num_gpus_per_worker)
        self._reader = ImageReaderStage(
            dali_batch_size=self.dali_batch_size,
            num_threads=self.reader_num_threads,
            num_gpus_per_worker=self.num_gpus_per_worker,
            verbose=self.verbose,
        )
        self._embedder = ImageEmbeddingStage(
            model_dir=self.model_dir,
            num_gpus_per_worker=self.num_gpus_per_worker,
            model_inference_batch_size=self.model_inference_batch_size,
            remove_image_data=self.remove_image_data,
            verbose=self.verbose,
        )

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["image_bytes", "image_path", "image_id", "embedding"]

    def setup_on_node(
        self,
        node_info: NodeInfo | None = None,
        worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        self._embedder.setup_on_node(node_info, worker_metadata)

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        self._embedder.setup(worker_metadata)

    def process(self, task: FileGroupTask) -> list[ImageBatch]:
        with self._time_metric("reader_process_time"):
            batches = self._reader.process(task)

        num_images = sum(len(batch.data) for batch in batches)
        with self._time_metric("embedding_process_time"):
            self._embedder.process_batch(batches)

        self._log_metric("num_images_processed", num_images)
        return batches


__all__ = ["ImageReaderEmbeddingStage"]
