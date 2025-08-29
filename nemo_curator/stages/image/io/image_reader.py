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

import pathlib
from collections.abc import Generator
from dataclasses import dataclass

import numpy as np
import torch
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import FileGroupTask, ImageBatch, ImageObject


@dataclass
class ImageReaderStage(ProcessingStage[FileGroupTask, ImageBatch]):
    """DALI-based reader that loads images from WebDataset tar shards.

    Works with DALI GPU (CUDA) or DALI CPU; decodes on GPU if CUDA is available,
    otherwise falls back to CPU decoding.
    """

    task_batch_size: int = 100
    verbose: bool = True
    num_threads: int = 8
    num_gpus_per_worker: float = 0.25
    _name: str = "image_reader"

    def __post_init__(self) -> None:
        # Allow both GPU and CPU DALI; log mode for visibility
        if torch.cuda.is_available():
            logger.info("ImageReaderStage using DALI GPU decode.")
        else:
            logger.info("CUDA not available; ImageReaderStage using DALI CPU decode.")

        if torch.cuda.is_available():
            self._resources = Resources(gpus=self.num_gpus_per_worker)
        else:
            self._resources = Resources()

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["image_data", "image_path", "image_id"]

    def _create_dali_pipeline(self, tar_paths: list[str]) -> object:
        try:
            from nvidia.dali import fn, pipeline_def, types
        except ModuleNotFoundError as exc:  # pragma: no cover
            msg = (
                "nvidia.dali is required to use ImageReaderStage. "
                "Install a compatible DALI build (GPU or CPU) for your environment."
            )
            raise RuntimeError(msg) from exc

        @pipeline_def(
            batch_size=self.task_batch_size,
            num_threads=self.num_threads,
            device_id=0,  # First device; unused for CPU-only DALI builds
        )
        def webdataset_pipeline() -> object:
            # Read only JPGs to avoid Python-side JSON parsing overhead
            img_raw = fn.readers.webdataset(
                paths=tar_paths,
                ext=["jpg"],
                missing_component_behavior="skip",
            )
            # Decode on GPU when available, otherwise on CPU; keep original sizes (no resize)
            decode_device = "mixed" if torch.cuda.is_available() else "cpu"
            return fn.decoders.image(img_raw, device=decode_device, output_type=types.RGB)

        pipe = webdataset_pipeline()
        pipe.build()
        return pipe

    def _read_tars_with_dali(self, tar_paths: list[pathlib.Path]) -> Generator[list[ImageObject], None, None]:
        """Yield lists of ImageObject per DALI run over one or more tar files."""
        pipe = self._create_dali_pipeline([str(p) for p in tar_paths])

        epoch_size_map = pipe.epoch_size()
        total_samples = epoch_size_map[next(iter(epoch_size_map.keys()))]

        samples_completed = 0
        # Use the tar filename stem as the id prefix for single shards; for grouped shards,
        # synthesize a group prefix and place generated image paths under the tars' parent dir.
        base_path = tar_paths[0] if len(tar_paths) == 1 else tar_paths[0].parent
        id_prefix = (
            tar_paths[0].stem
            if len(tar_paths) == 1
            else f"group_{tar_paths[0].stem}_x{len(tar_paths)}"
        )

        while samples_completed < total_samples:
            img_batch = pipe.run()
            if isinstance(img_batch, tuple):
                img_batch = img_batch[0]

            # Per-sample extraction to preserve original sizes
            img_cpu = img_batch.as_cpu()
            batch_size = len(img_cpu)
            remaining = total_samples - samples_completed
            effective = min(batch_size, remaining)

            image_objects: list[ImageObject] = []
            for i in range(effective):
                img_item = img_cpu.at(i)
                img_np = img_item if isinstance(img_item, np.ndarray) else img_item.as_array()
                image_objects.append(
                    ImageObject(
                        image_path=str(base_path / f"{id_prefix}_{samples_completed + i:06d}.jpg"),
                        image_id=f"{id_prefix}_{samples_completed + i:06d}",
                        image_data=img_np,
                    )
                )

            samples_completed += effective
            if image_objects:
                yield image_objects

    def _stream_batches(self, tar_files: list[pathlib.Path]) -> Generator[ImageBatch, None, None]:
        """Emit one ImageBatch per DALI run across all provided tar files."""
        for batch_id, image_objects in enumerate(self._read_tars_with_dali(tar_files)):
            yield ImageBatch(
                task_id=f"image_batch_{batch_id}",
                dataset_name="tar_files",
                data=image_objects,
            )

    def process(self, task: FileGroupTask) -> list[ImageBatch]:
        tar_file_paths = task.data
        if not tar_file_paths:
            msg = f"No tar file paths in task {task.task_id}"
            logger.error(msg)
            raise ValueError(msg)

        tar_files = [pathlib.Path(p) for p in tar_file_paths]

        return list(self._stream_batches(tar_files))
