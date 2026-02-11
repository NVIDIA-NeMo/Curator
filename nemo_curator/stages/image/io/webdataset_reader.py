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

from __future__ import annotations

import pathlib
from collections.abc import Generator
from dataclasses import dataclass, field

import numpy as np
import torch
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import FileGroupTask
from nemo_curator.tasks.webdataset import WebDatasetBatch, WebDatasetSample


@dataclass
class WebDatasetReaderStage(ProcessingStage[FileGroupTask, WebDatasetBatch]):
    """DALI-based reader that loads samples from WebDataset tar shards.

    Wraps NVIDIA DALI's ``fn.readers.webdataset()`` to decode one or more
    component extensions from tar files at distributed scale.  Each tar
    shard is read on a Ray worker, and the decoded samples are emitted as
    :class:`WebDatasetBatch` tasks for downstream processing.

    The stage works with DALI GPU (CUDA) or DALI CPU; decodes on GPU if
    CUDA is available, otherwise falls back to CPU decoding.

    Args:
        extensions: List of file extensions to read from each sample in
            the tar (e.g. ``["jpg"]``, ``["jpg", "json"]``).  Only
            extensions listed here will be extracted; others are skipped.
        dali_batch_size: Number of samples per DALI pipeline run.
        num_threads: Number of CPU threads used by the DALI pipeline.
        num_gpus_per_worker: Fractional GPU allocation per Ray worker.
        decode_images: Whether to decode image extensions
            (``"jpg"``, ``"jpeg"``, ``"png"``) into numpy arrays via
            ``fn.decoders.image``.  When *False* the raw bytes are
            returned.
        name: Stage name for the pipeline registry.
    """

    extensions: list[str] = field(default_factory=lambda: ["jpg"])
    dali_batch_size: int = 100
    num_threads: int = 8
    num_gpus_per_worker: float = 0.25
    decode_images: bool = True
    verbose: bool = True
    name: str = "webdataset_reader"

    # Well-known image extensions that DALI can hardware-decode
    _IMAGE_EXTENSIONS: frozenset[str] = frozenset({"jpg", "jpeg", "png", "bmp", "tiff", "tif"})

    def __post_init__(self) -> None:
        if torch.cuda.is_available():
            logger.info("WebDatasetReaderStage using DALI GPU decode.")
            self.resources = Resources(gpus=self.num_gpus_per_worker)
        else:
            logger.info("CUDA not available; WebDatasetReaderStage using DALI CPU decode.")
            self.resources = Resources()

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["key", "components", "shard_path"]

    # -- DALI pipeline construction -----------------------------------------------

    def _create_dali_pipeline(self, tar_paths: list[str]) -> object:
        """Build and return a DALI pipeline configured for the requested extensions."""
        try:
            from nvidia.dali import fn, pipeline_def, types
        except ModuleNotFoundError as exc:  # pragma: no cover
            msg = (
                "nvidia.dali is required to use WebDatasetReaderStage. "
                "Install a compatible DALI build (GPU or CPU) for your environment."
            )
            raise RuntimeError(msg) from exc

        ext_list = list(self.extensions)

        @pipeline_def(
            batch_size=self.dali_batch_size,
            num_threads=self.num_threads,
            device_id=0,
        )
        def _pipeline() -> object:
            outputs = fn.readers.webdataset(
                paths=tar_paths,
                ext=ext_list,
                missing_component_behavior="skip",
            )

            if not isinstance(outputs, tuple):
                outputs = (outputs,)

            result = []
            for i, ext in enumerate(ext_list):
                component = outputs[i]
                if ext in self._IMAGE_EXTENSIONS and self.decode_images:
                    decode_device = "mixed" if torch.cuda.is_available() else "cpu"
                    component = fn.decoders.image(component, device=decode_device, output_type=types.RGB)
                result.append(component)

            return tuple(result) if len(result) > 1 else result[0]

        pipe = _pipeline()
        pipe.build()
        return pipe

    # -- sample iteration ---------------------------------------------------------

    def _read_tars_with_dali(
        self, tar_paths: list[pathlib.Path]
    ) -> Generator[list[WebDatasetSample], None, None]:
        """Yield lists of WebDatasetSample per DALI run over one or more tar files."""
        pipe = self._create_dali_pipeline([str(p) for p in tar_paths])

        epoch_size_map = pipe.epoch_size()
        total_samples = epoch_size_map[next(iter(epoch_size_map.keys()))]

        shard_path = str(tar_paths[0]) if len(tar_paths) == 1 else str(tar_paths[0].parent)
        id_prefix = (
            tar_paths[0].stem
            if len(tar_paths) == 1
            else f"group_{tar_paths[0].stem}_x{len(tar_paths)}"
        )

        samples_completed = 0
        while samples_completed < total_samples:
            raw_output = pipe.run()

            # Normalise output to a tuple even when a single extension is used
            if not isinstance(raw_output, tuple):
                raw_output = (raw_output,)

            first_tl = raw_output[0]
            batch_tl = first_tl.as_cpu() if hasattr(first_tl, "as_cpu") else first_tl
            batch_size = len(batch_tl)
            remaining = total_samples - samples_completed
            effective = min(batch_size, remaining)

            wds_samples: list[WebDatasetSample] = []
            for i in range(effective):
                sample_key = f"{id_prefix}_{samples_completed + i:06d}"
                components: dict[str, object] = {}

                for ext_idx, ext in enumerate(self.extensions):
                    tl = raw_output[ext_idx]
                    cpu_tl = tl.as_cpu() if hasattr(tl, "as_cpu") else tl
                    item = cpu_tl.at(i)
                    if isinstance(item, np.ndarray):
                        components[ext] = item
                    elif hasattr(item, "as_array"):
                        components[ext] = item.as_array()
                    else:
                        components[ext] = item

                wds_samples.append(
                    WebDatasetSample(
                        key=sample_key,
                        components=components,
                        shard_path=shard_path,
                    )
                )

            samples_completed += effective
            if wds_samples:
                yield wds_samples

    def _stream_batches(self, tar_files: list[pathlib.Path]) -> Generator[WebDatasetBatch, None, None]:
        """Emit one WebDatasetBatch per DALI run across all provided tar files."""
        for batch_id, samples in enumerate(self._read_tars_with_dali(tar_files)):
            yield WebDatasetBatch(
                task_id=f"wds_batch_{batch_id}",
                dataset_name="webdataset",
                data=samples,
            )

    # -- ProcessingStage interface ------------------------------------------------

    def process(self, task: FileGroupTask) -> list[WebDatasetBatch]:
        tar_file_paths = task.data
        if not tar_file_paths:
            msg = f"No tar file paths in task {task.task_id}"
            logger.error(msg)
            raise ValueError(msg)

        tar_files = [pathlib.Path(p) for p in tar_file_paths]
        return list(self._stream_batches(tar_files))
