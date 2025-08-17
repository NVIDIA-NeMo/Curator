# ruff: noqa: I001
import os
import pathlib
import time
from collections.abc import Generator
from dataclasses import dataclass

from loguru import logger
import numpy as np
from typing import TYPE_CHECKING
import torch

from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.resources import Resources
from ray_curator.tasks import FileGroupTask, ImageBatch, ImageObject


@dataclass
class ImageReaderStage(ProcessingStage[FileGroupTask, ImageBatch]):
    """Strict DALI-based reader that loads images from WebDataset tar shards.

    This stage requires a visible CUDA device.
    """

    task_batch_size: int = 100
    verbose: bool = True
    num_threads: int = 8
    num_gpus_per_worker: float = 0.25
    _name: str = "image_reader"

    @property
    def resources(self) -> Resources:
        if torch.cuda.is_available():
            return Resources(gpus=self.num_gpus_per_worker)
        else:
            return Resources()

    def __post_init__(self) -> None:
        # Enforce DALI-only execution
        if not torch.cuda.is_available():
            msg = "ImageReaderStage requires CUDA for DALI."
            logger.error(msg)
            raise RuntimeError(msg)

        if self.verbose:
            logger.info("ImageReaderStage (DALI) configuration:")
            logger.info(f"  - task_batch_size: {self.task_batch_size}")
            logger.info(f"  - num_threads: {self.num_threads}")
            logger.info(f"  - CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
            logger.info(f"  - torch.cuda.device_count: {torch.cuda.device_count()}")
            logger.info(f"  - torch.cuda.current_device: {torch.cuda.current_device()}")

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["image_data", "image_path", "image_id"]

    def _create_dali_pipeline(self, tar_path: str):
        try:
            from nvidia.dali import fn, pipeline_def, types  # noqa: PLC0415
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError(
                "nvidia.dali is required to use ImageReaderStage. "
                "Install a compatible DALI build (GPU or CPU) for your environment."
            ) from exc

        @pipeline_def(
            batch_size=self.task_batch_size,
            num_threads=self.num_threads,
            device_id=0,  # Uses the first visible CUDA device for this worker
        )
        def webdataset_pipeline(_tar_path: str):
            # Read only JPGs to avoid Python-side JSON parsing overhead
            img_raw = fn.readers.webdataset(
                paths=_tar_path,
                ext=["jpg"],
                missing_component_behavior="skip",
            )
            # GPU-accelerated decode; keep original sizes (no resize)
            return fn.decoders.image(img_raw, device="mixed", output_type=types.RGB)

        pipe = webdataset_pipeline(tar_path)
        pipe.build()
        return pipe

    def _read_tar_with_dali(self, tar_path: pathlib.Path) -> Generator[list[ImageObject], None, None]:
        """Yield lists of ImageObject per DALI run (one batch -> one yield)."""
        pipe = self._create_dali_pipeline(str(tar_path))

        epoch_size_map = pipe.epoch_size()
        total_samples = epoch_size_map[next(iter(epoch_size_map.keys()))]

        samples_completed = 0
        tar_prefix = tar_path.stem
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
                        image_path=str(tar_path / f"{tar_prefix}_{samples_completed + i:06d}.jpg"),
                        image_id=f"{tar_prefix}_{samples_completed + i:06d}",
                        image_data=img_np,
                    )
                )

            samples_completed += effective
            if image_objects:
                yield image_objects

    def _stream_batches(self, tar_files: list[pathlib.Path]) -> Generator[ImageBatch, None, None]:
        """Emit one ImageBatch per DALI run; no intermediate accumulation."""
        batch_id = 0
        for tar_path in tar_files:
            for image_objects in self._read_tar_with_dali(tar_path):
                yield ImageBatch(
                    task_id=f"image_batch_{batch_id}",
                    dataset_name="tar_files",
                    data=image_objects,
                )
                batch_id += 1

    def process(self, task: FileGroupTask) -> list[ImageBatch]:
        process_start = time.time()

        tar_file_paths = task.data
        if not tar_file_paths:
            msg = f"No tar file paths in task {task.task_id}"
            logger.error(msg)
            raise ValueError(msg)

        tar_files = [pathlib.Path(p) for p in tar_file_paths]

        if self.verbose:
            logger.info(
                f"[PERF] Processing {len(tar_files)} tar files in task {task.task_id} with DALI"
            )

        batches = list(self._stream_batches(tar_files))

        if self.verbose:
            total_images = sum(len(b.data) for b in batches)
            total_time = time.time() - process_start
            logger.info(
                f"[PERF] Streamed {total_images} images in {len(batches)} batches | "
                f"Total time: {total_time:.3f}s | Throughput: {total_images/total_time:.1f} img/s"
            )

        return batches
