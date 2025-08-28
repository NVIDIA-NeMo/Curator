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

import hashlib
import io
import os
import tarfile
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks.file_group import FileGroupTask
from nemo_curator.tasks.image import ImageBatch


@dataclass
class ImageWriterStage(ProcessingStage[ImageBatch, FileGroupTask]):
    """Write images to tar files and corresponding metadata to a Parquet file.

    - Images are packed into tar archives with at most ``images_per_tar`` entries each.
    - Metadata for all written images in the batch is stored in a single Parquet file.
    - Tar filenames are unique across actors via an actor-scoped prefix.
    """

    output_dir: str
    images_per_tar: int = 1000
    verbose: bool = False
    deterministic_name: bool = True
    remove_image_data: bool = False
    _name: str = "image_writer"

    def __post_init__(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def construct_base_name(self, task: ImageBatch) -> str:
        """Construct a base name for tar files within this actor."""

        def get_deterministic_hash(inputs: list[str], seed: str = "") -> str:
            """Create a deterministic hash from inputs."""
            combined = "|".join(sorted(inputs)) + "|" + seed
            return hashlib.sha256(combined.encode()).hexdigest()[:12]

        if self.deterministic_name:
            image_paths = [img.image_path for img in task.data]
            base_name = f"images-{get_deterministic_hash(image_paths, task.task_id)}"
        else:
            base_name = f"images-{uuid.uuid4().hex[:16]}"
        return base_name

    def _encode_image_to_bytes(self, image: np.ndarray) -> tuple[bytes, str]:
        """Encode image array to JPEG bytes; always returns (bytes, ".jpg")."""

        from PIL import Image  # type: ignore[import-not-found]

        img = image
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        channels_gray = 2
        channels_rgb = 3
        channels_rgba = 4

        if img.ndim == channels_gray:
            mode = "L"
        elif img.shape[2] == channels_rgb:
            mode = "RGB"
        elif img.shape[2] == channels_rgba:
            mode = "RGBA"
        else:
            mode = "RGB"
            img = img[..., :channels_rgb]

        with io.BytesIO() as buffer:
            Image.fromarray(img, mode=mode).save(buffer, format="JPEG", quality=92)
            return buffer.getvalue(), ".jpg"

    def _write_tar(self, base_name: str, members: list[tuple[str, bytes]]) -> str:
        """Write a tar file with given (member_name, bytes) entries using provided base name.

        Returns tar path.
        """

        tar_filename = f"{base_name}.tar"
        tar_path = os.path.join(self.output_dir, tar_filename)

        # Assert to prevent accidental overwrite if a file with the same name already exists
        if os.path.exists(tar_path):
            err = f"Collision detected: refusing to overwrite existing tar file: {tar_path}"
            raise AssertionError(err)

        with open(tar_path, "wb") as fobj, tarfile.open(fileobj=fobj, mode="w") as tf:
            for member_name, payload in members:
                info = tarfile.TarInfo(name=member_name)
                info.size = len(payload)
                tf.addfile(info, io.BytesIO(payload))

        logger.debug(f"Wrote tar: {tar_path} with {len(members)} images")
        return tar_path

    def _write_parquet(self, base_name: str, rows: list[dict[str, Any]]) -> str:
        """Write metadata rows to a Parquet file for a specific tar and return its path.

        The Parquet file shares the same base name as the tar file: ``{base_name}.parquet``.
        """

        parquet_filename = f"{base_name}.parquet"
        parquet_path = os.path.join(self.output_dir, parquet_filename)

        # Assert to prevent accidental overwrite if a file with the same name already exists
        if os.path.exists(parquet_path):
            err = f"Collision detected: refusing to overwrite existing parquet file: {parquet_path}"
            raise AssertionError(err)

        # Convert rows to Arrow Table (assumes uniform keys across rows)
        table = pa.Table.from_pylist(rows)

        # Write directly to local filesystem
        pq.write_table(table, parquet_path)

        logger.debug(f"Wrote parquet: {parquet_path} with {len(rows)} rows")
        return parquet_path

    def process(self, task: ImageBatch) -> FileGroupTask:
        if task is None or not isinstance(task.data, list) or len(task.data) == 0:
            logger.warning("Empty ImageBatch provided to ImageWriterStage; writing empty metadata only")

        # Paths produced for this batch
        tar_paths: list[str] = []
        parquet_paths: list[str] = []

        # Iterate in chunks
        images = task.data
        for start in range(0, len(images), self.images_per_tar):
            chunk = images[start : start + self.images_per_tar]
            members: list[tuple[str, bytes]] = []
            for idx, img_obj in enumerate(chunk):
                if img_obj.image_data is None:
                    msg = "ImageObject.image_data is None; cannot write image bytes"
                    raise ValueError(msg)

                payload, ext = self._encode_image_to_bytes(img_obj.image_data)
                member_basename = img_obj.image_id or f"{start + idx:06d}"
                member_name = f"{member_basename}{ext}"
                members.append((member_name, payload))

            # Write tar and its corresponding parquet for this chunk
            if members:
                # Use per-task chunk index
                chunk_index = start // self.images_per_tar
                base_prefix = self.construct_base_name(task)
                base_name = f"{base_prefix}-{chunk_index:06d}"
                tar_path = self._write_tar(base_name, members)
                tar_paths.append(tar_path)

                # Build metadata rows only for this tar
                metadata_rows_for_tar: list[dict[str, Any]] = []
                for idx, img_obj in enumerate(chunk):
                    member_basename = img_obj.image_id or f"{start + idx:06d}"
                    metadata_rows_for_tar.append(
                        {
                            "image_id": member_basename,
                            "tar_file": tar_path,
                            "member_name": f"{member_basename}.jpg",
                            "original_path": img_obj.image_path,
                            # Store user metadata as JSON-ish via repr to avoid pandas dependency
                            "metadata": repr(img_obj.metadata)
                            if isinstance(img_obj.metadata, dict)
                            else str(img_obj.metadata),
                        }
                    )

                # Remove image data if requested
                # This is useful for:
                #  + Efficient downstream stages that don't need the image data
                #  + Finishing pipeline without gathering images data across actors
                if self.remove_image_data:
                    for img_obj in chunk:
                        img_obj.image_data = None

                parquet_path = self._write_parquet(base_name, metadata_rows_for_tar)
                parquet_paths.append(parquet_path)

        # Return FileGroupTask with produced files
        return FileGroupTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=[*tar_paths, *parquet_paths],
            _metadata={
                **task._metadata,
                "output_dir": self.output_dir,
                "images_per_tar": self.images_per_tar,
                "num_images": len(task.data),
            },
            _stage_perf=task._stage_perf,
        )
