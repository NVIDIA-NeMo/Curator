import io
import os
import tarfile
import uuid
from dataclasses import dataclass, field
from typing import Any
from loguru import logger
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import hashlib

from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.resources import Resources
from ray_curator.tasks.image import ImageBatch, ImageObject
from ray_curator.tasks.file_group import FileGroupTask


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
    # Local filesystem only; no cloud storage options

    _name: str = "image_writer"

    # Runtime fields
    _actor_id: str = field(default="", init=False, repr=False)
    _tar_seq: int = field(default=0, init=False, repr=False)

    @property
    def resources(self) -> Resources:
        # CPU-only writer
        return Resources()

    def __post_init__(self) -> None:  # noqa: D401
        os.makedirs(self.output_dir, exist_ok=True)

    def setup(self, worker_metadata=None) -> None:  # noqa: ANN001
        """Initialize unique actor prefix for output filenames.

        Uses provided ``worker_metadata.worker_id`` if available; otherwise falls back to
        ``hostname-pid-<short-uuid>``. Ensures filenames are unique across actors.
        """

        if getattr(worker_metadata, "worker_id", None):
            base = str(worker_metadata.worker_id)
        else:
            base = f"{os.getpid()}"
        self._actor_id = f"{base}-{uuid.uuid4().hex[:16]}"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def get_deterministic_hash(inputs: list[str], seed: str = "") -> str:
        """Create a deterministic hash from inputs."""
        combined = "|".join(sorted(inputs)) + "|" + seed
        return hashlib.sha256(combined.encode()).hexdigest()[:12]

    def construct_base_name(self, task: ImageBatch) -> str:
        """Construct a base name for tar files within this actor."""
        if self.deterministic_name:
            unique_image_paths = [hashlib.sha256(img.image_path.encode()).hexdigest()[:12] for img in task.data]
            base_name = f"images-{self.get_deterministic_hash(unique_image_paths)}"
        else:
            base_name = f"images-{uuid.uuid4().hex[:16]}"
        return base_name

    def construct_tar_base_name(self, task: ImageBatch) -> str:
        """Construct a base name for a tar file."""
        return f"images-{self._actor_id}-{self._tar_seq:06d}"

    def _encode_image_to_bytes(self, image: np.ndarray) -> tuple[bytes, str]:
        """Encode image array to JPEG bytes; always returns (bytes, ".jpg")."""

        from PIL import Image  # type: ignore

        img = image
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            mode = "L"
        elif img.shape[2] == 3:
            mode = "RGB"
        elif img.shape[2] == 4:
            mode = "RGBA"
        else:
            mode = "RGB"
            img = img[..., :3]

        with io.BytesIO() as buffer:
            Image.fromarray(img, mode=mode).save(buffer, format="JPEG", quality=92)
            return buffer.getvalue(), ".jpg"

    def _write_tar(self, base_name: str, members: list[tuple[str, bytes]]) -> str:
        """Write a tar file with given (member_name, bytes) entries using provided base name.

        Returns tar path.
        """

        tar_filename = f"{base_name}.tar"
        tar_path = os.path.join(self.output_dir, tar_filename)

        with open(tar_path, "wb") as fobj:
            with tarfile.open(fileobj=fobj, mode="w") as tf:
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
                    raise ValueError("ImageObject.image_data is None; cannot write image bytes")

                payload, ext = self._encode_image_to_bytes(img_obj.image_data)
                member_basename = img_obj.image_id or f"{start + idx:06d}"
                member_name = f"{member_basename}{ext}"
                members.append((member_name, payload))

            # Write tar and its corresponding parquet for this chunk
            if members:
                # Define the common base name once per tar/parquet pair and advance sequence
                tar_seq_used = self._tar_seq
                base_name = self.construct_tar_base_name(task)
                self._tar_seq += 1

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
                            "metadata": repr(img_obj.metadata) if isinstance(img_obj.metadata, dict) else str(img_obj.metadata),
                        }
                    )

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
