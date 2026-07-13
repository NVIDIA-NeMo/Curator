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

"""I/O stages for reading images from HF datasets and writing JSONL results."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from datasets import Dataset

    from nemo_curator.backends.base import WorkerMetadata

from loguru import logger
from PIL import Image

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import EmptyTask
from nemo_curator.tasks.image import ImageSampleTask, ImageTaskData

T_TaskData = TypeVar("T_TaskData", bound=ImageTaskData)


class HFDatasetImageReaderStage(ProcessingStage[EmptyTask, ImageSampleTask[T_TaskData]]):
    """Reads images from a HuggingFace dataset and creates image tasks.

    Accepts either a HF Hub dataset name or a local path.  Images are saved
    as JPEGs to ``image_dir`` on first run and reused on subsequent runs
    (idempotent).

    Args:
        dataset_name: HuggingFace Hub dataset id (e.g. ``"lmms-lab/textvqa"``) **or** a
            local path.  Local paths are detected automatically:

            * Directory containing ``dataset_info.json`` — loaded with
              ``load_from_disk()`` (saved via ``dataset.save_to_disk()``).
            * Any other existing directory — treated as an image folder and
              loaded with ``load_dataset("imagefolder", ...)``.
            * Anything else — loaded from the Hub with ``load_dataset()``.

        image_dir: Directory where extracted JPEG images are cached.  Images
            are written as ``<image_dir>/<image_id>.jpg``.  Already-present
            files are skipped so re-runs are cheap.
        split: Dataset split to load, e.g. ``"train"``, ``"validation"``.
            Ignored for ``load_from_disk`` paths (use the split key present in
            the saved dataset instead, or pass a leaf dataset directory).
        config_name: Optional dataset configuration / subset name, forwarded
            to ``load_dataset()`` as the second positional argument
            (e.g. ``"en"`` for multilingual datasets).  Ignored for local paths.
        image_column: Name of the column that holds the image.  The column
            value may be a PIL ``Image``, a ``{"bytes": ..., "path": ...}``
            dict (HF ``Image`` feature), a raw ``bytes`` object, or a file-path
            string.  All four are handled automatically.
        id_column: Column whose value is used as ``image_id``.  When multiple
            rows share the same id (e.g. one row per question in a VQA dataset)
            only the first occurrence is written; subsequent rows are deduplicated
            so that each physical image is processed exactly once.  If ``None``
            the row index is used (always unique).
        limit: Maximum number of *unique* images to load.  For Hub datasets
            this is passed directly into the HF split-slice notation
            (``"train[:N]"``) so only those records are downloaded — no wasted
            bandwidth.  For ``load_from_disk`` paths the limit is applied after
            loading via ``.select()``.
        task_type: Dataclass type instantiated for ``task.data``.  Must be a
            subclass of ``ImageTaskData``.  Defaults to ``ImageTaskData``; pass
            ``OCRData`` for the OCR pipeline.
    """

    name = "hf_dataset_image_reader"
    resources = Resources(cpus=1.0)

    def __init__(  # noqa: PLR0913
        self,
        dataset_name: str,
        image_dir: str | Path,
        split: str = "train",
        config_name: str | None = None,
        image_column: str = "image",
        id_column: str | None = None,
        limit: int | None = None,
        task_type: type[T_TaskData] = ImageTaskData,
    ) -> None:
        self.dataset_name = dataset_name
        self.image_dir = Path(image_dir)
        self.split = split
        self.config_name = config_name
        self.image_column = image_column
        self.id_column = id_column
        self.limit = limit
        self.task_type = task_type

    def ray_stage_spec(self) -> dict[str, Any]:
        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["image_path", "image_id"], []

    def _load_dataset(self) -> Dataset:
        """Load a HuggingFace Dataset from hub, save_to_disk dir, or imagefolder dir."""
        from datasets import load_dataset, load_from_disk

        local_path = Path(self.dataset_name)

        if local_path.exists():
            if (local_path / "dataset_info.json").exists():
                ds = load_from_disk(str(local_path))
                if hasattr(ds, "keys"):
                    if self.split not in ds:
                        available = list(ds.keys())
                        msg = (
                            f"Split '{self.split}' not found in dataset at {local_path}. Available splits: {available}"
                        )
                        raise ValueError(msg)
                    ds = ds[self.split]
                if self.limit is not None:
                    ds = ds.select(range(min(self.limit, len(ds))))
                return ds
            split_arg = self.split if self.limit is None else f"{self.split}[:{self.limit}]"
            return load_dataset("imagefolder", data_dir=str(local_path), split=split_arg)

        split_arg = self.split if self.limit is None else f"{self.split}[:{self.limit}]"
        return load_dataset(self.dataset_name, self.config_name, split=split_arg)

    @staticmethod
    def _to_pil(value: Any) -> Image.Image:  # noqa: ANN401
        """Convert various HF image column representations to a PIL Image."""
        if isinstance(value, Image.Image):
            return value
        if isinstance(value, dict):
            raw = value.get("bytes") or value.get("data")
            if raw:
                return Image.open(io.BytesIO(raw))
            path = value.get("path")
            if path:
                return Image.open(path)
        if isinstance(value, (bytes, bytearray)):
            return Image.open(io.BytesIO(value))
        if isinstance(value, str) and Path(value).exists():
            return Image.open(value)
        msg = f"Cannot convert value of type {type(value).__name__} to PIL Image. Expected a PIL Image, bytes, or a HF Image feature dict."
        raise ValueError(msg)

    def process(self, _: EmptyTask) -> list[ImageSampleTask[T_TaskData]]:
        self.image_dir.mkdir(parents=True, exist_ok=True)
        dataset = self._load_dataset()
        dataset_tag = Path(self.dataset_name).name.replace("/", "_")

        seen_ids: set[str] = set()
        tasks: list[ImageSampleTask[T_TaskData]] = []

        for idx, example in enumerate(dataset):
            image_id = str(example[self.id_column]) if self.id_column is not None else f"{idx:06d}"

            # Deduplicate images that appear in multiple rows (e.g. VQA datasets).
            if image_id in seen_ids:
                continue
            seen_ids.add(image_id)

            image_path = self.image_dir / f"{image_id}.jpg"
            if not image_path.exists():
                pil_image = self._to_pil(example[self.image_column])
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")
                pil_image.save(image_path, format="JPEG")

            tasks.append(
                ImageSampleTask(
                    dataset_name=dataset_tag,
                    data=self.task_type(
                        image_path=image_path,
                        image_id=image_id,
                    ),
                )
            )

        logger.info(
            f"hf_dataset_image_reader: {len(tasks)} unique images from "
            f"{self.dataset_name}/{self.split}" + (f" (limit={self.limit})" if self.limit else "")
        )
        return tasks


class JsonlSampleWriterStage(ProcessingStage[ImageSampleTask[T_TaskData], ImageSampleTask[T_TaskData]]):
    """Writes one JSONL line per sample task to a per-worker shard.

    Each worker writes to ``<stem>_worker<id><suffix>`` to avoid concurrent-write
    conflicts. Call :func:`merge_output_shards` after ``pipeline.run()`` to
    consolidate the shards into a single ``<stem><suffix>`` file.
    """

    name: str = "jsonl_sample_writer"
    resources = Resources(cpus=2.0)

    def __init__(
        self,
        output_path: str,
        valid_only: bool = True,
        image_parent: str | None = None,
    ) -> None:
        """Initialize the writer stage.

        Args:
            output_path: Base path for output JSONL shards.
            valid_only: If True, only write valid records.
            image_parent: If provided, make image paths relative to this directory.
        """
        self.output_path = output_path
        self.valid_only = valid_only
        self.image_parent = Path(image_parent) if image_parent else None
        self._file: Any = None
        self._saved_count: int = 0
        self._skipped_count: int = 0
        self._worker_id: str = ""

    def setup(self, worker_metadata: WorkerMetadata) -> None:
        self._worker_id = str(worker_metadata.worker_id)
        base_path = Path(self.output_path)
        suffix = base_path.suffix or ".jsonl"
        output = base_path.parent / f"{base_path.stem}_worker{self._worker_id}{suffix}"

        output.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(output, "w", encoding="utf-8")  # noqa: SIM115
        self._saved_count = 0
        self._skipped_count = 0
        logger.info(f"JsonlSampleWriter: opened {output} for writing")

    def _get_image_path_str(self, image_path: Path | None) -> str | None:
        if image_path is None:
            return None
        if self.image_parent is not None:
            try:
                return str(image_path.relative_to(self.image_parent))
            except ValueError:
                pass
        return str(image_path)

    def process(self, task: ImageSampleTask[T_TaskData]) -> ImageSampleTask[T_TaskData]:
        if not task.data.is_valid and self.valid_only:
            self._skipped_count += 1
            return task

        data = task.data.to_dict()
        data["image_path"] = self._get_image_path_str(task.data.image_path)
        # Keep empty lists/strings/False (e.g. OCR may legitimately be []); drop only None.
        self._file.write(
            json.dumps({k: v for k, v in data.items() if v is not None and k != "is_valid"}, default=str) + "\n"
        )
        self._file.flush()
        self._saved_count += 1
        return task

    def teardown(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
            logger.info(f"JsonlSampleWriter: wrote {self._saved_count} results, skipped {self._skipped_count}")

    @property
    def stats(self) -> dict[str, int]:
        return {
            "saved": self._saved_count,
            "skipped": self._skipped_count,
        }


def merge_output_shards(output_path: Path, *, delete_shards: bool = True) -> Path:
    """Merge per-worker JSONL shards from JsonlSampleWriterStage into a single file.

    JsonlSampleWriterStage writes one shard per worker named
    ``<stem>_worker<id><suffix>`` in the same directory as ``output_path``.
    Call this after ``pipeline.run()`` returns — by that point every worker
    has flushed its writes, so the merge is race-free regardless of node count.

    Args:
        output_path: The base output path passed to JsonlSampleWriterStage.
        delete_shards: Remove shard files after a successful merge (default True).

    Returns:
        Path to the merged file (``<stem><suffix>`` in the same directory).
    """
    import shutil

    suffix = output_path.suffix or ".jsonl"
    pattern = f"{output_path.stem}_worker*{suffix}"
    shards = sorted(output_path.parent.glob(pattern))

    if not shards:
        logger.info("merge_output_shards: no shards found, nothing to merge")
        return output_path

    merged = output_path.parent / f"{output_path.stem}{suffix}"
    with open(merged, "w", encoding="utf-8") as fout:
        for shard in shards:
            with open(shard, encoding="utf-8") as fin:
                shutil.copyfileobj(fin, fout)

    if delete_shards:
        for shard in shards:
            shard.unlink()

    logger.info(f"merge_output_shards: merged {len(shards)} shards → {merged}")
    return merged
