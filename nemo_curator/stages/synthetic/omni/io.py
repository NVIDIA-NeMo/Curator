"""I/O stages for reading and writing image caption tasks.

Based on NeMo Curator's VideoReader pattern.
"""

import base64
import io
import json
import re
import tarfile
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Generic, Type, TypeVar

from nemo_curator.backends.utils import RayStageSpecKeys
import pyarrow.parquet as pq
from loguru import logger
from PIL import Image

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import _EmptyTask
from nemo_curator.tasks.file_group import FileGroupTask

from nemo_curator.tasks.image import ImageTaskData, SingleDataTask


SUPPORTED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"]


T_TaskData = TypeVar("T_TaskData", bound=ImageTaskData)


class InputFormat(str, Enum):
    """Supported input formats for pipelines."""

    IMAGE_FOLDER = "image_folder"
    TAR = "tar"
    PARQUET = "parquet"
    JSONL_TAR = "jsonl_tar"


class ImageReaderStage(ProcessingStage[FileGroupTask, SingleDataTask[T_TaskData]], Generic[T_TaskData]):
    """Stage that reads image files and creates ImageCaptionTask objects.

    This stage processes image files by reading their binary content and
    creating ImageCaptionTask objects for downstream processing.

    Args:
        input_path: Base path for resolving relative image paths.
        verbose: If True, logs detailed image information after successful processing.
    """

    input_path: str | None = None
    verbose: bool = False
    name: str = "image_reader"
    resources = Resources(cpus=1.0)

    def __init__(self, input_path: str, verbose: bool = False, task_type: Type[T_TaskData] = ImageTaskData) -> None:
        """Initialize the image reader stage."""
        self.input_path = input_path
        self.verbose = verbose
        self.task_type = task_type

    def inputs(self) -> tuple[list[str], list[str]]:
        """Define the input attributes required by this stage."""
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        """Define the output attributes produced by this stage."""
        return ["image_path", "image_id"], []

    def process(self, task: FileGroupTask) -> SingleDataTask[T_TaskData]:
        """Process a file group task by creating an ImageCaptionTask.

        Args:
            task: FileGroupTask containing image file path(s).

        Returns:
            ImageCaptionTask with image_path and image_id populated.
        """
        if len(task.data) != 1:
            msg = f"Expected exactly 1 image file, got {len(task.data)}"
            raise ValueError(msg)

        image_path = Path(task.data[0])
        image_id = image_path.stem

        image_task = SingleDataTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=self.task_type(
                image_path=image_path,
                image_id=image_id,
            ),
        )

        if self.verbose:
            self._log_image_info(image_task)

        return image_task

    def _log_image_info(self, task: SingleDataTask[T_TaskData]) -> None:
        """Log image information after successful processing."""
        logger.info(f"Loaded image: {task.data.image_id} from {task.data.image_path}")


class ImageFolderReaderStage(CompositeStage[_EmptyTask, SingleDataTask[T_TaskData]], Generic[T_TaskData]):
    """Composite stage that reads image files from storage.

    This stage combines FilePartitioningStage and ImageReaderStage into a single
    high-level operation for reading image files from a directory.

    Args:
        input_image_path: Path to the directory containing image files.
        image_limit: Maximum number of images to process (None for unlimited).
        verbose: Whether to enable verbose logging during processing.
    """

    input_image_path: str
    image_limit: int | None = None
    verbose: bool = False
    name: str = "image_reader"

    def __init__(self, input_image_path: str, image_limit: int | None = None, verbose: bool = False, task_type: Type[T_TaskData] = ImageTaskData) -> None:
        """Initialize the image reader composite stage."""
        super().__init__()
        self.input_image_path = input_image_path
        self.image_limit = image_limit
        self.verbose = verbose
        self.task_type = task_type

    def decompose(self) -> list[ProcessingStage]:
        """Decompose into constituent execution stages.

        Returns:
            List of processing stages: [FilePartitioningStage, ImageReaderStage]
        """
        partitioning_stage = FilePartitioningStage(
            file_paths=self.input_image_path,
            files_per_partition=1,
            file_extensions=SUPPORTED_IMAGE_EXTENSIONS,
            limit=self.image_limit,
        )

        reader_stage = ImageReaderStage(
            input_path=self.input_image_path,
            verbose=self.verbose,
            task_type=self.task_type,
        )

        return [partitioning_stage, reader_stage]

    def get_description(self) -> str:
        """Get a description of what this composite stage does."""
        limit_str = str(self.image_limit) if self.image_limit is not None else "unlimited"
        return (
            f"Reads image files from '{self.input_image_path}' "
            f"(limit: {limit_str}) and creates ImageCaptionTask objects"
        )


class TarImageReaderStage(ProcessingStage[FileGroupTask, SingleDataTask[T_TaskData]], Generic[T_TaskData]):
    """Stage that reads images from WebDataset-style tar files.

    This stage processes tar archives containing images, extracting them
    and creating ImageCaptionTask objects.

    Args:
        input_path: Base path for resolving tar file paths.
        verbose: If True, logs detailed information.
    """

    input_path: str | None = None
    verbose: bool = False
    name: str = "tar_image_reader"
    resources = Resources(cpus=1.0)

    def __init__(
        self,
        input_path: str,
        verbose: bool = False,
        task_type: Type[T_TaskData] = ImageTaskData,
    ) -> None:
        """Initialize the tar image reader stage."""
        self.input_path = input_path
        self.verbose = verbose
        self.task_type = task_type

    def inputs(self) -> tuple[list[str], list[str]]:
        """Define the input attributes required by this stage."""
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        """Define the output attributes produced by this stage."""
        return ["image_path", "image_id"], []

    def _is_image(self, name: str) -> bool:
        """Check if file is a supported image format."""
        return Path(name).suffix.lower() in {ext.lower() for ext in SUPPORTED_IMAGE_EXTENSIONS}

    def process(self, task: FileGroupTask) -> list[SingleDataTask[T_TaskData]]:
        """Process a tar file and extract all images as ImageCaptionTasks.

        Args:
            task: FileGroupTask containing tar file path.

        Returns:
            List of ImageCaptionTask objects, one per image in the tar.
        """
        if len(task.data) != 1:
            msg = f"Expected exactly 1 tar file, got {len(task.data)}"
            raise ValueError(msg)

        tar_path = Path(task.data[0])
        tasks: list[SingleDataTask[T_TaskData]] = []

        with tarfile.open(tar_path, "r") as tar:
            for idx, member in enumerate(tar.getmembers()):
                if member.isfile() and self._is_image(member.name):
                    # Create a composite path: tar_path/member_name
                    image_path = tar_path / member.name
                    image_id = f"{tar_path.stem}/{Path(member.name).stem}"

                    image_task = SingleDataTask(
                        task_id=f"{task.task_id}_{idx}",
                        dataset_name=task.dataset_name,
                        data=self.task_type(
                            image_path=image_path,
                            image_id=image_id,
                        ),
                    )
                    tasks.append(image_task)

                    if self.verbose:
                        logger.info(f"Extracted image: {image_id} from {tar_path}")

        return tasks


class JsonlTarImageReaderStage(ProcessingStage[FileGroupTask | _EmptyTask, SingleDataTask[T_TaskData]], Generic[T_TaskData]):
    """Stage that reads tar-slice image paths from a JSONL file.

    Each JSON line should provide the tar shard name and byte range.
    When used after a file-partitioning stage, the JSONL path is taken from the
    incoming FileGroupTask (task.data[0]). Otherwise jsonl_path from init is used.

    Args:
        jsonl_path: Path to a single JSONL file (used when no preceding partitioning stage).
        tar_base_path: Base path for resolving tar file paths.
        verbose: If True, logs detailed information.
    """

    tar_base_path: Path
    verbose: bool = False
    name: str = "jsonl_tar_image_reader"
    resources = Resources(cpus=1.0)

    def ray_stage_spec(self) -> dict[str, Any]:
        """Ray stage specification for this stage."""
        return {
            RayStageSpecKeys.IS_FANOUT_STAGE: True,
        }


    def __init__(
        self,
        jsonl_path: str | None = None,
        tar_base_path: str | Path | None = None,
        verbose: bool = False,
        task_type: Type[T_TaskData] = ImageTaskData,
    ) -> None:
        """Initialize the JSONL tar image reader stage."""
        self.jsonl_path = jsonl_path
        self.tar_base_path = Path(tar_base_path) if tar_base_path is not None else Path(".")
        self.verbose = verbose
        self.task_type = task_type

    def inputs(self) -> tuple[list[str], list[str]]:
        """No required task attributes; path comes from task.data[0] or init jsonl_path."""
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        """Define the output attributes produced by this stage."""
        return ["image_path", "image_id"], []

    def process(self, task: FileGroupTask | _EmptyTask) -> list[SingleDataTask[T_TaskData]]:
        """Process a JSONL file and create ImageCaptionTasks.

        If task is a FileGroupTask with non-empty data, the JSONL path is taken
        from task.data[0]. Otherwise jsonl_path from init is used (legacy single-file mode).
        """
        if isinstance(task, FileGroupTask) and task.data:
            jsonl_path = Path(task.data[0])
        elif self.jsonl_path is not None:
            jsonl_path = Path(self.jsonl_path)
        else:
            msg = "Either use a file-partitioning stage before this stage or pass jsonl_path to JsonlTarImageReaderStage."
            raise ValueError(msg)
        tasks: list[SingleDataTask[T_TaskData]] = []

        with jsonl_path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue

                record = json.loads(line)
                image_name = record["image_path"]
                shard_name = record["img_shard_name"]
                byte_offset = int(record["img_byte_offset"])
                byte_size = int(record["img_byte_size"])

                tar_path = self.tar_base_path / shard_name
                tar_slice_path = f"{tar_path}/{byte_offset}:{byte_size}:{image_name}"
                image_path = Path(tar_slice_path)
                image_id = f"{tar_path.stem}/{Path(image_name).stem}"

                image_task = SingleDataTask(
                    task_id=f"jsonl_tar_image_reader_{idx}",
                    dataset_name=jsonl_path.stem,
                    data=self.task_type(
                        image_path=image_path,
                        image_id=image_id,
                    ),
                )
                tasks.append(image_task)

                if self.verbose:
                    logger.info(f"Indexed image: {image_id} from {jsonl_path}")

        return tasks


class HFDatasetImageReaderStage(ProcessingStage[_EmptyTask, SingleDataTask[T_TaskData]], Generic[T_TaskData]):
    """Reads images from a HuggingFace dataset and creates image tasks.

    Accepts either a HF Hub dataset name or a local path.  Images are saved
    as JPEGs to ``image_dir`` on first run and reused on subsequent runs
    (idempotent).  Replaces ``FilePartitioningStage`` + ``JsonlTarImageReaderStage``
    when the source is a HuggingFace dataset rather than tar shards.

    Args:
        dataset_name: HuggingFace Hub dataset id (e.g. ``"textvqa"``) **or** a
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

    def __init__(
        self,
        dataset_name: str,
        image_dir: str | Path,
        split: str = "train",
        config_name: str | None = None,
        image_column: str = "image",
        id_column: str | None = None,
        limit: int | None = None,
        task_type: Type[T_TaskData] = ImageTaskData,
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

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def _load_dataset(self):
        """Load a HuggingFace Dataset, handling hub, save_to_disk, and imagefolder paths."""
        from datasets import load_dataset, load_from_disk

        local_path = Path(self.dataset_name)

        if local_path.exists():
            if (local_path / "dataset_info.json").exists():
                # Saved with dataset.save_to_disk()
                ds = load_from_disk(str(local_path))
                # DatasetDict → pick the requested split
                if hasattr(ds, "keys"):
                    if self.split not in ds:
                        available = list(ds.keys())
                        raise ValueError(
                            f"Split '{self.split}' not found in dataset at {local_path}. "
                            f"Available splits: {available}"
                        )
                    ds = ds[self.split]
                if self.limit is not None:
                    ds = ds.select(range(min(self.limit, len(ds))))
                return ds
            else:
                # Raw image folder
                split_arg = self.split if self.limit is None else f"{self.split}[:{self.limit}]"
                return load_dataset("imagefolder", data_dir=str(local_path), split=split_arg)

        # HF Hub
        split_arg = self.split if self.limit is None else f"{self.split}[:{self.limit}]"
        return load_dataset(self.dataset_name, self.config_name, split=split_arg)

    # ------------------------------------------------------------------
    # Image normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _to_pil(value: Any) -> Image.Image:
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
        raise ValueError(
            f"Cannot convert value of type {type(value).__name__} to PIL Image. "
            "Expected a PIL Image, bytes, or a HF Image feature dict."
        )

    # ------------------------------------------------------------------
    # Stage entry point
    # ------------------------------------------------------------------

    def process(self, _: _EmptyTask) -> list[SingleDataTask[T_TaskData]]:
        self.image_dir.mkdir(parents=True, exist_ok=True)
        dataset = self._load_dataset()
        dataset_tag = Path(self.dataset_name).name.replace("/", "_")

        seen_ids: set[str] = set()
        tasks: list[SingleDataTask[T_TaskData]] = []

        for idx, example in enumerate(dataset):
            # Determine image_id
            if self.id_column is not None:
                image_id = str(example[self.id_column])
            else:
                image_id = f"{idx:06d}"

            # Deduplicate images that appear in multiple rows (e.g. VQA datasets)
            if image_id in seen_ids:
                continue
            seen_ids.add(image_id)

            # Save image to disk (skip if already present)
            image_path = self.image_dir / f"{image_id}.jpg"
            if not image_path.exists():
                pil_image = self._to_pil(example[self.image_column])
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")
                pil_image.save(image_path, format="JPEG")

            tasks.append(
                SingleDataTask(
                    task_id=f"hf_{idx}",
                    dataset_name=dataset_tag,
                    data=self.task_type(
                        image_path=image_path,
                        image_id=image_id,
                    ),
                )
            )

        logger.info(
            f"hf_dataset_image_reader: {len(tasks)} unique images from "
            f"{self.dataset_name}/{self.split}"
            + (f" (limit={self.limit})" if self.limit else "")
        )
        return tasks


class JsonlPipelineOutputReaderStage(ProcessingStage[FileGroupTask | _EmptyTask, SingleDataTask[T_TaskData]], Generic[T_TaskData]):
    """Stage that reads pipeline output JSONL files.

    Reads JSONL files produced by ResultWriterStage and reconstructs tasks
    with their data. The task_type must implement a `from_dict` class method.
    When used after a file-partitioning stage, the JSONL path is taken from
    the incoming FileGroupTask (task.data[0]). Otherwise jsonl_path from init is used.

    Args:
        jsonl_path: Path to the JSONL file to read (used when no preceding partitioning stage).
        verbose: If True, logs detailed information.
        task_type: The dataclass type to instantiate. Must have `from_dict(dict) -> Self`.
    """

    name: str = "jsonl_pipeline_output_reader"
    resources = Resources(cpus=1.0)

    def __init__(
        self,
        jsonl_path: str | Path | None = None,
        verbose: bool = False,
        task_type: Type[T_TaskData] = ImageTaskData,
    ) -> None:
        """Initialize the JSONL pipeline output reader stage."""
        self.jsonl_path = Path(jsonl_path) if jsonl_path is not None else None
        self.verbose = verbose
        self.task_type = task_type

        # Validate that task_type has from_dict method
        if not hasattr(task_type, "from_dict") or not callable(getattr(task_type, "from_dict")):
            msg = f"task_type {task_type.__name__} must implement a 'from_dict' class method"
            raise TypeError(msg)

    def inputs(self) -> tuple[list[str], list[str]]:
        """No required task attributes; path comes from task.data[0] or init jsonl_path."""
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        """Define the output attributes produced by this stage."""
        return ["image_path", "image_id"], []

    def process(self, task: FileGroupTask | _EmptyTask) -> list[SingleDataTask[T_TaskData]]:
        """Process a JSONL file and create tasks from pipeline output.

        If task is a FileGroupTask with non-empty data, the JSONL path is taken
        from task.data[0]. Otherwise jsonl_path from init is used (legacy single-file mode).
        """
        if isinstance(task, FileGroupTask) and task.data:
            jsonl_path = Path(task.data[0])
        elif self.jsonl_path is not None:
            jsonl_path = self.jsonl_path
        else:
            msg = "Either use a file-partitioning stage before this stage or pass jsonl_path to JsonlPipelineOutputReaderStage."
            raise ValueError(msg)

        tasks: list[SingleDataTask[T_TaskData]] = []

        with jsonl_path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue

                record = json.loads(line)
                task_data = self.task_type.from_dict(record)

                # Some writer paths (e.g. include-invalid outputs) may serialize invalid records
                # without an explicit "is_valid" field. If an "error" is present, treat the
                # record as invalid to prevent downstream stages from assuming required fields
                # (like conversation media) exist.
                if record.get("error") is not None and hasattr(task_data, "is_valid"):
                    task_data.is_valid = False
                    if hasattr(task_data, "error") and not getattr(task_data, "error"):
                        task_data.error = record.get("error")

                image_task = SingleDataTask(
                    task_id=f"jsonl_output_reader_{idx}",
                    dataset_name=jsonl_path.stem,
                    data=task_data,
                )
                tasks.append(image_task)

                if self.verbose:
                    logger.info(f"Loaded task {idx} from {jsonl_path}")

        logger.info(f"Loaded {len(tasks)} tasks from {jsonl_path}")
        return tasks


class TarImageReader(CompositeStage[_EmptyTask, SingleDataTask[T_TaskData]], Generic[T_TaskData]):
    """Composite stage that reads images from WebDataset-style tar shards.

    Args:
        input_tar_path: Path to directory containing tar files.
        image_limit: Maximum number of images to process (None for unlimited).
        verbose: Whether to enable verbose logging.
    """

    input_tar_path: str
    image_limit: int | None = None
    verbose: bool = False
    name: str = "tar_image_reader"

    def __init__(self, input_tar_path: str, image_limit: int | None = None, verbose: bool = False, task_type: Type[T_TaskData] = ImageTaskData) -> None:
        """Initialize the tar image reader composite stage."""
        super().__init__()
        self.input_tar_path = input_tar_path
        self.image_limit = image_limit
        self.verbose = verbose
        self.task_type = task_type

    def decompose(self) -> list[ProcessingStage]:
        """Decompose into constituent execution stages."""
        partitioning_stage = FilePartitioningStage(
            file_paths=self.input_tar_path,
            files_per_partition=1,
            file_extensions=[".tar"],
            limit=self.image_limit,
        )

        reader_stage = TarImageReaderStage(
            input_path=self.input_tar_path,
            verbose=self.verbose,
            task_type=self.task_type,
        )

        return [partitioning_stage, reader_stage]

    def get_description(self) -> str:
        """Get a description of what this composite stage does."""
        limit_str = str(self.image_limit) if self.image_limit is not None else "unlimited"
        return (
            f"Reads images from tar files in '{self.input_tar_path}' "
            f"(limit: {limit_str})"
        )


class ParquetImageReaderStage(ProcessingStage[FileGroupTask, SingleDataTask[T_TaskData]], Generic[T_TaskData]):
    """Stage that reads images from parquet files.

    Each parquet file contains rows with "id" and "image" columns, where
    "image" contains a dict with "bytes" key holding raw image data.

    Can either store references (parquet_path, parquet_index) for lazy loading,
    or extract images directly to disk if extract_dir is provided.

    Args:
        input_path: Base path for resolving parquet file paths.
        extract_dir: If provided, extract images to this directory and set image_path.
        verbose: If True, logs detailed information.
    """

    input_path: str | None = None
    extract_dir: str | None = None
    verbose: bool = False
    name: str = "parquet_image_reader"
    resources = Resources(cpus=1.0)

    def __init__(
        self,
        input_path: str,
        extract_dir: str | None = None,
        verbose: bool = False,
        task_type: Type[T_TaskData] = ImageTaskData,
    ) -> None:
        """Initialize the parquet image reader stage."""
        self.input_path = input_path
        self.extract_dir = Path(extract_dir) if extract_dir else None
        self.verbose = verbose
        self.task_type = task_type

    def inputs(self) -> tuple[list[str], list[str]]:
        """Define the input attributes required by this stage."""
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        """Define the output attributes produced by this stage."""
        return ["image_path", "parquet_path", "parquet_index"], []

    def _get_subfolder(self, parquet_path: Path) -> str:
        """Get subfolder name based on parquet file name."""
        if m := re.match(r"^.*-(\d+)-of-\d+\.parquet$", parquet_path.name):
            return m.group(1)
        return parquet_path.stem

    def process(self, task: FileGroupTask) -> list[SingleDataTask[T_TaskData]]:
        """Process a parquet file and create ImageCaptionTasks.

        If extract_dir is set, extracts images to disk. Otherwise stores
        parquet references for lazy loading.

        Args:
            task: FileGroupTask containing parquet file path.

        Returns:
            List of ImageCaptionTask objects, one per row in the parquet.
        """
        if len(task.data) != 1:
            msg = f"Expected exactly 1 parquet file, got {len(task.data)}"
            raise ValueError(msg)

        parquet_path = Path(task.data[0])
        tasks: list[SingleDataTask[T_TaskData]] = []

        if self.extract_dir:
            # Extract images directly to disk
            subfolder = self._get_subfolder(parquet_path)
            output_subdir = self.extract_dir / subfolder
            output_subdir.mkdir(parents=True, exist_ok=True)

            table = pq.read_table(parquet_path, columns=["id", "image"])
            for idx in range(table.num_rows):
                row_id = table["id"][idx].as_py()
                image_data = table["image"][idx].as_py()
                image_bytes = image_data["bytes"]

                # Detect format and save
                image = Image.open(io.BytesIO(image_bytes))
                fmt = image.format or "PNG"
                ext = fmt.lower()
                if ext == "jpeg":
                    ext = "jpg"

                output_path = (output_subdir / f"{row_id}.{ext}").absolute()
                if not output_path.exists() or output_path.stat().st_size != len(image_bytes):
                    output_path.write_bytes(image_bytes)
                    if self.verbose:
                        logger.info(f"Extracted image: {row_id} -> {output_path}")

                image_task = SingleDataTask(
                    task_id=f"{task.task_id}_{subfolder}_{row_id}",
                    dataset_name=task.dataset_name,
                    data=self.task_type(
                        image_path=output_path,
                    ),
                )
                tasks.append(image_task)
        else:
            # Store parquet references for lazy loading
            table = pq.read_table(parquet_path, columns=["id"])
            for idx in range(table.num_rows):
                row_id = table["id"][idx].as_py()

                image_task = SingleDataTask(
                    task_id=f"{task.task_id}_{row_id}",
                    dataset_name=task.dataset_name,
                    data=self.task_type(
                        image_path=(Path(parquet_path) / f"{idx:08d}").absolute(),
                    ),
                )
                tasks.append(image_task)

                if self.verbose:
                    logger.info(f"Indexed image: {row_id} from {parquet_path}[{idx}]")

        return tasks


class ParquetImageReader(CompositeStage[_EmptyTask, SingleDataTask[T_TaskData]], Generic[T_TaskData]):
    """Composite stage that reads images from parquet files.

    Each parquet file contains rows with "id" and "image" columns.

    Args:
        input_parquet_path: Path to directory containing parquet files.
        extract_dir: If provided, extract images to this directory during reading.
        image_limit: Maximum number of parquet files to process (None for unlimited).
        verbose: Whether to enable verbose logging.
    """

    input_parquet_path: str
    extract_dir: str | None = None
    image_limit: int | None = None
    verbose: bool = False
    name: str = "parquet_image_reader"

    def __init__(
        self,
        input_parquet_path: str,
        extract_dir: str | None = None,
        image_limit: int | None = None,
        verbose: bool = False,
        task_type: Type[T_TaskData] = ImageTaskData,
    ) -> None:
        """Initialize the parquet image reader composite stage."""
        super().__init__()
        self.input_parquet_path = input_parquet_path
        self.extract_dir = Path(extract_dir) if extract_dir else None
        self.image_limit = image_limit
        self.verbose = verbose
        self.task_type = task_type

    def decompose(self) -> list[ProcessingStage]:
        """Decompose into constituent execution stages."""
        partitioning_stage = FilePartitioningStage(
            file_paths=self.input_parquet_path,
            files_per_partition=1,
            file_extensions=[".parquet"],
            limit=self.image_limit,
        )

        reader_stage = ParquetImageReaderStage(
            input_path=self.input_parquet_path,
            extract_dir=self.extract_dir,
            verbose=self.verbose,
            task_type=self.task_type,
        )

        return [partitioning_stage, reader_stage]

    def get_description(self) -> str:
        """Get a description of what this composite stage does."""
        limit_str = str(self.image_limit) if self.image_limit is not None else "unlimited"
        extract_str = f", extracting to '{self.extract_dir}'" if self.extract_dir else ""
        return (
            f"Reads images from parquet files in '{self.input_parquet_path}' "
            f"(limit: {limit_str}){extract_str}"
        )


class SkipProcessedStage(ProcessingStage[SingleDataTask[T_TaskData], SingleDataTask[T_TaskData]], Generic[T_TaskData]):
    """Skip tasks that have already been processed in a prior run.

    This stage is intended for a simple "resume" mode: it reads an existing JSONL
    output (written by `ResultWriterStage`) and drops any incoming tasks whose
    `image_path` is already present in that file.

    Notes:
    - Skipping is keyed by the serialized `image_path` string in the JSONL.
    - If `image_parent` was used when writing outputs, the same `image_parent`
      must be provided here to normalize task keys consistently.
    - This stage returns `None` for skipped tasks, which is supported by the
      local streaming executor (`simple_pipeline_runner.py`).
    """

    name: str = "skip_processed"
    resources = Resources(cpus=1.0)

    batch_size: int = 1

    def __init__(
        self,
        output_path: str | Path,
        *,
        image_parent: str | Path | None = None,
        require_exists: bool = True,
    ) -> None:
        self.output_path = Path(output_path)
        self.image_parent = Path(image_parent) if image_parent else None
        self.require_exists = require_exists

        self._processed: set[str] = set()
        self._seen_in_run: set[str] = set()

        self._loaded_count: int = 0
        self._skipped_existing: int = 0
        self._skipped_duplicate: int = 0
        self._passed: int = 0

    def setup(self, worker_metadata: WorkerMetadata) -> None:  # noqa: ARG002
        if not self.output_path.exists():
            if self.require_exists:
                raise FileNotFoundError(f"SkipProcessed: output path does not exist: {self.output_path}")
            logger.info(f"SkipProcessed: no existing output at {self.output_path}; nothing to skip")
            return

        if self.output_path.is_dir():
            raise ValueError(f"SkipProcessed: output path is a directory, expected JSONL file: {self.output_path}")

        processed: set[str] = set()
        bad_json: int = 0
        missing_key: int = 0

        with self.output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    bad_json += 1
                    continue
                if not isinstance(record, dict):
                    missing_key += 1
                    continue
                image_path = record.get("image_path")
                if not isinstance(image_path, str) or not image_path:
                    missing_key += 1
                    continue
                processed.add(image_path)

        self._processed = processed
        self._loaded_count = len(self._processed)

        logger.info(
            f"SkipProcessed: loaded {self._loaded_count} processed keys from {self.output_path} "
            f"(bad_json={bad_json}, missing_image_path={missing_key})"
        )

    def _get_image_path_key(self, image_path: Path | None) -> str | None:
        """Get the task key used for resume comparison."""
        if image_path is None:
            return None
        if self.image_parent is not None:
            try:
                return str(image_path.relative_to(self.image_parent))
            except ValueError:
                pass
        return str(image_path)
    
    def process(self, task: SingleDataTask[T_TaskData]) -> SingleDataTask[T_TaskData]:
        x = self.process_batch([task])
        if len(x) == 0:
            return None
        return x[0]
    
    def process_batch(self, tasks: list[SingleDataTask[T_TaskData]]) -> list[SingleDataTask[T_TaskData]]:
        results = []
        for task in tasks:
            key = self._get_image_path_key(getattr(task.data, "image_path", None))
            if key is None:
                self._passed += 1
                results.append(task)
                continue
            if key in self._processed:
                self._skipped_existing += 1
                continue
            if key in self._seen_in_run:
                self._skipped_duplicate += 1
                continue
            self._seen_in_run.add(key)
            self._passed += 1
            results.append(task)
        return results

    def teardown(self) -> None:
        logger.info(
            f"SkipProcessed: loaded={self._loaded_count}, passed={self._passed}, "
            f"skipped_existing={self._skipped_existing}, skipped_duplicate={self._skipped_duplicate}"
        )

    @property
    def stats(self) -> dict[str, int]:
        return {
            "loaded": self._loaded_count,
            "passed": self._passed,
            "skipped_existing": self._skipped_existing,
            "skipped_duplicate": self._skipped_duplicate,
        }


class ResultWriterStage(ProcessingStage[SingleDataTask[T_TaskData], SingleDataTask[T_TaskData]], Generic[T_TaskData]):
    """Stage for writing pipeline results to JSONL file.

    In distributed execution, each worker writes to a separate file
    with a worker-specific suffix to avoid conflicts. Use single_file=True
    to write to exactly the specified output path without any suffix.
    """

    name: str = "result_writer"
    resources = Resources(cpus=2.0)

    def __init__(
        self,
        output_path: str,
        valid_only: bool = True,
        image_parent: str | None = None,
        single_file: bool = False,
        append: bool = False,
    ) -> None:
        """Initialize the result writer stage.

        Args:
            output_path: Base path for output JSONL file.
            valid_only: If True, only write valid records.
            image_parent: If provided, make image paths relative to this directory.
            single_file: If True, write to exactly output_path without worker_id suffix.
            append: If True, append to existing output file (resume mode).
        """
        self.output_path = output_path
        self.valid_only = valid_only
        self.image_parent = Path(image_parent) if image_parent else None
        self.single_file = single_file
        self.append = append
        self._file: Any = None
        self._saved_count: int = 0
        self._skipped_count: int = 0
        self._worker_id: str = ""

    def setup(self, worker_metadata: WorkerMetadata) -> None:
        """Open output file for writing.

        Creates a unique filename per worker for distributed execution,
        unless single_file mode is enabled.
        """
        # Get worker ID for unique filename in distributed mode
        self._worker_id = str(worker_metadata.worker_id)

        # Check if worker_metadata requests single_file mode
        use_single_file = self.single_file

        # Create output path (per-worker or single file).
        # With multiple workers, single_file=True would have every worker open the same path
        # with mode "w", causing truncation; use single_file=False for distributed runs.
        base_path = Path(self.output_path)
        if self._worker_id and not use_single_file:
            suffix = base_path.suffix or ".jsonl"
            output = base_path.parent / f"{base_path.stem}_worker{self._worker_id}{suffix}"
        else:
            output = base_path

        output.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self.append else "w"
        self._file = open(output, mode, encoding="utf-8")
        self._saved_count = 0
        self._skipped_count = 0
        logger.info(f"ResultWriter: opened {output} for writing (mode={mode})")

    def _get_image_path_str(self, image_path: Path | None) -> str | None:
        """Get image path string, optionally relative to image_parent."""
        if image_path is None:
            return None
        if self.image_parent is not None:
            try:
                return str(image_path.relative_to(self.image_parent))
            except ValueError:
                pass
        return str(image_path)

    def process(self, task: SingleDataTask[T_TaskData]) -> SingleDataTask[T_TaskData]:
        """Write task to output file.

        Args:
            task: ImageCaptionTask to write.

        Returns:
            The same task (pass-through).
        """
        if task.data.is_valid:
            data = task.data.to_dict()
            data["image_path"] = self._get_image_path_str(task.data.image_path)
            # Keep empty lists/strings/False (e.g. OCR may legitimately be []).
            # Only drop fields that are explicitly None, and always omit is_valid.
            self._file.write(json.dumps({k: v for k, v in data.items() if v is not None and k != "is_valid"}) + "\n")
        else:
            if self.valid_only:
                self._skipped_count += 1
                return task
            else:
                data = task.data.to_dict()
                data["image_path"] = self._get_image_path_str(task.data.image_path)
                self._file.write(json.dumps({k: v for k, v in data.items() if v is not None and k != "is_valid"}) + "\n")
        self._file.flush()  # Flush after each write for safety
        self._saved_count += 1
        return task

    def teardown(self) -> None:
        """Close output file."""
        if self._file:
            self._file.close()
            self._file = None
            logger.info(f"ResultWriter: wrote {self._saved_count} results, skipped {self._skipped_count}")

    @property
    def stats(self) -> dict[str, int]:
        """Return write statistics."""
        return {
            "saved": self._saved_count,
            "skipped": self._skipped_count,
        }


def merge_output_shards(output_path: Path, *, delete_shards: bool = True) -> Path:
    """Merge per-worker JSONL shards from ResultWriterStage into a single file.

    ResultWriterStage writes one shard per worker named
    ``<stem>_worker<id><suffix>`` in the same directory as ``output_path``.
    Call this after ``pipeline.run()`` returns — by that point every worker
    has flushed its writes, so the merge is race-free regardless of node count.

    Args:
        output_path: The base output path passed to ResultWriterStage.
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
    mode = "a" if merged.exists() else "w"
    with open(merged, mode, encoding="utf-8") as fout:
        for shard in shards:
            with open(shard, encoding="utf-8") as fin:
                shutil.copyfileobj(fin, fout)

    if delete_shards:
        for shard in shards:
            shard.unlink()

    logger.info(f"merge_output_shards: merged {len(shards)} shards → {merged}")
    return merged


class ImageWriterStage(ProcessingStage[SingleDataTask[T_TaskData], SingleDataTask[T_TaskData]], Generic[T_TaskData]):
    """Stage for writing images to disk alongside results.

    Images are organized in subfolders based on parquet file name to avoid
    having too many files in a single directory.

    Folder structure: output_dir/parquet_stem/image_id.{ext}
    """

    name: str = "image_writer"
    resources = Resources(cpus=1.0)

    def __init__(self, output_dir: str, valid_only: bool = True) -> None:
        """Initialize the image writer stage.

        Args:
            output_dir: Base directory for output images.
            valid_only: If True, only write images for valid records.
        """
        self.output_dir = Path(output_dir)
        self.valid_only = valid_only
        self._saved_count: int = 0
        self._skipped_count: int = 0
        self._worker_id: str = ""

    def setup(self, worker_metadata: WorkerMetadata) -> None:
        """Set up the image writer."""
        self._worker_id = str(worker_metadata.worker_id)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._saved_count = 0
        self._skipped_count = 0
        logger.info(f"ImageWriter: writing images to {self.output_dir}")

    def _get_subfolder(self, task: SingleDataTask[T_TaskData]) -> str:
        """Get subfolder name based on parquet file or other source.

        Args:
            task: The image caption task.

        Returns:
            Subfolder name string.
        """
        if task.data.image_path.parent.name.endswith(".parquet"):
            # Use parquet file stem (e.g., "train-00000-of-00314.parquet")
            if m := re.match(r"^.*-(\d+)-of-\d+\.parquet$", task.data.image_path.parent.name):
                return m.group(1)
            return task.data.image_path.parent.stem
        if (tar_slice := _parse_tar_slice_path(task.data.image_path)) is not None:
            tar_path, _, _, _ = tar_slice
            return tar_path.stem or "images"
        elif task.data.image_path is not None:
            # For tar files, use tar stem; for images, use parent dir name
            return task.data.image_path.parent.name or "images"
        else:
            return "images"

    @staticmethod
    def _get_output_stem(image_path: Path) -> str:
        if (tar_slice := _parse_tar_slice_path(image_path)) is not None:
            _, _, _, internal_name = tar_slice
            return Path(internal_name).stem or image_path.stem
        return image_path.stem

    def process(self, task: SingleDataTask[T_TaskData]) -> SingleDataTask[T_TaskData]:
        """Write image to disk.

        Args:
            task: ImageCaptionTask containing image data.

        Returns:
            The same task (pass-through).
        """
        if self.valid_only and not task.data.is_valid:
            self._skipped_count += 1
            return task

        try:
            # Load image
            image = load_image_from_task(task)

            # Determine output path with subfolder
            subfolder = self._get_subfolder(task)
            output_subdir = self.output_dir / subfolder
            output_subdir.mkdir(parents=True, exist_ok=True)

            # Determine file extension based on image format
            fmt = image.format or "PNG"
            ext = fmt.lower()
            if ext == "jpeg":
                ext = "jpg"

            output_stem = self._get_output_stem(task.data.image_path)
            output_path = output_subdir / f"{output_stem}.{ext}"
            image.save(output_path)

            self._saved_count += 1
            logger.debug(f"ImageWriter: saved {output_path}")

            task.data.image_path = output_path

        except Exception as e:
            logger.error(f"ImageWriter: failed to save image {task.data.image_path}: {e}")
            self._skipped_count += 1

        return task

    def teardown(self) -> None:
        """Log final statistics."""
        logger.info(f"ImageWriter: wrote {self._saved_count} images, skipped {self._skipped_count}")

    @property
    def stats(self) -> dict[str, int]:
        """Return write statistics."""
        return {
            "saved": self._saved_count,
            "skipped": self._skipped_count,
        }


class FileReader(ABC):
    """Abstract base class for reading files from various sources."""

    @abstractmethod
    def can_read(self, path: Path) -> bool:
        """Check if this reader can handle the given path.

        Args:
            path: Path to check.

        Returns:
            True if this reader can handle the path.
        """
        pass

    @abstractmethod
    def read_bytes(self, path: Path) -> bytes:
        """Read raw bytes from the given path.

        Args:
            path: Path to read from.

        Returns:
            Raw bytes of the file.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the path format is invalid.
        """
        pass

    def open_image(self, path: Path) -> Image.Image:
        """Open a PIL Image from the given path.

        Args:
            path: Path to read from.

        Returns:
            PIL Image object.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the path format is invalid.
        """
        image_bytes = self.read_bytes(path)
        return Image.open(io.BytesIO(image_bytes))

    def read_image_url(self, path: Path) -> str:
        """Read image as a data URL from the given path.

        Args:
            path: Path to read from.

        Returns:
            Data URL string (data:image/...;base64,...).

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the path format is invalid.
        """
        image_bytes = self.read_bytes(path)
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.format == "JPEG":
                data_type = "image/jpeg"
            elif image.format == "PNG":
                data_type = "image/png"
            elif image.format == "WEBP":
                data_type = "image/webp"
            else:
                data_type = "image/png"
            return f"data:{data_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}"


class RegularFileReader(FileReader):
    """Reader for regular file system files."""

    def can_read(self, path: Path) -> bool:
        """Check if path is a regular file."""
        return path.exists() and path.is_file()
    
    def open_image(self, path: Path) -> Image.Image:
        return Image.open(path)

    def read_bytes(self, path: Path) -> bytes:
        """Read bytes from a regular file."""
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return path.read_bytes()


class TarFileReader(FileReader):
    """Reader for tar slice paths.

    Tar slice paths have the format: /path/to/file.tar/<offset>:<length>:<name>
    """

    def can_read(self, path: Path) -> bool:
        """Check if path is a tar slice path."""
        return ".tar/" in str(path)

    def read_bytes(self, path: Path) -> bytes:
        """Read bytes from a tar slice."""
        parsed = self._parse_tar_slice_path(path)
        if parsed is None:
            raise ValueError(f"Invalid tar slice path: {path}")

        tar_path, byte_offset, byte_length, _ = parsed
        if not tar_path.exists():
            raise FileNotFoundError(f"Tar file not found: {tar_path}")

        with tar_path.open("rb") as tar_file:
            tar_file.seek(byte_offset)
            image_bytes = tar_file.read(byte_length)

        if len(image_bytes) != byte_length:
            msg = f"Short read from tar slice: expected {byte_length}, got {len(image_bytes)}"
            raise ValueError(msg)

        return image_bytes

    @staticmethod
    def _parse_tar_slice_path(image_path: Path) -> tuple[Path, int, int, str] | None:
        """Parse tar slice path into components."""
        marker = ".tar/"
        path_str = str(image_path)
        marker_index = path_str.find(marker)
        if marker_index == -1:
            return None

        tar_path_str = path_str[: marker_index + len(".tar")]
        slice_spec = path_str[marker_index + len(marker) :]
        parts = slice_spec.split(":", 2)
        if len(parts) != 3:
            msg = f"Invalid tar slice format: {image_path}"
            raise ValueError(msg)

        byte_offset = int(parts[0])
        byte_length = int(parts[1])
        internal_name = parts[2]
        return Path(tar_path_str), byte_offset, byte_length, internal_name


class ParquetFileReader(FileReader):
    """Reader for parquet file references.

    Parquet paths have the format: /path/to/file.parquet/<index>
    """

    _cache: dict[Path, pq.ParquetFile] = {}
    _cache_size: int = 10

    def can_read(self, path: Path) -> bool:
        """Check if path is a parquet reference."""
        return path.parent.name.endswith(".parquet")

    def read_bytes(self, path: Path) -> bytes:
        """Read bytes from a parquet file."""
        parquet_path = path.parent
        parquet_file = self._cache.get(parquet_path)

        if parquet_file is None:
            parquet_file = pq.ParquetFile(parquet_path)
            self._cache[parquet_path] = parquet_file
            if len(self._cache) > self._cache_size:
                self._cache.pop(next(iter(self._cache))).close()

        table = parquet_file.read(columns=["image"])
        image_data = table["image"][int(path.stem)].as_py()
        return image_data["bytes"]


_file_readers: list[FileReader] = [
    ParquetFileReader(),
    TarFileReader(),
    RegularFileReader(),
]


def _get_reader_for_path(path: Path) -> FileReader:
    """Get the appropriate reader for the given path.

    Args:
        path: Path to find a reader for.

    Returns:
        FileReader that can handle the path.

    Raises:
        ValueError: If no reader can handle the path.
    """
    for reader in _file_readers:
        if reader.can_read(path):
            return reader
    raise ValueError(f"No reader found for path: {path}")


def _parse_tar_slice_path(image_path: Path) -> tuple[Path, int, int, str] | None:
    """Parse tar slice path into components.

    Deprecated: Use TarFileReader._parse_tar_slice_path instead.
    """
    return TarFileReader._parse_tar_slice_path(image_path)


def load_image_from_task(task: SingleDataTask[ImageTaskData]) -> Image.Image:
    """Load PIL Image from an ImageCaptionTask.

    Handles parquet references, in-memory bytes, direct file paths, and tar-embedded paths.

    Args:
        task: ImageCaptionTask containing image source (parquet ref, bytes, or path).

    Returns:
        PIL Image object.
    """
    reader = _get_reader_for_path(task.data.image_path)
    return reader.open_image(task.data.image_path)


