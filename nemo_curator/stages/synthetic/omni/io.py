"""I/O stages for reading and writing image caption tasks.

Based on NeMo Curator's VideoReader pattern.
"""

from __future__ import annotations

import io
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from datasets import Dataset

    from nemo_curator.backends.base import WorkerMetadata

from loguru import logger
from PIL import Image

from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import _EmptyTask
from nemo_curator.tasks.image import ImageTaskData, SingleDataTask

T_TaskData = TypeVar("T_TaskData", bound=ImageTaskData)


class HFDatasetImageReaderStage(ProcessingStage[_EmptyTask, SingleDataTask[T_TaskData]], Generic[T_TaskData]):
    """Reads images from a HuggingFace dataset and creates image tasks.

    Accepts either a HF Hub dataset name or a local path.  Images are saved
    as JPEGs to ``image_dir`` on first run and reused on subsequent runs
    (idempotent).  Replaces ``FilePartitioningStage`` + ``JsonlTarImageReaderStage``
    when the source is a HuggingFace dataset rather than tar shards.

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

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def _load_dataset(self) -> Dataset:
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
                        msg = (
                            f"Split '{self.split}' not found in dataset at {local_path}. Available splits: {available}"
                        )
                        raise ValueError(msg)
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
            image_id = str(example[self.id_column]) if self.id_column is not None else f"{idx:06d}"

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
            f"{self.dataset_name}/{self.split}" + (f" (limit={self.limit})" if self.limit else "")
        )
        return tasks


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

    def setup(self, _worker_metadata: WorkerMetadata) -> None:
        # Determine which files to read:
        #   1. Merged file exists (output_path itself) — normal completed-run case.
        #   2. Only _worker* shards exist — pipeline was killed before merge ran.
        #   3. Neither exists — first run or require_exists controls the error.
        files_to_read: list[Path] = []

        if self.output_path.exists() and not self.output_path.is_dir():
            files_to_read = [self.output_path]
        else:
            suffix = self.output_path.suffix or ".jsonl"
            pattern = f"{self.output_path.stem}_worker*{suffix}"
            shard_files = sorted(self.output_path.parent.glob(pattern))
            if shard_files:
                logger.info(
                    f"SkipProcessed: merged file not found; reading {len(shard_files)} worker shards from "
                    f"{self.output_path.parent}"
                )
                files_to_read = shard_files
            elif self.require_exists:
                msg = f"SkipProcessed: output path does not exist: {self.output_path}"
                raise FileNotFoundError(msg)
            else:
                logger.info(f"SkipProcessed: no existing output at {self.output_path}; nothing to skip")
                return

        processed: set[str] = set()
        bad_json: int = 0
        missing_key: int = 0

        for path in files_to_read:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        record = json.loads(stripped)
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
            f"SkipProcessed: loaded {self._loaded_count} processed keys from {len(files_to_read)} file(s) "
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
        self._file = open(output, mode, encoding="utf-8")  # noqa: SIM115
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
            self._file.write(
                json.dumps({k: v for k, v in data.items() if v is not None and k != "is_valid"}, default=str) + "\n"
            )
        elif self.valid_only:
            self._skipped_count += 1
            return task
        else:
            data = task.data.to_dict()
            data["image_path"] = self._get_image_path_str(task.data.image_path)
            self._file.write(
                json.dumps({k: v for k, v in data.items() if v is not None and k != "is_valid"}, default=str) + "\n"
            )
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
            msg = f"Image not found: {path}"
            raise FileNotFoundError(msg)
        return path.read_bytes()


_file_reader = RegularFileReader()


def load_image_from_task(task: SingleDataTask[ImageTaskData]) -> Image.Image:
    """Load PIL Image from an ``ImageTaskData`` task.

    Args:
        task: Task whose ``data.image_path`` points to a regular image file on disk.

    Returns:
        PIL Image object.
    """
    return _file_reader.open_image(task.data.image_path)
