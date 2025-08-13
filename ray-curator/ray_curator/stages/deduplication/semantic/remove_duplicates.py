from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    import fsspec

    from ray_curator.backends.base import WorkerMetadata

import ray

from ray_curator.stages.base import ProcessingStage
from ray_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
from ray_curator.tasks import DocumentBatch
from ray_curator.utils.file_utils import get_fs


@dataclass
class RemoveDuplicatesByIdStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Filter out rows whose dedup IDs appear in the duplicates index.

    This stage expects the input `DocumentBatch.data` to already contain the
    `CURATOR_DEDUP_ID_STR` column. It reads the duplicates index written by
    `IdentifySemanticDuplicatesStage` and removes matching rows using an
    ID-range-limited read (predicate pushdown) to keep memory usage bounded.

    The ID range is derived from the current batch's `source_files` metadata via
    the global `id_generator` actor, mirroring how IDs were assigned in the
    `JsonlReaderStage`.
    """

    duplicates_path: str
    id_column: str = CURATOR_DEDUP_ID_STR
    verbose: bool = False
    input_storage_options: dict[str, Any] | None = None

    _name: str = "remove_duplicates_by_id"

    def setup(self, _: WorkerMetadata | None = None) -> None:  # type: ignore[name-defined]
        # Prepare filesystem for listing duplicate parquet files
        self.fs: fsspec.AbstractFileSystem = get_fs(self.duplicates_path, storage_options=self.input_storage_options)
        # Lazily resolved; only for logging/use

    def process(self, task: DocumentBatch) -> DocumentBatch:
        # Validate expectations
        if task.data is None:
            return task

        if self.id_column not in task.data.columns:  # type: ignore[attr-defined]
            msg = (
                f"Input batch is missing required id column '{self.id_column}'. "
                "Ensure the reader assigned IDs before running this stage."
            )
            raise ValueError(msg)

        # Determine the ID range for this batch using the same id_generator actor
        # that assigned IDs during read. Fallback to local min/max if actor is missing.
        source_files: list[str] = task._metadata.get("source_files", [])  # type: ignore[attr-defined]
        min_id: int
        max_id: int

        id_gen = ray.get_actor("id_generator", namespace="id_generator")
        min_id, max_id = ray.get(id_gen.get_batch_range.remote(source_files, None))
        # Collect duplicate IDs for this batch's ID range using parquet predicate pushdown
        duplicate_ids: set[int] = set()

        # List candidate parquet files under duplicates_path
        try:
            entries = self.fs.ls(self.duplicates_path, detail=True)
            files = [e["name"] if isinstance(e, dict) else e for e in entries]
        except Exception:  # noqa: BLE001
            # Some filesystems do not support ls on directory; attempt glob
            files = self.fs.glob(self.duplicates_path.rstrip("/") + "/**/*.parquet")

        parquet_files = [p for p in files if str(p).endswith(".parquet")]
        if not parquet_files:
            if self.verbose:
                logger.info("No duplicate index parquet files found under %s; skipping removal", self.duplicates_path)
            return task

        # Use pyarrow parquet reader with filters
        import pyarrow.parquet as pq
        from pyarrow.fs import FSSpecHandler, PyFileSystem

        pa_fs = PyFileSystem(FSSpecHandler(self.fs))

        filters = [(self.id_column, ">=", min_id), (self.id_column, "<=", max_id)]

        for file_path in parquet_files:
            try:
                # Inspect schema to pick an available id column
                pf = pq.ParquetFile(file_path, filesystem=pa_fs)
                schema = pf.schema_arrow
                candidate_cols = [self.id_column, "id"]
                read_col = next((c for c in candidate_cols if c in schema.names), None)
                if read_col is None:
                    if self.verbose:
                        logger.warning(
                            "Skipping %s as it lacks expected id columns (%s)",
                            file_path,
                            candidate_cols,
                        )
                    continue

                # Use read_table with filters instead of read_row_groups
                table_filters = (
                    filters if read_col == self.id_column else [(read_col, ">=", min_id), (read_col, "<=", max_id)]
                )
                table = pq.read_table(
                    file_path,
                    filesystem=pa_fs,
                    columns=[read_col],
                    filters=table_filters,
                )
            except Exception as exc:  # noqa: BLE001
                if self.verbose:
                    logger.warning("Failed to read duplicates from %s: %s", file_path, exc)
                continue

            if table.num_rows == 0:
                continue

            # Convert to Python ints and extend the set
            arr = table.column(0)
            # Handle chunked arrays efficiently
            for chunk in arr.chunks if hasattr(arr, "chunks") and arr.num_chunks > 1 else [arr]:
                duplicate_ids.update(int(v.as_py()) for v in chunk if v is not None)

        if not duplicate_ids:
            if self.verbose:
                logger.info("No duplicates found for id range [%d, %d]", min_id, max_id)
            return task

        # DEBUG: Log what duplicates were found
        if self.verbose:
            logger.info(
                "Found %d duplicates for id range [%d, %d]: %s",
                len(duplicate_ids),
                min_id,
                max_id,
                sorted(list(duplicate_ids))[:10],
            )

        # Apply anti-join filter to the in-memory batch
        # Choose which column to match in the source batch: prefer self.id_column,
        # but fall back to 'id' if needed
        match_column = (
            self.id_column if self.id_column in task.data.columns else ("id" if "id" in task.data.columns else None)
        )  # type: ignore[attr-defined]
        if match_column is None:
            raise ValueError(f"Neither '{self.id_column}' nor 'id' present in source batch; cannot match duplicates")

        try:
            import cudf  # type: ignore

            is_cudf = isinstance(task.data, cudf.DataFrame)  # type: ignore[attr-defined]
        except Exception:
            is_cudf = False

        if is_cudf:
            import cudf  # type: ignore

            dup_series = cudf.Series(list(duplicate_ids))
            mask = ~task.data[match_column].isin(dup_series)  # type: ignore[index]
            filtered = task.data.loc[mask]  # type: ignore[attr-defined]
        else:
            # Assume pandas-like dataframe
            filtered = task.data[~task.data[match_column].isin(duplicate_ids)]  # type: ignore[index]

        if self.verbose:
            removed = len(task.data) - len(filtered)  # type: ignore[arg-type]
            logger.info(
                "Removed %d duplicate rows out of %d (batch range [%d, %d])", removed, len(task.data), min_id, max_id
            )  # type: ignore[arg-type]

        # Return a new DocumentBatch with filtered data and same metadata
        return DocumentBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=filtered,
            _metadata=task._metadata,
        )
