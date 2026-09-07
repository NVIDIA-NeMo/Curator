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

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Any, Literal

import cupy as cp
import numpy as np

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.deduplication.io_utils import DeduplicationIO
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.embedders.utils import create_list_series_from_1d_or_2d_ar
from nemo_curator.tasks import EmptyTask, FileGroupTask
from nemo_curator.utils.file_utils import check_disallowed_kwargs, get_default_file_extensions

from .utils import (
    ParquetFileInfo,
    break_parquet_partition_into_groups,
    get_array_from_df,
    read_parquet_file_info,
)

if TYPE_CHECKING:
    import cudf

import gc
import os
import random
import time

from loguru import logger

# Column names
L2_DIST_TO_CENT_COL = "l2_dist_to_cent"
COSINE_DIST_TO_CENT_COL = "cosine_dist_to_cent"
_AUTO_FIT_MEMORY_FRACTION = 0.6


class KMeansReadFitWriteStage(ProcessingStage[FileGroupTask, EmptyTask], DeduplicationIO):
    """KMeans clustering stage that requires RAFT for distributed processing."""

    def __init__(  # noqa: PLR0913
        self,
        id_field: str,
        embedding_field: str,
        output_path: str,
        filetype: Literal["parquet", "jsonl"],
        # KMeans args
        n_clusters: int,
        metadata_fields: list[str] | None = None,
        verbose: bool = False,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int = 42,
        init: Literal["k-means||", "random"] | np.ndarray = "k-means||",
        n_init: int | Literal["auto"] = 1,
        oversampling_factor: float = 2.0,
        max_samples_per_batch: int = 1 << 15,
        fit_data_fraction: float | None = None,
        # I/O args
        cache_path: str | None = None,
        read_kwargs: dict[dict] | None = None,
        write_kwargs: dict[dict] | None = None,
    ):
        """KMeans clustering stage that requires RAFT for distributed processing.

        Args:
            id_field (str): The column name of the id column.
            embedding_field (str): The column name of the embedding column.
            output_path (str): The path to the output directory.
            n_clusters (int): The number of clusters to create.
            metadata_fields (list[str] | None): The columns to keep in the output. These columns can be used later to prioritize deduplication.
            verbose (bool): Whether to print verbose output.
            max_iter (int): The maximum number of iterations to run.
            tol (float): Tolerance for stopping criteria of the kmeans algorithm.
            random_state (int): Seed for the random number generator. Unseeded by default. Does not currently fully guarantee the exact same results.
            init (Literal["k-means||", "random"] | np.ndarray): 'scalable-k-means++' or 'k-means||': Uses fast and stable scalable kmeans++ initialization. 'random': Choose 'n_cluster' observations (rows) at random from data for the initial centroids. If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.
            n_init (int | Literal["auto"]): Number of times the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
            oversampling_factor (float): The amount of points to sample in scalable k-means++ initialization for potential centroids. Increasing this value can lead to better initial centroids at the cost of memory. The total number of centroids sampled in scalable k-means++ is oversampling_factor * n_clusters * 8.
            max_samples_per_batch (int): The number of data samples to use for batches of the pairwise distance computation. This computation is done throughout both fit predict. The default should suit most cases. The total number of elements in the batched pairwise distance computation is max_samples_per_batch * n_clusters. It might become necessary to lower this number when n_clusters becomes prohibitively large.
            fit_data_fraction (float | None): Fraction of whole files used to fit KMeans. When None,
                Parquet selects as many complete files as fit the live GPU-memory budget, while JSONL
                fits all input files in one pass.
            cache_path (str | None): The path to save the centroids to. If None, the centroids will not be saved.
            read_kwargs (dict[dict]): Keyword arguments for the read stage.
            write_kwargs (dict[dict]): Keyword arguments for the write stage.
        """
        self.id_field = id_field
        self.embedding_field = embedding_field
        self.output_path = output_path
        self.filetype = filetype
        self.n_clusters = n_clusters
        self.metadata_fields = metadata_fields if metadata_fields is not None else []
        self.verbose = verbose
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init = init
        self.n_init = n_init
        self.oversampling_factor = oversampling_factor
        self.max_samples_per_batch = max_samples_per_batch
        if fit_data_fraction is not None and not 0.0 < fit_data_fraction <= 1.0:
            msg = (
                f"fit_data_fraction must be in (0, 1], got {fit_data_fraction}; "
                "pass None to auto-size Parquet fitting or fit all JSONL input"
            )
            raise ValueError(msg)
        self.fit_data_fraction = fit_data_fraction
        self.cache_path = cache_path
        self.read_kwargs = read_kwargs.copy() if read_kwargs is not None else {}
        self.write_kwargs = write_kwargs.copy() if write_kwargs is not None else {}

        check_disallowed_kwargs(self.read_kwargs, ["columns", "assign_id"])
        check_disallowed_kwargs(self.write_kwargs, ["partition_file_name", "partition_cols", "index"])

        self.input_storage_options = self.read_kwargs.pop("storage_options", None)
        self.output_storage_options = self.write_kwargs.pop("storage_options", None)

        self.name = "KMeansStage"
        self.resources = Resources(cpus=1.0, gpus=1.0)

    def process(self, task: FileGroupTask) -> EmptyTask:
        msg = "KMeansReadFitWriteStage does not support single-task processing"
        raise NotImplementedError(msg)

    def process_batch(self, tasks: list[FileGroupTask]) -> list[EmptyTask]:
        """Fit cooperatively, then predict and write each bounded frame."""

        if not tasks:
            return []

        all_files = [file for task in tasks for file in task.data]
        if self.filetype == "parquet":
            return self._process_parquet(tasks, all_files)
        if self.filetype != "jsonl":
            msg = f"Unsupported filetype: {self.filetype}. Only jsonl and parquet are supported."
            raise ValueError(msg)

        if self.fit_data_fraction is not None:
            return self._process_jsonl_two_pass(tasks, all_files)
        return self._process_jsonl_single_pass(tasks, all_files)

    def _process_parquet(self, tasks: list[FileGroupTask], files: list[str]) -> list[EmptyTask]:  # noqa: PLR0915
        columns = list(dict.fromkeys([self.id_field, self.embedding_field, *self.metadata_fields]))
        footer_start = time.perf_counter()
        file_info = read_parquet_file_info(
            files,
            retained_columns=[self.id_field, *self.metadata_fields],
            embedding_column=self.embedding_field,
            storage_options=self.input_storage_options,
        )
        footer_time = time.perf_counter() - footer_start
        self._log_metric("kmeans_footer_scan_time", footer_time)

        fit_info, prediction_only_info = self._sample_fit_files(file_info)
        total_rows = sum(info.num_rows for info in file_info)
        fit_rows = sum(info.num_rows for info in fit_info)
        if fit_rows < self.n_clusters:
            msg = f"KMeans fit sample has {fit_rows} rows but requires at least {self.n_clusters}"
            raise ValueError(msg)

        fit_frames = iter(self._iter_parquet_frames(fit_info, columns))
        read_start = time.perf_counter()
        first_fit_frame = next(fit_frames)
        embedding_width = get_array_from_df(first_fit_frame, self.embedding_field).shape[1]
        fit_embeddings = cp.empty((fit_rows, embedding_width), dtype=cp.float32)
        sampled_chunks: list[tuple[cudf.DataFrame, int, int]] = []
        offset = 0
        for df in chain([first_fit_frame], fit_frames):
            stop = offset + len(df)
            embeddings = fit_embeddings[offset:stop]
            embeddings[...] = get_array_from_df(df, self.embedding_field)
            self._normalize_embeddings_in_place(embeddings)
            del df[self.embedding_field]
            sampled_chunks.append((df, offset, stop))
            offset = stop
        read_time = time.perf_counter() - read_start
        if offset != fit_rows:
            msg = f"Parquet footers reported {fit_rows} fit rows but the reader returned {offset}"
            raise RuntimeError(msg)
        del df, embeddings, first_fit_frame, fit_frames

        fit_start = time.perf_counter()
        self.kmeans.fit(fit_embeddings, sample_weight=None)
        fit_labels = cp.asarray(self.kmeans.labels_).astype(cp.int32, copy=False)
        fit_time = time.perf_counter() - fit_start
        self._log_metrics(
            {
                "kmeans_fit_time": fit_time,
                "kmeans_fit_rows": fit_rows,
                "kmeans_fit_files": len(fit_info),
                "kmeans_input_files": len(file_info),
                "kmeans_fit_data_fraction": fit_rows / total_rows,
                "kmeans_fit_file_fraction": len(fit_info) / len(file_info),
            }
        )
        centroids = cp.ascontiguousarray(cp.asarray(self.kmeans.cluster_centers_).copy())
        self._save_centroids(centroids)
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

        predict_time = 0.0
        write_time = 0.0
        predicted_rows = 0
        output_index = 0
        for metadata, start, stop in sampled_chunks:
            write_start = time.perf_counter()
            self._write_output_frame(
                f"{tasks[0].task_id}_{output_index}.parquet",
                metadata,
                fit_embeddings[start:stop],
                fit_labels[start:stop],
                centroids,
            )
            write_time += time.perf_counter() - write_start
            predicted_rows += len(metadata)
            output_index += 1

        sampled_chunks.clear()
        del fit_embeddings, fit_labels
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

        if prediction_only_info:
            read_start = time.perf_counter()
            for df in self._iter_parquet_frames(prediction_only_info, columns):
                read_time += time.perf_counter() - read_start
                embeddings = get_array_from_df(df, self.embedding_field).astype(cp.float32, copy=False)
                self._normalize_embeddings_in_place(embeddings)
                predict_start = time.perf_counter()
                labels = cp.asarray(self.kmeans.predict(embeddings, convert_dtype=False)).astype(cp.int32, copy=False)
                predict_time += time.perf_counter() - predict_start
                del df[self.embedding_field]
                write_start = time.perf_counter()
                self._write_output_frame(
                    f"{tasks[0].task_id}_{output_index}.parquet", df, embeddings, labels, centroids
                )
                write_time += time.perf_counter() - write_start
                predicted_rows += len(df)
                output_index += 1
                read_start = time.perf_counter()
            read_time += time.perf_counter() - read_start

        if predicted_rows != total_rows:
            msg = f"Parquet footers reported {total_rows} rows but prediction processed {predicted_rows}"
            raise RuntimeError(msg)
        self._log_metrics(
            {
                "kmeans_read_time": read_time,
                "kmeans_predict_time": predict_time,
                "kmeans_write_time": write_time,
                "num_rows": total_rows,
            }
        )
        actor_index = getattr(self, "_actor_index", 0)
        return [
            EmptyTask(
                dataset_name=f"kmeans_actor_{actor_index}",
                _metadata=None,
                _stage_perf=[],
                data=None,
            )
        ]

    def _sample_fit_files(
        self, file_info: list[ParquetFileInfo]
    ) -> tuple[list[ParquetFileInfo], list[ParquetFileInfo]]:
        rng = random.Random(self.random_state)  # noqa: S311
        shuffled = rng.sample(file_info, k=len(file_info))
        if self.fit_data_fraction is not None:
            count = max(1, round(len(file_info) * self.fit_data_fraction))
            fit = shuffled[:count]
        else:
            free_memory = cp.cuda.runtime.memGetInfo()[0]
            budget = int(free_memory * _AUTO_FIT_MEMORY_FRACTION)
            fit = []
            estimated_bytes = 0
            for info in shuffled:
                file_bytes = info.embedding_elements * cp.dtype(cp.float32).itemsize
                file_bytes += info.metadata_bytes
                if estimated_bytes + file_bytes > budget:
                    continue
                fit.append(info)
                estimated_bytes += file_bytes
            if not fit:
                msg = f"No complete Parquet file fits the automatic KMeans budget of {budget} bytes"
                raise MemoryError(msg)
        fit_paths = {info.path for info in fit}
        prediction_only = [info for info in file_info if info.path not in fit_paths]
        if self.fit_data_fraction is None and prediction_only:
            fit_rows = sum(info.num_rows for info in fit)
            total_rows = sum(info.num_rows for info in file_info)
            logger.warning(
                f"Automatic KMeans sizing selected {len(fit)}/{len(file_info)} files "
                f"({fit_rows}/{total_rows} rows) for fitting. Set fit_data_fraction=1.0 to fit all input rows."
            )
        else:
            logger.info(f"Selected {len(fit)}/{len(file_info)} complete files for KMeans fit")
        return fit, prediction_only

    def _iter_parquet_frames(self, file_info: list[ParquetFileInfo], columns: list[str]) -> Iterator["cudf.DataFrame"]:
        for group in break_parquet_partition_into_groups(file_info):
            yield self._read_group(group, columns)

    def _write_output_frame(
        self,
        output_filename: str,
        metadata: "cudf.DataFrame",
        embeddings: "cp.ndarray",
        labels: "cp.ndarray",
        centroids: "cp.ndarray",
    ) -> None:
        if not len(metadata):
            return
        frame = metadata.copy(deep=False)
        frame[self.embedding_field] = create_list_series_from_1d_or_2d_ar(embeddings, index=frame.index)
        frame["centroid"] = labels
        frame = self._assign_distances(frame, self.embedding_field, centroids)
        self.write_parquet(
            frame,
            self.output_path,
            partition_file_name=output_filename,
            partition_cols=["centroid"],
            index=False,
            storage_options=self.output_storage_options,
            **self.write_kwargs,
        )

    def _save_centroids(self, centroids: "cp.ndarray") -> None:
        if self.cache_path is not None and getattr(self, "_actor_index", 0) == 0:
            os.makedirs(self.cache_path, exist_ok=True)
            cp.save(f"{self.cache_path}/kmeans_centroids.npy", centroids)
            logger.info(f"Saved {self.n_clusters} KMeans centroids to {self.cache_path}/kmeans_centroids.npy")

    def _read_group(self, group: list[str], columns: list[str]) -> "cudf.DataFrame":
        """Read a group of files into a cudf DataFrame."""
        if self.filetype == "parquet":
            return self.read_parquet(
                group,
                columns=columns,
                storage_options=self.input_storage_options,
                assign_id=False,
                **self.read_kwargs,
            )
        if self.filetype == "jsonl":
            return self.read_jsonl(
                group,
                columns=columns,
                storage_options=self.input_storage_options,
                assign_id=False,
                **self.read_kwargs,
            )
        msg = f"Unsupported data type: {self.filetype}"
        raise ValueError(msg)

    def _process_jsonl_single_pass(self, tasks: list[FileGroupTask], files: list[str]) -> list["EmptyTask"]:
        """Read, fit, predict, and write JSONL input in one pass."""
        t0 = time.perf_counter()
        df = self._read_group(files, [self.id_field, self.embedding_field, *self.metadata_fields])
        embeddings = get_array_from_df(df, self.embedding_field).astype(cp.float32, copy=False)
        self._normalize_embeddings_in_place(embeddings)
        df[self.embedding_field] = create_list_series_from_1d_or_2d_ar(embeddings, index=df.index)

        t1 = time.perf_counter()
        self._log_metrics({"kmeans_read_time": t1 - t0, "num_rows": len(df)})
        logger.debug(f"Read time: {(t1 - t0):.2f} seconds")

        self.kmeans.fit(embeddings, sample_weight=None)
        self._save_centroids(cp.asarray(self.kmeans.cluster_centers_))
        df["centroid"] = self.kmeans.predict(embeddings).astype(cp.int32)

        t2 = time.perf_counter()
        self._log_metric("kmeans_fit_predict_time", t2 - t1)
        logger.info(f"KMeans fit+predict time: {(t2 - t1):.2f} seconds")

        df = self._assign_distances(df, self.embedding_field, self.kmeans.cluster_centers_)
        self.write_parquet(
            df,
            self.output_path,
            partition_file_name=f"{tasks[0].task_id}_0.parquet",
            partition_cols=["centroid"],
            index=False,
            storage_options=self.output_storage_options,
            **self.write_kwargs,
        )

        t3 = time.perf_counter()
        self._log_metric("kmeans_write_time", t3 - t2)
        logger.info(f"Write time: {(t3 - t2):.2f} seconds")

        return [EmptyTask(dataset_name="kmeans_group_0", _metadata=None, _stage_perf=[], data=None)]

    def _process_jsonl_two_pass(self, tasks: list[FileGroupTask], files: list[str]) -> list["EmptyTask"]:
        """Fit on sampled JSONL files, then predict and write every file."""
        pass1_read_time = self._fit_pass(files)
        results, pass2_read_time, total_rows = self._predict_write_pass(tasks, files)
        self._log_metrics(
            {
                "kmeans_read_time": pass1_read_time + pass2_read_time,
                "num_rows": total_rows,
            }
        )
        return results

    def _fit_pass(self, files: list[str]) -> float:
        """Sample JSONL files, read embeddings, fit KMeans, and save centroids.

        Returns:
            Wall-clock seconds spent reading sampled files (for the combined
            kmeans_read_time metric reported by the orchestrator).
        """
        fraction = self.fit_data_fraction

        target_n_files = round(len(files) * fraction)
        n_files = max(1, target_n_files)
        if target_n_files < 1:
            # RAFT's cooperative fit needs every actor to contribute at least one row,
            # so we pull up to 1. Warn loudly: the user asked for less than that, and
            # if many actors hit this floor the realized sample is much larger than
            # fit_data_fraction would suggest.
            logger.warning(
                f"fit_data_fraction={fraction} on {len(files)} files would sample "
                f"0 files for this actor; bumping to 1 to keep it in the cooperative "
                f"fit. Increase fit_data_fraction (or pass None for full data) if you "
                f"care about pass-1 cost."
            )
        rng = random.Random(self.random_state)  # noqa: S311
        fit_files = rng.sample(files, n_files)

        t0 = time.perf_counter()
        df = self._read_group(fit_files, [self.embedding_field])
        sampled_rows = len(df)
        concatenated_samples = get_array_from_df(df, self.embedding_field).astype(cp.float32, copy=True)
        self._normalize_embeddings_in_place(concatenated_samples)
        del df
        gc.collect()

        t1 = time.perf_counter()
        pass1_read_time = t1 - t0
        logger.debug(
            f"Pass 1 (sampling) time: {pass1_read_time:.2f}s, "
            f"read {len(fit_files)}/{len(files)} files = {sampled_rows} rows"
        )

        logger.info(
            f"Fitting KMeans on {len(concatenated_samples)} sampled rows "
            f"(fit_data_fraction={fraction:.4f}, {len(fit_files)}/{len(files)} files)"
        )

        self.kmeans.fit(concatenated_samples, sample_weight=None)
        del concatenated_samples
        gc.collect()
        # Stop the fit-time clock before centroid I/O so the metric isn't skewed
        # by disk write latency on actor 0.
        t_fit_done = time.perf_counter()
        self._log_metric("kmeans_fit_time", t_fit_done - t1)
        logger.info(f"KMeans fit time: {(t_fit_done - t1):.2f} seconds")

        self._save_centroids(cp.asarray(self.kmeans.cluster_centers_))

        return pass1_read_time

    def _predict_write_pass(
        self, tasks: list[FileGroupTask], files: list[str]
    ) -> tuple[list["EmptyTask"], float, int]:
        """Read all JSONL files, predict labels, and write results.

        Returns:
            (results, pass2_read_time, total_rows). The orchestrator combines
            pass2_read_time with pass1_read_time into kmeans_read_time, and
            reports total_rows as num_rows.
        """
        t_start = time.perf_counter()
        df = self._read_group(files, [self.id_field, self.embedding_field, *self.metadata_fields])
        embeddings = get_array_from_df(df, self.embedding_field).astype(cp.float32, copy=False)
        self._normalize_embeddings_in_place(embeddings)
        df[self.embedding_field] = create_list_series_from_1d_or_2d_ar(embeddings, index=df.index)
        pass2_read_time = time.perf_counter() - t_start
        total_rows = len(df)

        labels = self.kmeans.predict(embeddings).astype(cp.int32)
        df["centroid"] = labels
        df = self._assign_distances(df, self.embedding_field, self.kmeans.cluster_centers_)
        self.write_parquet(
            df,
            self.output_path,
            partition_file_name=f"{tasks[0].task_id}_0.parquet",
            partition_cols=["centroid"],
            index=False,
            storage_options=self.output_storage_options,
            **self.write_kwargs,
        )

        t_end = time.perf_counter()
        self._log_metric("kmeans_predict_write_time", (t_end - t_start) - pass2_read_time)
        logger.info(
            f"Pass 2 total time: {(t_end - t_start):.2f} seconds "
            f"(read: {pass2_read_time:.2f}s, predict+write: {(t_end - t_start) - pass2_read_time:.2f}s)"
        )

        return (
            [EmptyTask(dataset_name="kmeans_group_0", _metadata=None, _stage_perf=[], data=None)],
            pass2_read_time,
            total_rows,
        )

    def setup(self, _: WorkerMetadata | None = None) -> None:
        from cuml.cluster.kmeans_mg import KMeansMG as cumlKMeans

        if not hasattr(self, "_raft_handle"):
            msg = "RAFT handle not found. Make sure the stage is initialized with RAFT"
            raise ValueError(msg)

        self.kmeans = cumlKMeans(
            handle=self._raft_handle,
            output_type="cupy",
            init=self.init,
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            verbose=self.verbose,
            n_init=self.n_init,
            oversampling_factor=self.oversampling_factor,
            max_samples_per_batch=self.max_samples_per_batch,
        )

    @staticmethod
    def _normalize_embeddings_in_place(embeddings: "cp.ndarray") -> None:
        embeddings /= cp.linalg.norm(embeddings, axis=1, keepdims=True)

    @staticmethod
    def _assign_distances(df: "cudf.DataFrame", embedding_col: str, centroids: "cp.ndarray") -> "cudf.DataFrame":
        """
        Computes the L2 distance to nearest centroid to each embedding in the DataFrame.
        Embeddings are normalized. For cosine we'll need to normalize the centroids as well.
        """
        normalized_embeddings = get_array_from_df(df, embedding_col)
        # We normalize the centroids as well for cosine distance
        normalized_centroids = centroids / cp.linalg.norm(centroids, axis=1, keepdims=True)

        df[L2_DIST_TO_CENT_COL] = cp.sqrt(
            cp.sum((normalized_embeddings - centroids[df["centroid"].values]) ** 2, axis=1)
        )
        df[COSINE_DIST_TO_CENT_COL] = 1 - (
            cp.sum(
                normalized_embeddings * normalized_centroids[df["centroid"].values],
                axis=1,
            )
        )
        return df

    def ray_stage_spec(self) -> dict[str, Any]:
        return {
            "is_raft_actor": True,
        }


@dataclass
class KMeansStage(CompositeStage[EmptyTask, EmptyTask]):
    """KMeans clustering stage that requires RAFT for distributed processing."""

    n_clusters: int
    id_field: str
    embedding_field: str
    input_path: str | list[str]
    output_path: str
    metadata_fields: list[str] | None = None
    verbose: bool = False
    # I/O args
    input_filetype: Literal["jsonl", "parquet"] = "parquet"
    input_file_extensions: list[str] | None = None
    read_kwargs: dict[dict] | None = None
    write_kwargs: dict[dict] | None = None
    # KMeans args
    max_iter: int = 300
    tol: float = 1e-4
    random_state: int = 42
    init: Literal["k-means||", "random"] | np.ndarray = "k-means||"
    n_init: int | Literal["auto"] = 1
    oversampling_factor: float = 2.0
    max_samples_per_batch: int = 1 << 15
    fit_data_fraction: float | None = None
    cache_path: str | None = None
    """KMeans clustering stage that requires RAFT for distributed processing.

    Args:
        n_clusters (int): The number of clusters to create.
        id_field (str): The column name of the id column.
        embedding_field (str): The column name of the embedding column.
        input_path (str | list[str]): The path to the input directory.
        output_path (str): The path to the output directory.
        metadata_fields (list[str] | None): The columns to keep in the output. These columns can be used later to prioritize deduplication.
        verbose (bool): Whether to print verbose output.
        input_filetype (Literal["jsonl", "parquet"]): The type of the input file
        read_kwargs (dict[dict]): Keyword arguments for the read stage.
        write_kwargs (dict[dict]): Keyword arguments for the write stage.
        max_iter (int): The maximum number of iterations to run.
        tol (float): Tolerance for stopping criteria of the kmeans algorithm.
        random_state (int): Seed for the random number generator. Unseeded by default. Does not currently fully guarantee the exact same results.
        init (Literal["k-means||", "random"] | np.ndarray): 'scalable-k-means++' or 'k-means||': Uses fast and stable scalable kmeans++ initialization. 'random': Choose 'n_cluster' observations (rows) at random from data for the initial centroids. If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.
        n_init (int | Literal["auto"]): Number of times the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
        oversampling_factor (float): The amount of points to sample in scalable k-means++ initialization for potential centroids. Increasing this value can lead to better initial centroids at the cost of memory. The total number of centroids sampled in scalable k-means++ is oversampling_factor * n_clusters * 8.
        max_samples_per_batch (int): The number of data samples to use for batches of the pairwise distance computation. This computation is done throughout both fit predict. The default should suit most cases. The total number of elements in the batched pairwise distance computation is max_samples_per_batch * n_clusters. It might become necessary to lower this number when n_clusters becomes prohibitively large.
        fit_data_fraction (float | None): Fraction of whole files used for fitting. When None, Parquet
            sizes the sample automatically from free GPU memory, while JSONL fits all input files in
            one pass.
        cache_path (str | None): The path to save the centroids to. If None, the centroids will not be saved.
    """

    def __post_init__(self):
        """Initialize parent class after dataclass initialization."""
        super().__init__()
        # Validate eagerly so bad values surface at construction, not later in
        # decompose() / on a worker.
        if self.fit_data_fraction is not None and not 0.0 < self.fit_data_fraction <= 1.0:
            msg = (
                f"fit_data_fraction must be in (0, 1], got {self.fit_data_fraction}; "
                "pass None to auto-size Parquet fitting or fit all JSONL input"
            )
            raise ValueError(msg)
        if self.fit_data_fraction is None and self.input_filetype == "jsonl":
            logger.warning(
                "fit_data_fraction=None fits all JSONL input in one pass; automatic GPU-memory sizing is only "
                "available for Parquet input"
            )

    def decompose(self) -> list[ProcessingStage]:
        # Set default file extensions based on input_filetype if not provided
        file_extensions = self.input_file_extensions or get_default_file_extensions(self.input_filetype)

        return [
            FilePartitioningStage(
                file_paths=self.input_path,
                file_extensions=file_extensions,
                files_per_partition=1,  # We set this to one, and then the RaftActor will break it up into smaller groups
                storage_options=self.read_kwargs.get("storage_options") if self.read_kwargs is not None else None,
            ),
            KMeansReadFitWriteStage(
                id_field=self.id_field,
                embedding_field=self.embedding_field,
                output_path=self.output_path,
                filetype=self.input_filetype,
                n_clusters=self.n_clusters,
                metadata_fields=self.metadata_fields,
                verbose=self.verbose,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
                init=self.init,
                n_init=self.n_init,
                oversampling_factor=self.oversampling_factor,
                max_samples_per_batch=self.max_samples_per_batch,
                fit_data_fraction=self.fit_data_fraction,
                read_kwargs=self.read_kwargs,
                write_kwargs=self.write_kwargs,
                cache_path=self.cache_path,
            ),
        ]
