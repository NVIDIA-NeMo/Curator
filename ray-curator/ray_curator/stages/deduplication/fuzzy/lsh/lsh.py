# Copyright (c) 2025, NVIDIA CORPORATION.
"""
This module implements a Locality Sensitive Hashing (LSH) actor for Ray that inherits from a base shuffling actor.
"""

# ruff: noqa: ERA001
from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import cudf
from rapidsmpf.utils.cudf import pylibcudf_to_cudf_dataframe

from ray_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
from ray_curator.stages.shuffler.rapidsmpf_shuffler import BulkRapidsMPFShuffler

if TYPE_CHECKING:
    from collections.abc import Iterator

# import ray
# from ray_curator.utils.file_utils import get_all_files_paths_under
# import math
# import time
# import os
# from rapidsmpf.integrations.ray import setup_ray_ucxx_cluster


class LSHActor(BulkRapidsMPFShuffler):
    """
    Actor that performs LSH operations and shuffling using Ray.

    Parameters
    ----------
    nranks
        Number of ranks in the communication group.
    total_nparts
        Total number of output partitions.
    shuffle_on
        List of column names to shuffle on.
    num_bands
        Number of LSH bands.
    minhashes_per_band
        Number of minhashes per band.
    id_column
        Name of the ID column in input data.
    minhash_field
        Name of the minhash field in input data.
    output_path
        Path to write output files.
    rmm_pool_size
        Size of the RMM memory pool in bytes.
    spill_device
        Device memory limit for spilling to host in bytes.
    enable_statistics
        Whether to collect statistics.

    Notes
    -----
    Architecture and Processing Flow:

    This implementation follows a clean separation of responsibilities with distinct methods
    for each part of the pipeline:

    Input Phase:
    - `read_minhash`: Reads minhash files and returns a DataFrame

    Processing Phase:
    - `minhash_to_bands`: Transforms a single minhash DataFrame into LSH bands
    - `read_and_insert`: Orchestrates reading, band creation, and insertion

    Output Phase:
    - `extract_and_group`: Extracts and groups shuffled data, yielding results as a generator
    - `extract_and_write`: Processes each yielded result and writes to output files immediately

    1. Files are read using `read_minhash`
    2. Data is processed with `minhash_to_bands` to extract LSH bucket IDs
    3. Processed data is immediately inserted into the shuffler
    4. Results are extracted and processed one partition at a time using generators
    5. Each partition is written to disk as soon as it's processed, without accumulating in memory
    """

    def __init__(  # noqa: PLR0913
        self,
        nranks: int,
        total_nparts: int,
        shuffle_on: list[str],
        num_bands: int,
        minhashes_per_band: int,
        id_column: str = CURATOR_DEDUP_ID_STR,
        minhash_field: str = "_minhash_signature",
        output_path: str = "./",
        rmm_pool_size: int = 1024 * 1024 * 1024,
        spill_device: int | str | None = "auto",
        *,
        enable_statistics: bool = False,
    ):
        super().__init__(
            nranks=nranks,
            total_nparts=total_nparts,
            shuffle_on=shuffle_on,
            output_path=output_path,
            rmm_pool_size=rmm_pool_size,
            spill_device=spill_device,
            enable_statistics=enable_statistics,
        )
        self.num_bands = num_bands
        self.minhashes_per_band = minhashes_per_band
        self.id_column = id_column
        self.minhash_field = minhash_field

    @staticmethod
    def _generate_band_ranges(num_bands: int, minhashes_per_band: int) -> list[list[int]]:
        """
        Generates a list of indices for the minhash ranges given num_bands &
        minhashes_per_band.
        eg: num_bands=3, minhashes_per_band=2
        [[0, 1], [2, 3], [4, 5]]
        """
        return [list(range(band * minhashes_per_band, (band + 1) * minhashes_per_band)) for band in range(num_bands)]

    def read_minhash(self, filepaths: list[str]) -> cudf.DataFrame:
        """
        Read minhash data from parquet files.

        Parameters
        ----------
        filepaths
            List of paths to minhash files.

        Returns
        -------
            DataFrame containing minhash data from all input files.
        """
        return cudf.read_parquet(filepaths, columns=[self.id_column, self.minhash_field])

    def minhash_to_bands(self, minhash_df: cudf.DataFrame, band_range: tuple[int, int]) -> cudf.DataFrame:
        """
        Process a single minhash DataFrame to extract LSH band data.

        Parameters
        ----------
        minhash_df
            DataFrame containing minhash data.
        band_range
            Tuple of (start_band, end_band) to process.

        Returns
        -------
            DataFrame with document IDs and their corresponding bucket IDs.
        """
        if minhash_df is None or len(minhash_df) == 0:
            return None

        # Get the band ranges for the specified band range
        band_ranges = self._generate_band_ranges(num_bands=self.num_bands, minhashes_per_band=self.minhashes_per_band)[
            band_range[0] : band_range[1]
        ]

        # Create a dataframe with just the ID column
        id_df = minhash_df[[self.id_column]]

        for i, h in enumerate(band_ranges):
            indices = cudf.Series([h]).repeat(len(id_df))
            id_df[f"_bucket_{i}"] = f"b{i}_" + minhash_df[self.minhash_field].list.take(indices).hash_values(
                method="md5"
            )

        value_vars = [f"_bucket_{i}" for i in range(len(band_ranges))]
        melted_df = id_df.melt(id_vars=[self.id_column], value_name="_bucket_id", value_vars=value_vars)

        # Keep only the columns we need
        return melted_df[[self.id_column, "_bucket_id"]]

    def read_and_insert(self, filepaths: list[str], band_range: tuple[int, int]) -> list[str]:
        """
        Read minhashes from files, create LSH bands, and insert into the shuffler.

        This method orchestrates the full processing pipeline:
        1. Reads minhash data from parquet files in batches
        2. Processes each batch to extract LSH bands
        3. Inserts the bands into the shuffler for distribution

        Parameters
        ----------
        filepaths
            List of paths to minhash files.
        band_range
            Tuple of (start_band, end_band) to process.

        Returns
        -------
            Column names of the output table.
        """
        # Get consistent column names for all batches
        column_names = [self.id_column, "_bucket_id"]

        if not filepaths:
            # Return column names even if no paths were provided
            return column_names

        if band_range[0] < 0 or band_range[1] > self.num_bands or band_range[0] >= band_range[1]:
            msg = f"Invalid band range: {band_range}, must be in range [0, {self.num_bands})"
            raise ValueError(msg)

        # Process files in batches
        minhash_df = self.read_minhash(filepaths)
        # Skip processing if the batch is empty
        if minhash_df is None or len(minhash_df) == 0:
            print("Skipping empty batch")
            return column_names

        # Process this batch of minhashes to get band data
        band_df = self.minhash_to_bands(minhash_df, band_range)

        # Call parent's insert_chunk method
        self.insert_chunk(band_df, list(band_df.columns))
        # Clear memory after processing a batch
        del minhash_df, band_df

        return column_names

    def group_by_bucket(self, df: cudf.DataFrame, include_singles: bool = False) -> cudf.DataFrame:
        """
        Group items by bucket ID and aggregate IDs into lists.

        Parameters
        ----------
        df
            DataFrame containing bucket IDs and document IDs.
        include_singles
            If True, include buckets with only one document. Default is False, which
            excludes single-document buckets as they cannot form duplicates. Set to True
            when building an LSH index that needs to maintain all documents.

        Returns
        -------
            DataFrame with bucket IDs and lists of document IDs.
        """
        if not include_singles:
            # Find bucket_ids that appear more than once (have multiple documents)
            # Keep only rows with buckets that are duplicated
            df = df[df["_bucket_id"].duplicated(keep=False)]
        # Group by bucket_id and aggregate document IDs
        return df.groupby("_bucket_id")[self.id_column].agg(list).list.sort_values().reset_index()

    def extract_and_group(self) -> Iterator[tuple[int, cudf.DataFrame]]:
        """
        Extract shuffled partitions and group by bucket ID, yielding results one by one.

        This generator approach allows processing each partition immediately after it's ready,
        which is more memory-efficient than collecting all partitions first.

        Yields
        ------
        tuple
            A tuple of (partition_id, grouped_df) where grouped_df contains bucket IDs
            and their corresponding document ID lists.
        """
        # Fixed column names for pylibcudf conversion
        column_names = [self.id_column, "_bucket_id"]
        for partition_id, partition in self.extract():
            # Convert to cuDF DataFrame
            df = pylibcudf_to_cudf_dataframe(partition, column_names=column_names)
            # Group by bucket ID
            grouped_df = self.group_by_bucket(df)

            # Yield the result immediately instead of collecting in a list
            yield partition_id, grouped_df
            # Clean up memory
            del df, grouped_df

    def extract_and_write(self) -> list[str]:
        """
        Extract shuffled partitions, group by bucket ID, and write results to files.

        This method orchestrates the post-processing pipeline:
        1. Extracts partitioned data from the shuffler using extract_and_group
        2. Writes each grouped partition to a parquet file as soon as it's available

        This generator-based approach is more memory-efficient since it processes
        one partition at a time rather than collecting all partitions in memory.
        """
        output_paths = []
        # Process each partition as it becomes available
        for partition_id, grouped_df in self.extract_and_group():
            path = f"{self.output_path}/part.{partition_id}.parquet"

            # Write to file immediately
            grouped_df.to_parquet(path, index=False)
            output_paths.append(path)
            # Clean up to release memory
            del grouped_df

        return output_paths


# def bulk_lsh(
#     minhash_dir: str,
#     output_path: str,
#     num_bands: int = 20,
#     minhashes_per_band: int = 13,
#     bands_per_iter: int = 5,
#     id_column: str = CURATOR_DEDUP_ID_STR,
#     minhash_field: str = "_minhash_signature",
#     num_workers: int | None = None,
#     batchsize: int = 1,
#     num_output_files: int | None = None,
#     rmm_pool_size: int = 1024 * 1024 * 1024,
#     spill_device: int | None = None,
#     *,
#     enable_statistics: bool = False,
# ) -> None:
#     """
#     Perform LSH bucketing and grouping using Ray and UCXX communication.

#     Parameters
#     ----------
#     minhash_dir
#         Directory containing minhash files.
#     output_path
#         Path to write output files.
#     num_bands
#         Number of LSH bands.
#     minhashes_per_band
#         Number of minhashes per band.
#     bands_per_iter
#         Number of bands to process per iteration.
#     id_column
#         Name of the ID column.
#     minhash_field
#         Name of the minhash field.
#     num_workers
#         Number of workers to use.
#     batchsize
#         Number of files to process in a batch. This controls memory usage by determining
#         how many parquet files are read and processed at once before being inserted
#         into the shuffler.
#     num_output_files
#         Number of output files to write. If None, will be determined based on input size.
#     rmm_pool_size
#         Size of the RMM memory pool in bytes.
#     spill_device
#         Device memory limit for spilling to host in bytes.
#     enable_statistics
#         Whether to collect statistics.
#     """
#     # Initialize parameters

#     print(f"Using {num_workers} workers")
#     paths = get_all_files_paths_under(minhash_dir)
#     num_input_files = len(paths)
#     input_batches = [paths[i : i + batchsize] for i in range(0, len(paths), batchsize)]
#     num_output_files = num_output_files or len(input_batches)
#     num_iterations = math.ceil(num_bands / bands_per_iter)
#     # num output partitions is the closest power of 2 less than or equal to the number of input batches
#     total_num_partitions = max(1, 2 ** math.floor(math.log2(num_output_files)))
#     print(f"Processing {num_input_files} files with {num_workers} workers")
#     print(f"Running {num_iterations} band iterations ({bands_per_iter} bands per iteration)")
#     print(f"Processing files in batches of {batchsize}")

#     start_time = time.time()
#     # Process each band range
#     for curr_iter, band_id in enumerate(range(0, num_bands, bands_per_iter)):
#         band_range = (band_id, min(band_id + bands_per_iter, num_bands))
#         print(f"Processing band range {band_range[0]}-{band_range[1]} ({curr_iter + 1}/{num_iterations})")
#         output_dir = f"{output_path}/bands_{band_range[0]}-{band_range[1]}"

#         # Create directory if it doesn't exist
#         os.makedirs(output_dir, exist_ok=True)

#         # Initialize LSH actors
#         actors = setup_ray_ucxx_cluster(
#             LSHActor,
#             num_workers=num_workers,
#             total_nparts=total_num_partitions,
#             shuffle_on=["_bucket_id"],
#             num_bands=num_bands,
#             minhashes_per_band=minhashes_per_band,
#             id_column=id_column,
#             minhash_field=minhash_field,
#             output_path=output_dir,
#             enable_statistics=enable_statistics,
#             rmm_pool_size=rmm_pool_size,
#             spill_device=spill_device,
#         )
#         from ray.util.actor_pool import ActorPool

#         actor_pool = ActorPool(actors)
#         iter_start_time = time.time()

#         results = actor_pool.map_unordered(
#             lambda actor, file_batch: actor.read_and_insert.remote(file_batch, band_range),
#             input_batches,
#         )
#         results = list(results)

#         # Signal that we're done inserting
#         ray.get([actor.insert_finished.remote() for actor in actors])

#         print(
#             f"Reading and inserting minhashes for band range {band_range[0]}-{band_range[1]} took {time.time() - iter_start_time} seconds"
#         )

#         ray.get([actor.extract_and_write.remote() for actor in actors])

#         iter_end_time = time.time()
#         print(f"Time taken for bands {band_range[0]}-{band_range[1]}: {iter_end_time - iter_start_time} seconds")
#         # Cleanup actors
#         ray.get([actor.cleanup.remote() for actor in actors])
#         for actor in actors:
#             ray.kill(actor)

#     end_time = time.time()
#     print(f"Total time taken for LSH: {end_time - start_time} seconds")


# def main() -> None:
#     """
#     Main function to run the LSH implementation with default parameters.
#     """
#     # Initialize Ray
#     ray.init(dashboard_host="0.0.0.0", )#={"env_vars":{"RAPIDSMPF_LOG":"INFO"}})

#     # Define input and output paths
#     minhash_path = "/raid/adattagupta/NeMo-Curator/ray-curator/ray_curator/fuzzy_dedup_cache/MinHashStage"
#     output_path = "/raid/adattagupta/NeMo-Curator/ray-curator/ray_curator/fuzzy_dedup_cache/LSHStage"
#     # Use 90% of available device memory
#     rmm_pool_size = (70 * 1024 * 1024 * 1024 // 256) * 256
#     spill_device = (55 * 1024 * 1024 * 1024 // 256) * 256

#     # Note shuffle perf is sensitive to batchsize try to use larger batchsize (based on input size) instead of fpp
#     files_per_partition = 8
#     num_output_partitions = None  # automatically set based on input files and batchsize

#     bulk_lsh(
#         minhash_dir=minhash_path,
#         num_output_files=num_output_partitions,
#         output_path=output_path,
#         num_bands=20,
#         minhashes_per_band=13,
#         bands_per_iter=20,
#         id_column=CURATOR_DEDUP_ID_STR,
#         minhash_field="_minhash_signature",
#         num_workers=4,
#         batchsize=files_per_partition,  # Use calculated batch size
#         enable_statistics=False,
#         rmm_pool_size=rmm_pool_size,
#         spill_device=spill_device,
#     )


# if __name__ == "__main__":
#     main()
