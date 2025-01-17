# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import logging
import os
import time
from typing import List, Union

import cudf
import dask_cudf
from tqdm import tqdm

from nemo_curator.log import create_logger
from nemo_curator.utils.distributed_utils import (
    get_current_client,
    get_num_workers,
    performance_report_if_with_ts_suffix,
)
from nemo_curator.utils.fuzzy_dedup_utils.id_mapping import int_ids_to_str
from nemo_curator.utils.fuzzy_dedup_utils.io_utils import (
    aggregated_anchor_docs_with_bk_read,
    get_restart_offsets,
    update_restart_offsets,
)
from nemo_curator.utils.fuzzy_dedup_utils.merge_utils import (
    extract_partitioning_index,
    filter_text_rows_by_bucket_batch,
    merge_left_to_shuffled_right,
)
from nemo_curator.utils.fuzzy_dedup_utils.shuffle_utils import write_partitioned_file


class _Shuffle:
    def __init__(
        self,
        id_fields: Union[str, list] = "id",
        text_field: str = "text",
        logger: Union[logging.LoggerAdapter, str] = "./",
        profile_dir: str = None,
        int_to_str_id: str = None,
    ):
        if isinstance(logger, str):
            self._logger = create_logger(
                rank=0,
                log_file=os.path.join(logger, "LSH.log"),
                name="LSH",
            )
        else:
            self._logger = logger

        self.id_fields = id_fields
        self.text_field = text_field
        self.profile_dir = profile_dir
        self.int_to_str_id = int_to_str_id

    def shuffle_docs_on_buckets(
        self,
        documents_df: dask_cudf.DataFrame,
        bucket_w_anchors_path: str,
        output_shuffled_docs_path: str,
        bucket_mapping_df_blocksize,
        parts_per_worker: int = 1,
        bucket_parts_per_worker: int = 8,
        partition_on: str = "_output_partition_id",
    ):

        ddf_anchor_docs_with_bk, bk_mapping = aggregated_anchor_docs_with_bk_read(
            path=bucket_w_anchors_path,
            blocksize=bucket_mapping_df_blocksize,
        )
        self._logger.info("Getting ddf_anchor_docs_with_bk completed")
        self._logger.debug(
            f"ddf_anchor_docs_with_bk.npartitions = {ddf_anchor_docs_with_bk.npartitions}"
        )
        st = time.time()
        num_workers = get_num_workers(get_current_client())
        parts_per_batch = num_workers * parts_per_worker
        self._logger.debug(f"parts_per_batch  = {parts_per_batch}")
        parts_per_bucket_batch = num_workers * bucket_parts_per_worker
        self._logger.debug(f"parts_per_bucket_batch  = {parts_per_bucket_batch}")

        dask_profile_name = (
            "suffle_docs"
            + f"-parts_per_batch-{parts_per_batch}"
            + f"-parts_per_bucket_batch-{parts_per_bucket_batch}"
        )
        documents_df = documents_df[self.id_fields + [self.text_field]]

        with performance_report_if_with_ts_suffix(self.profile_dir, dask_profile_name):
            self._batched_merge_and_write(
                left_df=documents_df,
                right_df=ddf_anchor_docs_with_bk,
                output_path=output_shuffled_docs_path,
                merge_on=self.id_fields,
                partition_on=partition_on,
                parts_per_text_batch=parts_per_batch,
                parts_per_bucket_batch=parts_per_bucket_batch,
                bk_mapping=bk_mapping,
                num_workers=num_workers,
            )
        self._logger.info(
            f"Time taken for Shuffle = {time.time()-st}s and output written at {output_shuffled_docs_path}"
        )

    def _batched_merge_and_write(
        self,
        left_df: dask_cudf.DataFrame,
        right_df: dask_cudf.DataFrame,
        output_path: str,
        merge_on: List[str],
        partition_on: str,
        parts_per_text_batch: int,
        parts_per_bucket_batch: int,
        bk_mapping,
        num_workers: int = None,
    ):
        total_text_partitions = left_df.npartitions
        total_bucket_partitions = right_df.npartitions

        # Extract global partitioning index
        left_df, global_partitioning_index = extract_partitioning_index(
            left_df,
            merge_on,
            bk_mapping,
            parts_per_bucket_batch,
            total_bucket_partitions,
        )

        # Set start offsets
        bucket_part_start_offset, text_part_start_offset = get_restart_offsets(
            output_path
        )

        # Set end offsets
        # NOTE: These end offsets are always set to the end
        # of the data. However, we may want to be able to set
        # both the start and end offsets from the command line
        # in the future.
        bucket_part_end_offset = total_bucket_partitions
        text_part_end_offset = total_text_partitions

        # Check that offsets are valid
        assert bucket_part_start_offset % parts_per_bucket_batch == 0
        assert bucket_part_end_offset > bucket_part_start_offset
        assert text_part_end_offset > text_part_start_offset

        # Initialize "retry" variables
        #
        # - retry_count: The number of successive batches that
        #     we have already performed at a reduced batch size.
        # - retry_threshold: The number of successive batches
        #     for which we should keep the batch size low
        #     before attempting the default batch size again.
        #     Every time we return to the default batch size
        #     and immediately fail, retry_threshold will double.
        parts_per_text_batch_retry = None
        retry_count, retry_threshold = 0, 1

        self._logger.info(
            f"Starting at bucket-map partition {bucket_part_start_offset}"
            f" and text-df partition {text_part_start_offset}",
        )

        for bucket_part_offset in tqdm(
            range(
                bucket_part_start_offset, bucket_part_end_offset, parts_per_bucket_batch
            )
        ):

            # Outer loop over batches of "bucket-map" partitions
            end_bucket_offset = min(
                bucket_part_offset + parts_per_bucket_batch, bucket_part_end_offset
            )
            print(
                f"\nStarted processing bucket-map partitions {bucket_part_offset} "
                f"through {end_bucket_offset} of {bucket_part_end_offset}",
                flush=True,
            )
            st_bucket = time.time()

            # Select our bucket-mapping batch
            subset_bucket_df = right_df.partitions[bucket_part_offset:end_bucket_offset]
            subset_bucket_df = subset_bucket_df.persist()

            # Filter out rows of left_df that we know cannot
            # align with any rows of subset_bucket_df
            left_df_use = filter_text_rows_by_bucket_batch(
                left_df,
                global_partitioning_index,
                bucket_part_offset,
                bucket_part_end_offset,
                total_bucket_partitions,
            )

            text_part_offset = text_part_start_offset
            while text_part_offset < text_part_end_offset:

                # Check if we are "retrying" with a smaller "parts_per_text_batch"
                if parts_per_text_batch_retry:
                    parts_per_text_batch_use = parts_per_text_batch_retry
                else:
                    st_text = time.time()
                    parts_per_text_batch_use = parts_per_text_batch
                print(f"Using {parts_per_text_batch_use} text partitions.", flush=True)

                # Select partitions for our text batch
                end_text_offset = min(
                    text_part_offset + parts_per_text_batch_use, text_part_end_offset
                )
                subset_text_df = left_df_use.partitions[
                    text_part_offset:end_text_offset
                ]
                subset_merged_df = merge_left_to_shuffled_right(
                    subset_text_df,
                    subset_bucket_df,
                    merge_on,
                )
                output_df = subset_merged_df.shuffle(on=partition_on)

                if self.int_to_str_id is not None and output_df is not None:
                    output_df = output_df.map_partitions(
                        int_ids_to_str, id_column=self.int_to_str_id
                    )
                batch_label = f"{end_bucket_offset}_{end_text_offset}"
                if output_df is not None:
                    written_files = output_df.map_partitions(
                        write_partitioned_file,
                        output_path,
                        partition_on,
                        batch_label,
                        meta=cudf.Series([True]),
                    )
                    written_files = written_files.compute()
                update_restart_offsets(output_path, bucket_part_offset, end_text_offset)
                del output_df

                print(
                    "Text-df partition ",
                    f"{end_text_offset}/{text_part_end_offset} "
                    f"completed in {time.time()-st_text}",
                    flush=True,
                )

                # Update loop control-flow variables
                if parts_per_text_batch_use == parts_per_text_batch:
                    # We succeeded at the default batch size.
                    # Reset the retry count
                    retry_count, retry_threshold = 0, 1
                else:
                    # We succeeded at a lower batch size
                    retry_count += 1
                    if retry_count >= retry_threshold:
                        # Go back to the default text-batch size,
                        # but increase the retry_threshold in
                        # case we fail again
                        parts_per_text_batch_retry = None
                        retry_count, retry_threshold = 0, min(retry_threshold * 2, 16)
                text_part_offset += parts_per_text_batch_use

            update_restart_offsets(output_path, end_bucket_offset, end_text_offset)
            print(
                "Bucket partition ",
                f"{end_bucket_offset}/{bucket_part_end_offset} "
                f"completed in {time.time()-st_bucket}",
                flush=True,
            )

            # Need to reset text_part_start_offset to 0 after
            # a single bucket-batch pass (only matters if we are
            # breaking the bucket-mapping df into multiple batches)
            text_part_start_offset = 0
