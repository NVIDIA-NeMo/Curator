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

import os
import time

import cudf
from tqdm import tqdm

from nemo_curator.gpu_deduplication.jaccard_utils.batch_shuffle_utils import (
    text_bytes_aware_shuffle,
)
from nemo_curator.gpu_deduplication.jaccard_utils.doc_id_mapping import (
    combine_back_adlr_ids,
)
from nemo_curator.gpu_deduplication.jaccard_utils.io_utils import (
    aggregated_anchor_docs_with_bk_read,
    get_restart_offsets,
    get_text_ddf_from_json_path_with_blocksize,
    update_restart_offsets,
)
from nemo_curator.gpu_deduplication.jaccard_utils.merge_utils import (
    extract_partitioning_index,
    filter_text_rows_by_bucket_batch,
    merge_left_to_shuffled_right,
)
from nemo_curator.gpu_deduplication.utils import (
    get_client,
    get_num_workers,
    parse_nc_args,
    performance_report_if,
)


def write_partitioned_file(df, output_path, partition_on, batch_id):
    if len(df) == 0:
        return cudf.Series([True])

    cudf.io.parquet.write_to_dataset(
        df,
        output_path,
        partition_cols=[partition_on],
        filename=f"batch_{batch_id}.parquet",
    )
    return cudf.Series([True])


def batched_merge_and_write(
    left_df,
    right_df,
    merge_on,
    partition_on,
    output_path,
    parts_per_text_batch,
    parts_per_bucket_batch,
    bk_mapping,
    num_workers=None,
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
    bucket_part_start_offset, text_part_start_offset = get_restart_offsets(output_path)

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

    print(
        f"Starting at bucket-map partition {bucket_part_start_offset}"
        f" and text-df partition {text_part_start_offset}",
        flush=True,
    )

    for bucket_part_offset in tqdm(
        range(bucket_part_start_offset, bucket_part_end_offset, parts_per_bucket_batch)
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
            subset_text_df = left_df_use.partitions[text_part_offset:end_text_offset]

            try:
                # NOTE: If we have more text-df partitions than bucket-map
                # partitions, we are more likely to see an OverflowError
                output_df = text_bytes_aware_shuffle(
                    merge_left_to_shuffled_right(
                        subset_text_df,
                        subset_bucket_df,
                        merge_on,
                    ),
                    partition_on,
                    num_workers=num_workers,
                )
            except OverflowError as err:
                # We encountered an overflow error!
                # Let's try again with less text data
                parts_per_text_batch_retry = int(parts_per_text_batch_use / 2)
                if parts_per_text_batch_retry < 1:
                    raise err
                print(
                    f"\nWe encountered an OverflowError and will retry "
                    f"the current batch with {parts_per_text_batch_retry} "
                    f"text partitions instead of {parts_per_text_batch_use}.",
                    flush=True,
                )
                continue

            output_df = output_df.map_partitions(combine_back_adlr_ids)
            batch_label = f"{end_bucket_offset}_{end_text_offset}"
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


def jaccard_shuffling_workflow(
    client,
    input_data_paths,
    input_anchor_docs_with_bk_dir,
    output_shuffled_docs_path,
    text_ddf_blocksize,
    bucket_mapping_ddf_blocksize,
    num_files,
    parts_per_worker,
    profile_path,
    bucket_parts_per_worker,
):
    """'
    Args:
        client: dask client
        input_data_paths: paths to input data
        input_anchor_docs_with_bk_dir: path to input anchor docs with buckets
        output_shuffled_docs_path: path to output shuffled docs
        text_ddf_blocksize: block size for chunking jsonl files for text ddf
        bucket_mapping_ddf_blocksize: block size for chunking parquet files
                                      for anchor_docs_with_bk ddf
        num_files: number of files to process
        parts_per_worker: parts per worker to process in a batch
        profile_path: dask profile path
        bucket_parts_per_worker: bucket parts per worker to process in a batch
    """
    # Part1. Reading+Shuffling Data
    # Read Text from Data from jsonl files

    text_ddf = get_text_ddf_from_json_path_with_blocksize(
        input_data_paths=input_data_paths,
        num_files=num_files,
        blocksize=text_ddf_blocksize,
    )
    print(
        "Graph creation for get_text_ddf_from_json_path_with_blocksize" " complete.",
        flush=True,
    )
    print(f"text_ddf.npartitions  = {text_ddf.npartitions}", flush=True)
    st = time.time()
    ddf_anchor_docs_with_bk, bk_mapping = aggregated_anchor_docs_with_bk_read(
        input_anchor_docs_with_bk_dir,
        blocksize=bucket_mapping_ddf_blocksize,
    )
    print("Getting ddf_anchor_docs_with_bk completed")
    print(
        f"ddf_anchor_docs_with_bk.npartitions = {ddf_anchor_docs_with_bk.npartitions}",
        flush=True,
    )
    st = time.time()
    num_workers = get_num_workers(client)
    parts_per_batch = num_workers * parts_per_worker
    print(f"parts_per_batch  = {parts_per_batch}")
    parts_per_bucket_batch = num_workers * bucket_parts_per_worker
    print(f"parts_per_bucket_batch  = {parts_per_bucket_batch}")
    dask_profile_name = f"blocksize-{text_ddf_blocksize}"
    dask_profile_name = dask_profile_name + f"parts_per_batch-{parts_per_batch}"
    dask_profile_name = (
        dask_profile_name + f"-parts_per_bucket_batch-{parts_per_bucket_batch}"
    )
    dask_profile_name = dask_profile_name + f"-jaccard-n_input_files-{num_files}.html"

    text_ddf = text_ddf[["dataset_id", "doc_id", "text"]]

    with performance_report_if(profile_path, dask_profile_name):
        # Merge and write the dataframes
        batched_merge_and_write(
            text_ddf,
            ddf_anchor_docs_with_bk,
            output_path=output_shuffled_docs_path,
            merge_on=["dataset_id", "doc_id"],
            partition_on="output_partition_id",
            parts_per_text_batch=parts_per_batch,
            parts_per_bucket_batch=parts_per_bucket_batch,
            bk_mapping=bk_mapping,
            num_workers=num_workers,
        )
        print(f"Writing+Shuffling data took = {time.time()-st} s", flush=True)


def main(args):
    input_data_paths = args.input_data_dirs
    input_anchor_docs_with_bk_dir = args.input_bucket_mapping_dir
    OUTPUT_PATH = args.output_dir
    output_anchor_docs_with_bk_path = os.path.join(
        OUTPUT_PATH, "anchor_docs_with_bk.parquet"
    )
    output_shuffled_docs_path = os.path.join(OUTPUT_PATH, "shuffled_docs.parquet")
    client = get_client(args)
    print(f"Num Workers = {get_num_workers(client)}", flush=True)
    print("Connected to dask cluster", flush=True)
    print("Running jaccard shuffle script", flush=True)
    print(f"Args = {args}")
    st = time.time()
    jaccard_shuffling_workflow(
        client=client,
        input_data_paths=input_data_paths,
        input_anchor_docs_with_bk_dir=input_anchor_docs_with_bk_dir,
        output_shuffled_docs_path=output_shuffled_docs_path,
        text_ddf_blocksize=args.text_ddf_blocksize,
        bucket_mapping_ddf_blocksize=args.bucket_mapping_ddf_blocksize,
        num_files=args.num_files,
        parts_per_worker=args.parts_per_worker,
        profile_path=args.profile_path,
        bucket_parts_per_worker=args.bucket_parts_per_worker,
    )
    et = time.time()
    print(f"Jaccard Shuffle E2E time taken = {et-st} s")


def attach_args(parser=None):
    description = """Shuffles input text documents based on the given bucket
    map. The output is a partitioned parquet dataset with the documents
    shuffled by buckets
    """
    if not parser:
        parser = parse_nc_args(description=description)

    parser.add_argument(
        "--input-bucket-mapping-dir",
        type=str,
        help="The directory containing anchor docs with bk files",
    )
    parser.add_argument(
        "--text-ddf-blocksize",
        type=int,
        default=256,
        help="The block size for chunking jsonl files for text ddf in mb",
    )
    parser.add_argument(
        "--bucket-mapping-ddf-blocksize",
        type=int,
        default=256,
        help="The block size for for anchor_docs_with_bk ddf in mb",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="The output directory to write results in",
    )
    parser.add_argument(
        "--parts-per-worker",
        default=2,
        type=int,
        help="The number of parts to process per worker per batch",
    )
    parser.add_argument(
        "--bucket-parts-per-worker",
        default=8,
        type=int,
        help="The number of bucket parts to process per worker per batch",
    )
    return parser


def console_script():
    main(attach_args().parse_args())


if __name__ == "__main__":
    main(attach_args().parse_args())
