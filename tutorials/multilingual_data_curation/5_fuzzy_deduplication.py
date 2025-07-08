import argparse
import os

import dask

from nemo_curator import FuzzyDuplicates, FuzzyDuplicatesConfig
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.script_utils import ArgumentHelper


def pre_imports() -> None:
    import cudf  # noqa: F401


def main(args: argparse.Namespace) -> None:
    #  Parameters
    exact_dedup_dir = args.exact_dedup_dir
    json_blocksize = args.json_blocksize
    json_files_per_partition = args.json_files_per_partition
    fuzzy_dedup_cache_dir = args.fuzzy_dedup_cache_dir
    fuzzy_dedup_log_dir = args.fuzzy_dedup_log_dir
    remove_duplicates = args.remove_duplicates
    fuzzy_dedup_dir = args.fuzzy_dedup_dir

    with dask.config.set({"dataframe.backend": "cudf"}):
        # Set up GPU-based Dask client
        client = get_client(scheduler_file=os.environ.get("SCHEDULER_FILE"))
        client.run(pre_imports)
        print(f"Dask client object: {client}")

        # If neither is set, set the default blocksize to 1GB
        if json_blocksize is None and json_files_per_partition is None:
            json_blocksize = "1gb"

        dataset = DocumentDataset.read_json(
            exact_dedup_dir, backend="cudf", blocksize=json_blocksize, files_per_partition=json_files_per_partition
        )

        fuzzy_dedup_config = FuzzyDuplicatesConfig(
            cache_dir=fuzzy_dedup_cache_dir,
            id_field="id",
            text_field="text",
            # Decides whether output of the module is a deduplicated dataset or the IDs of the duplicates
            perform_removal=False,
            seed=42,
            char_ngrams=24,
            num_buckets=20,
            hashes_per_bucket=13,
            use_64_bit_hash=False,
            buckets_per_shuffle=5,  # set to a smaller value if encountering OOMs during LSH
            false_positive_check=False,
        )
        fuzzy_dupes = FuzzyDuplicates(logger=fuzzy_dedup_log_dir, config=fuzzy_dedup_config)

        # When perform_removal=False, it will only call .identify_duplicates() and return the list of duplicate IDs.
        # When perform_removal=True, then exact_dup outputs the dataset with the duplicates removed.
        # It will behave by calling .identify_duplicates() and .remove() in sequence.
        duplicates = fuzzy_dupes(dataset=dataset)  # or fuzzy_dupes.identify_duplicates(dataset)

        if duplicates is None:
            print("No duplicates found")
            return

        if remove_duplicates:
            result = fuzzy_dupes.remove(dataset, duplicates)
            result.to_json(fuzzy_dedup_dir)
            print(f"Saved fuzzy deduplicated data to {fuzzy_dedup_dir}")
        else:
            print(f"Fuzzy duplicate IDs saved to {fuzzy_dedup_cache_dir}/connected_components.parquet")

        client.close()


def attach_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    arg_helper = ArgumentHelper(parser).add_distributed_args()
    # Recommend not using this argument, keeping it False (default)
    arg_helper.attach_bool_arg(
        parser, "remove-duplicates", default=False, help="Removes all duplicates in the dataset."
    )

    parser.add_argument(
        "--exact-dedup-dir",
        type=str,
        help="Path to the exact deduplicated data directory.",
        required=True,
    )
    parser.add_argument(
        "--json-blocksize",
        type=str,
        help="Blocksize to use for reading the JSONL files.",
        required=False,
    )
    parser.add_argument(
        "--json-files-per-partition",
        type=int,
        help="The number of JSONL files for each partition of the DocumentDataset.",
        required=False,
    )
    parser.add_argument(
        "--fuzzy-dedup-cache-dir",
        type=str,
        help="Path to the fuzzy deduplication cache directory.",
        required=True,
    )
    parser.add_argument(
        "--fuzzy-dedup-log-dir",
        type=str,
        help="Path to the fuzzy deduplication log directory.",
        required=True,
    )
    parser.add_argument(
        "--fuzzy-dedup-dir",
        type=str,
        help="Path to the fuzzy deduplicated data directory.",
        required=True,
    )

    return parser


if __name__ == "__main__":
    main(attach_args(argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)).parse_args())
