import argparse
import os

from nemo_curator.datasets import DocumentDataset
from nemo_curator.modules import ExactDuplicates
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.script_utils import ArgumentHelper


def pre_imports() -> None:
    import cudf  # noqa: F401


def main(args: argparse.Namespace) -> None:
    #  Parameters
    id_data_dir = args.id_data_dir
    json_blocksize = args.json_blocksize
    json_files_per_partition = args.json_files_per_partition
    exact_dedup_log_dir = args.exact_dedup_log_dir
    exact_dedup_cache_dir = args.exact_dedup_cache_dir
    exact_dedup_dir = args.exact_dedup_dir

    # Set up GPU-based Dask client
    client = get_client(scheduler_file=os.environ.get("SCHEDULER_FILE"))
    client.run(pre_imports)
    print(f"Dask client object: {client}")

    # If neither is set, set the default blocksize to 1GB
    if json_blocksize is None and json_files_per_partition is None:
        json_blocksize = "1gb"

    dataset = DocumentDataset.read_json(
        id_data_dir, backend="cudf", blocksize=json_blocksize, files_per_partition=json_files_per_partition
    )

    exact_dupes = ExactDuplicates(
        logger=exact_dedup_log_dir,
        id_field="id",
        text_field="text",
        # Decides whether output of the module is deduplicated dataset or duplicates
        # If true, you should set cache_dir for performance improvement
        perform_removal=False,
        cache_dir=exact_dedup_cache_dir,  # Optionally write the output to disk
    )

    # When perform_removal=False, it will only call .identify_duplicates() and return the list of duplicate IDs.
    # When perform_removal=True, then exact_dupes outputs the dataset with the duplicates removed.
    # It will behave by calling .identify_duplicates() and .remove() in sequence.
    duplicates = exact_dupes(dataset=dataset)  # or exact_dupes.identify_duplicates(dataset)

    # If caching, result is a path to the output dataset.
    if isinstance(duplicates, str):
        duplicates = DocumentDataset.read_parquet(duplicates, backend="cudf")

    dataset = exact_dupes.remove(dataset, duplicates)
    dataset.to_json(exact_dedup_dir)

    print(f"Saved exact deduplicated data to {exact_dedup_dir}")
    client.close()


def attach_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    arg_helper = ArgumentHelper(parser).add_distributed_args()  # noqa: F841

    parser.add_argument(
        "--id-data-dir",
        type=str,
        help="Path to the ID data directory.",
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
        "--exact-dedup-log-dir",
        type=str,
        help="Path to the exact deduplication log directory.",
        required=True,
    )
    parser.add_argument(
        "--exact-dedup-cache-dir",
        type=str,
        help="Path to the exact deduplication cache directory.",
        required=True,
    )
    parser.add_argument(
        "--exact-dedup-dir",
        type=str,
        help="Path to the exact deduplicated data directory.",
        required=True,
    )

    return parser


if __name__ == "__main__":
    main(attach_args(argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)).parse_args())
