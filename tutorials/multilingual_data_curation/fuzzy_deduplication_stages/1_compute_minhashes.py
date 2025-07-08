import argparse
import os

from nemo_curator import MinHash
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
    compute_minhashes_log_dir = args.compute_minhashes_log_dir
    compute_minhashes_dir = args.compute_minhashes_dir

    # Set up GPU-based Dask client
    client = get_client(scheduler_file=os.environ.get("SCHEDULER_FILE"))
    client.run(pre_imports)
    print(f"Dask client object: {client}")

    minhasher = MinHash(
        seed=42,
        num_hashes=260,
        char_ngrams=24,
        use_64bit_hash=False,
        logger=compute_minhashes_log_dir,
        id_field="id",
        text_field="text",
    )

    # If neither is set, set the default blocksize to 1GB
    if json_blocksize is None and json_files_per_partition is None:
        json_blocksize = "1gb"

    dataset = DocumentDataset.read_json(
        exact_dedup_dir, backend="cudf", blocksize=json_blocksize, files_per_partition=json_files_per_partition
    )

    result = minhasher(dataset).df
    write_path = os.path.join(compute_minhashes_dir, os.path.basename(exact_dedup_dir), "minhashes.parquet")
    result.to_parquet(write_path, write_index=False)
    print(f"Minhashes saved to {write_path}")

    client.close()


def attach_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    arg_helper = ArgumentHelper(parser).add_distributed_args()  # noqa: F841

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
        "--compute-minhashes-log-dir",
        type=str,
        help="Path to the compute minhashes log directory.",
        required=True,
    )
    parser.add_argument(
        "--compute-minhashes-dir",
        type=str,
        help="Path to the computed minhashes data directory.",
        required=True,
    )

    return parser


if __name__ == "__main__":
    main(attach_args(argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)).parse_args())
