import argparse
import os

import dask_cudf

from nemo_curator import BucketsToEdges
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.script_utils import ArgumentHelper


def main(args: argparse.Namespace) -> None:
    #  Parameters
    minhash_lsh_cache_dir = args.minhash_lsh_cache_dir
    buckets_to_edges_log_dir = args.buckets_to_edges_log_dir

    # Set up GPU-based Dask client
    client = get_client(scheduler_file=os.environ.get("SCHEDULER_FILE"))
    print(f"Dask client object: {client}")

    buckets_to_edges = BucketsToEdges(
        cache_dir=minhash_lsh_cache_dir,
        id_fields=["dataset_id", "doc_id"],
        str_id_name="id",
        bucket_field="_bucket_id",
        logger=buckets_to_edges_log_dir,
    )

    minhash_lsh_path = os.path.join(minhash_lsh_cache_dir, "_buckets.parquet")
    buckets_df = DocumentDataset(dask_cudf.read_parquet(minhash_lsh_path, split_row_groups=False))

    _ = buckets_to_edges(buckets_df)
    print(f"Buckets to edges results saved to {minhash_lsh_cache_dir}/_edges.parquet")

    client.close()


def attach_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    arg_helper = ArgumentHelper(parser).add_distributed_args()  # noqa: F841

    parser.add_argument(
        "--minhash-lsh-cache-dir",
        type=str,
        help="Path to the minhash LSH cache directory.",
        required=True,
    )
    parser.add_argument(
        "--buckets-to-edges-log-dir",
        type=str,
        help="Path to the buckets to edges log directory.",
        required=True,
    )

    return parser


if __name__ == "__main__":
    main(attach_args(argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)).parse_args())
