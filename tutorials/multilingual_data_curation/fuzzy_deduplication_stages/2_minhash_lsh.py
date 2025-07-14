import argparse
import os

import cudf
import dask_cudf
import numpy as np

from nemo_curator import LSH
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.fuzzy_dedup_utils.id_mapping import convert_str_id_to_int
from nemo_curator.utils.script_utils import ArgumentHelper


def pre_imports() -> None:
    import cudf  # noqa: F401


def main(args: argparse.Namespace) -> None:
    #  Parameters
    compute_minhashes_dir = args.compute_minhashes_dir
    minhash_field = "_minhash_signature"
    minhash_lsh_cache_dir = args.minhash_lsh_cache_dir
    minhash_lsh_log_dir = args.minhash_lsh_log_dir

    # Set up GPU-based Dask client
    client = get_client(scheduler_file=os.environ.get("SCHEDULER_FILE"))
    client.run(pre_imports)
    print(f"Dask client object: {client}")

    df = dask_cudf.read_parquet(compute_minhashes_dir, blocksize="2GB", aggregate_files=True)
    df = df[~df["id"].isna()]
    df = df.map_partitions(
        convert_str_id_to_int,
        id_column="id",
        meta=cudf.DataFrame({minhash_field: [[1, 2, 3]], "doc_id": [1], "dataset_id": np.uint32(1)}),
    )

    lsh = LSH(
        cache_dir=minhash_lsh_cache_dir,
        num_hashes=260,
        num_buckets=20,
        buckets_per_shuffle=5,  # set to a smaller value if encountering OOMs during LSH
        id_fields=["dataset_id", "doc_id"],
        minhash_field=minhash_field,
        false_positive_check=False,
        logger=minhash_lsh_log_dir,
    )

    _ = lsh(DocumentDataset(df))
    print(f"LSH results saved to {minhash_lsh_cache_dir}/_buckets.parquet")

    client.close()


def attach_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    arg_helper = ArgumentHelper(parser).add_distributed_args()  # noqa: F841

    parser.add_argument(
        "--compute-minhashes-dir",
        type=str,
        help="Path to the computed minhashes directory.",
        required=True,
    )
    parser.add_argument(
        "--minhash-lsh-cache-dir",
        type=str,
        help="Path to the minhash LSH cache directory.",
        required=True,
    )
    parser.add_argument(
        "--minhash-lsh-log-dir",
        type=str,
        help="Path to the minhash LSH log directory.",
        required=True,
    )

    return parser


if __name__ == "__main__":
    main(attach_args(argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)).parse_args())
