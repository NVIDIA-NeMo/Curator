import argparse
import os

from nemo_curator import ConnectedComponents
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.script_utils import ArgumentHelper


def main(args: argparse.Namespace) -> None:
    #  Parameters
    minhash_lsh_cache_dir = args.minhash_lsh_cache_dir
    connected_components_log_dir = args.connected_components_log_dir

    # Set up GPU-based Dask client
    client = get_client(scheduler_file=os.environ.get("SCHEDULER_FILE"))
    print(f"Dask client object: {client}")

    components_stage = ConnectedComponents(
        cache_dir=minhash_lsh_cache_dir,
        jaccard_pairs_path=os.path.join(minhash_lsh_cache_dir, "_edges.parquet"),
        id_column="id",
        jaccard_threshold=0.8,
        logger=connected_components_log_dir,
    )
    components_stage.cc_workflow(output_path=minhash_lsh_cache_dir)
    print(f"Connected components results saved to {minhash_lsh_cache_dir}/connected_components.parquet")

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
        "--connected-components-log-dir",
        type=str,
        help="Path to the connected components log directory.",
        required=True,
    )

    return parser


if __name__ == "__main__":
    main(attach_args(argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)).parse_args())
