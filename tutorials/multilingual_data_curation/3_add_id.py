import argparse
import os

from nemo_curator import AddId
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.script_utils import ArgumentHelper


def main(args: argparse.Namespace) -> None:
    # Parameters
    lang_data_dir = args.lang_data_dir
    id_data_dir = args.id_data_dir

    # Set up CPU-based Dask client
    client = get_client(scheduler_file=os.environ.get("SCHEDULER_FILE"))
    print(f"Dask client object: {client}")

    dataset = DocumentDataset.read_json(lang_data_dir, backend="cudf", files_per_partition=20, blocksize=None)

    # Step 3: Add ID column
    add_id = AddId(id_field="id", id_prefix="cc_")
    dataset = add_id(dataset)

    dataset.to_json(id_data_dir)

    print(f"Saved ID data to {id_data_dir}")
    client.close()


def attach_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    arg_helper = ArgumentHelper(parser).add_distributed_args()  # noqa: F841

    parser.add_argument(
        "--lang-data-dir",
        type=str,
        help="Path to the language data directory.",
        required=True,
    )
    parser.add_argument(
        "--id-data-dir",
        type=str,
        help="Path to the ID data directory.",
        required=True,
    )

    return parser


if __name__ == "__main__":
    main(attach_args(argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)).parse_args())
