import argparse
import os

from nemo_curator.download import (
    JusTextExtractor,
    ResiliparseExtractor,
    TrafilaturaExtractor,
    download_common_crawl,
)
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.script_utils import ArgumentHelper


def main(args: argparse.Namespace) -> None:
    # Extraction parameters
    extracted_data_dir = args.extracted_data_dir
    snapshot = args.snapshot
    extractor = args.extractor

    # Set up CPU-based Dask client
    client = get_client(scheduler_file=os.environ.get("SCHEDULER_FILE"))
    print(f"Dask client object: {client}")

    # Set up the extractor
    if extractor.lower() == "resiliparse":
        extraction_algorithm = ResiliparseExtractor()
    elif extractor.lower() == "trafilatura":
        extraction_algorithm = TrafilaturaExtractor()
    elif extractor.lower() == "justext":
        extraction_algorithm = JusTextExtractor()
    else:
        msg = f"Invalid extractor: {extractor}"
        raise ValueError(msg)

    # Step 1: Download and extract Common Crawl data
    common_crawl_dataset = download_common_crawl(
        output_path=extracted_data_dir,
        start_snapshot=snapshot,
        end_snapshot=snapshot,
        output_type="jsonl",
        algorithm=extraction_algorithm,
    )
    common_crawl_dataset.to_json(output_path=extracted_data_dir, write_to_filename=True)

    print(f"Downloaded Common Crawl data to {extracted_data_dir}")
    client.close()


def attach_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    arg_helper = ArgumentHelper(parser).add_distributed_args()  # noqa: F841

    parser.add_argument(
        "--extracted-data-dir",
        type=str,
        help="Path to the extracted data directory.",
        required=True,
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        default="2025-18",  # April 2025
        help="Snapshot to download.",
    )
    parser.add_argument(
        "--extractor",
        type=str,
        default="justext",
        help="Extractor to use. Can be jusText, Resiliparse, or Trafilatura.",
    )

    return parser


if __name__ == "__main__":
    main(attach_args(argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)).parse_args())
