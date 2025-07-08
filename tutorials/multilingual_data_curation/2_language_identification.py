import argparse
import os

from nemo_curator import ScoreFilter
from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import FastTextLangId
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.script_utils import ArgumentHelper


def main(args: argparse.Namespace) -> None:
    # Parameters
    extracted_data_dir = args.extracted_data_dir
    lang_id_model_path = args.lang_id_model_path
    lang_data_dir = args.lang_data_dir

    # Set up CPU-based Dask client
    client = get_client(scheduler_file=os.environ.get("SCHEDULER_FILE"))
    print(f"Dask client object: {client}")

    dataset = DocumentDataset.read_json(extracted_data_dir)

    # Step 2: Language identification
    lang_id = FastTextLangId(lang_id_model_path)
    language_id_pipeline = ScoreFilter(lang_id, score_field="fasttext_language", score_type="object")
    dataset = language_id_pipeline(dataset)

    # Drop the score and keep the language label
    dataset.df["fasttext_language"] = dataset.df["fasttext_language"].apply(
        lambda score: score[1], meta=("fasttext_language", "object")
    )

    dataset.to_json(lang_data_dir)

    print(f"Saved language data to {lang_data_dir}")
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
        "--lang-id-model-path",
        type=str,
        help="Path to the FastText language identification model",
        required=True,
    )
    parser.add_argument(
        "--lang-data-dir",
        type=str,
        help="Path to the language data directory.",
        required=True,
    )

    return parser


if __name__ == "__main__":
    main(attach_args(argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)).parse_args())
