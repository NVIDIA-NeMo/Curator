import os

from nemo_curator import AddId
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.file_utils import get_all_files_paths_under


def ensure_directory_exists(filename: str) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)


def process_data(input_folder: str, output_folder: str, prefix: str, partition_size: str = "512MB") -> None:
    raw_files = get_all_files_paths_under(input_folder)
    raw_data = DocumentDataset.read_parquet(raw_files)
    raw_data_rep = DocumentDataset(
        raw_data.df.repartition(partition_size=partition_size),
    )
    add_id = AddId(id_field="nemo_id", id_prefix=prefix)
    data_with_id = add_id(raw_data_rep)
    ensure_directory_exists(output_folder)
    data_with_id.to_parquet(output_folder)
