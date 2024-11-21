import json
import logging
import os

import cudf
import dask_cudf
import tqdm

from nemo_curator.utils.distributed_utils import get_client, get_num_workers

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)


def list_deepest_folders(path):
    deepest_folders = []

    # Walk through the directory structure
    for root, dirs, files in os.walk(
        path, topdown=False
    ):  # `topdown=False` starts from the deepest folder
        if not dirs:  # Keep only folders that don't have subdirectories
            deepest_folders.append(root)

    return deepest_folders


def read_data_subset(dir_paths: list):
    """
    Reads 1 row from each dataset assuming each file has a unique dataset prefix
    """
    dfs = []
    for dir_path in tqdm.tqdm(dir_paths):
        file_name = sorted([x for x in os.listdir(dir_path) if ".parquet" in x])[0]
        file_path = os.path.join(dir_path, file_name)
        x = dask_cudf.read_parquet(file_path).head(1)  # read 1 rows from each file
        dfs.append(x)
    x = cudf.concat(dfs)
    return x


def generate_mapping(x: cudf.DataFrame):
    dx = x.nemo_id.str.rsplit("-", n=1, expand=True)
    x["dataset"] = dx[0]
    x["dataset_id"] = x.dataset.hash_values()
    mapping_df = x[["dataset", "dataset_id"]]
    mapping_df = mapping_df.drop_duplicates()
    mapping_df["dataset_id"] = mapping_df.dataset_id.astype(str)
    dataset_id_mapping = mapping_df.set_index("dataset_id")["dataset"].to_dict()
    return dataset_id_mapping


def convert_cc_ids(cc_df: dask_cudf.DataFrame, doc_id_mapping: dict, pad_width=10):
    cc_df["doc_id"] = (
        cc_df["doc_id"].astype(str).str.pad(width=pad_width, side="left", fillchar="0")
    )
    cc_df["dataset_id"] = cc_df.dataset_id.astype(str).replace(doc_id_mapping)
    cc_df["original_id"] = cc_df.dataset_id + "-" + cc_df.doc_id
    return cc_df[["original_id", "group"]]


DATA_BASE = os.environ.get("DATA_BASE")
SCHEDULER_FILE = os.environ.get("SCHEDULER_FILE")
ID_MAPPING = os.path.join(DATA_BASE, "dataset_id_mapping.json")


if __name__ == "__main__":
    client = get_client(scheduler_file=SCHEDULER_FILE)
    logging.info(f"Number of dask workers: {get_num_workers(client)}")

    all_folders = sorted(list_deepest_folders(os.path.join(DATA_BASE, "processed")))
    df_subset = read_data_subset(all_folders)
    dataset_id_mapping = generate_mapping(df_subset)

    with open(ID_MAPPING, "w") as f:
        f.write(json.dumps(dataset_id_mapping))
