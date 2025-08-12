import json
import logging
import os

import dask.dataframe as dd
import pandas as pd
from dask.distributed import Client, LocalCluster

from nemo_curator.utils.distributed_utils import get_num_workers

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
DATA_BASE = os.environ.get("DATA_BASE")
CC_BASE = os.path.join(DATA_BASE, "fuzzy/cc/")
CC_FOLDER = os.path.join(CC_BASE, "connected_components.parquet")
CC_CONVERTED_FOLDER = os.path.join(CC_BASE, "connected_components_converted.parquet")
ID_MAPPING = os.path.join(DATA_BASE, "dataset_id_mapping.json")
CPU_WORKERS = os.environ.get("CPU_WORKERS")


if __name__ == "__main__":
    cluster = LocalCluster(n_workers=CPU_WORKERS, processes=True)
    client = Client(cluster)
    logger.info(f"Number of dask workers: {get_num_workers(client)}")

    cc_df = dd.read_parquet(CC_FOLDER, split_row_groups=False)

    with open(ID_MAPPING) as f:
        dataset_id_mapping = json.loads(f.read())

    global_dataset_id_mapping = {}
    for key, val in dataset_id_mapping.items():
        if "dclm" in val:
            global_dataset_id_mapping[key] = "dclm"
        elif "fwe2" in val:
            global_dataset_id_mapping[key] = "fwe2"
        elif "zyda" in val:
            global_dataset_id_mapping[key] = "zyda"
        elif "dolma-cc" in val:
            global_dataset_id_mapping[key] = "dolma-cc"
        else:
            print(f"Unknown value {val} for key {key}")

    def convert_cc_ids(
        cc_df: pd.DataFrame,
        dataset_id_mapping: dict[str, str],
        global_dataset_id_mapping: dict[str, str],
        doc_id_len: int = 10,
    ) -> pd.DataFrame:
        cc_df["global_dataset_id"] = cc_df.dataset_id.astype(str).replace(
            global_dataset_id_mapping,
        )
        cc_df["dataset_id"] = cc_df.dataset_id.astype(str).replace(dataset_id_mapping)
        cc_df["doc_id"] = cc_df["doc_id"].astype(str).str.pad(width=doc_id_len, side="left", fillchar="0")
        cc_df["original_id"] = cc_df.dataset_id + "-" + cc_df.doc_id
        return cc_df[["global_dataset_id", "dataset_id", "original_id", "group"]]

    cc_df_converted = convert_cc_ids(
        cc_df,
        dataset_id_mapping,
        global_dataset_id_mapping,
    )
    cc_df_converted.to_parquet(CC_CONVERTED_FOLDER, overwrite=True, write_index=False)
    logger.info("Done converting!")
