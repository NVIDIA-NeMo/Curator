import json
import logging
import os

import dask.dataframe as dd
import pandas as pd
from dask.distributed import Client, LocalCluster

from nemo_curator.utils.distributed_utils import get_num_workers
from nemo_curator.utils.module_utils import count_digits

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)

DATA_BASE = os.environ.get("DATA_BASE")
RAW_DATA_BASE = os.path.join(DATA_BASE, "processed")
CC_BASE = os.path.join(DATA_BASE, "fuzzy/cc/")
CC_FOLDER = os.path.join(CC_BASE, "connected_components.parquet")
CC_GROUPED_COUNTS_FOLDER = os.path.join(
    CC_BASE, "connected_components_grouped_counts.parquet"
)

DUPES_BASE = os.path.join(CC_BASE, "dupes")
DUPES_IDS_GROUPED_IN_COLUMNS = os.path.join(
    DUPES_BASE, "dupes_ids_grouped_in_columns.parquet"
)

CPU_WORKERS = os.environ.get("CPU_WORKERS")


if __name__ == "__main__":
    cluster = LocalCluster(n_workers=CPU_WORKERS, processes=True)
    client = Client(cluster)
    logging.info(f"Number of dask workers: {get_num_workers(client)}")

    paths = {
        "dclm": os.path.join(RAW_DATA_BASE, "dclm-baseline-1.0-parquet/filtered"),
        "dolma-cc": os.path.join(RAW_DATA_BASE, "dolma-v1_7-cc-parquet"),
        "fwe2": os.path.join(RAW_DATA_BASE, "fineweb-edu-score-2/data"),
        "zyda": os.path.join(RAW_DATA_BASE, "data/zyda_no_starcoder"),
    }

    dclm_id2dir = {
        "gs1": "global-shard_01_of_10",
        "gs2": "global-shard_02_of_10",
        "gs3": "global-shard_03_of_10",
        "gs4": "global-shard_04_of_10",
        "gs5": "global-shard_05_of_10",
        "gs6": "global-shard_06_of_10",
        "gs7": "global-shard_07_of_10",
        "gs8": "global-shard_08_of_10",
        "gs9": "global-shard_09_of_10",
        "gs10": "global-shard_10_of_10",
    }

    dclm_dir2id = {}
    for key, val in dclm_id2dir.items():
        dclm_dir2id[val] = key

    # Counting digits
    dclm_digits = {}
    for dir in sorted(os.listdir(paths["dclm"])):
        files = [
            x for x in os.listdir(os.path.join(paths["dclm"], dir)) if ".parquet" in x
        ]
        dclm_digits[dclm_dir2id[dir]] = count_digits(len(files))

    dolma_digits = count_digits(
        len([x for x in os.listdir(paths["dolma-cc"]) if ".parquet" in x])
    )

    zyda_digits = {}
    for dir in sorted(os.listdir(paths["zyda"])):
        files = [
            x for x in os.listdir(os.path.join(paths["zyda"], dir)) if ".parquet" in x
        ]
        zyda_digits[dir] = count_digits(len(files))

    fwe2_digits = {}
    for dir in sorted(os.listdir(paths["fwe2"])):
        files = [
            x for x in os.listdir(os.path.join(paths["fwe2"], dir)) if ".parquet" in x
        ]
        fwe2_digits[dir] = count_digits(len(files))

    cc_grouped_counts_df = dd.read_parquet(
        CC_GROUPED_COUNTS_FOLDER, split_row_groups=False
    )
    cc_grouped_counts_filtered_df = cc_grouped_counts_df[
        cc_grouped_counts_df["size"] > 1
    ]

    cc_groups_counts_inter_df = cc_grouped_counts_filtered_df[
        cc_grouped_counts_filtered_df["size"] != cc_grouped_counts_filtered_df["dclm"]
    ]
    cc_groups_counts_inter_df = cc_groups_counts_inter_df[
        cc_groups_counts_inter_df["size"] != cc_groups_counts_inter_df["fwe2"]
    ]
    cc_groups_counts_inter_df = cc_groups_counts_inter_df[
        cc_groups_counts_inter_df["size"] != cc_groups_counts_inter_df["dolma-cc"]
    ]
    cc_groups_counts_inter_df = cc_groups_counts_inter_df[
        cc_groups_counts_inter_df["size"] != cc_groups_counts_inter_df["zyda"]
    ]

    def select_dupes(partition):
        # Removes all overlaps with fwe2
        partition_fwe2 = partition[partition["fwe2"] > 0]
        partition_fwe2["dupes_to_remove"] = partition_fwe2["original_id"].apply(
            lambda x: json.dumps([id for id in json.loads(x) if "fwe2" not in id])
        )

        # Removes all overlaps with fwe2 (after fwe2 overlaps are removed)
        partition_dclm = partition[partition["fwe2"] == 0]
        partition_dclm = partition_dclm[partition_dclm["dclm"] > 0]
        partition_dclm["dupes_to_remove"] = partition_dclm["original_id"].apply(
            lambda x: json.dumps([id for id in json.loads(x) if "dclm" not in id])
        )

        # Removes all overlaps with zyda (after dclm, fwe2 overlaps are removed)
        partition_zyda = partition[partition["fwe2"] == 0]
        partition_zyda = partition_zyda[partition_zyda["dclm"] == 0]
        partition_zyda = partition_zyda[partition_zyda["zyda"] > 0]
        partition_zyda["dupes_to_remove"] = partition_zyda["original_id"].apply(
            lambda x: json.dumps([id for id in json.loads(x) if "zyda" not in id])
        )

        return pd.concat([partition_dclm, partition_fwe2, partition_zyda])

    meta = {
        "group": int,
        "global_dataset_id": str,
        "dataset_id": str,
        "original_id": str,
        "size": int,
        "dclm": int,
        "fwe2": int,
        "dolma-cc": int,
        "zyda": int,
        "dupes_to_remove": str,
    }
    dupes_df = cc_groups_counts_inter_df.map_partitions(select_dupes, meta=meta)

    def group_dupes(partition):
        partition["dclm_dupes"] = partition["dupes_to_remove"].apply(
            lambda x: json.dumps([id for id in json.loads(x) if "dclm" in id])
        )
        partition["zyda_dupes"] = partition["dupes_to_remove"].apply(
            lambda x: json.dumps([id for id in json.loads(x) if "zyda" in id])
        )
        partition["dolma_dupes"] = partition["dupes_to_remove"].apply(
            lambda x: json.dumps([id for id in json.loads(x) if "dolma" in id])
        )

        return partition[
            [
                "group",
                "size",
                "dclm",
                "fwe2",
                "dolma-cc",
                "zyda",
                "dclm_dupes",
                "zyda_dupes",
                "dolma_dupes",
            ]
        ]

    meta = {
        "group": int,
        "size": int,
        "dclm": int,
        "fwe2": int,
        "dolma-cc": int,
        "zyda": int,
        "dclm_dupes": str,
        "zyda_dupes": str,
        "dolma_dupes": str,
    }

    grouped_dupes_df = dupes_df.map_partitions(group_dupes, meta=meta)
    grouped_dupes_df.to_parquet(
        DUPES_IDS_GROUPED_IN_COLUMNS, write_index=False, overwrite=True
    )
