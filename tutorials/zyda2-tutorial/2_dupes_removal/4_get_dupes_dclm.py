import json
import logging
import os

import dask.dataframe as dd
from dask.distributed import Client, LocalCluster

from nemo_curator.utils.distributed_utils import get_num_workers
from nemo_curator.utils.module_utils import count_digits

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)

DATA_BASE = os.environ.get("DATA_BASE")
RAW_DATA_BASE = os.path.join(DATA_BASE, "processed")
CC_BASE = os.path.join(DATA_BASE, "fuzzy/cc/")

DUPES_BASE = os.path.join(CC_BASE, "dupes")
DUPES_IDS_GROUPED_IN_COLUMNS = os.path.join(
    DUPES_BASE, "dupes_ids_grouped_in_columns.parquet"
)

DCLM_EXPLODED = os.path.join(DUPES_BASE, "dupes_dclm_exploded.parquet")
DUPES_DCLM_TO_REMOVE = os.path.join(DUPES_BASE, "dupes_dclm_to_remove.jsonl")

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

    # Processing DCLM dupes
    grouped_dupes_df = dd.read_parquet(
        DUPES_IDS_GROUPED_IN_COLUMNS, split_row_groups=False
    )
    dclm_df = grouped_dupes_df[grouped_dupes_df["dclm_dupes"] != "[]"]

    def decode_and_explode(partition, column):
        # Decode JSON strings to lists
        partition["id_list"] = partition[column].apply(json.loads)
        # Explode the lists
        return partition.explode("id_list")[["group", "id_list"]]

    meta = {
        "group": int,
        "id_list": str,
    }
    dclm_exploded_df = dclm_df.map_partitions(
        decode_and_explode, "dclm_dupes", meta=meta
    ).reset_index(drop=True)
    dclm_exploded_df = dclm_exploded_df.rename(columns={"id_list": "id"})

    def split_id(df, id_column="id"):
        dx = df[id_column].str.rsplit("-", n=1, expand=True)
        df["doc_id"] = dx[1].astype("str")
        dy = dx[0].str.split("-", n=1, expand=True)
        df["global_dataset_id"] = dy[0].astype("str")
        df["folder"] = dy[1].astype("str")
        return df

    meta = {
        "group": int,
        "id": str,
        "doc_id": str,
        "global_dataset_id": str,
        "folder": str,
    }
    dclm_exploded_df = dclm_exploded_df.map_partitions(split_id, meta=meta)

    def split_doc_id(df):
        df["row"] = df.apply(
            lambda x: int(x["doc_id"][: -dclm_digits[x["folder"]]]), axis=1
        )
        df["partition"] = df.apply(
            lambda x: int(x["doc_id"][-dclm_digits[x["folder"]] :]), axis=1
        )
        return df

    meta = {
        "group": int,
        "id": str,
        "doc_id": str,
        "global_dataset_id": str,
        "folder": str,
        "row": int,
        "partition": int,
    }
    dclm_exploded_df = dclm_exploded_df.map_partitions(split_doc_id, meta=meta)
    dclm_exploded_df.to_parquet(DCLM_EXPLODED, write_index=False, overwrite=True)

    dclm_exploded_df = dd.read_parquet(DCLM_EXPLODED, split_row_groups=False)

    def write_group_to_jsonl(group):
        folder_id, partition = group.name

        zipped = zip(list(group["row"]), list(group["group"]), list(group["id"]))

        zipped = sorted(zipped, key=lambda x: x[0])

        rows, groups, ids = list(zip(*zipped))

        partition_dict = {
            "rows": rows,
            "groups": groups,
            "ids": ids,
        }

        # Writing rows
        file_path = os.path.join(
            DUPES_DCLM_TO_REMOVE, dclm_id2dir[folder_id], f"{partition}.jsonl"
        )
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(json.dumps(partition_dict))

    dclm_exploded_df.groupby(["folder", "partition"]).apply(
        write_group_to_jsonl, meta=()
    ).compute()
