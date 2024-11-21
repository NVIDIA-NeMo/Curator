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

ZYDA_EXPLODED = os.path.join(DUPES_BASE, "dupes_zyda_exploded.parquet")
DUPES_ZYDA_TO_REMOVE = os.path.join(DUPES_BASE, "dupes_zyda_to_remove.jsonl")

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

    # Counting digits
    zyda_digits = {}
    for dir in sorted(os.listdir(paths["zyda"])):
        files = [
            x for x in os.listdir(os.path.join(paths["zyda"], dir)) if ".parquet" in x
        ]
        zyda_digits[dir] = count_digits(len(files))

    grouped_dupes_df = dd.read_parquet(
        DUPES_IDS_GROUPED_IN_COLUMNS, split_row_groups=False
    )
    zyda_df = grouped_dupes_df[grouped_dupes_df["zyda_dupes"] != "[]"][
        ["group", "size", "dclm", "fwe2", "zyda", "dolma-cc", "zyda_dupes"]
    ]

    def decode_and_explode(partition, column):
        partition["id_list"] = partition[column].apply(json.loads)
        return partition.explode("id_list")[["group", "id_list"]]

    meta = {
        "group": int,
        "id_list": str,
    }
    zyda_exploded_df = zyda_df.map_partitions(
        decode_and_explode, "zyda_dupes", meta=meta
    ).reset_index(drop=True)
    zyda_exploded_df = zyda_exploded_df.rename(columns={"id_list": "id"})

    def split_id(df, id_column="id"):
        dx = df[id_column].str.rsplit("-", n=1, expand=True)
        df["doc_id"] = dx[1].astype("str")
        df["folder"] = dx[0].astype("str")
        return df

    meta = {
        "group": int,
        "id": str,
        "doc_id": str,
        "folder": str,
    }
    zyda_exploded_df = zyda_exploded_df.map_partitions(split_id, meta=meta)

    def split_doc_id(df):
        df["row"] = df.apply(
            lambda x: int(x["doc_id"][: -zyda_digits[x["folder"]]]), axis=1
        )
        df["partition"] = df.apply(
            lambda x: int(x["doc_id"][-zyda_digits[x["folder"]] :]), axis=1
        )
        return df

    meta = {
        "group": int,
        "id": str,
        "doc_id": str,
        "folder": str,
        "row": int,
        "partition": int,
    }
    zyda_exploded_df = zyda_exploded_df.map_partitions(split_doc_id, meta=meta)
    zyda_exploded_df.to_parquet(ZYDA_EXPLODED, write_index=False, overwrite=True)

    zyda_exploded_df = dd.read_parquet(ZYDA_EXPLODED, split_row_groups=False)

    def write_group_to_jsonl(group):
        folder_name, partition = group.name

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
            DUPES_ZYDA_TO_REMOVE, folder_name, f"{partition}.jsonl"
        )
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(json.dumps(partition_dict))

    zyda_exploded_df.groupby(["folder", "partition"]).apply(
        write_group_to_jsonl, meta=()
    ).compute()
