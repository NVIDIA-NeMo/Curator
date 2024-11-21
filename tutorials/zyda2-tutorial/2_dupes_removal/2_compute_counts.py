import json
import logging
import os
import time

import dask.dataframe as dd
import pandas as pd
from dask.distributed import Client, LocalCluster

from nemo_curator.utils.distributed_utils import get_num_workers

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)

DATA_BASE = os.environ.get("DATA_BASE")
CC_BASE = os.path.join(DATA_BASE, "fuzzy/cc/")
CC_FOLDER = os.path.join(CC_BASE, "connected_components.parquet")
CC_CONVERTED_FOLDER = os.path.join(CC_BASE, "connected_components_converted.parquet")
CC_GROUPED_FOLDER = os.path.join(CC_BASE, "connected_components_grouped.parquet")
CC_GROUPED_COUNTS_FOLDER = os.path.join(
    CC_BASE, "connected_components_grouped_counts.parquet"
)
CPU_WORKERS = os.environ.get("CPU_WORKERS")


if __name__ == "__main__":
    cluster = LocalCluster(n_workers=CPU_WORKERS, processes=True)
    client = Client(cluster)
    logging.info(f"Number of dask workers: {get_num_workers(client)}")

    cc_df_converted = dd.read_parquet(CC_CONVERTED_FOLDER, split_row_groups=False)

    logging.info("Grouping by cluster id")
    t0 = time.time()

    def group_partition(partition):
        sizes = partition.groupby("group").size().reset_index()

        grouped = (
            partition.groupby("group")
            .agg(
                {
                    "global_dataset_id": lambda x: json.dumps(list(x)),
                    "dataset_id": lambda x: json.dumps(list(x)),
                    "original_id": lambda x: json.dumps(list(x)),
                }
            )
            .reset_index()
        )

        result = pd.merge(sizes, grouped, on="group")

        return result[
            ["group", "global_dataset_id", "dataset_id", "original_id", "size"]
        ]

    meta = {
        "group": int,
        "global_dataset_id": str,
        "dataset_id": str,
        "original_id": str,
        "size": int,
    }
    cc_df_grouped = cc_df_converted.map_partitions(group_partition, meta=meta)
    cc_df_grouped.to_parquet(CC_GROUPED_FOLDER, write_index=False, overwrite=True)
    print(f"Done grouping in {time.time() - t0:.2f} sec")

    logging.info("Computing counts")
    t0 = time.time()
    global_dataset_ids = ["dclm", "fwe2", "dolma-cc", "zyda"]

    def count_occurrences_in_partition(partition):
        for id in global_dataset_ids:
            partition[id] = partition["global_dataset_id"].apply(
                lambda x: json.loads(x).count(id)
            )
        return partition

    meta = {
        "group": "int",
        "global_dataset_id": "str",
        "dataset_id": "str",
        "original_id": "str",
        "size": "int",
    }
    for id in global_dataset_ids:
        meta[id] = "int"
    cc_grouped_counts_df = cc_df_grouped.map_partitions(
        count_occurrences_in_partition,
        meta=meta,
    )
    cc_grouped_counts_df.to_parquet(CC_GROUPED_COUNTS_FOLDER, overwrite=True)

    logging.info(f"Done computing counts in {time.time() - t0:.2f} sec")
