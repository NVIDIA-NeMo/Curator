import argparse
import json
import logging
import os
import time

import dask.dataframe as dd
import pandas as pd
from dask.distributed import Client, LocalCluster

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NeMo quality classifier.")
    parser.add_argument(
        "--dupes-path",
        type=str,
        required=True,
        help="Path to the folder with dupes indices",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the folder with input dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to where write the result",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=64,
        help="Number of CPU Dask workers",
    )
    args = parser.parse_args()

    t0 = time.time()
    cluster = LocalCluster(
        n_workers=args.n_workers,
        threads_per_worker=2,
        processes=True,
    )
    client = Client(cluster)
    logger.info(f"Dask client: {client}")
    logger.info(f"Dashboard link: {client.dashboard_link}")

    dupes_files = sorted([x for x in os.listdir(args.dupes_path) if ".jsonl" in x])
    ind_2_fullpath = {}
    for file in dupes_files:
        ind = int(file.split(".")[0])
        ind_2_fullpath[ind] = os.path.join(args.dupes_path, file)
    logger.info(f"Found {len(dupes_files)} files with dupes")

    ddf = dd.read_parquet(args.input, split_row_groups=False)

    def drop_dupes(partition: pd.DataFrame, partition_info: dict | None = None) -> pd.DataFrame:
        ind = partition_info["number"]

        dupes_file_path = ind_2_fullpath.get(ind)
        if not dupes_file_path:
            return partition

        with open(dupes_file_path) as f:
            dupe_inds = json.loads(f.read())["rows"]
            return partition.drop(index=dupe_inds)

    meta = ddf.dtypes.to_dict()
    ddf_deduped = ddf.map_partitions(drop_dupes, meta=meta)
    logger.info("Removing dupes...")
    ddf_deduped.to_parquet(args.output, write_index=False, overwrite=True)
    logger.info(f"Done in {time.time() - t0:.2f}sec")

    client.cluster.close()
    client.shutdown()
