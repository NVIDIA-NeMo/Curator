import logging
import os
import time

import dask.dataframe as dd
import pyarrow as pa
from dask.distributed import Client, LocalCluster

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)

DATA_BASE = os.environ.get("DATA_BASE")
CPU_WORKERS = os.environ.get("CPU_WORKERS")


if __name__ == "__main__":
    t0 = time.time()
    cluster = LocalCluster(n_workers=CPU_WORKERS, threads_per_worker=2, processes=True)
    client = Client(cluster)

    logging.info(f"Filtering...")
    INPUT_BASE = os.path.join(DATA_BASE, "processed/fineweb-edu-score-2")
    OUTPUT_BASE = os.path.join(DATA_BASE, "zyda2/fwe3")
    folders = sorted(os.listdir(INPUT_BASE))
    for folder in folders:
        print(f"\nProcessing {folder}")
        ddf = dd.read_parquet(os.path.join(INPUT_BASE, folder))
        ddf_filtered = ddf[ddf["int_score"] >= 3].repartition(partition_size="512M")
        out_folder = os.path.join(OUTPUT_BASE, folder)
        print(f"Saving to {out_folder}")
        ddf_filtered.to_parquet(out_folder, write_index=False, overwrite=True)
    logging.info(f"Done in {time.time() - t0:.2f} sec")
