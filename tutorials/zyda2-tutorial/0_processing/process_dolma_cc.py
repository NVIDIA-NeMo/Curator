import logging
import os

from dask.distributed import Client, LocalCluster
from helper import process_data

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)

DATA_BASE = os.environ.get("DATA_BASE")
INPUT_BASE = os.path.join(DATA_BASE, "raw/dolma-v1_7-cc-parquet")
OUTPUT_BASE = os.path.join(DATA_BASE, "processed/dolma-v1_7-cc-parquet")
CPU_WORKERS = os.environ.get("CPU_WORKERS")


if __name__ == "__main__":
    logging.info("Starting Dask cluster")
    cluster = LocalCluster(n_workers=CPU_WORKERS, processes=True, memory_limit="48GB")
    client = Client(cluster)
    logging.info(client)

    logging.info(f"Processing Dolma-CC")
    process_data(input_folder=INPUT_BASE, output_folder=OUTPUT_BASE, prefix="dolma-cc")
    logging.info("Done!")

    client.cluster.close()
    client.shutdown()
