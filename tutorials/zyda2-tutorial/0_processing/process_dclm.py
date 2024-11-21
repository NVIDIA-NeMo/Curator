import logging
import os

from dask.distributed import Client, LocalCluster
from helper import process_data

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)

DATA_BASE = os.environ.get("DATA_BASE")
INPUT_BASE = os.path.join(DATA_BASE, "raw/dclm-baseline-1.0-parquet/filtered")
OUTPUT_BASE = os.path.join(DATA_BASE, "processed/dclm-baseline-1.0-parquet")
CPU_WORKERS = os.environ.get("CPU_WORKERS")


if __name__ == "__main__":
    logging.info("Starting Dask cluster")
    cluster = LocalCluster(n_workers=CPU_WORKERS, processes=True, memory_limit="48GB")
    client = Client(cluster)
    logging.info(client)

    components = [
        "global-shard_01_of_10",
        "global-shard_02_of_10",
        "global-shard_03_of_10",
        "global-shard_04_of_10",
        "global-shard_05_of_10",
        "global-shard_06_of_10",
        "global-shard_07_of_10",
        "global-shard_08_of_10",
        "global-shard_09_of_10",
        "global-shard_10_of_10",
    ]

    for i, component in enumerate(components, start=1):
        input_path = os.path.join(INPUT_BASE, component)
        if not os.path.exists(input_path):
            continue
        output_path = os.path.join(OUTPUT_BASE, component)
        logging.info(f"Processing {component}")
        process_data(
            input_folder=input_path,
            output_folder=output_path,
            prefix=f"dclm-gs{i}",
        )
        logging.info("Done!")

    client.cluster.close()
    client.shutdown()
