import logging
import os

from dask.distributed import Client, LocalCluster
from helper import process_data

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
DATA_BASE = os.environ.get("DATA_BASE")
INPUT_BASE = os.path.join(DATA_BASE, "raw/data/zyda_no_starcoder")
OUTPUT_BASE = os.path.join(DATA_BASE, "processed/zyda-parquet")
CPU_WORKERS = os.environ.get("CPU_WORKERS")


if __name__ == "__main__":
    logger.info("Starting Dask cluster")
    cluster = LocalCluster(n_workers=CPU_WORKERS, processes=True, memory_limit="48GB")
    client = Client(cluster)
    logger.info(client)

    components = [
        "zyda_arxiv",
        "zyda_peS2o",
        "zyda_pile-uncopyrighted",
        "zyda_slimpajama",
        "zyda_c4-en",
        "zyda_refinedweb",
    ]

    for component in components:
        input_path = os.path.join(INPUT_BASE, component)
        if not os.path.exists(input_path):
            continue
        output_path = os.path.join(OUTPUT_BASE, component)
        logger.info(f"Processing {component}")
        process_data(
            input_folder=input_path,
            output_folder=output_path,
            prefix=component,
        )
        logger.info("Done!")

    client.cluster.close()
    client.shutdown()
