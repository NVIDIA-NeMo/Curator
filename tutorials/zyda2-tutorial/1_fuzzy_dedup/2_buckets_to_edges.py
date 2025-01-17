import logging
import os
import time

import dask_cudf

from nemo_curator import BucketsToEdges
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client, get_num_workers

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)


DATA_BASE = os.environ.get("DATA_BASE")
SCHEDULER_FILE = os.environ.get("SCHEDULER_FILE")


if __name__ == "__main__":
    client = get_client(scheduler_file=SCHEDULER_FILE)
    logging.info(f"Number of dask workers: {get_num_workers(client)}")

    # Input
    lsh_base_output_path = os.path.join(DATA_BASE, "fuzzy/lsh")
    lsh_buckets_output_path = os.path.join(
        lsh_base_output_path, "data/_buckets.parquet"
    )

    # Output
    buckets_to_edges_out = os.path.join(DATA_BASE, "fuzzy/buckets_to_edges/data")

    t0 = time.time()

    ddf_bk = dask_cudf.read_parquet(
        lsh_buckets_output_path,
        split_row_groups=False,
    )

    buckets_to_edges = BucketsToEdges(
        cache_dir=buckets_to_edges_out,
        id_fields=["dataset_id", "doc_id"],
    )

    ddf_b2e = buckets_to_edges(DocumentDataset(ddf_bk))

    logging.info(f"Time taken for Buckets to Edges: {time.time() - t0} s")
