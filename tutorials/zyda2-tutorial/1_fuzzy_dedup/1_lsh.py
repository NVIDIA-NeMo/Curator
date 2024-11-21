import logging
import os
import time

import cudf
import dask_cudf
import numpy as np

from nemo_curator import LSH
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client, get_num_workers
from nemo_curator.utils.fuzzy_dedup_utils.id_mapping import convert_str_id_to_int

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)


DATA_BASE = os.environ.get("DATA_BASE")
SCHEDULER_FILE = os.environ.get("SCHEDULER_FILE")


if __name__ == "__main__":
    client = get_client(scheduler_file=SCHEDULER_FILE)
    logging.info(f"Number of dask workers: {get_num_workers(client)}")

    minhash_base_output_path = os.path.join(DATA_BASE, "fuzzy/minhash")
    minhash_output_dir = os.path.join(minhash_base_output_path, "data")

    # Input
    lsh_input_data_path = minhash_output_dir

    # Output
    lsh_base_output_path = os.path.join(DATA_BASE, "fuzzy/lsh")
    lsh_output_dir = os.path.join(lsh_base_output_path, "data")

    # Relevant parameters
    lsh_id_field = "nemo_id"
    minhash_field = "_minhash_signature"
    minhash_length = 128
    num_bands = 8
    buckets_per_shuffle = 8

    t0 = time.time()

    # Load MinHash output
    logging.info("Converting ids")
    df = dask_cudf.read_parquet(lsh_input_data_path, backend="cudf")
    df = df.map_partitions(
        convert_str_id_to_int,
        id_column=lsh_id_field,
        meta=cudf.DataFrame(
            {minhash_field: [[1, 2, 3]], "doc_id": [1], "dataset_id": np.uint32(1)}
        ),
    )
    # Run LSH()
    lsh = LSH(
        cache_dir=lsh_output_dir,
        num_hashes=minhash_length,
        num_buckets=num_bands,
        buckets_per_shuffle=buckets_per_shuffle,
        id_fields=["dataset_id", "doc_id"],
        minhash_field=minhash_field,
    )
    res = lsh(DocumentDataset(df))

    t1 = time.time()
    logging.info(f"Time taken for LSH: {time.time() - t0} s")
