import logging
import os
import time

import dask_cudf

from nemo_curator import MinHash
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client, get_num_workers
from nemo_curator.utils.file_utils import get_all_files_paths_under

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)


def read_folder(input_folder, columns=["nemo_id", "text"]):
    data_paths = get_all_files_paths_under(input_folder)
    data_paths = [f for f in data_paths if f.endswith(".parquet")]
    data_paths.sort()
    logging.info(f"Number of files being read: {len(data_paths)}")
    text_ddf = dask_cudf.read_parquet(
        data_paths,
        columns=columns,
    )
    return text_ddf


DATA_BASE = os.environ.get("DATA_BASE")
SCHEDULER_FILE = os.environ.get("SCHEDULER_FILE")


if __name__ == "__main__":
    client = get_client(scheduler_file=SCHEDULER_FILE)
    logging.info(f"Number of dask workers: {get_num_workers(client)}")

    minhash_base_output_path = os.path.join(DATA_BASE, "fuzzy/minhash")
    minhash_output_dir = os.path.join(minhash_base_output_path, "data")

    # Relevant parameters
    minhash_id_field = "nemo_id"
    minhash_text_field = "text"
    seed = 10
    minhash_length = 128
    char_ngram = 25
    use_64bit_hash = False

    # Reading all the data
    text_ddf = read_folder(
        input_folder=os.path.join(DATA_BASE, "processed"),
        columns=[minhash_id_field, minhash_text_field],
    )

    # Computing minhashes
    t0 = time.time()
    minhasher = MinHash(
        seed=seed,
        num_hashes=minhash_length,
        char_ngrams=char_ngram,
        use_64bit_hash=use_64bit_hash,
        id_field=minhash_id_field,
        text_field=minhash_text_field,
        cache_dir=minhash_output_dir,
    )
    res = minhasher(DocumentDataset(text_ddf)).df
    logging.info(f"Time taken for MinHash: {time.time()-t0:.2f}sec.")
