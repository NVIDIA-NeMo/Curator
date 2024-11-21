import ctypes
import gc
import logging
import os
from pathlib import Path

from dask.distributed import Client, LocalCluster
from helper import process_data

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)

DATA_BASE = os.environ.get("DATA_BASE")
INPUT_BASE = os.path.join(DATA_BASE, "raw/fineweb-edu-score-2/data")
OUTPUT_BASE = os.path.join(DATA_BASE, "processed/fineweb-edu-score-2")
CPU_WORKERS = os.environ.get("CPU_WORKERS")


def trim_memory() -> int:
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)


def get_folder_size(folder_path):
    return sum(
        file.stat().st_size for file in Path(folder_path).rglob("*") if file.is_file()
    )


def sort_folders_by_size(parent_directory):
    folders = [
        f
        for f in os.listdir(parent_directory)
        if os.path.isdir(os.path.join(parent_directory, f))
    ]
    folder_sizes = [
        (folder, get_folder_size(os.path.join(parent_directory, folder)))
        for folder in folders
    ]
    return sorted(folder_sizes, key=lambda x: x[1])


def bytes_to_human_readable(size_in_bytes):
    suffixes = ["B", "KB", "MB", "GB", "TB", "PB"]
    suffix_index = 0
    size = float(size_in_bytes)
    while size >= 1024 and suffix_index < len(suffixes) - 1:
        size /= 1024.0
        suffix_index += 1
    return f"{size:.2f} {suffixes[suffix_index]}"


if __name__ == "__main__":
    logging.info("Starting Dask cluster")
    cluster = LocalCluster(n_workers=CPU_WORKERS, processes=True, memory_limit="240GB")
    client = Client(cluster)
    logging.info(client)

    components_with_sizes = sort_folders_by_size(INPUT_BASE)

    for component, component_size in components_with_sizes:
        input_path = os.path.join(INPUT_BASE, component)
        if not os.path.exists(input_path) or not os.path.isdir(input_path):
            continue
        output_path = os.path.join(OUTPUT_BASE, component)
        logging.info(
            f"Processing {component}, size = {bytes_to_human_readable(component_size)}"
        )
        process_data(
            input_folder=input_path,
            output_folder=output_path,
            prefix=f"fwe2-{component}",
        )
        logging.info("Trimming memory")
        gc.collect()
        client.run(trim_memory)
        logging.info("Done!")

    client.cluster.close()
    client.shutdown()
