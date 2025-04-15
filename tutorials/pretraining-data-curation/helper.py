import gzip
import json
import os

import cudf
import dask.bag as db
import dask.dataframe as dd


def convert_single_file(input_output_paths: tuple[str, str]) -> None:
    input_path, output_path = input_output_paths

    with gzip.open(input_path, "rt", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            try:
                # Parse each line as a separate JSON object
                item = json.loads(line)
                # Write the JSON object to the .jsonl file
                json.dump(item, f_out)
                f_out.write("\n")
            except json.JSONDecodeError as e:  # noqa: PERF203
                print(f"Error decoding JSON in file {input_path}: {e}")
                continue


def convert_json_gz_to_jsonl(input_dir: str, output_dir: str, partition_size: int = 2) -> None:
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List all .json.gz files in the input directory
    file_paths = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".json.gz"):
            input_path = os.path.join(input_dir, filename)
            output_filename = os.path.splitext(os.path.splitext(filename)[0])[0] + ".jsonl"
            output_path = os.path.join(output_dir, output_filename)
            file_paths.append((input_path, output_path))

    # Create a Dask bag from the file paths and apply the function in parallel
    bag = db.from_sequence(file_paths, partition_size=partition_size)
    bag.map(convert_single_file).compute()


def convert_str_id_to_int(df: cudf.DataFrame, id_column: str = "id") -> cudf.DataFrame:
    """
    Converts the legacy id format "dataset_name-0000034"
    type of ID into 2 int based ID's
    """
    dx = df[id_column].str.rsplit("-", n=1, expand=True)
    df["doc_id"] = dx[1].astype("int64").values
    df["dataset_id"] = dx[0].hash_values()
    return df


def get_dataframe_complement(original_df: dd.DataFrame, filtered_df: dd.DataFrame) -> dd.DataFrame:
    def partition_complement(part_original_df: cudf.DataFrame, partition_info: dict | None = None) -> cudf.DataFrame:
        if not partition_info:
            return part_original_df
        part_filtered_df = filtered_df.get_partition(partition_info["number"])
        complement_mask = ~part_original_df.index.isin(part_filtered_df.index.persist())
        return part_original_df[complement_mask]

    return original_df.map_partitions(partition_complement)
