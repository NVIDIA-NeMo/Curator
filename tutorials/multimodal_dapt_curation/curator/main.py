# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import base64
import io
import json
import os

os.environ["CUDF_SPILL"] = "on"
import tarfile
from pathlib import Path
from typing import Any

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from utils import (
    TextLineCountFilter,
    clean_and_unify,
    exact_dedupe,
    filter_text,
    fuzzy_dedupe,
    redact_pii,
    rm_dir,
    semantic_dedupe,
)

from nemo_curator import ClusteringModel, ScoreFilter, SemanticClusterLevelDedup, Sequential
from nemo_curator.datasets import DocumentDataset, ImageTextPairDataset
from nemo_curator.image.embedders import TimmImageEmbedder
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.script_utils import ArgumentHelper

SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR_PATH, "data")
IMG_DIR = os.path.join(SCRIPT_DIR_PATH, "image_dataset")
CONFIG_DIR = os.path.join(SCRIPT_DIR_PATH, "configs")


def process_textual_data(textual_data: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Process a list of textual data entries and extract structured information
    into a pandas DataFrame.

    Args:
        textual_data (List[Dict[str, Any]]): A list of dictionaries containing
            metadata and content for each text document.

    Returns:
        pd.DataFrame: A DataFrame containing processed text, metadata, and derived info.
    """
    processed_records = []

    for text_data in textual_data:
        encodings = "None"  # Placeholder, can be updated if encodings are later added
        text = text_data["metadata"]["content"]
        category = text_data["document_type"]
        line_count = len(text.split("\n"))
        source_metadata = text_data["metadata"]["source_metadata"]
        doc_id = source_metadata["source_id"]
        file_path = source_metadata["source_name"]
        filename = Path(file_path).stem
        file_extension = Path(file_path).suffix

        processed_records.append(
            [
                encodings,
                text,
                doc_id,
                file_extension,
                filename,
                category,
                line_count,
                file_path,
            ]
        )

    return pd.DataFrame(
        processed_records,
        columns=[
            "encodings",
            "text",
            "id",
            "file_extension",
            "file_name",
            "category",
            "line_count",
            "path",
        ],
    )


def process_structured_data(structured_data: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Process a list of structured data entries (e.g., tables) and extract relevant
    information into a structured pandas DataFrame.

    Args:
        structured_data (List[Dict[str, Any]]): A list of dictionaries containing
            metadata and table content for each structured document.

    Returns:
        pd.DataFrame: A DataFrame with extracted metadata and table-related information.
    """
    processed_records = []

    for struct_data in structured_data:
        encodings = struct_data["metadata"]["content"]
        text = struct_data["metadata"]["table_metadata"]["table_content"]
        category = struct_data["document_type"]
        line_count = len(text.split("\n"))
        source_metadata = struct_data["metadata"]["source_metadata"]
        doc_id = source_metadata["source_id"]
        file_path = source_metadata["source_name"]
        filename = Path(file_path).stem
        file_extension = Path(file_path).suffix

        processed_records.append(
            [
                encodings,
                text,
                doc_id,
                file_extension,
                filename,
                category,
                line_count,
                file_path,
            ]
        )

    return pd.DataFrame(
        processed_records,
        columns=[
            "encodings",
            "text",
            "id",
            "file_extension",
            "file_name",
            "category",
            "line_count",
            "path",
        ],
    )


def process_data(data_type_map: dict[str, list]) -> tuple[DocumentDataset, DocumentDataset]:
    """
    Process different types of document data (text, image, structured) and convert them into
    DocumentDataset-wrapped Dask DataFrames.

    Args:
        data_type_map (Dict[str, list]): A dictionary with keys "text", "image", and "structured",
            each mapping to a list of document records.

    Returns:
        Tuple[DocumentDataset, DocumentDataset, DocumentDataset]: A tuple containing
            DocumentDatasets for text, image, and structured data respectively.
    """
    text_df = process_textual_data(data_type_map["text"])
    struct_df = process_structured_data(data_type_map["structured"])

    text_ddf = DocumentDataset(dd.from_pandas(text_df, npartitions=2))
    struct_ddf = DocumentDataset(dd.from_pandas(struct_df, npartitions=2))

    return text_ddf, struct_ddf


def run_text_curation_pipeline(args: dict, text_ddf: DocumentDataset, struct_ddf: DocumentDataset) -> None:  # noqa: PLR0915
    """
    Run the curation pipeline on the Wiki+Arxiv+Github datasets.

    Args:
        args (Any): Command-line arguments.
        jsonl_dir (str): Directory path where the JSONL files are stored.
    """

    # Define data curation steps for text and pdf files
    curation_steps_text = Sequential(
        [
            clean_and_unify,
            ScoreFilter(TextLineCountFilter(), text_field="file_type_count", score_type=bool),
            filter_text,
            exact_dedupe,
            redact_pii,
        ]
    )

    # create a field combining fields file type and line count
    text_ddf.df["file_type_count"] = text_ddf.df["category"] + " : " + text_ddf.df["line_count"].astype(str)

    struct_ddf.df["file_type_count"] = struct_ddf.df["category"] + " : " + struct_ddf.df["line_count"].astype(str)

    print("Executing the curation pipeline...")
    dataset_text = curation_steps_text(text_ddf)
    dataset_struct = curation_steps_text(struct_ddf)

    datasets = [dataset_text, dataset_struct]
    counts = {"person_count": 0, "email_count": 0}

    for dataset in datasets:
        counts["person_count"] += dataset.df["text"].str.count(r"\bPERSON\b").sum().compute()
        counts["email_count"] += dataset.df["text"].str.count(r"\bEMAIL_ADDRESS\b").sum().compute()

    person_count, email_count = counts["person_count"], counts["email_count"]

    print(f"Original dataset length for text: {len(text_ddf.df)}")
    print(f"Original dataset length for tables and charts: {len(struct_ddf.df)}")

    print(f"After preprocessing text data: {len(dataset_text.df)}")
    print(f"After preprocessing tables and charts: {len(dataset_struct.df)}")

    print(f"Redacted names and email address for text, tables and charts: {person_count, email_count}")

    if args.device == "gpu":
        print("Executing the semantic dedupe pipeline...")
        gpu_dataset_text = DocumentDataset(dataset_text.df.to_backend("cudf"))
        gpu_dataset_struct = DocumentDataset(dataset_struct.df.to_backend("cudf"))

        text_sem_dedupe_config_yaml_path = os.path.join(CONFIG_DIR, "text_semantic_dedupe_config.yaml")
        struct_sem_dedupe_config_yaml_path = os.path.join(CONFIG_DIR, "struct_semantic_dedupe_config.yaml")

        cache_dir_txt = os.path.join(SCRIPT_DIR_PATH, "cache", "semdedup_cache", "text")
        cache_dir_struct = os.path.join(SCRIPT_DIR_PATH, "cache", "semdedup_cache", "struct")

        rm_dir(cache_dir_txt)
        rm_dir(cache_dir_struct)

        semantic_dataset_text = semantic_dedupe(
            dataset=gpu_dataset_text,
            sem_dedupe_config_yaml_path=text_sem_dedupe_config_yaml_path,
        )
        semantic_dataset_struct = semantic_dedupe(
            dataset=gpu_dataset_struct,
            sem_dedupe_config_yaml_path=struct_sem_dedupe_config_yaml_path,
        )

        print(f"After semantic dedupe for text: {len(semantic_dataset_text.df)}")
        print(f"After semantic dedupe for tables and charts: {len(semantic_dataset_struct.df)}")

        print("Executing the fuzzy dedupe pipeline...")
        cache_dir_txt = os.path.join(SCRIPT_DIR_PATH, "cache", "fuzzy_dedupe", "text")
        cache_dir_struct = os.path.join(SCRIPT_DIR_PATH, "cache", "fuzzy_dedupe", "struct")

        rm_dir(cache_dir_txt)
        rm_dir(cache_dir_struct)

        fuzzy_dataset_text = fuzzy_dedupe(dataset=semantic_dataset_text, cache=cache_dir_txt)
        fuzzy_dataset_struct = fuzzy_dedupe(dataset=semantic_dataset_struct, cache=cache_dir_struct)

        dataset_text.df = fuzzy_dataset_text.df.to_backend("pandas")
        dataset_struct.df = fuzzy_dataset_struct.df.to_backend("pandas")
        print(f"After fuzzy dedupe for text files: {len(dataset_text.df)}")
        print(f"After fuzzy dedupe for tables and charts files: {len(dataset_struct.df)}")

    final_dataset_text = dataset_text.persist()
    final_dataset_struct = dataset_struct.persist()

    print("Writing the results to disk...")
    # Overwrite existing files in the curated directory.
    out_path = os.path.join(DATA_DIR, "curated")
    rm_dir(out_path)
    os.makedirs(out_path)
    final_dataset_text.to_json(out_path, write_to_filename=True)
    final_dataset_struct.to_json(out_path, write_to_filename=True)
    print("Writing results to disk completed")


def save_image(base64_str: str, output_path: str) -> None:
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image.save(output_path, format="JPEG")


def process_image_data(data_type_map: dict[str, Any]) -> ImageTextPairDataset:
    image_data = data_type_map["image"]
    shard_size = 1000
    parquet_rows = []
    os.makedirs(IMG_DIR, exist_ok=True)

    # Create tar file path
    shard_name = "00000"
    tar_path = os.path.join(IMG_DIR, f"{shard_name}.tar")

    with tarfile.open(tar_path, "w") as tar:
        for i, img_data in enumerate(image_data):
            shard_id = i // shard_size
            record_id = i % shard_size
            global_id = shard_id * shard_size + record_id
            shard_name = f"{shard_id:05d}"
            file_id = f"{global_id:09d}"

            # Extract fields
            image_b64 = img_data["metadata"]["content"]
            caption = img_data["metadata"]["image_metadata"]["caption"]
            unique_key = file_id  # or use any hash or ID here
            url = f"{unique_key}.jpg"

            # Prepare in-memory files
            img_buffer = io.BytesIO()
            save_image(image_b64, img_buffer)
            img_buffer.seek(0)

            caption_bytes = caption.encode("utf-8")
            json_bytes = json.dumps({"url": url, "caption": caption, "key": unique_key}).encode("utf-8")

            # Add files to tar
            # with tarfile.open(tar_path, "a") as tar:
            tarinfo = tarfile.TarInfo(f"{file_id}.jpg")
            tarinfo.size = len(img_buffer.getvalue())
            tar.addfile(tarinfo, img_buffer)

            tarinfo = tarfile.TarInfo(f"{file_id}.txt")
            tarinfo.size = len(caption_bytes)
            tar.addfile(tarinfo, io.BytesIO(caption_bytes))

            tarinfo = tarfile.TarInfo(f"{file_id}.json")
            tarinfo.size = len(json_bytes)
            tar.addfile(tarinfo, io.BytesIO(json_bytes))

            # # Collect row for Parquet
            parquet_rows.append({"url": f"{file_id}.jpg", "caption": caption, "key": unique_key})

            # # Write parquet at shard boundary or end
            if (i + 1) % shard_size == 0 or (i + 1) == len(image_data):
                table = pa.Table.from_pylist(parquet_rows)
                pq.write_table(table, os.path.join(IMG_DIR, f"{shard_name}.parquet"))
                parquet_rows.clear()

    return ImageTextPairDataset.from_webdataset(path=IMG_DIR, id_col="key")


def run_image_curation_pipeline(dataset: DocumentDataset) -> None:
    embedding_model = TimmImageEmbedder(
        "vit_large_patch14_clip_quickgelu_224.openai",
        pretrained=True,
        batch_size=1024,
        num_threads_per_worker=16,
        normalize_embeddings=True,
        autocast=False,
    )
    dataset = embedding_model(dataset)
    embeddings_dataset = DocumentDataset(dataset.metadata)

    # Use absolute paths to avoid Dask worker directory mismatch issues
    semantic_dedup_outputs = os.path.abspath("semantic_deduplication")
    os.makedirs(semantic_dedup_outputs, exist_ok=True)

    # Run clustering
    clustering_output = os.path.join(semantic_dedup_outputs, "cluster_output")
    clustering_model = ClusteringModel(
        id_column="key",
        embedding_column="image_embedding",
        max_iter=10,
        n_clusters=1,
        random_state=42,
        clustering_output_dir=clustering_output,
    )
    clustered_dataset = clustering_model(embeddings_dataset)
    if clustered_dataset:
        print("Clustering successful")

    # Run cluster-level dedup
    emb_by_cluster_output = os.path.join(clustering_output, "embs_by_nearest_center")
    duplicate_output = os.path.join(semantic_dedup_outputs, "duplicates")

    semantic_dedup = SemanticClusterLevelDedup(
        n_clusters=1,
        emb_by_clust_dir=emb_by_cluster_output,
        id_column="key",
        which_to_keep="hard",
        embedding_column="image_embedding",
        batched_cosine_similarity=1024,
        output_dir=duplicate_output,
    )
    semantic_dedup.compute_semantic_match_dfs()
    deduplicated_dataset_ids = semantic_dedup.extract_dedup_data(eps_to_extract=1.0)

    deduplicated_dataset_path = os.path.abspath("./deduplicated_dataset")
    dataset.metadata["is_unique"] = dataset.metadata["key"].isin(deduplicated_dataset_ids.df["key"].compute())
    dataset.to_webdataset(deduplicated_dataset_path, "is_unique")


def main() -> None:
    """
    Main function to run the data curation pipeline for text and image datasets.

    Args:
        None
    """
    # Parse arguments
    parser = argparse.ArgumentParser()
    args = ArgumentHelper(parser).add_distributed_args().parse_args()

    # Limit the total number of workers to ensure we don't run out of memory.
    args.n_workers = min(args.n_workers, 2)
    args.device = "gpu"
    print("Args: ", args)

    # Load the data type map
    nv_ingest_path = os.path.abspath("../ingest/sources/separated_extracted_data/data_type_map.json")
    with open(nv_ingest_path, encoding="utf-8") as f:
        data_type_map = json.load(f)

    # Process the data
    text_ddf, struct_ddf = process_data(data_type_map)
    image_ddf = process_image_data(data_type_map)

    # Initialize the client and run the curation pipelines
    client = get_client(**ArgumentHelper.parse_client_args(args), set_torch_to_use_rmm=True)
    run_text_curation_pipeline(args, text_ddf, struct_ddf)
    run_image_curation_pipeline(image_ddf)
    client.close()


if __name__ == "__main__":
    main()
