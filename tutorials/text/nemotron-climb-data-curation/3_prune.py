# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
import os
import re
import shutil
from dataclasses import dataclass

import fasttext
import networkx as nx
import numpy as np
import pandas as pd
import ray
from loguru import logger

import nemo_curator.stages.text.io.writer.utils as writer_utils
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline.pipeline import Pipeline
from nemo_curator.stages.text.filters import DocumentFilter, Score
from nemo_curator.stages.text.io.reader import JsonlReader, ParquetReader
from nemo_curator.stages.text.io.writer.base import BaseWriter
from nemo_curator.tasks import DocumentBatch, FileGroupTask
from nemo_curator.utils.client_utils import is_remote_url


def preprocess_text(text: str) -> str:
    text = text.replace("\n", "<newline>")
    # Add spaces before and after punctuation
    text = re.sub(r"([.\!?,\'/()])", r" \1 ", text)
    # Convert to lowercase
    text = text.lower()
    # Merge multiple spaces into a single space
    return " ".join(text.split())


class FastTextQualityLabeler(DocumentFilter):
    def __init__(self, model_path: str | None = None):
        if model_path is None:
            msg = "Must provide a valid path to a FastText model to compute document scores with this filter"
            raise ValueError(msg)
        self._model_path = model_path
        self._name = "fasttext_quality_labeler"

    def model_check_or_download(self) -> None:
        if not os.path.exists(self._model_path):
            msg = f"Model file {self._model_path} not found"
            raise FileNotFoundError(msg)

    def load_model(self) -> None:
        self._fasttext_quality_model = fasttext.load_model(self._model_path)
        # Assert the model labels
        model_labels = [
            "__label__-1",
            "__label__0",
            "__label__1",
            "__label__2",
            "__label__3",
            "__label__4",
            "__label__5",
        ]
        assert sorted(self._fasttext_quality_model.labels) == sorted(model_labels), (  # noqa: S101
            "Incompatible fasttext model labels"
        )

    def score_document(self, text: str) -> float:
        # See setup() function in modules/filter.py
        model = self._fasttext_quality_model

        text = preprocess_text(text)

        # prediction returns a tuple (label, probability)
        prediction = model.predict([text])[0]
        label = prediction[0]
        return float(label[0].replace("__label__", ""))

    def keep_document(self, _: float) -> bool:
        # Always keep the document
        return True


@dataclass
class ParquetClusterWriter(BaseWriter):
    file_extension: str = "parquet"
    name: str = "parquet_cluster_writer"

    def write_data(self, task: DocumentBatch, file_path: str) -> None:
        df = task.to_pandas()
        df.to_parquet(file_path, index=None)

    def process(self, task: DocumentBatch) -> FileGroupTask:
        # Get source files from metadata for deterministic naming
        if source_files := task._metadata.get("source_files"):
            assert len(source_files) == 1, "Only one source file is allowed"  # noqa: S101
            source_file = source_files[0]
            assert "centroid=" in source_file, "Source file must contain a 'centroid=' directory"  # noqa: S101

            centroid = source_file.split("/")[-2].split("=")[1]
            filename = f"centroid={centroid}/{writer_utils.get_deterministic_hash(source_files, task.task_id)}"
        else:
            msg = "The task either does not have source_files in metadata or source_files does not contain a 'centroid=' directory"
            raise RuntimeError(msg)

        # Generate filename with appropriate extension using normalized fs path
        file_extension = self.get_file_extension()
        file_path = self.fs.sep.join([self._fs_path, f"{filename}.{file_extension}"])

        # For remote URLs, restore the protocol prefix so downstream code can infer the filesystem
        file_path_with_protocol = self.fs.unstrip_protocol(file_path) if is_remote_url(self.path) else file_path

        if self.fs.exists(file_path):
            logger.debug(f"File {file_path_with_protocol} already exists, overwriting it")

        self.write_data(task, file_path_with_protocol)
        logger.debug(f"Written {task.num_items} records to {file_path_with_protocol}")

        # Create FileGroupTask with written files using the full protocol-prefixed path
        return FileGroupTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=[file_path_with_protocol],
            _metadata={
                **task._metadata,
                "format": self.get_file_extension(),
            },
            _stage_perf=task._stage_perf,
        )


@ray.remote
def compute_average_score_per_cluster(cluster_path: str, score_field: str, threshold: float = 1.0) -> str | None:
    # Compute average score per cluster
    scores = []
    for file_path in os.listdir(cluster_path):
        if file_path.endswith(".parquet"):
            df = pd.read_parquet(
                os.path.join(cluster_path, file_path), columns=[score_field], engine="pyarrow", dtype_backend="pyarrow"
            )
            scores.extend(df[score_field].to_numpy().tolist())

    # Return paths to remove
    avg_score = sum(scores) / len(scores)
    if avg_score >= threshold:
        return None
    else:
        return cluster_path


def main(args: argparse.Namespace) -> None:  # noqa: C901, PLR0912
    subdirectories = [os.path.join(args.input_path, d) for d in os.listdir(args.input_path)]
    # Initialize empty subdirectories in output path
    for subdirectory in subdirectories:
        if not os.path.exists(os.path.join(args.output_path, subdirectory.split("/")[-1])):
            os.makedirs(os.path.join(args.output_path, subdirectory.split("/")[-1]))

    ray_client = RayClient(num_cpus=args.num_cpus, num_gpus=args.num_gpus)
    ray_client.start()

    if args.input_filetype == "jsonl":
        reader = JsonlReader
    elif args.input_filetype == "parquet":
        reader = ParquetReader
    else:
        msg = f"Invalid input file type: {args.input_filetype}"
        raise ValueError(msg)

    pipeline = Pipeline(name="3_prune")

    pipeline.add_stage(reader(file_paths=args.input_path, files_per_partition=1, fields=[args.text_field]))

    assert len(args.fasttext_model_paths) == len(args.score_fields), (  # noqa: S101
        "Number of fasttext model paths and score fields must match"
    )
    for fasttext_model_path, score_field in zip(args.fasttext_model_paths, args.score_fields, strict=True):
        pipeline.add_stage(
            Score(
                score_fn=FastTextQualityLabeler(model_path=fasttext_model_path),
                score_field=score_field,
                text_field=args.text_field,
            )
        )

    pipeline.add_stage(ParquetClusterWriter(path=args.output_path))

    pipeline.run()

    removed_clusters = []
    for score_field in args.score_fields:
        # List all subdirectories under the output path and compute average score per cluster
        subdirectories = [os.path.join(args.output_path, d) for d in os.listdir(args.output_path)]
        paths_to_remove = ray.get(
            [
                compute_average_score_per_cluster.remote(subdirectory, score_field, args.pruning_threshold)
                for subdirectory in subdirectories
            ]
        )

        # Remove the clusters with average score < pruning threshold
        for path_to_remove in paths_to_remove:
            if path_to_remove is not None:
                shutil.rmtree(path_to_remove)
                # Grab the number from the path, e.g., output_path/centroid=999 grabs 999
                removed_clusters.append(int(path_to_remove.split("/")[-1].split("=")[1]))

    centroids = np.load(os.path.join(args.centroids_path, "kmeans_centroids.npy"))
    # Compute the Euclidean distance between the centroids
    diff = centroids[:, None, :] - centroids[None, :, :]
    l2_distances = np.sqrt(np.sum(diff**2, axis=2))

    # Build graph of mergeable centroids
    graph = nx.Graph()
    num_centroids = centroids.shape[0]
    graph.add_nodes_from(range(num_centroids))

    pairs = np.argwhere((l2_distances < args.merge_threshold) & (l2_distances > 0))
    for i, j in pairs:
        # Skip if i or j is in removed_clusters
        if i in removed_clusters or j in removed_clusters:
            continue
        graph.add_edge(i, j)

    # Find connected components, each component is a merged cluster
    for comp in nx.connected_components(graph):
        component = list(comp)
        if len(component) <= 1:
            continue
        # Merge all directories in this component into the first one
        main = component[0]
        for other in component[1:]:
            src_dir = f"{args.output_path}/centroid={other}"
            dst_dir = f"{args.output_path}/centroid={main}"
            for f in os.listdir(src_dir):
                shutil.move(os.path.join(src_dir, f), dst_dir)
            os.rmdir(src_dir)

    ray_client.stop()


def attach_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Ray cluster args
    parser.add_argument("--num-cpus", type=int, default=None)
    parser.add_argument("--num-gpus", type=int, default=0)

    # Reader args
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--input-filetype", type=str, default="parquet", choices=["parquet", "jsonl"])

    # FastText args
    parser.add_argument("--fasttext-model-paths", nargs="+", required=True)
    parser.add_argument("--score-fields", nargs="+", required=True)
    parser.add_argument("--text-field", type=str, default="text")

    # Writer args
    parser.add_argument("--output-path", type=str, required=True)

    # Pruning args
    parser.add_argument("--pruning-threshold", type=float, default=1.0)

    # Cluster merging args
    parser.add_argument("--centroids-path", type=str, required=True)
    parser.add_argument("--merge-threshold", type=float, default=1.5)

    return parser


if __name__ == "__main__":
    main(attach_args().parse_args())
