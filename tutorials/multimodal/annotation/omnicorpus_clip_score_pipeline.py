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

"""OmniCorpus WebDataset -> parquet with CLIP max image–text score per content row.

:class:`InterleavedCLIPScoreExportStage` mirrors :class:`InterleavedCLIPScoreFilterStage`
scoring: for each image row, ``scores[i].max()`` over CLIP similarities to all text rows
in the sample. Writes ``sample_id``, ``position``, ``clip_score`` for content rows.

Requires ``omni_corpus_annotation`` on ``PYTHONPATH`` like :mod:`omnicorpus_annotation_pipeline`.
"""

from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from fsspec.core import url_to_fs
from omni_corpus_annotation.stages.materialize import OmniCorpusMaterializeStage
from omni_corpus_annotation.stages.omnicorpus_reader import OmniCorpusReaderStage

import nemo_curator.stages.text.io.writer.utils as writer_utils
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.interleaved.filter import InterleavedCLIPScoreFilterStage
from nemo_curator.stages.interleaved.utils import image_bytes_to_array, resolve_storage_options
from nemo_curator.tasks import InterleavedBatch
from nemo_curator.utils.client_utils import is_remote_url
from nemo_curator.utils.file_utils import check_output_mode

ANNOTATION_METADATA_KEY = "annotation"
CLIP_MODEL_DIR = "./model_weights"


def _sample_texts_for_sample(df: pd.DataFrame, sample_id: str) -> list[str]:
    if "text_content" not in df.columns or "modality" not in df.columns:
        return []
    subset = df[(df["sample_id"] == sample_id) & (df["modality"] == "text")]
    if subset.empty:
        return []
    return [s.strip() for s in subset["text_content"].dropna().astype(str).tolist() if s.strip()]


@dataclass
class InterleavedCLIPScoreExportStage(InterleavedCLIPScoreFilterStage):
    """CLIP setup identical to the filter stage; exposes per-image ``scores[i].max()`` (no thresholding)."""

    name: str = "interleaved_clip_score_export"

    def clip_max_scores(self, task: InterleavedBatch, df: pd.DataFrame) -> pd.Series:
        max_scores = pd.Series(np.nan, index=df.index, dtype=float)
        image_mask = df["modality"] == "image"
        if not image_mask.any():
            return max_scores

        sample_id_to_rows: dict[str, list[tuple[int, bytes]]] = {}
        for idx, image_bytes in self.iter_materialized_bytes(task=task, df=df, row_mask=image_mask):
            if image_bytes is None:
                continue
            sample_id = df.loc[idx, "sample_id"]
            sample_id_to_rows.setdefault(sample_id, []).append((idx, image_bytes))

        for sample_id, rows in sample_id_to_rows.items():
            texts = _sample_texts_for_sample(df, sample_id)
            if not texts:
                continue
            indices, images = [], []
            for idx, b in rows:
                indices.append(idx)
                images.append(image_bytes_to_array(b))
            img_emb = self._model(images)
            text_emb = self._model.encode_text(texts)
            scores = img_emb @ text_emb.T
            for i, idx in enumerate(indices):
                max_scores.loc[idx] = scores[i].max().item()

        return max_scores


@dataclass
class InterleavedCLIPScoreAnnotationStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    clip_stage: InterleavedCLIPScoreExportStage
    name: str = "interleaved_clip_score_annotation"

    def __post_init__(self) -> None:
        self.resources = self.clip_stage.resources

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: InterleavedBatch) -> InterleavedBatch:
        df = task.to_pandas()
        if df.empty:
            task._metadata[ANNOTATION_METADATA_KEY] = None
            return task
        scores = self.clip_stage.clip_max_scores(task, df)
        content = (df["modality"] != "metadata") & (df["position"] >= 0)
        sub = df.loc[content]
        out = sub[["sample_id", "position"]].copy()
        out["clip_score"] = scores.loc[sub.index].to_numpy(dtype=float, copy=True)
        task._metadata[ANNOTATION_METADATA_KEY] = out if not out.empty else None
        return task


@dataclass
class InterleavedClipScoreParquetWriterStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    path: str
    write_kwargs: dict[str, Any] | None = None
    mode: Literal["ignore", "overwrite", "append", "error"] = "ignore"
    name: str = "interleaved_clip_score_parquet_writer"

    def __post_init__(self) -> None:
        self.write_kwargs = self.write_kwargs or {}
        self.storage_options = self.write_kwargs.get("storage_options", {})
        self.fs, self._fs_path = url_to_fs(self.path, **self.storage_options)
        check_output_mode(self.mode, self.fs, self._fs_path, append_mode_implemented=False)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: InterleavedBatch) -> InterleavedBatch:
        to_write = task._metadata.get(ANNOTATION_METADATA_KEY)
        if to_write is None or to_write.empty:
            return task
        if source_files := task._metadata.get("source_files"):
            filename = writer_utils.get_deterministic_hash(source_files, task.task_id)
        else:
            filename = uuid.uuid4().hex
        file_path = self.fs.sep.join([self._fs_path, f"{filename}.parquet"])
        file_path_with_protocol = self.fs.unstrip_protocol(file_path) if is_remote_url(self.path) else file_path
        write_kwargs = dict(self.write_kwargs)
        write_kwargs.setdefault("compression", "snappy")
        write_kwargs.setdefault("row_group_size", 128_000)
        write_kwargs["index"] = False
        to_write.to_parquet(file_path_with_protocol, **write_kwargs)
        return task


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    read_kwargs: dict[str, Any] = {}
    write_kwargs: dict[str, Any] = {}
    if args.storage_options_json:
        storage_options = json.loads(args.storage_options_json)
        read_kwargs["storage_options"] = storage_options
        write_kwargs["storage_options"] = storage_options

    storage_opts = resolve_storage_options(io_kwargs=read_kwargs)

    pipe = Pipeline(
        name="omnicorpus_clip_score_multimodal",
        description="OmniCorpus WebDataset -> sample_id, position, clip_score (CLIP max over sample texts)",
    )
    pipe.add_stage(
        FilePartitioningStage(
            file_paths=args.input_path,
            files_per_partition=args.files_per_partition,
            blocksize=args.input_blocksize,
            file_extensions=[".tar"],
            storage_options=storage_opts,
        )
    )
    pipe.add_stage(
        OmniCorpusReaderStage(
            max_batch_bytes=args.output_max_batch_bytes,
            read_kwargs=read_kwargs,
            include_general_metadata=args.include_general_metadata,
        )
    )
    pipe.add_stage(OmniCorpusMaterializeStage())
    pipe.add_stage(
        InterleavedCLIPScoreAnnotationStage(
            clip_stage=InterleavedCLIPScoreExportStage(model_dir=CLIP_MODEL_DIR),
        )
    )
    pipe.add_stage(
        InterleavedClipScoreParquetWriterStage(
            path=args.output_path,
            write_kwargs=write_kwargs,
            mode=args.mode,
        )
    )
    return pipe


def main(args: argparse.Namespace) -> None:
    ray_client = RayClient()
    ray_client.start()
    pipeline = build_pipeline(args)
    print(pipeline.describe())
    pipeline.run()
    ray_client.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OmniCorpus: write per-row CLIP max image–text score (scores[i].max()) to parquet",
    )
    parser.add_argument("--input-path", type=str, required=True, help="Input tar shard path or directory")
    parser.add_argument("--output-path", type=str, required=True, help="Output directory for parquet")
    parser.add_argument("--files-per-partition", type=int, default=1)
    parser.add_argument("--input-blocksize", type=str, default=None)
    parser.add_argument("--output-max-batch-bytes", type=int, default=None)
    parser.add_argument(
        "--include-general-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Join url/safety fields from .general_metadata.pkl onto rows (OmniCorpus reader default).",
    )
    parser.add_argument("--mode", type=str, default="ignore", choices=["ignore", "overwrite", "append", "error"])
    parser.add_argument(
        "--storage-options-json",
        type=str,
        default=None,
        help="JSON-encoded fsspec storage options for cloud paths",
    )
    main(parser.parse_args())
