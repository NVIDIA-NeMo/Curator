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

"""OmniCorpus WebDataset -> interleaved score filters -> Parquet.

Reads OmniCorpus-CC tar shards via :class:`OmniCorpusReaderStage`, optionally
runs :class:`OmniCorpusMaterializeStage` when ``blur``, ``qrcode``, or ``clip``
filters are selected. Each score filter stage leaves all original columns on
``task.data`` and appends its score columns; the final writer persists that
full table to Parquet (dict-valued CLIP scores are JSON-serialized for Parquet).

Requires the ``omni_corpus_annotation`` package (e.g. multi-modal-data-curation repo)
on ``PYTHONPATH`` or installed in the same environment as NeMo Curator.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd
from omni_corpus_annotation.stages.materialize import OmniCorpusMaterializeStage
from omni_corpus_annotation.stages.omnicorpus_reader import OmniCorpusReaderStage

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.interleaved.filter import (
    InterleavedBlurFilterStage,
    InterleavedCLIPScoreFilterStage,
    InterleavedImageToTextRatioFilterStage,
    InterleavedQRCodeFilterStage,
)
from nemo_curator.stages.interleaved.io import InterleavedParquetWriterStage
from nemo_curator.stages.interleaved.utils import resolve_storage_options

if TYPE_CHECKING:
    from nemo_curator.tasks import InterleavedBatch

FILTER_CHOICES = ("blur", "qrcode", "clip", "ratio")
CLIP_MODEL_DIR = "./model_weights"

_IMAGE_BYTE_FILTERS = frozenset({"blur", "qrcode", "clip"})


def _annotation_dict_to_json_for_parquet(v: Any) -> Any:
    """Serialize CLIP per-text-position scores (dict or dict-like Series) for Parquet."""
    if isinstance(v, dict):
        ordered = sorted(((int(k), float(val)) for k, val in v.items()), key=lambda kv: kv[0])
        return json.dumps({str(k): val for k, val in ordered})
    if isinstance(v, pd.Series):
        pairs: list[tuple[int, float]] = []
        for k, val in v.items():
            if pd.isna(val):
                continue
            pairs.append((int(k), float(val)))
        pairs.sort(key=lambda kv: kv[0])
        return json.dumps({str(k): val for k, val in pairs})
    return v


@dataclass
class OmnicorpusScoredParquetWriterStage(InterleavedParquetWriterStage):
    """Writes the scored interleaved table; JSON-serializes dict cells (e.g. CLIP) for Parquet."""

    name: str = "omnicorpus_scored_parquet_writer"

    def write_data(self, task: InterleavedBatch, file_path: str) -> None:
        with self._time_metric("materialize_dataframe_total_s"):
            df = self._materialize_dataframe(task)
        df = df.copy()
        for col in df.columns:
            if df[col].dtype != object:
                continue
            if not df[col].map(lambda v: isinstance(v, (dict, pd.Series))).any():
                continue
            df[col] = df[col].map(_annotation_dict_to_json_for_parquet)
        df = self._align_output(df)
        self._write_dataframe(df, file_path, self._effective_write_kwargs)


def add_score_filter_stages(pipe: Pipeline, args: argparse.Namespace) -> None:
    """Append score filter stages in ``--filters`` order."""
    for name in args.filters:
        if name == "blur":
            score = args.score if args.score is not None else 100.0
            pipe.add_stage(InterleavedBlurFilterStage(score_threshold=score))
        elif name == "qrcode":
            score = args.score if args.score is not None else 0.05
            pipe.add_stage(InterleavedQRCodeFilterStage(score_threshold=score))
        elif name == "clip":
            score = args.score if args.score is not None else 0.15
            pipe.add_stage(
                InterleavedCLIPScoreFilterStage(
                    model_dir=CLIP_MODEL_DIR,
                    min_score=score,
                )
            )
        elif name == "ratio":
            max_ratio = args.max_ratio if args.max_ratio is not None else float("inf")
            pipe.add_stage(
                InterleavedImageToTextRatioFilterStage(
                    min_ratio=args.min_ratio,
                    max_ratio=max_ratio,
                )
            )


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    read_kwargs: dict[str, Any] = {}
    write_kwargs: dict[str, Any] = {}
    if args.storage_options_json:
        storage_options = json.loads(args.storage_options_json)
        read_kwargs["storage_options"] = storage_options
        write_kwargs["storage_options"] = storage_options

    storage_opts = resolve_storage_options(io_kwargs=read_kwargs)

    pipe = Pipeline(
        name="omnicorpus_annotation_multimodal",
        description="OmniCorpus WebDataset -> score filter stages -> scored interleaved Parquet",
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
    if _IMAGE_BYTE_FILTERS.intersection(args.filters):
        pipe.add_stage(OmniCorpusMaterializeStage())

    add_score_filter_stages(pipe, args)
    pipe.add_stage(
        OmnicorpusScoredParquetWriterStage(
            path=args.output_path,
            materialize_on_write=False,
            mode=args.mode,
            write_kwargs=write_kwargs,
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
        description=(
            "OmniCorpus multimodal pipeline: score filters append columns to each batch, "
            "then write the full interleaved table to Parquet."
        )
    )
    parser.add_argument("--input-path", type=str, required=True, help="Input tar shard path or directory")
    parser.add_argument("--output-path", type=str, required=True, help="Output directory for Parquet files")
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
    parser.add_argument(
        "--filters",
        nargs="+",
        choices=list(FILTER_CHOICES),
        default=["blur"],
        help="Interleaved score filters in order (each appends its score columns to task.data). CLIP loads from ./model_weights.",
    )
    parser.add_argument(
        "--score",
        type=float,
        default=None,
        help=(
            "Threshold for blur (min sharpness), qrcode (max QR area ratio), and clip (min similarity). "
            "If omitted, each filter uses its stage default (blur 100, qrcode 0.05, clip 0.15)."
        ),
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=0.0,
        dest="min_ratio",
        help="Ratio filter: min image_count / max(text_word_count, 1) for pass_mask semantics on score columns.",
    )
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=None,
        dest="max_ratio",
        help="Ratio filter: max image_count / max(text_word_count, 1) (omit for no upper bound).",
    )
    main(parser.parse_args())
