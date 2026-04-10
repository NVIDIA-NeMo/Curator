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

"""Pipeline that runs the same filters as mint1t_mvp_pipeline but saves annotation rows
for every content (sample_id, original position) with a cumulative ``keep_mask`` boolean.
Downstream tools can compute filtered vs kept counts from parquet alone without re-scanning
the WebDataset input. Position is the original index so preview/restore can match rows
to the original dataset without re-running the filter.
"""

import argparse
import json
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from fsspec.core import url_to_fs

import nemo_curator.stages.text.io.writer.utils as writer_utils
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.interleaved.filter import (
    InterleavedBlurFilterStage,
    InterleavedCLIPScoreFilterStage,
    InterleavedImageToTextRatioFilterStage,
    InterleavedQRCodeFilterStage,
    interleaved_score_pass_mask,
)
from nemo_curator.stages.interleaved.io import WebdatasetReader
from nemo_curator.stages.interleaved.stages import BaseInterleavedScoreFilterStage
from nemo_curator.tasks import InterleavedBatch
from nemo_curator.utils.client_utils import is_remote_url
from nemo_curator.utils.file_utils import check_output_mode

ANNOTATION_METADATA_KEY = "annotation"

FILTER_CHOICES = ("blur", "qrcode", "clip", "ratio")
CLIP_MODEL_DIR = "./model_weights"


def add_annotation_filters(pipe: Pipeline, args: argparse.Namespace) -> None:
    """Append InterleavedAnnotationFilterStage instances in the order given by --filters."""
    for name in args.filters:
        if name == "blur":
            score = args.score if args.score is not None else 100.0
            pipe.add_stage(
                InterleavedAnnotationFilterStage(
                    filter_stage=InterleavedBlurFilterStage(score_threshold=score),
                )
            )
        elif name == "qrcode":
            score = args.score if args.score is not None else 0.05
            pipe.add_stage(
                InterleavedAnnotationFilterStage(
                    filter_stage=InterleavedQRCodeFilterStage(score_threshold=score),
                )
            )
        elif name == "clip":
            score = args.score if args.score is not None else 0.15
            pipe.add_stage(
                InterleavedAnnotationFilterStage(
                    filter_stage=InterleavedCLIPScoreFilterStage(
                        model_dir=CLIP_MODEL_DIR,
                        min_score=score,
                    ),
                )
            )
        elif name == "ratio":
            max_ratio = args.max_ratio if args.max_ratio is not None else float("inf")
            pipe.add_stage(
                InterleavedAnnotationFilterStage(
                    filter_stage=InterleavedImageToTextRatioFilterStage(
                        min_ratio=args.min_ratio,
                        max_ratio=max_ratio,
                    ),
                )
            )


@dataclass
class InterleavedAnnotationFilterStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    """Runs a single interleaved filter and updates annotation for all content rows.

    Each row has ``sample_id``, ``position``, and cumulative ``keep_mask`` (AND across
    filter stages so far). Task ``data`` is unchanged; only ``_metadata[annotation]`` is updated.
    Uses the same resources as the wrapped filter_stage.
    """

    filter_stage: BaseInterleavedScoreFilterStage
    name: str = "interleaved_annotation_filter"

    def __post_init__(self) -> None:
        self.resources = self.filter_stage.resources

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: InterleavedBatch) -> InterleavedBatch:
        df = task.to_pandas()
        if df.empty:
            task._metadata[ANNOTATION_METADATA_KEY] = None
            return task
        keep_mask = interleaved_score_pass_mask(self.filter_stage, task, df, drop_invalid_rows=True)
        content = (df["modality"] != "metadata") & (df["position"] >= 0)
        sub = df.loc[content]
        base = sub[["sample_id", "position"]].copy()
        base["keep_this"] = keep_mask.loc[sub.index].fillna(False).to_numpy(dtype=bool)

        current = task._metadata.get(ANNOTATION_METADATA_KEY)
        if current is not None and not current.empty:
            prev = current[["sample_id", "position"]].copy()
            if "keep_mask" in current.columns:
                prev["keep_mask"] = current["keep_mask"].fillna(False).astype(bool)
            else:
                prev["keep_mask"] = True
            merged = base.merge(prev, on=["sample_id", "position"], how="left")
            merged["keep_mask"] = merged["keep_this"] & merged["keep_mask"].fillna(False)
        else:
            merged = base
            merged["keep_mask"] = merged["keep_this"]

        annotation = merged[["sample_id", "position", "keep_mask"]].drop_duplicates(
            subset=["sample_id", "position"], keep="last"
        )
        task._metadata[ANNOTATION_METADATA_KEY] = annotation if not annotation.empty else None
        return task


@dataclass
class InterleavedAnnotationParquetWriterStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    """Writes annotation ``sample_id``, ``position``, ``keep_mask`` from task metadata to parquet.

    Pass-through stage. Expects annotation from ``InterleavedAnnotationFilterStage``.
    """

    path: str
    write_kwargs: dict[str, Any] | None = None
    mode: Literal["ignore", "overwrite", "append", "error"] = "ignore"
    name: str = "interleaved_annotation_parquet_writer"

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
    read_kwargs = {}
    write_kwargs = {}
    if args.storage_options_json:
        storage_options = json.loads(args.storage_options_json)
        read_kwargs["storage_options"] = storage_options
        write_kwargs["storage_options"] = storage_options

    pipe = Pipeline(
        name="mint1t_annotation_multimodal",
        description="WebDataset MINT1T -> annotation (sample_id, position, keep_mask) for all content rows",
    )
    pipe.add_stage(
        WebdatasetReader(
            source_id_field="pdf_name",
            file_paths=args.input_path,
            files_per_partition=args.files_per_partition,
            blocksize=args.input_blocksize,
            max_batch_bytes=args.output_max_batch_bytes,
            read_kwargs=read_kwargs,
            materialize_on_read=args.materialize_on_read,
            fields=tuple(args.fields) if args.fields else None,
            per_image_fields=tuple(args.per_image_fields) if args.per_image_fields else (),
            per_text_fields=tuple(args.per_text_fields) if args.per_text_fields else (),
        )
    )
    # One annotation filter stage per --filters entry; annotation is intersected across stages
    add_annotation_filters(pipe, args)
    pipe.add_stage(
        InterleavedAnnotationParquetWriterStage(
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
        description="MINT1T multimodal pipeline: save annotation (sample_id, position, keep_mask) for all content rows"
    )
    parser.add_argument("--input-path", type=str, required=True, help="Input tar shard path or directory")
    parser.add_argument("--output-path", type=str, required=True, help="Output directory for annotation parquet")
    parser.add_argument("--files-per-partition", type=int, default=1)
    parser.add_argument("--input-blocksize", type=str, default=None)
    parser.add_argument("--output-max-batch-bytes", type=int, default=None)
    parser.add_argument("--materialize-on-read", action="store_true", dest="materialize_on_read")
    parser.add_argument("--no-materialize-on-read", action="store_false", dest="materialize_on_read")
    parser.set_defaults(materialize_on_read=False)
    parser.add_argument("--mode", type=str, default="ignore", choices=["ignore", "overwrite", "append", "error"])
    parser.add_argument("--fields", nargs="*", default=None)
    parser.add_argument("--per-image-fields", nargs="*", default=["image_metadata"])
    parser.add_argument("--per-text-fields", nargs="*", default=[])
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
        default=["qrcode"],
        help="Interleaved filters to run (order matters). clip loads weights from ./model_weights.",
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
        help="Image-to-text ratio filter: min images-per-word for a sample",
    )
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=None,
        dest="max_ratio",
        help="Image-to-text ratio filter: max images-per-word (omit for no upper bound)",
    )
    main(parser.parse_args())
