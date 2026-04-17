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

"""Filter scored Parquet files produced by omnicorpus_annotation_pipeline.py.

Reads each Parquet file from --input-path (restoring CLIP score JSON cells to
dict[int, float]), applies InterleavedAnnotationThresholdFilterStage, then
writes filtered rows to --output-path (re-encoding CLIP dicts to JSON for
Parquet storage). One output file is written per input file.

Example (run from /opt/Curator inside the container):
    python tutorials/multimodal/run_annotation_threshold_filter.py \\
        --input-path  ./output/omni_annotation \\
        --output-path ./output/omni_filtered

Disable individual criteria with --no-blur / --no-clip / --no-qrcode / --no-image-text.
Override thresholds with --blur-min-sharpness, --clip-min-score, etc.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from nemo_curator.stages.interleaved.filter.annotation_threshold_filter import (
    InterleavedAnnotationThresholdFilterStage,
)
from nemo_curator.stages.interleaved.filter.blur_filter import DEFAULT_BLUR_SCORE_THRESHOLD
from nemo_curator.stages.interleaved.filter.clip_score_filter import DEFAULT_CLIP_MIN_SCORE
from nemo_curator.stages.interleaved.filter.image_to_text_ratio_filter import (
    DEFAULT_IMAGE_TO_TEXT_MAX_RATIO,
    DEFAULT_IMAGE_TO_TEXT_MIN_RATIO,
)
from nemo_curator.stages.interleaved.filter.qrcode_filter import DEFAULT_QRCODE_SCORE_THRESHOLD
from nemo_curator.tasks import InterleavedBatch


def _is_clip_scores_column(name: str) -> bool:
    return name == "clip_scores" or name.endswith("clip_scores")


def _parse_clip_cell(v: Any) -> Any:  # noqa: ANN401
    """JSON string -> dict[int, float]; pass-through nulls and already-decoded dicts."""
    if v is None:
        return v
    if isinstance(v, dict):
        return {int(k): float(val) for k, val in v.items()}
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return pd.NA
        parsed = json.loads(s)
        return {int(k): float(val) for k, val in parsed.items()} if isinstance(parsed, dict) else v
    try:
        if pd.isna(v):
            return pd.NA
    except (TypeError, ValueError):
        pass
    return v


def _serialize_clip_cell(v: Any) -> Any:  # noqa: ANN401
    """dict / pd.Series -> JSON string for Parquet; pass-through everything else."""
    if isinstance(v, dict):
        ordered = sorted(((int(k), float(val)) for k, val in v.items()), key=lambda kv: kv[0])
        return json.dumps({str(k): val for k, val in ordered})
    if isinstance(v, pd.Series):
        pairs = [(int(k), float(val)) for k, val in v.items() if pd.notna(val)]
        pairs.sort(key=lambda kv: kv[0])
        return json.dumps({str(k): val for k, val in pairs})
    return v


def _read_scored_parquet(path: str) -> InterleavedBatch:
    df = pq.read_table(path).to_pandas()
    for col in df.columns:
        if _is_clip_scores_column(col):
            df[col] = df[col].map(_parse_clip_cell)
    return InterleavedBatch(task_id=Path(path).stem, dataset_name="omnicorpus_scored", data=df)


def _write_scored_parquet(df: pd.DataFrame, path: str) -> None:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object and out[col].map(lambda v: isinstance(v, (dict, pd.Series))).any():
            out[col] = out[col].map(_serialize_clip_cell)
    out.to_parquet(path, compression="snappy", index=False)


def build_stage(args: argparse.Namespace) -> InterleavedAnnotationThresholdFilterStage:
    return InterleavedAnnotationThresholdFilterStage(
        blur_min_sharpness=args.blur_min_sharpness,
        blur_score_column=None if args.no_blur else "sharpness",
        clip_min_score=args.clip_min_score,
        clip_scores_column=None if args.no_clip else "clip_scores",
        qrcode_max_area_ratio=args.qrcode_max_ratio,
        qrcode_ratio_column=None if args.no_qrcode else "qr_area_ratio",
        image_text_min_ratio=args.image_text_min_ratio,
        image_text_max_ratio=args.image_text_max_ratio,
        image_text_image_num_column=None if args.no_image_text else "image_num",
        image_text_word_num_column=None if args.no_image_text else "text_word_num",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply InterleavedAnnotationThresholdFilterStage to scored Parquet files "
            "produced by omnicorpus_annotation_pipeline.py."
        )
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Directory of scored Parquet files (output of omnicorpus_annotation_pipeline.py).",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Directory to write filtered Parquet files (one file per input file).",
    )
    parser.add_argument(
        "--mode",
        default="ignore",
        choices=["ignore", "overwrite", "error"],
        help="How to handle existing output files (default: ignore).",
    )

    parser.add_argument(
        "--blur-min-sharpness",
        type=float,
        default=DEFAULT_BLUR_SCORE_THRESHOLD,
        metavar="N",
        help=f"Min Laplacian sharpness to keep an image (default: {DEFAULT_BLUR_SCORE_THRESHOLD}).",
    )
    parser.add_argument("--no-blur", action="store_true", help="Disable blur filter.")

    parser.add_argument(
        "--clip-min-score",
        type=float,
        default=DEFAULT_CLIP_MIN_SCORE,
        metavar="N",
        help=f"Min CLIP image-text similarity to keep an image (default: {DEFAULT_CLIP_MIN_SCORE}).",
    )
    parser.add_argument("--no-clip", action="store_true", help="Disable CLIP score filter.")

    parser.add_argument(
        "--qrcode-max-ratio",
        type=float,
        default=DEFAULT_QRCODE_SCORE_THRESHOLD,
        metavar="N",
        help=f"Max QR code area ratio to keep an image (default: {DEFAULT_QRCODE_SCORE_THRESHOLD}).",
    )
    parser.add_argument("--no-qrcode", action="store_true", help="Disable QR code filter.")

    parser.add_argument(
        "--image-text-min-ratio",
        type=float,
        default=DEFAULT_IMAGE_TO_TEXT_MIN_RATIO,
        metavar="N",
        help=f"Min images-per-word ratio per sample (default: {DEFAULT_IMAGE_TO_TEXT_MIN_RATIO}).",
    )
    parser.add_argument(
        "--image-text-max-ratio",
        type=float,
        default=DEFAULT_IMAGE_TO_TEXT_MAX_RATIO,
        metavar="N",
        help=f"Max images-per-word ratio per sample (default: {DEFAULT_IMAGE_TO_TEXT_MAX_RATIO}).",
    )
    parser.add_argument("--no-image-text", action="store_true", help="Disable image-to-text ratio filter.")

    args = parser.parse_args()

    input_dir = Path(args.input_path)
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(input_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"No Parquet files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    stage = build_stage(args)
    print(f"Stage: {stage}")
    print(f"Processing {len(parquet_files)} file(s) from {input_dir}\n")

    total_in = total_out = 0
    for pf in parquet_files:
        out_path = output_dir / pf.name
        if out_path.exists():
            if args.mode == "error":
                print(f"Error: output already exists: {out_path}", file=sys.stderr)
                sys.exit(1)
            if args.mode == "ignore":
                print(f"  skip (exists): {out_path.name}")
                continue

        batch = _read_scored_parquet(str(pf))
        n_in = len(batch.to_pandas())
        filtered = stage.process(batch)
        df_out = filtered.to_pandas()
        n_out = len(df_out)
        total_in += n_in
        total_out += n_out

        _write_scored_parquet(df_out, str(out_path))
        print(f"  {pf.name}: {n_in} rows -> {n_out} rows  (dropped {n_in - n_out})")

    print(f"\nTotal: {total_in} rows -> {total_out} rows  (dropped {total_in - total_out})")


if __name__ == "__main__":
    main()
