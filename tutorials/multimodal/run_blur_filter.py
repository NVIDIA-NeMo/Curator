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

"""Run interleaved score filters on a demo batch; optionally write scored rows to Parquet."""

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import pandas as pd

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.interleaved.filter import (
    InterleavedBlurFilterStage,
    InterleavedCLIPScoreFilterStage,
    InterleavedImageToTextRatioFilterStage,
    InterleavedQRCodeFilterStage,
)
from nemo_curator.stages.interleaved.io import InterleavedParquetWriterStage
from nemo_curator.tasks import InterleavedBatch


def _clip_dict_cell_to_json_for_parquet(v: Any) -> Any:  # noqa: ANN401
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


def _image_bytes_from_path(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _image_bytes_from_url(url: str) -> tuple[bytes, str]:
    """Fetch image from http(s) URL. Returns (bytes, content_type)."""
    with urlopen(url) as resp:  # noqa: S310
        data = resp.read()
        content_type = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if content_type in ("image/jpeg", "image/jpg"):
            return data, "image/jpeg"
        if content_type == "image/png":
            return data, "image/png"
        # Infer from URL path if Content-Type missing or unknown
        path_lower = url.split("?")[0].lower()
        if path_lower.endswith((".jpg", ".jpeg")):
            return data, "image/jpeg"
        if path_lower.endswith(".png"):
            return data, "image/png"
        return data, content_type or "image/jpeg"


def _make_demo_image_bytes() -> bytes:
    """Return a small in-memory JPEG (no file I/O). Requires cv2."""
    import cv2
    import numpy as np

    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[:] = (128, 128, 128)
    _, buf = cv2.imencode(".jpg", arr)
    return buf.tobytes()


def _load_one_image(source: str) -> tuple[bytes, str]:
    """Load image from local path or http(s) URL. Returns (bytes, content_type)."""
    s = source.strip().lower()
    if s.startswith(("http://", "https://")):
        return _image_bytes_from_url(source)
    path = Path(source)
    if not path.is_file():
        msg = f"Not a file: {path}"
        raise SystemExit(msg)
    data = _image_bytes_from_path(source)
    ct = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return data, ct


def build_task(images: list[tuple[bytes, str]], num_repeats: int = 1) -> InterleavedBatch:
    """Build an InterleavedBatch with repeated samples; each repeat has a unique sample_id."""
    rows = []
    for r in range(num_repeats):
        sample_id = f"demo_{r}"
        pos = 0
        rows.append(
            {
                "sample_id": sample_id,
                "position": pos,
                "modality": "text",
                "content_type": "text/plain",
                "text_content": "Human and dog",
                "binary_content": None,
                "source_ref": None,
                "materialize_error": None,
            }
        )
        pos += 1
        for image_bytes, content_type in images:
            rows.append(
                {
                    "sample_id": sample_id,
                    "position": pos,
                    "modality": "image",
                    "content_type": content_type,
                    "text_content": None,
                    "binary_content": image_bytes,
                    "source_ref": None,
                    "materialize_error": None,
                }
            )
            pos += 1
        rows.append(
            {
                "sample_id": sample_id,
                "position": pos,
                "modality": "text",
                "content_type": "text/plain",
                "text_content": "Several flowers",
                "binary_content": None,
                "source_ref": None,
                "materialize_error": None,
            }
        )
    df = pd.DataFrame(rows)
    return InterleavedBatch(task_id="blur_demo", dataset_name="demo", data=df)


def main() -> None:  # noqa: C901, PLR0915
    parser = argparse.ArgumentParser(
        description="Run InterleavedBlurAnnotatorStage and InterleavedQRCodeAnnotatorStage on one or more images."
    )
    parser.add_argument(
        "image_paths",
        type=str,
        nargs="*",
        help="Paths to image files or http(s) URLs (JPEG/PNG). If omitted, one small synthetic image is used.",
    )
    parser.add_argument(
        "--num-repeats",
        type=int,
        default=3,
        help="Number of repeated samples (each with a unique sample_id); default 3.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=100.0,
        help="Blur filter: minimum sharpness (Laplacian variance) to keep an image (default: 100.0).",
    )
    parser.add_argument(
        "--qrcode-threshold",
        type=float,
        default=0.05,
        help="QR filter: max QR area ratio to keep an image; above this the image is dropped (default: 0.05).",
    )
    parser.add_argument(
        "--min-image-to-text-ratio",
        type=float,
        default=0.0,
        help="Image-to-text ratio filter: min ratio (images per word) to keep sample (default: 0.0).",
    )
    parser.add_argument(
        "--max-image-to-text-ratio",
        type=float,
        default=float("inf"),
        help="Image-to-text ratio filter: max ratio (images per word) to keep sample (default: inf).",
    )
    parser.add_argument(
        "--clip-model-dir",
        type=str,
        default=None,
        help="CLIP score filter: directory with CLIP weights; if set, filter by image-text relevance.",
    )
    parser.add_argument(
        "--clip-min-score",
        type=float,
        default=0.15,
        help="CLIP score filter: min image-text similarity to keep an image (default: 0.15).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save kept images (default: kept_images). Set to empty to skip saving.",
    )
    parser.add_argument(
        "--parquet-path",
        type=str,
        default=None,
        help="If set, write the scored interleaved table to this directory as Parquet (one file per batch).",
    )
    args = parser.parse_args()

    args.image_paths = args.image_paths or [
        "https://images.iphonephotographyschool.com/22704/1120/How-To-Blur-Background-On-iPhone.jpg",
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        "https://cdn0070.qrcodechimp.com/images/qr_code_with_logo.png",
    ]

    args.min_image_to_text_ratio = 0.001
    args.max_image_to_text_ratio = 0.8

    if args.image_paths:
        images = [_load_one_image(p) for p in args.image_paths]
    else:
        images = [(_make_demo_image_bytes(), "image/jpeg")]

    args.num_repeats = 1
    task = build_task(images, num_repeats=args.num_repeats)
    num_images_per_sample = len(images)
    total_input_images = args.num_repeats * num_images_per_sample
    blur_stage = InterleavedBlurFilterStage(score_threshold=args.score_threshold)
    qrcode_stage = InterleavedQRCodeFilterStage(score_threshold=args.qrcode_threshold)
    ratio_stage = InterleavedImageToTextRatioFilterStage(
        min_ratio=args.min_image_to_text_ratio,
        max_ratio=args.max_image_to_text_ratio,
    )

    samples_info = f"{args.num_repeats} samples x (1 text + {num_images_per_sample} images + 1 text)"
    print(f"Input rows: {len(task.data)} ({samples_info}), total images: {total_input_images}")
    task = blur_stage.process(task)
    task = qrcode_stage.process(task)
    task = ratio_stage.process(task)
    if args.clip_model_dir is not None:
        clip_stage = InterleavedCLIPScoreFilterStage(
            model_dir=args.clip_model_dir,
            min_score=args.clip_min_score,
        )
        clip_stage.setup_on_node(NodeInfo(), WorkerMetadata())
        clip_stage.setup()
        task = clip_stage.process(task)
    out_df = task.to_pandas()
    image_rows = out_df[out_df["modality"] == "image"]
    kept = len(image_rows)
    filtered = total_input_images - kept
    print(f"Output rows: {len(task.data)}, images kept: {kept}, images filtered out: {filtered}")
    print(task.data)

    if args.parquet_path:
        df = task.to_pandas().copy()
        for col in df.columns:
            if df[col].dtype != object:
                continue
            if not df[col].map(lambda v: isinstance(v, (dict, pd.Series))).any():
                continue
            df[col] = df[col].map(_clip_dict_cell_to_json_for_parquet)
        write_task = InterleavedBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=df,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )
        writer = InterleavedParquetWriterStage(
            path=args.parquet_path,
            materialize_on_write=False,
            mode="overwrite",
            write_kwargs={},
        )
        out_files = writer.process(write_task)
        print(f"Parquet written: {out_files.data}")

    if kept > 0 and args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ext_map = {"image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png"}
        saved = []
        for i, (_, row) in enumerate(image_rows.iterrows()):
            raw = row.get("binary_content")
            if raw is None:
                continue
            data = raw if isinstance(raw, bytes) else bytes(raw)
            ct = row.get("content_type", "image/jpeg") or "image/jpeg"
            ext = ext_map.get(ct, ".jpg")
            path = out_dir / f"kept_{i}{ext}"
            path.write_bytes(data)
            saved.append(str(path))
        if saved:
            print("Kept images saved to:")
            for p in saved:
                print(f"  {p}")


if __name__ == "__main__":
    main()
