#!/usr/bin/env python3
"""Categorize OCR pipeline results into per-image review folders.

Structure:
  <output>/
    success/   <image_id>/  image.jpg (annotated with bboxes)  result.json
    empty_ocr/ <image_id>/  image.jpg (raw)                    result.json
    error/     <image_id>/  image.jpg (raw)                    result.json

Usage:
    python ocr_categorize_results.py --input <glob> --output <dir>
"""

from __future__ import annotations

import argparse
import io
import json
import re
from glob import glob
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


_TAR_RE = re.compile(r"^(.*\.tar)/(\d+):(\d+):(.+)$")
_SKIP_KEYS = {
    "ocr_language_route_prompt",
    "ocr_language_route_response_raw",
    "ocr_dense_prompt",
    "ocr_verification_prompt",
}


def _read_image(image_path: str) -> Image.Image | None:
    m = _TAR_RE.match(image_path)
    if not m:
        return None
    with open(m.group(1), "rb") as tf:
        tf.seek(int(m.group(2)))
        data = tf.read(int(m.group(3)))
    return Image.open(io.BytesIO(data)).convert("RGB")


def _annotate(img: Image.Image, words: list, font) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size
    for w in words:
        b = w["bbox_2d"]
        x0, y0, x1, y1 = int(b[0]/1000*W), int(b[1]/1000*H), int(b[2]/1000*W), int(b[3]/1000*H)
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
        draw.text((x0, max(0, y0 - 16)), w["text_content"], fill="red", font=font)
    return img


def _image_id(record: dict) -> str:
    img_id = record.get("image_id", "")
    if img_id:
        return img_id.replace("/", "_")
    m = _TAR_RE.match(record.get("image_path", ""))
    return Path(m.group(4)).stem if m else "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Glob pattern or directory for output JSONL files")
    ap.add_argument("--output", required=True, help="Root output directory")
    args = ap.parse_args()

    input_path = Path(args.input)
    files = sorted(input_path.glob("*.jsonl")) if input_path.is_dir() else [Path(p) for p in sorted(glob(args.input))]
    if not files:
        print(f"No JSONL files found: {args.input}")
        return

    out = Path(args.output)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    counts = {"success": 0, "empty_ocr": 0, "error": 0}

    for fpath in files:
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)

                error = record.get("error")
                words = record.get("ocr_dense")

                if error:
                    cat = "error"
                elif words:
                    cat = "success"
                else:
                    cat = "empty_ocr"

                img_id = _image_id(record)
                folder = out / cat / img_id
                folder.mkdir(parents=True, exist_ok=True)

                img = _read_image(record.get("image_path", ""))
                if img is not None:
                    img.save(folder / "image.jpg")
                    if words:
                        _annotate(img, words, font).save(folder / "annotated.jpg")

                result = {k: v for k, v in record.items() if k not in _SKIP_KEYS}
                (folder / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
                counts[cat] += 1

    print(f"Done: {sum(counts.values())} total -> {out}")
    for cat, n in counts.items():
        print(f"  {cat}/: {n}")


if __name__ == "__main__":
    main()
