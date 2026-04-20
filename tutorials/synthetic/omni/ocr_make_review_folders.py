#!/usr/bin/env python3
"""Build per-image review folders from OCR pipeline JSONL output.

For each record creates:
  <out>/<category>/<image_id>/
    image.jpg          — raw image extracted from tar
    annotated.jpg      — image with OCR bboxes drawn  (if OCR data present)
    result.json        — key fields only (no raw prompts)

Categories:
  pass/    — passed OCR verification
  fail/    — failed OCR verification (bbox errors)
  skip/    — skipped by routing (unsupported language etc.)

Usage:
    python ocr_make_review_folders.py \\
        --input  /path/to/shard00000_verified_*.jsonl \\
        --output /path/to/review \\
        [--limit N]          # max images per category (default 100)
        [--no-annotate]      # skip drawing bboxes
"""

from __future__ import annotations

import argparse
import io
import json
import re
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


_TAR_PATH_RE = re.compile(r"^(.*\.tar)/(\d+):(\d+):(.+)$")


def _read_image_from_path(image_path: str) -> Image.Image:
    m = _TAR_PATH_RE.match(image_path)
    if m:
        tar_path, offset, size, _ = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        with open(tar_path, "rb") as f:
            f.seek(offset)
            data = f.read(size)
        return Image.open(io.BytesIO(data)).convert("RGB")
    return Image.open(image_path).convert("RGB")


def _draw_bboxes(img: Image.Image, words: list[dict]) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for w in words:
        bbox = w.get("bbox_2d") or w.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x0 = int(bbox[0] / 1000 * W)
        y0 = int(bbox[1] / 1000 * H)
        x1 = int(bbox[2] / 1000 * W)
        y1 = int(bbox[3] / 1000 * H)
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
        text = str(w.get("text_content") or "")
        if text:
            draw.text((x0, max(0, y0 - 16)), text, fill="red", font=font)
    return img


def _result_json(record: dict) -> dict:
    """Extract key fields for result.json (drop raw prompts to keep it readable)."""
    skip = {
        "ocr_dense_prompt", "ocr_verification_prompt",
    }
    return {k: v for k, v in record.items() if k not in skip}


def _image_id(record: dict) -> str:
    img_id = record.get("image_id")
    if img_id:
        return img_id.replace("/", "_")
    path = record.get("image_path", "unknown")
    m = _TAR_PATH_RE.match(path)
    if m:
        stem = Path(m.group(4)).stem
        tar_stem = Path(m.group(1)).stem
        return f"{tar_stem}_{stem}"
    return Path(path).stem


def _categorize(record: dict) -> str:
    error = record.get("error", "")
    if not error:
        return "pass"
    if error.startswith("ocr_verification"):
        return "fail"
    return "skip"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Glob pattern or directory for verified JSONL files")
    ap.add_argument("--output", required=True, help="Root output directory")
    ap.add_argument("--limit", type=int, default=100, help="Max images per category (default 100)")
    ap.add_argument("--no-annotate", action="store_true", help="Skip drawing bboxes")
    args = ap.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        files = sorted(input_path.glob("*.jsonl"))
    else:
        from glob import glob
        files = [Path(p) for p in sorted(glob(str(input_path)))]
    if not files:
        print(f"No JSONL files found: {args.input}")
        return

    out_root = Path(args.output)
    counts: dict[str, int] = {"pass": 0, "fail": 0, "skip": 0}
    errors_seen: dict[str, int] = {}

    for fpath in files:
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                cat = _categorize(record)
                if counts[cat] >= args.limit:
                    continue

                img_id = _image_id(record)
                folder = out_root / cat / img_id
                folder.mkdir(parents=True, exist_ok=True)

                # Extract and save raw image
                try:
                    img = _read_image_from_path(record["image_path"])
                    img.save(folder / "image.jpg")
                except Exception as e:
                    (folder / "image_error.txt").write_text(str(e))
                    img = None

                # Draw annotated image if OCR data present
                if img is not None and not args.no_annotate:
                    words = record.get("ocr_dense") or []
                    if words:
                        try:
                            annotated = _draw_bboxes(img, words)
                            annotated.save(folder / "annotated.jpg")
                        except Exception as e:
                            (folder / "annotate_error.txt").write_text(str(e))

                # Save result JSON (no raw prompts)
                result = _result_json(record)
                (folder / "result.json").write_text(
                    json.dumps(result, indent=2, ensure_ascii=False)
                )

                counts[cat] += 1

        # Stop early if all categories are full
        if all(counts[c] >= args.limit for c in counts):
            break

    total = sum(counts.values())
    print(f"Written {total} review folders to {out_root}")
    for cat, n in counts.items():
        print(f"  {cat}/  : {n}")


if __name__ == "__main__":
    main()
