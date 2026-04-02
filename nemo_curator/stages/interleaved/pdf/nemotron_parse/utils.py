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

"""Utility functions for Nemotron-Parse PDF processing.

Provides output parsing, image canvas construction, bbox cropping, and
element reordering used by the preprocess / postprocess stages.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import zipfile
from typing import Any

from PIL import Image

DEFAULT_MIN_CROP_PX = 10
DEFAULT_MAX_PAGES = 50


def _render_scale_to_fit(page: Any, base_scale: float, max_wh: tuple[int, int] | None) -> float:  # noqa: ANN401
    """Return the render scale capped so the output fits within max_wh pixels.

    Mirrors NeMo-Retriever's ``_compute_render_scale_to_fit``: uses the
    standard fit-to-box formula min(target_w/page_w, target_h/page_h) and
    clamps to a minimum of 1e-3 to avoid degenerate renders.  When max_wh is
    None the base_scale is returned unchanged.
    """
    if max_wh is None:
        return base_scale
    target_w, target_h = max_wh
    if target_w <= 0 or target_h <= 0:
        return base_scale
    page_w, page_h = float(page.get_width()), float(page.get_height())
    if page_w <= 0.0 or page_h <= 0.0:
        return base_scale
    fit_scale = max(min(target_w / page_w, target_h / page_h), 1e-3)
    return min(base_scale, fit_scale)


def _bitmap_to_rgb(bitmap: Any) -> Image.Image:  # noqa: ANN401
    """Convert a pypdfium2 bitmap to an RGB PIL image using OpenCV."""
    import cv2

    arr = bitmap.to_numpy().copy()
    mode = bitmap.mode
    if mode in {"BGRA", "BGRX"}:
        cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA, dst=arr)
        img = Image.fromarray(arr, "RGBA").convert("RGB")
    elif mode == "BGR":
        cv2.cvtColor(arr, cv2.COLOR_BGR2RGB, dst=arr)
        img = Image.fromarray(arr, "RGB")
    else:
        img = Image.fromarray(arr)
        if img.mode != "RGB":
            img = img.convert("RGB")
    return img


def _render_page(doc: Any, page_num: int, base_scale: float, max_size: tuple[int, int] | None) -> Image.Image | None:  # noqa: ANN401
    """Render a single PDF page; returns None on any error."""
    page = None
    bitmap = None
    try:
        page = doc[page_num]
        scale = _render_scale_to_fit(page, base_scale, max_size)
        bitmap = page.render(scale=scale)
        return _bitmap_to_rgb(bitmap)
    except Exception:  # noqa: BLE001
        return None
    finally:
        with contextlib.suppress(Exception):
            if bitmap is not None:
                bitmap.close()
        with contextlib.suppress(Exception):
            if page is not None:
                page.close()


def render_pdf_pages(
    pdf_bytes: bytes,
    dpi: int = 300,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_size: tuple[int, int] | None = (1664, 2048),
) -> list[Image.Image]:
    """Render PDF pages to PIL images using pypdfium2.

    Follows the same pattern as NeMo-Retriever to avoid two pdfium pitfalls:
    1. Explicitly close each page/bitmap after use so the weakref finalizer
       never fires (avoids SIGABRT in _close_impl during GC).
    2. Use ``bitmap.to_numpy().copy()`` + OpenCV for BGR->RGB conversion
       instead of pdfium's ``rev_byteorder`` flag, which triggers a
       non-thread-safe code path in CFX_AggDeviceDriver::GetDIBits().

    The render scale is capped per page via ``_render_scale_to_fit`` so that
    no rendered image exceeds ``max_size`` pixels (default: 1664x2048 =
    Nemotron-Parse processor size).  This bounds the bitmap size regardless of
    how large the PDF page dimensions are, eliminating decompression-bomb
    errors downstream and keeping render time predictable.
    """
    import pypdfium2 as pdfium

    images: list[Image.Image] = []
    doc = None
    with contextlib.suppress(Exception):
        doc = pdfium.PdfDocument(pdf_bytes)
        base_scale = dpi / 72.0
        for page_num in range(min(len(doc), max_pages)):
            img = _render_page(doc, page_num, base_scale, max_size)
            if img is not None:
                images.append(img)
    with contextlib.suppress(Exception):
        if doc is not None:
            doc.close()
    return images


def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    """Serialize a PIL Image to bytes."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def parse_nemotron_output(raw_text: str) -> list[dict[str, Any]]:
    """Parse Nemotron-Parse raw output into structured elements.

    Each element is a dict with keys ``class``, ``text``, and ``bbox``
    (normalized [x1, y1, x2, y2]).
    """
    elements: list[dict[str, Any]] = []
    pattern = re.compile(
        r"<x_([\d.]+)><y_([\d.]+)>"
        r"(.*?)"
        r"<x_([\d.]+)><y_([\d.]+)>"
        r"<class_([^>]+)>",
        re.DOTALL,
    )
    for match in pattern.finditer(raw_text):
        x1, y1 = float(match.group(1)), float(match.group(2))
        x2, y2 = float(match.group(4)), float(match.group(5))
        cls = match.group(6)
        text = re.sub(r"<[^>]+>", "", match.group(3)).strip()
        bbox = [x1, y1, x2, y2]
        if text or cls == "Picture":
            elements.append({"class": cls, "text": text, "bbox": bbox})

    if not elements and raw_text.strip():
        cleaned = re.sub(r"<[^>]+>", "", raw_text).strip()
        if cleaned:
            elements.append({"class": "Text", "text": cleaned, "bbox": None})
    return elements


def build_canvas(page_img: Image.Image, proc_size: tuple[int, int]) -> Image.Image:
    """Replicate the model processor's resize-then-center-pad to build the canvas.

    This lets us crop bboxes directly in the model's coordinate space.
    """
    import cv2
    import numpy as np

    proc_h, proc_w = proc_size
    orig_w, orig_h = page_img.size
    arr = np.asarray(page_img)

    ar = orig_w / orig_h
    new_h, new_w = orig_h, orig_w
    if new_h > proc_h:
        new_h = proc_h
        new_w = int(new_h * ar)
    if new_w > proc_w:
        new_w = proc_w
        new_h = int(new_w / ar)

    if (new_w, new_h) != (orig_w, orig_h):
        arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_h = max(0, proc_h - arr.shape[0])
    pad_w = max(0, proc_w - arr.shape[1])
    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        arr = np.pad(
            arr,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=255,
        )

    return Image.fromarray(arr)


def crop_to_bbox(
    canvas: Image.Image,
    bbox: list[float] | None,
    proc_size: tuple[int, int],
    min_crop_px: int = DEFAULT_MIN_CROP_PX,
) -> Image.Image | None:
    """Crop a region from the padded canvas using normalized bbox coordinates.

    Returns None if the crop is too small (likely a degenerate bbox).
    """
    if bbox is None:
        return canvas
    proc_h, proc_w = proc_size
    x0 = int(bbox[0] * proc_w)
    y0 = int(bbox[1] * proc_h)
    x1 = int(bbox[2] * proc_w)
    y1 = int(bbox[3] * proc_h)
    x0, x1 = max(0, min(x0, x1)), min(proc_w, max(x0, x1))
    y0, y1 = max(0, min(y0, y1)), min(proc_h, max(y0, y1))
    if x1 - x0 < min_crop_px or y1 - y0 < min_crop_px:
        return None
    return canvas.crop((x0, y0, x1, y1))


def _bbox_center_y(bbox: list[float] | None) -> float:
    if bbox is None:
        return 0.0
    return (bbox[1] + bbox[3]) / 2.0


def _pair_pictures_and_captions(
    floaters: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Group each Caption with its nearest Picture by bbox proximity."""
    pictures = [(i, f) for i, f in enumerate(floaters) if f["class"] == "Picture"]
    captions = [(i, f) for i, f in enumerate(floaters) if f["class"] == "Caption"]

    pic_taken: set[int] = set()
    cap_to_pic: dict[int, int] = {}

    for ci, cap in captions:
        cap_y = _bbox_center_y(cap.get("bbox"))
        best_pi = None
        best_dist = float("inf")
        for pi, pic in pictures:
            if pi in pic_taken:
                continue
            dist = abs(_bbox_center_y(pic.get("bbox")) - cap_y)
            if dist < best_dist:
                best_dist = dist
                best_pi = pi
        if best_pi is not None:
            cap_to_pic[ci] = best_pi
            pic_taken.add(best_pi)

    groups: list[list[dict[str, Any]]] = []
    used_caps: set[int] = set(cap_to_pic.keys())

    for pi, pic in pictures:
        group = [pic]
        matched_cap = [(ci, cap) for ci, cap in captions if cap_to_pic.get(ci) == pi]
        for _ci, cap in matched_cap:
            group.append(cap)
        groups.append(group)

    for ci, cap in captions:
        if ci not in used_caps:
            groups.append([cap])

    groups.sort(key=lambda g: _bbox_center_y(g[0].get("bbox")))
    return groups


def interleave_floaters(
    anchored: list[dict[str, Any]],
    floaters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Insert floater elements (Pictures/Captions) next to the closest anchor.

    Anchored elements keep their original model output order.  Pictures and
    Captions are first paired, then each pair is inserted after the anchored
    element whose bbox center-y is closest.

    This is needed for Nemotron-Parse v1.1 which emits Picture/Caption at the
    end of the page output rather than in reading order.  v1.2+ outputs them
    in correct reading order so this reordering can be skipped.
    """
    if not floaters:
        return list(anchored)
    if not anchored:
        result: list[dict[str, Any]] = []
        for group in _pair_pictures_and_captions(floaters):
            result.extend(group)
        return result

    groups = _pair_pictures_and_captions(floaters)
    anchor_ys = [_bbox_center_y(e.get("bbox")) for e in anchored]

    insert_map: dict[int, list[list[dict[str, Any]]]] = {}
    for group in groups:
        gy = _bbox_center_y(group[0].get("bbox"))
        best_idx = min(range(len(anchor_ys)), key=lambda i: abs(anchor_ys[i] - gy))
        insert_map.setdefault(best_idx, []).append(group)

    for groups_at_idx in insert_map.values():
        groups_at_idx.sort(key=lambda g: _bbox_center_y(g[0].get("bbox")))

    result = []
    for i, elem in enumerate(anchored):
        result.append(elem)
        if i in insert_map:
            for group in insert_map[i]:
                result.extend(group)
    return result


def build_interleaved_rows(  # noqa: PLR0913
    sample_id: str,
    url: str,
    pdf_name: str,
    page_images: list[Image.Image],
    page_outputs: list[str],
    proc_size: tuple[int, int] = (2048, 1664),
    reorder_floaters: bool = True,
    min_crop_px: int = DEFAULT_MIN_CROP_PX,
) -> list[dict[str, Any]]:
    """Convert Nemotron-Parse page outputs into interleaved-schema rows.

    Args:
        sample_id: Unique identifier for this PDF.
        url: Source URL of the PDF.
        pdf_name: Original PDF filename.
        page_images: Rendered page images.
        page_outputs: Raw Nemotron-Parse output per page.
        proc_size: Model processor's expected (height, width).
        reorder_floaters: If True, re-insert Pictures/Captions in reading order
            (needed for v1.1).  If False, preserve raw model output order (v1.2+).
        min_crop_px: Minimum pixel dimension for image crops.
    """
    rows: list[dict[str, Any]] = [
        {
            "sample_id": sample_id,
            "position": -1,
            "modality": "metadata",
            "content_type": "application/json",
            "text_content": json.dumps({"url": url, "pdf_name": pdf_name, "num_pages": len(page_images)}),
            "binary_content": None,
            "source_ref": None,
            "url": url,
            "page_number": None,
            "pdf_name": pdf_name,
            "element_class": None,
        }
    ]

    position = 0
    for page_num, (page_img, raw_output) in enumerate(zip(page_images, page_outputs, strict=True)):
        canvas = build_canvas(page_img, proc_size)
        elements = parse_nemotron_output(raw_output)

        if reorder_floaters:
            anchored = [e for e in elements if e["class"] not in ("Picture", "Caption")]
            floaters = [e for e in elements if e["class"] in ("Picture", "Caption")]
            ordered = interleave_floaters(anchored, floaters)
        else:
            ordered = elements

        for elem in ordered:
            cls = elem["class"]
            bbox = elem.get("bbox")
            source_ref = json.dumps({"page": page_num, "bbox": bbox})

            if cls == "Picture":
                modality, content_type = "image", "image/png"
                cropped = crop_to_bbox(canvas, bbox, proc_size, min_crop_px)
                if cropped is None:
                    continue
                binary, text = image_to_bytes(cropped), elem.get("text")
            elif cls == "Table":
                modality, content_type = "table", "text/markdown"
                binary, text = None, elem["text"]
            else:
                modality, content_type = "text", "text/markdown"
                binary, text = None, elem["text"]

            rows.append(
                {
                    "sample_id": sample_id,
                    "position": position,
                    "modality": modality,
                    "content_type": content_type,
                    "text_content": text,
                    "binary_content": binary,
                    "source_ref": source_ref,
                    "url": url,
                    "page_number": page_num,
                    "pdf_name": pdf_name,
                    "element_class": cls,
                }
            )
            position += 1

    return rows


# ---------------------------------------------------------------------------
# CC-MAIN PDF zip archive helpers
# ---------------------------------------------------------------------------


def resolve_cc_pdf_zip_path(file_name: str, zip_base_dir: str) -> tuple[str, str]:
    """Map a CC-MAIN PDF filename to its zip archive path and member name.

    The CC-MAIN-2021-31-PDF-UNTRUNCATED dataset organises PDFs into zip
    archives using a two-level numeric grouping::

        <zip_base_dir>/0000-0999/0001.zip  → contains 0001000.pdf .. 0001999.pdf
        <zip_base_dir>/1000-1999/1234.zip  → contains 1234000.pdf .. 1234999.pdf

    Args:
        file_name: PDF filename (e.g. ``"0001234.pdf"``).
        zip_base_dir: Root directory containing the zip archive hierarchy.

    Returns:
        Tuple of (zip_path, member_name).
    """
    num = int(file_name.replace(".pdf", ""))
    zip_num = num // 1000
    group_start = (zip_num // 1000) * 1000
    group_end = group_start + 999
    return (
        os.path.join(zip_base_dir, f"{group_start:04d}-{group_end:04d}", f"{zip_num:04d}.zip"),
        file_name,
    )


def extract_pdf_from_zip(file_name: str, zip_base_dir: str) -> bytes | None:
    """Extract a PDF file from a CC-MAIN zip archive.

    Returns None if extraction fails.
    """
    try:
        zip_path, member = resolve_cc_pdf_zip_path(file_name, zip_base_dir)
    except ValueError:
        return None
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            return zf.read(member)
    except (OSError, KeyError, zipfile.BadZipFile):
        return None


def extract_pdf_from_jsonl(
    jsonl_file: str,
    line_idx: int | None = None,
    byte_offset: int | None = None,
) -> bytes | None:
    """Extract a base64-encoded PDF from a JSONL file.

    Used for GitHub-style PDF datasets where each line contains a JSON object
    with a ``content`` field holding a base64-encoded PDF.

    Prefer ``byte_offset`` (O(1) seek) over ``line_idx`` (O(N) linear scan).
    When both are absent, returns None.
    """
    import base64

    try:
        if byte_offset is not None:
            with open(jsonl_file, "rb") as f:
                f.seek(byte_offset)
                line = f.readline()
                record = json.loads(line)
                return base64.b64decode(record["content"])
        if line_idx is not None:
            with open(jsonl_file) as f:
                for i, line in enumerate(f):
                    if i == line_idx:
                        record = json.loads(line)
                        return base64.b64decode(record["content"])
    except Exception:  # noqa: BLE001
        return None
    return None


def extract_pdfs_from_jsonl_batch(
    jsonl_file: str,
    offsets: list[int],
) -> dict[int, bytes | None]:
    """Extract multiple PDFs from a JSONL file in a single file open.

    Opens the file once and seeks to each byte offset in sorted order.
    Returns a dict mapping byte_offset -> pdf_bytes (None on error).
    """
    import base64

    results: dict[int, bytes | None] = {}
    try:
        with open(jsonl_file, "rb") as f:
            for offset in sorted(offsets):
                result: bytes | None = None
                with contextlib.suppress(Exception):
                    f.seek(offset)
                    line = f.readline()
                    record = json.loads(line)
                    result = base64.b64decode(record["content"])
                results[offset] = result
    except OSError:
        for offset in offsets:
            results[offset] = None
    return results

