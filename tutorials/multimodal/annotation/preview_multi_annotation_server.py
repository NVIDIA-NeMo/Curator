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

"""Multi-stage annotation preview: filtered counts/ratios from annotation parquet (keep_mask); sample rows from tars.

``--input-source webdataset`` (default) uses MINT-style WebDataset tars; ``omnicorpus`` uses OmniCorpus-CC shards
via ``preview_annotation_standalone`` (same as ``preview_annotation_server``).
"""

from __future__ import annotations

import argparse
import base64
import glob
import html
import http.server
import io
import json
import os
import re
import socketserver
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

try:
    from PIL import Image
except ImportError:
    Image = None

from preview_annotation_server import (  # noqa: E402
    _content_mask,
    _load_annotation_all_and_kept_sets,
    _materialize_rows_parallel,
    _safe_str,
    _to_bytes,
)
from preview_annotation_standalone import (  # noqa: E402
    DEFAULT_JSON_EXTENSIONS,
    DEFAULT_WEBDATASET_EXTENSIONS,
    FileGroupTask,
    OmniCorpusReaderStage,
    WebdatasetReaderStage,
    get_all_file_paths_under,
    materialize_omnicorpus_binary_content,
)

PER_STAGE_SAMPLES = 5
DEFAULT_WORKERS = 16
DISPLAY_IMAGE_HEIGHT_PX = 400
THUMBNAIL_JPEG_QUALITY = 100

# Trailing numeric suffix: ``a_60``, ``a_0.1`` (float branch matched before lone int).
_TRAILING_NUM_RE = re.compile(r"(\d+\.\d+|\d+)$")


def _format_filtered_ratio(n_filtered: int, n_total: int) -> str:
    if n_total <= 0:
        return "—"
    return f"{100.0 * n_filtered / n_total:.1f}%"


def _resolve_tar_paths(input_path: str | list[str], read_kwargs: dict, max_tars: int | None) -> list[str]:
    read_kwargs = read_kwargs or {}
    if isinstance(input_path, str):
        paths = get_all_file_paths_under(
            input_path,
            recurse_subdirectories=True,
            keep_extensions=list(DEFAULT_WEBDATASET_EXTENSIONS),
            storage_options=read_kwargs.get("storage_options"),
        )
    else:
        paths = list(input_path)
    if max_tars is not None and max_tars > 0:
        paths = paths[:max_tars]
    return paths


def _stage_newly_filtered_plan(
    all_keys_per_stage: list[set[tuple[str, int]]],
    kept_sets: list[set[tuple[str, int]]],
    annotation_dirs: list[str],
    per_stage: int,
) -> list[tuple[str, int, list[tuple[str, int]], int]]:
    """Per stage: (annotation_dir, n_total_filtered, sample_keys_for_tars, total keys in parquet).

    ``n_total_filtered`` is |all_keys − kept| for that stage. ``sample_keys_for_tars`` are up to
    ``per_stage`` keys that are newly filtered vs the previous stage (for thumbnails only).
    """
    out: list[tuple[str, int, list[tuple[str, int]], int]] = []
    for i, ann_dir in enumerate(annotation_dirs):
        all_k = all_keys_per_stage[i]
        kept_i = kept_sets[i]
        n_total_filtered = len(all_k - kept_i)
        new_keys = all_k - kept_i if i == 0 else kept_sets[i - 1] - kept_i & all_k
        picked = sorted(new_keys)[:per_stage]
        out.append((ann_dir, n_total_filtered, picked, len(all_k)))
    return out


def _read_one_tar_rows_and_image_count(
    index: int,
    tar_path: str,
    read_kwargs: dict,
    need: frozenset[tuple[str, int]],
) -> tuple[int, int, dict[tuple[str, int], pd.Series]]:
    """Read one tar once: count image rows and collect needed content rows by key."""
    reader = WebdatasetReaderStage(
        source_id_field="pdf_name",
        read_kwargs=read_kwargs,
        materialize_on_read=False,
        max_batch_bytes=None,
        json_extensions=tuple(DEFAULT_JSON_EXTENSIONS),
    )
    task = FileGroupTask(
        task_id=f"scan_{index}",
        dataset_name="multi_preview",
        data=[tar_path],
        _metadata={"source_files": [tar_path]},
    )
    out = reader.process(task)
    batches = out if isinstance(out, list) else [out]
    found: dict[tuple[str, int], pd.Series] = {}
    img_rows = 0
    for batch in batches:
        df = batch.to_pandas()
        if df.empty:
            continue
        img = (df["modality"].astype(str) == "image") & (df["position"] >= 0)
        img_rows += int(img.sum())
        if not need:
            continue
        content = _content_mask(df)
        sub = df.loc[content]
        if sub.empty:
            continue
        for _, row in sub.iterrows():
            k = (str(row["sample_id"]), int(row["position"]))
            if k in need and k not in found:
                found[k] = row
    return index, img_rows, found


def _scan_tars_for_rows_and_image_count(
    paths: list[str],
    read_kwargs: dict,
    need_keys: set[tuple[str, int]],
    num_workers: int,
) -> tuple[dict[tuple[str, int], pd.Series], int]:
    """Single pass over all tars: merge rows for ``need_keys`` and sum image row counts."""
    if not paths:
        return {}, 0
    need = frozenset(need_keys)
    if num_workers <= 1:
        merged: dict[tuple[str, int], pd.Series] = {}
        total_img = 0
        for i, tar_path in enumerate(paths):
            _, nimg, part = _read_one_tar_rows_and_image_count(i, tar_path, read_kwargs, need)
            total_img += nimg
            for k, row in part.items():
                if k not in merged:
                    merged[k] = row
        return merged, total_img

    workers = min(num_workers, len(paths))
    per_index: dict[int, tuple[int, dict[tuple[str, int], pd.Series]]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_read_one_tar_rows_and_image_count, i, p, read_kwargs, need): i for i, p in enumerate(paths)
        }
        for fut in as_completed(futures):
            idx, nimg, part = fut.result()
            per_index[idx] = (nimg, part)
    merged: dict[tuple[str, int], pd.Series] = {}
    total_img = 0
    for i in range(len(paths)):
        nimg, part = per_index.get(i, (0, {}))
        total_img += nimg
        for k, row in part.items():
            if k not in merged:
                merged[k] = row
    return merged, total_img


def _read_one_tar_rows_and_image_count_omnicorpus(
    index: int,
    tar_path: str,
    read_kwargs: dict,
    need: frozenset[tuple[str, int]],
    include_general_metadata: bool,
    max_batch_bytes: int | None,
) -> tuple[int, int, dict[tuple[str, int], pd.Series]]:
    """Read one OmniCorpus tar: count image rows and collect needed content rows by key."""
    reader = OmniCorpusReaderStage(
        max_batch_bytes=max_batch_bytes,
        read_kwargs=read_kwargs,
        include_general_metadata=include_general_metadata,
    )
    task = FileGroupTask(
        task_id=f"scan_{index}",
        dataset_name="multi_preview",
        data=[tar_path],
        _metadata={"source_files": [tar_path]},
    )
    out = reader.process(task)
    batches = out if isinstance(out, list) else [out]
    found: dict[tuple[str, int], pd.Series] = {}
    img_rows = 0
    for batch in batches:
        mat = materialize_omnicorpus_binary_content(batch, io_kwargs=read_kwargs)
        df = mat.to_pandas()
        if df.empty:
            continue
        img = (df["modality"].astype(str) == "image") & (df["position"] >= 0)
        img_rows += int(img.sum())
        if not need:
            continue
        content = _content_mask(df)
        sub = df.loc[content]
        if sub.empty:
            continue
        for _, row in sub.iterrows():
            k = (str(row["sample_id"]), int(row["position"]))
            if k in need and k not in found:
                found[k] = row
    return index, img_rows, found


def _scan_tars_for_rows_and_image_count_omnicorpus(
    paths: list[str],
    read_kwargs: dict,
    need_keys: set[tuple[str, int]],
    num_workers: int,
    include_general_metadata: bool,
    omni_max_batch_bytes: int | None,
) -> tuple[dict[tuple[str, int], pd.Series], int]:
    """Single pass over OmniCorpus tars: merge rows for ``need_keys`` and sum image row counts."""
    if not paths:
        return {}, 0
    need = frozenset(need_keys)
    if num_workers <= 1:
        merged: dict[tuple[str, int], pd.Series] = {}
        total_img = 0
        for i, tar_path in enumerate(paths):
            _, nimg, part = _read_one_tar_rows_and_image_count_omnicorpus(
                i, tar_path, read_kwargs, need, include_general_metadata, omni_max_batch_bytes
            )
            total_img += nimg
            for k, row in part.items():
                if k not in merged:
                    merged[k] = row
        return merged, total_img

    workers = min(num_workers, len(paths))
    per_index: dict[int, tuple[int, dict[tuple[str, int], pd.Series]]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                _read_one_tar_rows_and_image_count_omnicorpus,
                i,
                p,
                read_kwargs,
                need,
                include_general_metadata,
                omni_max_batch_bytes,
            ): i
            for i, p in enumerate(paths)
        }
        for fut in as_completed(futures):
            idx, nimg, part = fut.result()
            per_index[idx] = (nimg, part)
    merged = {}
    total_img = 0
    for i in range(len(paths)):
        nimg, part = per_index.get(i, (0, {}))
        total_img += nimg
        for k, row in part.items():
            if k not in merged:
                merged[k] = row
    return merged, total_img


def _rows_dict_to_dataframe(
    keys_order: list[tuple[str, int]], by_key: dict[tuple[str, int], pd.Series]
) -> pd.DataFrame:
    rows = [by_key[k] for k in keys_order if k in by_key]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).reset_index(drop=True)


def _scale_image_fixed_height(b: bytes, height_px: int) -> tuple[bytes, str] | None:
    if Image is None:
        return None
    try:
        with Image.open(io.BytesIO(b)) as img:
            img = img.convert("RGB")
            w, h = img.size
            if h <= 0:
                return None
            new_w = max(1, round(w * height_px / h))
            resample = getattr(Image, "Resampling", Image).LANCZOS
            if (w, h) != (new_w, height_px):
                img = img.resize((new_w, height_px), resample)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=THUMBNAIL_JPEG_QUALITY, optimize=True)
            return buf.getvalue(), "image/jpeg"
    except (OSError, ValueError, TypeError):
        return None


def _row_to_html(row: pd.Series, index: int, _kind: str) -> str:
    sid = html.escape(_safe_str(row.get("sample_id")))
    pos = row.get("position", "")
    mod = html.escape(_safe_str(row.get("modality")))
    text_val = row.get("text_content")
    text = html.escape(_safe_str(text_val))

    parts = [f"<div class='row'><span class='meta'>#{index} sample_id={sid} position={pos} modality={mod}</span>"]
    if _safe_str(row.get("modality")) == "image":
        raw = row.get("binary_content")
        b = _to_bytes(raw)
        if b and len(b) > 0:
            scaled = _scale_image_fixed_height(b, DISPLAY_IMAGE_HEIGHT_PX)
            if scaled is not None:
                thumb_b, ct = scaled
                b64 = base64.b64encode(thumb_b).decode("ascii")
                parts.append(f'<img src="data:{ct};base64,{b64}" alt="image" />')
            else:
                parts.append("<span class='noimg'>[image not loaded or invalid]</span>")
        else:
            parts.append("<span class='noimg'>[image not loaded]</span>")
    else:
        parts.append(f"<pre class='text'>{text}</pre>")
    parts.append("</div>")
    return "\n".join(parts)


def _shared_styles() -> str:
    return """
    body { font-family: system-ui, sans-serif; margin: 1rem 2rem; }
    h1 { font-size: 1.2rem; }
    h2 { font-size: 1rem; margin-top: 1.5rem; color: #06c; }
    h3 { font-size: 0.95rem; margin-top: 1rem; }
    section.sample { margin-bottom: 2rem; padding-bottom: 1.5rem; border-bottom: 1px solid #ccc; }
    .row { margin: 0.5rem 0; padding: 0.5rem; border: 1px solid #ddd; border-radius: 4px; }
    .meta { color: #666; font-size: 0.85rem; }
    .row img { display: block; margin-top: 0.25rem; }
    .text { margin: 0.25rem 0; white-space: pre-wrap; word-break: break-word; font-size: 0.9rem; }
    .noimg { color: #999; }
    .empty { color: #999; font-style: italic; }
    .summary-table { border-collapse: collapse; margin-top: 1rem; }
    .summary-table th, .summary-table td { padding: 0.4rem 0.8rem; text-align: left; border: 1px solid #ddd; }
    .summary-table th.num, .summary-table td.num { text-align: right; }
    .summary-table a { color: #06c; }
    .back-link { display: inline-block; margin-bottom: 1rem; }
    .img-stats { margin: 0.5rem 0 1rem; padding: 0.5rem 0.75rem; background: #f5f5f5; border-radius: 4px; font-size: 0.95rem; }
    .img-stats strong { color: #333; }
"""


def _annotation_dir_sort_key(path: str) -> tuple:
    base = os.path.basename(path.rstrip(os.sep))
    m = _TRAILING_NUM_RE.search(base)
    if m:
        return (0, float(m.group(1)), path.lower())
    return (1, base.lower(), path)


def _expand_annotation_dirs(pattern: str) -> list[str]:
    matches = glob.glob(pattern)
    dirs = [os.path.abspath(m) for m in matches if os.path.isdir(m)]
    dirs.sort(key=_annotation_dir_sort_key)
    if not matches:
        msg = f"No paths matched annotation glob: {pattern!r}"
        raise SystemExit(msg)
    if not dirs:
        msg = f"Annotation glob matched {len(matches)} path(s) but none are directories: {pattern!r}"
        raise SystemExit(msg)
    return dirs


def _stage_page_extra_css() -> str:
    return """
    .stage-block { margin-bottom: 2rem; padding: 1rem; border: 1px solid #ccc; border-radius: 6px; }
    .stage-title { font-size: 1.05rem; color: #06c; margin-bottom: 0.5rem; }
    .stage-path { font-size: 0.8rem; color: #666; word-break: break-all; }
    .thumb-grid { display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: flex-start; }
    .explain { color: #555; font-size: 0.9rem; margin: 0.5rem 0 1rem; }
    """


def _build_main_summary_html(
    stage_rows: list[tuple[str, pd.DataFrame, int, int]],
    annotation_pattern: str,
    per_stage: int,
    total_input_images: int,
    input_path_display: str,
    input_source: str,
) -> str:
    style = _shared_styles()
    pat_esc = html.escape(annotation_pattern)
    in_esc = html.escape(input_path_display)
    shard_label = "WebDataset" if input_source == "webdataset" else "OmniCorpus-CC"
    table_rows: list[str] = []
    for i, (ann_dir, _df, n_total_filtered, total_keys) in enumerate(stage_rows):
        label = html.escape(os.path.basename(ann_dir.rstrip(os.sep)) or ann_dir)
        path_esc = html.escape(ann_dir)
        ratio_s = html.escape(_format_filtered_ratio(n_total_filtered, total_keys))
        link = f"/?stage={i}"
        table_rows.append(
            f"    <tr><td class='num'>{i}</td><td>{label}</td><td class='path-cell'>{path_esc}</td>"
            f"<td class='num'>{total_keys}</td><td class='num'>{n_total_filtered}</td><td class='num'>{ratio_s}</td>"
            f'<td><a href="{html.escape(link)}">View images</a></td></tr>'
        )
    tbody = "\n".join(table_rows) if table_rows else "    <tr><td colspan='7'>No stages</td></tr>"
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Multi-stage annotation — summary</title>
  <style>{style}
    .path-cell {{ font-size: 0.8rem; color: #444; max-width: 28rem; word-break: break-all; }}
  </style>
</head>
<body>
  <h1>Annotation stages — filtered summary</h1>
  <p class='img-stats'><strong>Total image rows</strong> in scanned {shard_label} tars: <strong>{total_input_images}</strong> (<code>modality=image</code>, position &gt;= 0). Source: <code>{in_esc}</code> (honors <code>--max-tars</code>; <code>--input-source</code>).</p>
  <p class='img-stats'>Filtered counts below come from each stage&rsquo;s parquet (<code>keep_mask</code>). <strong>View images</strong> links open a detail page with up to {per_stage} sample thumbnails per stage.</p>
  <p class='explain'>Glob: <code>{pat_esc}</code>. Stage index matches sort order (trailing int or float in basename, e.g. <code>a_0.1</code>, <code>a_60</code>).</p>
  <table class='summary-table' id='stage-summary'>
    <thead>
      <tr>
        <th class='num'>#</th>
        <th>Stage</th>
        <th>Path</th>
        <th class='num'>Total keys</th>
        <th class='num'>Total filtered</th>
        <th class='num'>Filtered %</th>
        <th>Images</th>
      </tr>
    </thead>
    <tbody>
{tbody}
    </tbody>
  </table>
</body>
</html>"""


def _build_stage_detail_html(
    stage_index: int,
    ann_dir: str,
    df: pd.DataFrame,
    n_total_filtered: int,
    total_keys: int,
    annotation_pattern: str,
    per_stage: int,
) -> str:
    style = _shared_styles()
    extra = _stage_page_extra_css()
    label = html.escape(os.path.basename(ann_dir.rstrip(os.sep)) or ann_dir)
    path_esc = html.escape(ann_dir)
    pat_esc = html.escape(annotation_pattern)
    ratio_s = html.escape(_format_filtered_ratio(n_total_filtered, total_keys))
    stats_esc = (
        f"<p class='img-stats'><strong>Content keys in this stage&rsquo;s parquet:</strong> total {total_keys}; "
        f"<strong>total filtered</strong> (<code>keep_mask</code> false): {n_total_filtered}; "
        f"<strong>filtered ratio</strong>: {ratio_s}</p>"
    )
    if df.empty:
        body = (
            f"<p class='empty'>No rows loaded from <code>--input-path</code> for up to {per_stage} "
            "picked keys (preview assumes image-like payloads where applicable).</p>"
        )
    else:
        thumbs = "".join(_row_to_html(row, j, "newly_filtered") for j, (_, row) in enumerate(df.iterrows(), start=1))
        body = f"<div class='thumb-grid'>{thumbs}</div>"
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Stage {stage_index}: {label}</title>
  <style>{style}{extra}</style>
</head>
<body>
  <a class='back-link' href='/'>&larr; Back to summary</a>
  <h1>Stage {stage_index}: {label}</h1>
  <div class='stage-path'>{path_esc}</div>
  <p class='explain'>Annotation glob: <code>{pat_esc}</code>. Sample rows: newly filtered vs previous stage (max {per_stage}). Images at fixed height {DISPLAY_IMAGE_HEIGHT_PX}px.</p>
  {stats_esc}
  <section class='stage-block'>
    <h2 class='stage-title'>Samples</h2>
    {body}
  </section>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-stage annotation preview: full content-key stats; sample rows from input."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Tar path or directory (format depends on --input-source; loads row payloads for preview samples)",
    )
    parser.add_argument(
        "--input-source",
        type=str,
        choices=("webdataset", "omnicorpus"),
        default="webdataset",
        help=(
            "Shard format: webdataset=MINT-style; omnicorpus=OmniCorpus-CC "
            "(.json_image_text + .images pickle via preview_annotation_standalone)."
        ),
    )
    parser.add_argument(
        "--include-general-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="OmniCorpus only: join url/safety fields from .general_metadata.pkl.",
    )
    parser.add_argument(
        "--omni-max-batch-bytes",
        type=int,
        default=None,
        help="OmniCorpus only: OmniCorpusReaderStage.max_batch_bytes (default: None).",
    )
    parser.add_argument(
        "--annotation-pattern",
        type=str,
        required=True,
        help=(
            "Glob matching one directory per stage (e.g. .../a_*). "
            "Dirs are ordered by trailing number in the name (int or float, e.g. a_60, a_0.1). Each dir is scanned for parquet."
        ),
    )
    parser.add_argument("--port", type=int, default=8080, help="HTTP port")
    parser.add_argument(
        "--storage-options-json",
        type=str,
        default=None,
        help="JSON fsspec storage options for cloud paths",
    )
    parser.add_argument(
        "--max-tars",
        type=int,
        default=None,
        metavar="N",
        help="Optional: scan at most N tar files (sorted paths). Default: all tars under --input-path.",
    )
    parser.add_argument(
        "--per-stage",
        type=int,
        default=PER_STAGE_SAMPLES,
        help=f"Max sample rows to load per stage from input (default: {PER_STAGE_SAMPLES})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Thread workers (default: {DEFAULT_WORKERS})",
    )
    args = parser.parse_args()

    read_kwargs: dict = {}
    if args.storage_options_json:
        read_kwargs["storage_options"] = json.loads(args.storage_options_json)

    workers = max(1, args.workers)
    per_stage = max(1, args.per_stage)

    annotation_dirs = _expand_annotation_dirs(args.annotation_pattern)
    ak_and_kept = [
        _load_annotation_all_and_kept_sets(d, read_kwargs.get("storage_options"), workers) for d in annotation_dirs
    ]
    all_keys_per_stage = [t[0] for t in ak_and_kept]
    kept_sets = [t[1] for t in ak_and_kept]
    if not any(all_keys_per_stage):
        print("No annotation rows found in parquets under matched directories.")
        raise SystemExit(1)

    paths = _resolve_tar_paths(args.input_path, read_kwargs, args.max_tars)
    if not paths:
        print("No tar files found under --input-path.")
        raise SystemExit(1)

    plan = _stage_newly_filtered_plan(all_keys_per_stage, kept_sets, annotation_dirs, per_stage)
    need_keys: set[tuple[str, int]] = set()
    for _, _, keys, _ in plan:
        need_keys.update(keys)

    if args.input_source == "omnicorpus":
        by_key, total_input_images = _scan_tars_for_rows_and_image_count_omnicorpus(
            paths,
            read_kwargs,
            need_keys,
            workers,
            args.include_general_metadata,
            args.omni_max_batch_bytes,
        )
    else:
        by_key, total_input_images = _scan_tars_for_rows_and_image_count(paths, read_kwargs, need_keys, workers)
    stage_pick: list[tuple[str, pd.DataFrame, int, int]] = [
        (ann_dir, _rows_dict_to_dataframe(picked, by_key), n_total_f, total_k)
        for ann_dir, n_total_f, picked, total_k in plan
    ]

    if args.input_source == "webdataset":
        to_mat = [df for _, df, _, _ in stage_pick if not df.empty]
        if to_mat:
            merged = pd.concat(to_mat, ignore_index=True)
            merged_mat = _materialize_rows_parallel(merged, read_kwargs, workers)
            rebuilt: list[tuple[str, pd.DataFrame, int, int]] = []
            start = 0
            for ann_dir, df, n_total_f, total_k in stage_pick:
                n = len(df)
                if n == 0:
                    rebuilt.append((ann_dir, df, n_total_f, total_k))
                else:
                    rebuilt.append((ann_dir, merged_mat.iloc[start : start + n].copy(), n_total_f, total_k))
                    start += n
            stage_pick = rebuilt

    main_html = _build_main_summary_html(
        stage_pick,
        args.annotation_pattern,
        per_stage,
        total_input_images,
        args.input_path,
        args.input_source,
    )
    detail_by_stage: dict[int, str] = {
        i: _build_stage_detail_html(
            i,
            ann_dir,
            df,
            n_total_f,
            total_k,
            args.annotation_pattern,
            per_stage,
        )
        for i, (ann_dir, df, n_total_f, total_k) in enumerate(stage_pick)
    }

    def handler_factory(main: str, details: dict[int, str]):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                parsed = urllib.parse.urlparse(self.path)
                req_path = parsed.path or "/"
                if req_path != "/":
                    self.send_response(404)
                    self.end_headers()
                    return
                qs = urllib.parse.parse_qs(parsed.query)
                stage_vals = qs.get("stage", [])
                if stage_vals:
                    try:
                        idx = int(stage_vals[0])
                    except ValueError:
                        self.send_response(404)
                        self.end_headers()
                        return
                    body = details.get(idx)
                    if body is None:
                        self.send_response(404)
                        self.end_headers()
                        return
                else:
                    body = main
                self.send_response(200)
                self.send_header("Content-type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(body.encode("utf-8"))

            def log_message(self, format: str, *args: object) -> None:
                print(args[0] if args else "")

        return Handler

    Handler = handler_factory(main_html, detail_by_stage)

    class ThreadingHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        daemon_threads = True

    with ThreadingHTTPServer(("", args.port), Handler) as httpd:
        print(
            f"Open http://localhost:{args.port} — {len(annotation_dirs)} stage(s), glob {args.annotation_pattern!r}; "
            f"input-source={args.input_source!r}; input-path image rows: {total_input_images}"
        )
        try:
            httpd.serve_forever()
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    main()
