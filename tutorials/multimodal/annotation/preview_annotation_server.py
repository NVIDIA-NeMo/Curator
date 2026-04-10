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

"""Serve a web page showing kept vs filtered rows by sample_id using annotation parquet and original data.

Supports:

- **webdataset** (default): MINT-style tars via :class:`WebdatasetReaderStage` in
  ``preview_annotation_standalone`` (annotation from ``mint1t_annotation_pipeline``).
- **omnicorpus**: OmniCorpus-CC tars via :class:`OmniCorpusReaderStage` in
  ``preview_annotation_standalone`` (annotation from ``omnicorpus_annotation_pipeline``).
  Same stack as WebDataset preview: no ``nemo_curator`` or ``omni_corpus_annotation`` imports.
"""

from __future__ import annotations

import argparse
import base64
import html
import http.server
import io
import json
import socketserver
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# Sibling module (no nemo_curator); allow running from repo root or this directory.
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

try:
    from PIL import Image
except ImportError:
    Image = None

from preview_annotation_standalone import (  # noqa: E402
    DEFAULT_JSON_EXTENSIONS,
    DEFAULT_WEBDATASET_EXTENSIONS,
    FileGroupTask,
    InterleavedBatch,
    WebdatasetReaderStage,
    collect_omnicorpus_kept_filtered_chunks,
    get_all_file_paths_under,
    materialize_task_binary_content,
)

LIMIT_SAMPLES = 50
MAX_TARS = 30
# Scale images to longest side <= this before embedding (avoids huge data URIs).
MAX_IMAGE_PX = 800
# JPEG quality for scaled thumbnails (1-100).
THUMBNAIL_JPEG_QUALITY = 100


def _content_mask(df: pd.DataFrame) -> pd.Series:
    return (df["modality"] != "metadata") & (df["position"] >= 0)


def _read_one_annotation_parquet(path: str, storage_options: dict | None) -> pd.DataFrame | None:
    try:
        df = pd.read_parquet(path, storage_options=storage_options)
    except Exception:
        df = pd.read_parquet(path)
    if "sample_id" not in df.columns or "position" not in df.columns:
        return None
    cols = ["sample_id", "position"]
    if "keep_mask" in df.columns:
        cols.append("keep_mask")
    return df[cols]


def _combine_annotation_parquets(annotation_path: str, storage_options: dict | None, num_workers: int) -> pd.DataFrame:
    """Concatenate all annotation parquet files under ``annotation_path``."""
    paths = get_all_file_paths_under(
        annotation_path,
        recurse_subdirectories=True,
        keep_extensions=[".parquet"],
        storage_options=storage_options,
    )
    if not paths:
        return pd.DataFrame()
    dfs: list[pd.DataFrame] = []
    if num_workers <= 1:
        for p in paths:
            part = _read_one_annotation_parquet(p, storage_options)
            if part is not None:
                dfs.append(part)
    else:
        w = min(num_workers, len(paths))

        def _read_annotation(p: str) -> pd.DataFrame | None:
            return _read_one_annotation_parquet(p, storage_options)

        with ThreadPoolExecutor(max_workers=w) as ex:
            for part in ex.map(_read_annotation, paths):
                if part is not None:
                    dfs.append(part)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def _load_kept_set(annotation_path: str, storage_options: dict | None, num_workers: int) -> set[tuple[str, int]]:
    """Load annotation parquet files and return set of (sample_id, position) for kept rows.

    If ``keep_mask`` is present, only rows with True are kept. Legacy parquets without
    ``keep_mask`` treat every row as kept.
    """
    combined = _combine_annotation_parquets(annotation_path, storage_options, num_workers)
    if combined.empty:
        return set()
    if "keep_mask" in combined.columns:
        combined = combined.loc[combined["keep_mask"].fillna(False).astype(bool)]
    return set(zip(combined["sample_id"].astype(str), combined["position"].astype(int), strict=True))


def _load_annotation_all_and_kept_sets(
    annotation_path: str,
    storage_options: dict | None,
    num_workers: int,
) -> tuple[set[tuple[str, int]], set[tuple[str, int]]]:
    """Return (all content keys in parquet, kept keys) without reading WebDataset input.

    Legacy parquets without ``keep_mask``: all keys are treated as kept only; all_keys == kept_keys.
    """
    combined = _combine_annotation_parquets(annotation_path, storage_options, num_workers)
    if combined.empty:
        return set(), set()
    sid = combined["sample_id"].astype(str)
    pos = combined["position"].astype(int)
    all_keys = set(zip(sid, pos, strict=True))
    if "keep_mask" in combined.columns:
        km = combined["keep_mask"].fillna(False).astype(bool)
        kept = set(zip(sid[km], pos[km], strict=True))
    else:
        kept = all_keys
    return all_keys, kept


def _read_one_tar_kept_filtered(
    index: int,
    tar_path: str,
    read_kwargs: dict,
    kept_set: set[tuple[str, int]],
) -> tuple[int, pd.DataFrame, pd.DataFrame]:
    """Parse one tar into kept / filtered content rows (thread-local reader)."""
    reader = WebdatasetReaderStage(
        source_id_field="pdf_name",
        read_kwargs=read_kwargs,
        materialize_on_read=False,
        max_batch_bytes=None,
        json_extensions=tuple(DEFAULT_JSON_EXTENSIONS),
    )
    task = FileGroupTask(
        task_id=f"preview_{index}",
        dataset_name="preview",
        data=[tar_path],
        _metadata={"source_files": [tar_path]},
    )
    out = reader.process(task)
    batches = out if isinstance(out, list) else [out]
    kept_parts: list[pd.DataFrame] = []
    filtered_parts: list[pd.DataFrame] = []
    for batch in batches:
        df = batch.to_pandas()
        if df.empty:
            continue
        content = _content_mask(df)
        row_keys = list(zip(df["sample_id"].astype(str), df["position"].astype(int), strict=True))
        in_kept = pd.Series([k in kept_set for k in row_keys], index=df.index)
        kept_parts.append(df[content & in_kept])
        filtered_parts.append(df[content & ~in_kept])
    k = pd.concat(kept_parts, ignore_index=True) if kept_parts else pd.DataFrame()
    f = pd.concat(filtered_parts, ignore_index=True) if filtered_parts else pd.DataFrame()
    return index, k, f


def _collect_kept_and_filtered(
    input_path: str,
    annotation_path: str,
    read_kwargs: dict | None,
    max_tars: int,
    limit_samples: int,
    num_workers: int,
    *,
    input_source: str = "webdataset",
    include_general_metadata: bool = True,
    omni_max_batch_bytes: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    read_kwargs = read_kwargs or {}
    kept_set = _load_kept_set(annotation_path, read_kwargs.get("storage_options"), num_workers)

    if isinstance(input_path, str):
        paths = get_all_file_paths_under(
            input_path,
            recurse_subdirectories=True,
            keep_extensions=list(DEFAULT_WEBDATASET_EXTENSIONS),
            storage_options=read_kwargs.get("storage_options"),
        )
    else:
        paths = list(input_path)
    paths = paths[:max_tars]
    if not paths:
        return pd.DataFrame(), pd.DataFrame()

    kept_chunks: list[pd.DataFrame] = []
    filtered_chunks: list[pd.DataFrame] = []

    if input_source == "omnicorpus":
        kept_chunks, filtered_chunks = collect_omnicorpus_kept_filtered_chunks(
            paths,
            read_kwargs,
            kept_set,
            limit_samples,
            num_workers,
            include_general_metadata,
            omni_max_batch_bytes,
        )
    elif num_workers <= 1:
        reader = WebdatasetReaderStage(
            source_id_field="pdf_name",
            read_kwargs=read_kwargs,
            materialize_on_read=False,
            max_batch_bytes=None,
            json_extensions=tuple(DEFAULT_JSON_EXTENSIONS),
        )

        for i, tar_path in enumerate(paths):
            task = FileGroupTask(
                task_id=f"preview_{i}",
                dataset_name="preview",
                data=[tar_path],
                _metadata={"source_files": [tar_path]},
            )
            out = reader.process(task)
            batches = out if isinstance(out, list) else [out]
            for batch in batches:
                df = batch.to_pandas()
                if df.empty:
                    continue
                content = _content_mask(df)
                row_keys = list(zip(df["sample_id"].astype(str), df["position"].astype(int), strict=True))
                in_kept = pd.Series([k in kept_set for k in row_keys], index=df.index)
                kept_content = df[content & in_kept]
                filtered_content = df[content & ~in_kept]
                kept_chunks.append(kept_content)
                filtered_chunks.append(filtered_content)

            kept_df_so_far = pd.concat(kept_chunks, ignore_index=True) if kept_chunks else pd.DataFrame()
            filtered_df_so_far = pd.concat(filtered_chunks, ignore_index=True) if filtered_chunks else pd.DataFrame()
            if kept_df_so_far.empty and filtered_df_so_far.empty:
                continue
            all_sids = sorted(
                pd.unique(
                    list(kept_df_so_far["sample_id"].dropna().astype(str))
                    + list(filtered_df_so_far["sample_id"].dropna().astype(str))
                ).tolist()
            )
            all_kept_sids = [
                s for s in all_sids if len(filtered_df_so_far[filtered_df_so_far["sample_id"].astype(str) == s]) == 0
            ]
            has_filtered_sids = [
                s for s in all_sids if len(filtered_df_so_far[filtered_df_so_far["sample_id"].astype(str) == s]) > 0
            ]
            if len(all_kept_sids) >= limit_samples and len(has_filtered_sids) >= limit_samples:
                break
    else:
        workers = min(num_workers, len(paths))
        per_index: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(_read_one_tar_kept_filtered, i, tar_path, read_kwargs, kept_set): i
                for i, tar_path in enumerate(paths)
            }
            for fut in as_completed(futures):
                idx, k, f = fut.result()
                per_index[idx] = (k, f)
        for i in range(len(paths)):
            k, f = per_index.get(i, (pd.DataFrame(), pd.DataFrame()))
            kept_chunks.append(k)
            filtered_chunks.append(f)

    kept_df = pd.concat(kept_chunks, ignore_index=True) if kept_chunks else pd.DataFrame()
    filtered_df = pd.concat(filtered_chunks, ignore_index=True) if filtered_chunks else pd.DataFrame()
    all_sids = sorted(
        pd.unique(
            list(kept_df["sample_id"].dropna().astype(str)) + list(filtered_df["sample_id"].dropna().astype(str))
        ).tolist()
    )
    all_kept_sids = [s for s in all_sids if len(filtered_df[filtered_df["sample_id"].astype(str) == s]) == 0]
    has_filtered_sids = [s for s in all_sids if len(filtered_df[filtered_df["sample_id"].astype(str) == s]) > 0]
    samples_to_keep = list(dict.fromkeys(all_kept_sids[:limit_samples] + has_filtered_sids[:limit_samples]))
    if samples_to_keep:
        kept_df = kept_df[kept_df["sample_id"].astype(str).isin(samples_to_keep)]
        filtered_df = filtered_df[filtered_df["sample_id"].astype(str).isin(samples_to_keep)]
    return kept_df, filtered_df


def _materialize_rows(df: pd.DataFrame, io_kwargs: dict | None) -> pd.DataFrame:
    if df.empty:
        return df
    task = InterleavedBatch(
        task_id="preview_materialize",
        dataset_name="preview",
        data=df,
        _metadata={},
    )
    out = materialize_task_binary_content(task, only_missing_binary=True, io_kwargs=io_kwargs)
    return out.to_pandas()


def _split_dataframe(df: pd.DataFrame, n: int) -> list[pd.DataFrame]:
    if n <= 1 or len(df) <= 1:
        return [df]
    n = min(n, len(df))
    base, rem = divmod(len(df), n)
    parts: list[pd.DataFrame] = []
    start = 0
    for i in range(n):
        sz = base + (1 if i < rem else 0)
        if sz:
            parts.append(df.iloc[start : start + sz].copy())
            start += sz
    return parts if parts else [df]


def _materialize_rows_parallel(df: pd.DataFrame, io_kwargs: dict | None, num_workers: int) -> pd.DataFrame:
    if df.empty or num_workers <= 1:
        return _materialize_rows(df, io_kwargs)
    parts = _split_dataframe(df, num_workers)
    if len(parts) <= 1:
        return _materialize_rows(df, io_kwargs)
    workers = min(num_workers, len(parts))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_materialize_rows, p, io_kwargs) for p in parts]
        merged = [f.result() for f in futures]
    return pd.concat(merged, ignore_index=True)


def _safe_str(val: object) -> str:
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val)


def _to_bytes(raw: object) -> bytes | None:
    """Convert cell value to bytes for data URI. Handles pandas/pyarrow/numpy types."""
    if raw is None:
        return None
    try:
        if pd.isna(raw):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(raw, bytes):
        return raw
    if isinstance(raw, bytearray):
        return bytes(raw)
    if isinstance(raw, memoryview):
        return raw.tobytes()
    try:
        if hasattr(raw, "as_py"):
            raw = raw.as_py()
        if hasattr(raw, "tobytes"):
            return raw.tobytes()
    except Exception:
        pass
    try:
        return bytes(raw)
    except (TypeError, ValueError):
        return None


def _scale_image_for_display(b: bytes) -> tuple[bytes, str] | None:
    """Load image, scale down to max MAX_IMAGE_PX on longest side, return (jpeg_bytes, 'image/jpeg')."""
    if Image is None:
        return None
    try:
        with Image.open(io.BytesIO(b)) as img:
            img = img.convert("RGB")
            img.thumbnail((MAX_IMAGE_PX, MAX_IMAGE_PX), getattr(Image, "Resampling", Image).LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=THUMBNAIL_JPEG_QUALITY, optimize=True)
            return buf.getvalue(), "image/jpeg"
    except (OSError, ValueError, TypeError):
        return None


def _row_to_html(row: pd.Series, index: int, kind: str) -> str:
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
            scaled = _scale_image_for_display(b)
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
    h3.kept { color: #0a0; }
    h3.filtered { color: #c00; }
    section.sample { margin-bottom: 2rem; padding-bottom: 1.5rem; border-bottom: 1px solid #ccc; }
    .row { margin: 0.5rem 0; padding: 0.5rem; border: 1px solid #ddd; border-radius: 4px; }
    .meta { color: #666; font-size: 0.85rem; }
    .row img { display: block; margin-top: 0.25rem; }
    .text { margin: 0.25rem 0; white-space: pre-wrap; word-break: break-word; font-size: 0.9rem; }
    .noimg { color: #999; }
    .empty { color: #999; font-style: italic; }
    .summary-table { border-collapse: collapse; margin-top: 1rem; }
    .summary-table th, .summary-table td { padding: 0.4rem 0.8rem; text-align: left; border: 1px solid #ddd; }
    .summary-table a { color: #06c; }
    .back-link { display: inline-block; margin-bottom: 1rem; }
    .section-head { margin-top: 1.5rem; color: #06c; }
    .section-head.filtered { color: #c00; }
    .section-desc { color: #666; font-size: 0.9rem; margin-top: 0.25rem; }
"""


def _build_detail_html(sid: str, k: pd.DataFrame, f: pd.DataFrame) -> str:
    kept_rows = "".join(_row_to_html(row, i, "kept") for i, (_, row) in enumerate(k.iterrows(), start=1))
    filtered_rows = "".join(_row_to_html(row, i, "filtered") for i, (_, row) in enumerate(f.iterrows(), start=1))
    sid_esc = html.escape(sid)
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Sample: {sid_esc}</title>
  <style>{_shared_styles()}</style>
</head>
<body>
  <a class="back-link" href="/">&larr; Back to summary</a>
  <section class="sample">
    <h2>sample_id: {sid_esc}</h2>
    <h3 class="kept">Kept ({len(k)} rows)</h3>
    {kept_rows if not k.empty else "<p class='empty'>none</p>"}
    <h3 class="filtered">Filtered ({len(f)} rows)</h3>
    {filtered_rows if not f.empty else "<p class='empty'>none</p>"}
  </section>
</body>
</html>"""


def _build_main_html(kept_df: pd.DataFrame, filtered_df: pd.DataFrame) -> str:
    sample_ids = sorted(
        pd.unique(
            list(kept_df["sample_id"].dropna().astype(str)) + list(filtered_df["sample_id"].dropna().astype(str))
        ).tolist()
    )
    all_kept: list[str] = []
    has_filtered: list[str] = []
    for sid in sample_ids:
        kept_df[kept_df["sample_id"].astype(str) == sid]
        f = filtered_df[filtered_df["sample_id"].astype(str) == sid]
        n_f = len(f)
        if n_f == 0:
            all_kept.append(sid)
        else:
            has_filtered.append(sid)

    def table_rows(sids: list[str]) -> str:
        out = []
        for sid in sids:
            k = kept_df[kept_df["sample_id"].astype(str) == sid]
            f = filtered_df[filtered_df["sample_id"].astype(str) == sid]
            sid_esc = html.escape(sid)
            link = "/?sample=" + urllib.parse.quote(sid, safe="")
            out.append(
                f'    <tr><td>{sid_esc}</td><td>{len(k)}</td><td>{len(f)}</td><td><a href="{html.escape(link)}">Details</a></td></tr>'
            )
        return "\n".join(out) if out else "    <tr><td colspan='4'>None</td></tr>"

    all_kept_body = table_rows(all_kept)
    has_filtered_body = table_rows(has_filtered)

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Annotation preview – by sample_id</title>
  <style>{_shared_styles()}</style>
</head>
<body>
  <h1>Preview: {len(sample_ids)} samples ({len(kept_df)} kept, {len(filtered_df)} filtered rows)</h1>
  <p>Kept/filtered from annotation-path (sample_id, position). Data from original shards. Click Details for per-sample rows.</p>
  <h2 class="section-head">Samples with all kept ({len(all_kept)} samples)</h2>
  <p class="section-desc">No rows filtered out for these samples.</p>
  <table class="summary-table">
    <thead><tr><th>sample_id</th><th>kept</th><th>filtered</th><th></th></tr></thead>
    <tbody>
{all_kept_body}
    </tbody>
  </table>
  <h2 class="section-head filtered">Samples with filtered ({len(has_filtered)} samples)</h2>
  <p class="section-desc">At least one row was filtered out for these samples.</p>
  <table class="summary-table">
    <thead><tr><th>sample_id</th><th>kept</th><th>filtered</th><th></th></tr></thead>
    <tbody>
{has_filtered_body}
    </tbody>
  </table>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Web server showing kept vs filtered rows by sample_id (default: 10 samples)"
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Original tar path or directory (format depends on --input-source)",
    )
    parser.add_argument(
        "--annotation-path",
        type=str,
        required=True,
        help="Annotation parquet dir (from mint1t_annotation_pipeline or omnicorpus_annotation_pipeline)",
    )
    parser.add_argument(
        "--input-source",
        type=str,
        choices=("webdataset", "omnicorpus"),
        default="webdataset",
        help=(
            "Shard format: webdataset=MINT-style json+image members; "
            "omnicorpus=OmniCorpus-CC (.json_image_text + .images pickle via preview_annotation_standalone)."
        ),
    )
    parser.add_argument(
        "--include-general-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="OmniCorpus only: join url/safety fields from .general_metadata.pkl (matches omnicorpus_annotation_pipeline default).",
    )
    parser.add_argument(
        "--omni-max-batch-bytes",
        type=int,
        default=None,
        help="OmniCorpus only: passed to OmniCorpusReaderStage.max_batch_bytes (default: None = one batch per tar).",
    )
    parser.add_argument("--port", type=int, default=8080, help="Port for HTTP server")
    parser.add_argument(
        "--storage-options-json",
        type=str,
        default=None,
        help="JSON-encoded fsspec storage options for cloud paths",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=LIMIT_SAMPLES,
        help="Max samples per list: all-kept and with-filtered (default: 50)",
    )
    parser.add_argument("--max-tars", type=int, default=MAX_TARS, help="Max tar files to scan")
    default_workers = 16
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=(
            "Thread pool for annotation parquet read, tar read, materialize shards, and detail HTML "
            f"(default: {default_workers}; use 1 for sequential tar scan with early stop)"
        ),
    )
    args = parser.parse_args()

    read_kwargs = {}
    if args.storage_options_json:
        read_kwargs["storage_options"] = json.loads(args.storage_options_json)

    workers = max(1, args.workers)
    kept_df, filtered_df = _collect_kept_and_filtered(
        args.input_path,
        args.annotation_path,
        read_kwargs,
        max_tars=args.max_tars,
        limit_samples=args.max_samples,
        num_workers=workers,
        input_source=args.input_source,
        include_general_metadata=args.include_general_metadata,
        omni_max_batch_bytes=args.omni_max_batch_bytes,
    )
    if kept_df.empty and filtered_df.empty:
        hint = (
            "OmniCorpus shards (.json_image_text)."
            if args.input_source == "omnicorpus"
            else "WebDataset samples (json + image members)."
        )
        print(f"No content rows found. Check --input-path and that tars contain {hint}")
        return

    if args.input_source == "webdataset":
        kept_df = _materialize_rows_parallel(kept_df, read_kwargs, workers)
        filtered_df = _materialize_rows_parallel(filtered_df, read_kwargs, workers)

    main_html = _build_main_html(kept_df, filtered_df)
    sample_ids = sorted(
        pd.unique(
            list(kept_df["sample_id"].dropna().astype(str)) + list(filtered_df["sample_id"].dropna().astype(str))
        ).tolist()
    )
    detail_html_by_id: dict[str, str] = {}
    if workers <= 1:
        for sid in sample_ids:
            k = kept_df[kept_df["sample_id"].astype(str) == sid]
            f = filtered_df[filtered_df["sample_id"].astype(str) == sid]
            detail_html_by_id[sid] = _build_detail_html(sid, k, f)
    else:
        w = min(workers, max(1, len(sample_ids)))
        work: list[tuple[str, pd.DataFrame, pd.DataFrame]] = [
            (
                sid,
                kept_df[kept_df["sample_id"].astype(str) == sid].copy(),
                filtered_df[filtered_df["sample_id"].astype(str) == sid].copy(),
            )
            for sid in sample_ids
        ]

        def _detail_from_triple(triple: tuple[str, pd.DataFrame, pd.DataFrame]) -> tuple[str, str]:
            sid, k, f = triple
            return sid, _build_detail_html(sid, k, f)

        with ThreadPoolExecutor(max_workers=w) as ex:
            for sid, page in ex.map(_detail_from_triple, work):
                detail_html_by_id[sid] = page

    def handler_factory(main: str, details: dict[str, str]):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                path = self.path.split("?")[0] or "/"
                query = self.path.split("?", 1)[1] if "?" in self.path else ""
                params = urllib.parse.parse_qs(query)
                sample_id = params.get("sample", [None])[0]
                if path != "/":
                    self.send_response(404)
                    self.end_headers()
                    return
                if sample_id is not None:
                    sample_id = urllib.parse.unquote(sample_id)
                    body = details.get(sample_id)
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

    Handler = handler_factory(main_html, detail_html_by_id)

    class ThreadingHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        daemon_threads = True

    with ThreadingHTTPServer(("", args.port), Handler) as httpd:
        print(f"Open http://localhost:{args.port} for preview ({workers} load worker thread(s)).")
        try:
            httpd.serve_forever()
        finally:
            print("\nShutting down.")
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    main()
