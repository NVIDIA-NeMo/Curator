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

"""HTTP server: show text + images for one sample_id from WebDataset tars.

Loads only rows matching the given --sample-id. Tars are scanned in path order;
search stops after the first tar that contains that sample_id (all rows for a
sample are assumed to live in a single task / tar shard).
Same reader stack as preview_annotation_server.py (preview_annotation_standalone).
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

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
    get_all_file_paths_under,
    materialize_task_binary_content,
)

MAX_IMAGE_PX = 800
THUMBNAIL_JPEG_QUALITY = 100


def _content_mask(df: pd.DataFrame) -> pd.Series:
    return (df["modality"] != "metadata") & (df["position"] >= 0)


def _read_single_tar(
    index: int,
    tar_path: str,
    read_kwargs: dict,
    target_sample_id: str,
) -> list[pd.DataFrame]:
    """Read one tar; keep only content rows for target_sample_id (thread-local reader)."""
    reader = WebdatasetReaderStage(
        source_id_field="pdf_name",
        read_kwargs=read_kwargs,
        materialize_on_read=False,
        max_batch_bytes=None,
        json_extensions=tuple(DEFAULT_JSON_EXTENSIONS),
    )
    task = FileGroupTask(
        task_id=f"sample_preview_{index}",
        dataset_name="sample_preview",
        data=[tar_path],
        _metadata={"source_files": [tar_path]},
    )
    out = reader.process(task)
    batches = out if isinstance(out, list) else [out]
    chunks: list[pd.DataFrame] = []
    sid_match = target_sample_id  # compare as str
    for batch in batches:
        df = batch.to_pandas()
        if df.empty:
            continue
        content = _content_mask(df)
        sub = df[content]
        if sub.empty:
            continue
        sub = sub[sub["sample_id"].astype(str) == sid_match]
        if not sub.empty:
            chunks.append(sub.copy())
    return chunks


def _load_content_dataframe_for_sample(
    input_path: str,
    read_kwargs: dict | None,
    target_sample_id: str,
) -> pd.DataFrame:
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
    if not paths:
        return pd.DataFrame()

    chunks: list[pd.DataFrame] = []
    for i, tar_path in enumerate(paths):
        tar_chunks = _read_single_tar(i, tar_path, read_kwargs, target_sample_id)
        if tar_chunks:
            chunks.extend(tar_chunks)
            break

    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def _materialize_rows(df: pd.DataFrame, io_kwargs: dict | None) -> pd.DataFrame:
    if df.empty:
        return df
    task = InterleavedBatch(
        task_id="sample_preview_materialize",
        dataset_name="sample_preview",
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


def _row_to_html(row: pd.Series, index: int) -> str:
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
    h2 { font-size: 1rem; margin-top: 1rem; color: #06c; }
    section.sample { margin-bottom: 2rem; padding-bottom: 1.5rem; border-bottom: 1px solid #ccc; }
    .row { margin: 0.5rem 0; padding: 0.5rem; border: 1px solid #ddd; border-radius: 4px; }
    .meta { color: #666; font-size: 0.85rem; }
    .row img { display: block; margin-top: 0.25rem; max-width: 100%; }
    .text { margin: 0.25rem 0; white-space: pre-wrap; word-break: break-word; font-size: 0.9rem; }
    .noimg { color: #999; }
    .empty { color: #999; font-style: italic; }
"""


def _build_page_html(sid: str, sample_df: pd.DataFrame) -> str:
    if sample_df.empty:
        rows_html = "<p class='empty'>No rows for this sample_id in the scanned tars.</p>"
    else:
        ordered = sample_df.sort_values(by="position", kind="mergesort")
        rows_html = "\n".join(_row_to_html(row, i) for i, (_, row) in enumerate(ordered.iterrows(), start=1))
    sid_esc = html.escape(sid)
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Sample: {sid_esc}</title>
  <style>{_shared_styles()}</style>
</head>
<body>
  <section class="sample">
    <h2>sample_id: {sid_esc}</h2>
    <p>{len(sample_df)} content row(s), ordered by position.</p>
    {rows_html}
  </section>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Web server: show one WebDataset sample_id (text + images). "
            "Stops after the first tar that contains that sample_id."
        ),
    )
    parser.add_argument("--input-path", type=str, required=True, help="WebDataset tar path or directory")
    parser.add_argument(
        "--sample-id",
        type=str,
        required=True,
        help="sample_id to load; tars are scanned in order and reading stops once this id is found in a tar",
    )
    parser.add_argument("--port", type=int, default=8080, help="HTTP listen port")
    parser.add_argument(
        "--storage-options-json",
        type=str,
        default=None,
        help="JSON-encoded fsspec storage options for cloud paths",
    )
    default_workers = 16
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=(
            "Thread pool size for parallel materialization of row shards "
            f"(default: {default_workers}; use 1 to disable)"
        ),
    )
    args = parser.parse_args()

    read_kwargs: dict = {}
    if args.storage_options_json:
        read_kwargs["storage_options"] = json.loads(args.storage_options_json)

    if args.sample_id == "" or args.sample_id is None:
        print("Please provide a sample_id using --sample-id")
        sys.exit()

    workers = max(1, args.workers)
    target_sid = str(args.sample_id)
    df = _load_content_dataframe_for_sample(
        args.input_path,
        read_kwargs,
        target_sample_id=target_sid,
    )
    if df.empty:
        print(
            f"No rows for sample_id={target_sid!r} under --input-path "
            "(scanned tars in order until the list ended). Check --input-path and --sample-id."
        )

    df = _materialize_rows_parallel(df, read_kwargs, workers)
    page_html = _build_page_html(target_sid, df)

    def handler_factory(body: str):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                path = self.path.split("?")[0] or "/"
                if path != "/":
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(body.encode("utf-8"))

            def log_message(self, format: str, *args: object) -> None:
                print(args[0] if args else "")

        return Handler

    Handler = handler_factory(page_html)

    class ThreadingHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        daemon_threads = True

    with ThreadingHTTPServer(("", args.port), Handler) as httpd:
        print(
            f"Open http://localhost:{args.port} for sample_id={target_sid!r} "
            f"({workers} worker thread(s) for materialize)."
        )
        try:
            httpd.serve_forever()
        finally:
            print("\nShutting down.")
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    main()
