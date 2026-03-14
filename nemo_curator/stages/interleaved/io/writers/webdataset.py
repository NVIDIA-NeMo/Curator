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

from __future__ import annotations

import io
import json
import mimetypes
import tarfile
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import fsspec
import pandas as pd

from nemo_curator.tasks.interleaved import RESERVED_COLUMNS

from .base import BaseInterleavedWriter

if TYPE_CHECKING:
    from nemo_curator.tasks import InterleavedBatch

_MIME_TO_EXT: dict[str, str] = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/tiff": "tiff",
    "image/gif": "gif",
    "image/webp": "webp",
    "image/bmp": "bmp",
}


def _escape_key(raw: str) -> str:
    """Produce a safe WebDataset tar key from a sample_id.

    WDS readers group tar members by stem (``Path(member.name).stem``), so
    the key must not contain characters that break this convention:

    - ``.`` -- WDS key/extension separator
    - ``/``, ``\\`` -- path separators that create nested tar members
    - ``:`` -- invalid on Windows filesystems

    All unsafe characters are percent-encoded (e.g. ``.`` -> ``%2E``).

    Note: ``%`` itself is NOT escaped, so collisions are theoretically possible
    (e.g. ``"a.b"`` and ``"a%2Eb"`` both map to ``"a%2Eb"``), but rare in
    practice with real sample IDs.
    """
    out: list[str] = []
    for ch in raw:
        if ch in (".", "/", "\\", ":"):
            out.append(f"%{ord(ch):02X}")
        else:
            out.append(ch)
    return "".join(out)


def _ext_from_content_type(content_type: object) -> str:
    if isinstance(content_type, str) and content_type in _MIME_TO_EXT:
        return _MIME_TO_EXT[content_type]
    if isinstance(content_type, str):
        ext = mimetypes.guess_extension(content_type, strict=False)
        if ext:
            return ext.lstrip(".")
    return "bin"


def _add_tar_member(tf: tarfile.TarFile, name: str, data: bytes, mtime: float) -> None:
    ti = tarfile.TarInfo(name=name)
    ti.size = len(data)
    ti.mtime = mtime
    ti.mode = 0o0444
    ti.uname = "bigdata"
    ti.gname = "bigdata"
    tf.addfile(ti, io.BytesIO(data))


def _is_valid_binary(val: object) -> bool:
    """Return True if *val* is non-null binary content."""
    if val is None:
        return False
    if isinstance(val, (bytes, bytearray)):
        return True
    try:
        return not pd.isna(val)
    except (TypeError, ValueError):
        return False


def _build_index(sid_col: list | pd.Series) -> list[tuple[str, list[int]]]:
    """Return (sample_id, row_indices) pairs in first-occurrence order."""
    sid_to_indices: dict[str, list[int]] = {}
    insertion_order: list[str] = []
    for i, raw in enumerate(sid_col):
        sid = str(raw)
        if sid not in sid_to_indices:
            sid_to_indices[sid] = []
            insertion_order.append(sid)
        sid_to_indices[sid].append(i)
    return [(sid, sid_to_indices[sid]) for sid in insertion_order]


def _has_value(val: object) -> bool:
    """Return True if *val* is non-null."""
    if val is None:
        return False
    try:
        return not pd.isna(val)
    except (TypeError, ValueError):
        return True


def _safe_json_value(val: object) -> object:
    """Convert a stored value back to a JSON-serializable Python object.

    The reader stores dicts/lists as JSON strings; this reverses that encoding
    so the output JSON matches the original structure.
    """
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(val, (bytes, bytearray)):
        return None
    if isinstance(val, (dict, list, int, float, bool)):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, (dict, list)):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return val if isinstance(val, str) else str(val)


def _to_text_value(v: object) -> str | None:
    """Coerce a text_content value to str or None, handling pyarrow NA types."""
    if isinstance(v, str):
        return v
    if v is None:
        return None
    try:
        return None if pd.isna(v) else str(v)
    except (TypeError, ValueError):
        return str(v)


class _SampleData:
    """Intermediate representation of a single WDS sample's row-level data."""

    __slots__ = ("image_at_pos", "image_extra", "payload", "text_at_pos", "text_extra")

    def __init__(self) -> None:
        self.payload: dict[str, Any] = {}
        self.text_at_pos: dict[int, object] = {}
        self.image_at_pos: dict[int, tuple[object, object]] = {}
        self.image_extra: dict[int, dict[str, Any]] = {}
        self.text_extra: dict[int, dict[str, Any]] = {}


def _rebuild_per_modality_lists(
    payload: dict[str, Any],
    modality_positions: dict[int, Any],
    extras_at_pos: dict[int, dict[str, Any]],
) -> None:
    """Reconstruct per-modality fields as parallel lists in the JSON payload.

    Each list is indexed 1:1 with the non-None entries for that modality
    (sorted by position), matching the original WDS convention.
    """
    if not extras_at_pos:
        return
    field_names: set[str] = set()
    for ext in extras_at_pos.values():
        field_names.update(ext)
    for name in sorted(field_names):
        values = [extras_at_pos.get(pos, {}).get(name) for pos in sorted(modality_positions)]
        payload[name] = values


class _TarWriteContext:
    """Pre-extracted column data for fast O(1) per-row access during tar writing."""

    __slots__ = ("cols", "extra_col_names", "mtime")

    def __init__(self, cols: dict[str, list], extra_col_names: list[str], mtime: float) -> None:
        self.cols = cols
        self.extra_col_names = extra_col_names
        self.mtime = mtime


def _collect_sample_from_cols(indices: list[int], ctx: _TarWriteContext) -> _SampleData:
    """Collect row data for one sample using pre-extracted column lists (O(1) per row)."""
    cols, extra_col_names = ctx.cols, ctx.extra_col_names
    sd = _SampleData()
    for i in indices:
        mod = cols["modality"][i]
        pos = cols["position"][i]
        if mod == "metadata":
            sd.payload.update({n: _safe_json_value(cols[n][i]) for n in extra_col_names} if extra_col_names else {})
        elif mod == "text":
            sd.text_at_pos[pos] = cols["text_content"][i]
            ext = {n: _safe_json_value(cols[n][i]) for n in extra_col_names if _has_value(cols[n][i])}
            if ext:
                sd.text_extra[pos] = ext
        elif mod == "image":
            sd.image_at_pos[pos] = (cols["binary_content"][i], cols["content_type"][i])
            ext = {n: _safe_json_value(cols[n][i]) for n in extra_col_names if _has_value(cols[n][i])}
            if ext:
                sd.image_extra[pos] = ext
        else:
            msg = f"Unsupported modality '{mod}'. Supported modalities: 'metadata', 'text', 'image'"
            raise ValueError(msg)
    return sd


def _write_sample_from_cols(
    tf: tarfile.TarFile,
    key: str,
    indices: list[int],
    ctx: _TarWriteContext,
) -> None:
    """Write one sample using pre-extracted column lists (O(1) per-row access)."""
    sd = _collect_sample_from_cols(indices, ctx)

    all_positions = set(sd.text_at_pos) | set(sd.image_at_pos)
    n = max(all_positions) + 1 if all_positions else 0
    texts: list[str | None] = [None] * n
    images: list[str | None] = [None] * n
    binaries: list[tuple[str, bytes]] = []

    for pos in range(n):
        if pos in sd.text_at_pos:
            texts[pos] = _to_text_value(sd.text_at_pos[pos])
        if pos in sd.image_at_pos:
            binary, content_type = sd.image_at_pos[pos]
            ext = _ext_from_content_type(content_type)
            images[pos] = f"{key}.{pos}.{ext}"
            if _is_valid_binary(binary):
                binaries.append((f"{key}.{pos}.{ext}", bytes(binary)))

    _rebuild_per_modality_lists(sd.payload, sd.image_at_pos, sd.image_extra)
    _rebuild_per_modality_lists(sd.payload, sd.text_at_pos, sd.text_extra)
    sd.payload["texts"] = texts
    sd.payload["images"] = images
    json_bytes = json.dumps(sd.payload, ensure_ascii=True).encode("utf-8")
    _add_tar_member(tf, f"{key}.json", json_bytes, ctx.mtime)
    for member_name, binary_data in binaries:
        _add_tar_member(tf, member_name, binary_data, ctx.mtime)


@dataclass
class InterleavedWebdatasetWriterStage(BaseInterleavedWriter):
    """Write interleaved rows to WebDataset tar shards."""

    file_extension: str = "tar"
    name: str = "interleaved_webdataset_writer"

    def _write_dataframe(self, df: pd.DataFrame, file_path: str, write_kwargs: dict[str, Any]) -> None:
        pass

    def write_data(self, task: InterleavedBatch, file_path: str) -> None:
        with self._time_metric("materialize_dataframe_total_s"):
            df = self._materialize_dataframe(task)
        df = self._align_output(df)

        with self._time_metric("webdataset_write_s"):
            self._write_tar(df, file_path)

    def _write_tar(self, df: pd.DataFrame, file_path: str) -> None:
        import pyarrow as pa

        mtime = time.time()
        samples_written = 0

        table = pa.Table.from_pandas(df, preserve_index=False)
        col_names = table.schema.names
        extra_col_indices = [i for i, n in enumerate(col_names) if n not in RESERVED_COLUMNS]
        extra_col_names = [col_names[i] for i in extra_col_indices]

        cols = {name: table.column(name).to_pylist() for name in col_names}
        sample_index = _build_index(cols["sample_id"])

        with (
            fsspec.open(file_path, mode="wb", **self.storage_options) as fobj,
            tarfile.open(fileobj=fobj, mode="w") as tf,
        ):
            ctx = _TarWriteContext(cols, extra_col_names, mtime)
            for sid, indices in sample_index:
                _write_sample_from_cols(tf, _escape_key(sid), indices, ctx)
                samples_written += 1

        self._log_metric("samples_written", float(samples_written))
