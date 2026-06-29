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

import json
import mimetypes
import tarfile
import urllib.parse
from dataclasses import dataclass
from io import BytesIO
from typing import Any, ClassVar

import fsspec
import pandas as pd

from nemo_curator.tasks.interleaved import RESERVED_COLUMNS

from .base import BaseInterleavedWriter

# ---------------------------------------------------------------------------
# Module-level helpers (importable in tests)
# ---------------------------------------------------------------------------

_CONTENT_TYPE_TO_EXT: dict[str, str] = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/tiff": "tiff",
    "image/webp": "webp",
    "image/gif": "gif",
    "image/bmp": "bmp",
    "image/avif": "avif",
}


def _escape_key(sample_id: str) -> str:
    """Percent-encode a sample_id so it is safe as a tar member name stem."""
    return urllib.parse.quote(sample_id, safe="")


def _ext_from_content_type(content_type: str | None) -> str:
    """Return a file extension for *content_type*, falling back to ``"bin"``."""
    if content_type:
        ext = _CONTENT_TYPE_TO_EXT.get(content_type)
        if ext:
            return ext
        guessed = mimetypes.guess_extension(content_type, strict=False)
        if guessed:
            return guessed.lstrip(".")
    return "bin"


def _is_null(value: object) -> bool:
    """Return True for Python/pandas null-ish scalars."""
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    try:
        # Handles pd.NA, pd.NaT, and ArrowDtype scalars
        return pd.isna(value)
    except (TypeError, ValueError):
        return False


def _series_to_nullable_list(series: pd.Series) -> list:
    """Return a Python list from *series*, replacing any null-like value with ``None``."""
    na_mask = series.isna()
    return [None if is_na else v for is_na, v in zip(na_mask, series, strict=True)]


def _per_modality_passthrough(
    passthrough_cols: list[str],
    meta_keys: set[str],
    image_rows: pd.DataFrame,
    text_rows: pd.DataFrame,
    content_rows: pd.DataFrame,
) -> dict[str, Any]:
    """Build per-modality passthrough lists from content rows.

    For each passthrough column not already in *meta_keys*:

    * **Pure per-image** (non-null only in image rows): emitted as a list with
      one entry per image row (None preserved for sparse nulls).
    * **Pure per-text** (non-null only in text rows): emitted as a list with
      one entry per text row (None preserved for sparse nulls).
    * **Mixed** (non-null in both image and text rows): emitted as a
      position-aligned list with one entry per content row in position order
      (None where the value is absent).  When read back without declaring the
      field in ``per_image_fields`` / ``per_text_fields``, the reader treats
      the entire list as a sample-level passthrough on the metadata row.
    """
    result: dict[str, Any] = {}
    for col in passthrough_cols:
        if col in meta_keys:
            continue
        image_has_values = not image_rows.empty and not image_rows[col].isna().all()
        text_has_values = not text_rows.empty and not text_rows[col].isna().all()

        if image_has_values and not text_has_values:
            result[col] = _series_to_nullable_list(image_rows[col])
        elif text_has_values and not image_has_values:
            result[col] = _series_to_nullable_list(text_rows[col])
        elif image_has_values and text_has_values:
            result[col] = _series_to_nullable_list(content_rows[col])
    return result


def _write_sample(
    tf: tarfile.TarFile,
    sample_df: pd.DataFrame,
    sample_id: str,
    passthrough_cols: list[str],
) -> None:
    """Write one sample (JSON + image binaries) into *tf*."""
    escaped = _escape_key(sample_id)
    content_rows = sample_df[sample_df["position"] >= 0].sort_values("position")
    max_pos = int(content_rows["position"].max()) if not content_rows.empty else -1
    texts: list[str | None] = [None] * (max_pos + 1)
    images: list[str | None] = [None] * (max_pos + 1)

    for row in content_rows.itertuples(index=False):
        pos = int(row.position)
        if row.modality == "text":
            texts[pos] = row.text_content
        elif row.modality == "image":
            ext = _ext_from_content_type(row.content_type)
            member_name = f"{escaped}.{pos}.{ext}"
            images[pos] = member_name
            raw = row.binary_content
            if not _is_null(raw):
                img_bytes = bytes(raw)
                info = tarfile.TarInfo(name=member_name)
                info.size = len(img_bytes)
                tf.addfile(info, BytesIO(img_bytes))

    passthrough: dict[str, Any] = {}
    meta_rows = sample_df[sample_df["position"] == -1]
    if not meta_rows.empty:
        meta_row = meta_rows.iloc[0]
        for col in passthrough_cols:
            val = meta_row[col]
            if not _is_null(val):
                passthrough[col] = val

    image_rows = content_rows[content_rows["modality"] == "image"]
    text_rows = content_rows[content_rows["modality"] == "text"]
    passthrough.update(
        _per_modality_passthrough(passthrough_cols, set(passthrough), image_rows, text_rows, content_rows)
    )

    payload = {"sample_id": sample_id, "texts": texts, "images": images, **passthrough}
    json_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    info = tarfile.TarInfo(name=f"{escaped}.json")
    info.size = len(json_bytes)
    tf.addfile(info, BytesIO(json_bytes))


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


@dataclass
class InterleavedWebdatasetWriterStage(BaseInterleavedWriter):
    """Write an ``InterleavedBatch`` as a MINT-1T-style WebDataset tar shard.

    Each sample is reconstructed from its row-based representation:

    * ``metadata`` rows supply passthrough fields embedded in the JSON.
    * ``text`` rows are assembled into the ``"texts"`` list (``None`` at gaps).
    * ``image`` rows are assembled into the ``"images"`` list and written as
      individual tar members; ``binary_content`` must be populated (either by
      the upstream pipeline or via ``materialize_on_write=True``).

    The JSON member key is ``urllib.parse.quote(sample_id, safe="")`` so that
    roundtripping via :class:`InterleavedWebdatasetReaderStage` with
    ``sample_id_field="sample_id"`` recovers the original sample_id.

    Only ``"metadata"``, ``"text"``, and ``"image"`` modalities are supported.
    Any other modality raises ``ValueError`` at write time.
    """

    file_extension: str = "tar"
    name: str = "interleaved_webdataset_writer"

    _SUPPORTED_MODALITIES: ClassVar[frozenset[str]] = frozenset({"metadata", "text", "image"})

    def _write_dataframe(self, df: pd.DataFrame, file_path: str, _write_kwargs: dict[str, Any]) -> None:
        unsupported = set(df["modality"].dropna().unique()) - self._SUPPORTED_MODALITIES
        if unsupported:
            msg = f"Unsupported modality {sorted(unsupported)!r}. Supported: {sorted(self._SUPPORTED_MODALITIES)}"
            raise ValueError(msg)

        passthrough_cols = [c for c in df.columns if c not in RESERVED_COLUMNS]

        with (
            fsspec.open(file_path, "wb", **self.storage_options) as fobj,
            tarfile.open(fileobj=fobj, mode="w:") as tf,
        ):
            sample_count = 0
            for sample_id, sample_df in df.groupby("sample_id", sort=False):
                _write_sample(tf, sample_df, sample_id, passthrough_cols)
                sample_count += 1

        self._log_metric("samples_written", float(sample_count))
