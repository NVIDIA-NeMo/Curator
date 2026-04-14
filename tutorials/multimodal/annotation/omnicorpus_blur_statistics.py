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

"""Histograms over Parquet from ``omnicorpus_annotation_pipeline``.

**Blur (Laplacian sharpness)** — default **11** bins ``[0,10)``, …, ``[90,100)``, ``[100,\\infty)``
labeled ``0-10`` … ``90-100``, ``100+``, plus ``(no sharpness)`` for null scores on image rows.

**CLIP** — per image, **max** similarity over stored text-position scores; **10** equal bins on
``[0, 0.5]`` (width ``0.05``): ``[0,0.05)``, …, ``[0.45, 0.5)``. Scores are clipped to ``[0, 0.5]``
for binning (values above ``0.5`` count in the top bin). Plus ``(no clip score)`` when the cell is
null / empty / unparseable. Additional **by ``sample_id``** stats: per sample, **max** and **min** of per-image ``clip_max``
over image rows in that sample; same bin layout as per-image CLIP; no image export for these.

**Image / text balance per ``sample_id``**: counts match
:func:`~nemo_curator.stages.interleaved.filter.image_to_text_ratio_filter.per_row_image_word_counts_broadcast`.
When Parquet has ``image_num`` / ``text_word_num`` from
:class:`~nemo_curator.stages.interleaved.filter.image_to_text_ratio_filter.InterleavedImageToTextRatioFilterStage`,
non-null values override the broadcast (sparse rows at ``position == 0``). Ratio
``images / words`` for samples with at least one image and one word; **10** equal bins on ``[0, 0.1]``
(width ``0.01``), values clipped to ``[0, 0.1]`` for binning (above ``0.1`` counts in the top bin).
Extra row **``0 (no images)``** when ``n_images == 0`` and ``n_words > 0``. Missing row when
``n_words == 0`` (includes both counts zero).

``pct`` is the share of **all** image rows in each row when the corresponding ``(no …)`` row is
included (sums to 100).

With ``--export-dir``, writes ``distribution.json`` (histograms + sample manifests) and up to
``--samples-per-bin`` files per **scored** bin under ``<export-dir>/blur/<bin>/`` and
``<export-dir>/clip/<bin>/`` (when blur / CLIP are present). For **CLIP**, each sample is an
**image + text pair** export: the image file plus a ``*_pair.json`` listing every text row in the
same ``sample_id`` with ``clip_similarity`` from the CLIP score dict for that image. Use
``--export-overwrite`` to remove an existing ``--export-dir`` before writing. The ``(no sharpness)``
/ ``(no clip score)`` bins are counted in the tables but **not** written as sample folders or
manifest entries. Rows without ``binary_content`` are still listed in manifests for scored bins.

Each positional input may be a Parquet file, a remote URI to one file, or a **directory**; directories
are walked recursively for ``*.parquet`` (case-insensitive suffix). Multiple files are read in
parallel using thread workers (``--read-workers``).

Requires the same environment as NeMo Curator (pandas, pyarrow, fsspec).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from fsspec.core import url_to_fs
from pyarrow.fs import FSSpecHandler, PyFileSystem

from nemo_curator.stages.interleaved.filter.image_to_text_ratio_filter import (
    per_row_image_word_counts_broadcast,
)
from nemo_curator.stages.interleaved.utils import resolve_storage_options

MISSING_BIN_LABEL_BLUR = "(no sharpness)"
MISSING_BIN_LABEL_CLIP = "(no clip score)"

CLIP_HIST_MAX = 0.5
CLIP_N_BINS = 10
CLIP_BIN_WIDTH = CLIP_HIST_MAX / CLIP_N_BINS
CLIP_EDGES = np.linspace(0.0, CLIP_HIST_MAX, CLIP_N_BINS + 1, dtype=np.float64)
CLIP_BIN_LABELS = [
    "0-0.05",
    "0.05-0.1",
    "0.1-0.15",
    "0.15-0.2",
    "0.2-0.25",
    "0.25-0.3",
    "0.3-0.35",
    "0.35-0.4",
    "0.4-0.45",
    "0.45-0.5",
]

MISSING_BIN_LABEL_IMAGE_WORD_RATIO = "(no ratio: zero words, or zero images and zero words)"
ZERO_IMAGES_BIN_LABEL = "0 (no images)"
RATIO_HIST_MAX = 0.1
RATIO_HIST_N_BINS = 10
RATIO_BIN_WIDTH = RATIO_HIST_MAX / RATIO_HIST_N_BINS
RATIO_HIST_EDGES = np.linspace(0.0, RATIO_HIST_MAX, RATIO_HIST_N_BINS + 1, dtype=np.float64)
RATIO_BIN_LABELS = [
    "0-0.01",
    "0.01-0.02",
    "0.02-0.03",
    "0.03-0.04",
    "0.04-0.05",
    "0.05-0.06",
    "0.06-0.07",
    "0.07-0.08",
    "0.08-0.09",
    "0.09-0.1",
]


def absolute_bin_edges_and_labels(*, bin_width: float = 10.0, n_bins: int = 11) -> tuple[np.ndarray, list[str]]:
    """Edges for ``n_bins`` blur intervals; see module docstring."""
    if n_bins < 2:
        msg = "n_bins must be at least 2 (finite intervals + one tail)"
        raise ValueError(msg)
    w = float(bin_width)
    finite = np.arange(0, n_bins * w, w, dtype=np.float64)
    if len(finite) != n_bins:
        msg = "bin_width and n_bins must yield n_bins finite lower edges (use tail as last bin)"
        raise ValueError(msg)
    edges = np.append(finite, np.inf)
    labels: list[str] = [f"{int(i * w)}-{int((i + 1) * w)}" for i in range(n_bins - 1)]
    labels.append(f"{int((n_bins - 1) * w)}+")
    return edges, labels


def blur_sharpness_column(df: pd.DataFrame) -> str | None:
    """Return ``sharpness`` if present, else ``None``."""
    return "sharpness" if "sharpness" in df.columns else None


def clip_score_column(df: pd.DataFrame) -> str | None:
    """Return ``clip_scores`` if present, else ``None``."""
    return "clip_scores" if "clip_scores" in df.columns else None


def image_to_text_ratio_image_num_column(df: pd.DataFrame) -> str | None:
    """Return ``image_num`` if present, else ``None``."""
    return "image_num" if "image_num" in df.columns else None


def image_to_text_ratio_text_word_num_column(df: pd.DataFrame) -> str | None:
    """Return ``text_word_num`` if present, else ``None``."""
    return "text_word_num" if "text_word_num" in df.columns else None


def clip_max_score_from_cell(v: Any) -> Any:
    """Max CLIP similarity for one cell (dict, JSON string, or dict-like Series); ``pd.NA`` if none."""
    if v is None or v is pd.NA:
        return pd.NA
    if pd.api.types.is_scalar(v) and pd.isna(v):
        return pd.NA
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return pd.NA
        try:
            parsed: Any = json.loads(s)
        except json.JSONDecodeError:
            return pd.NA
        if not isinstance(parsed, dict) or not parsed:
            return pd.NA
        return max(float(x) for x in parsed.values())
    if isinstance(v, dict):
        if not v:
            return pd.NA
        return max(float(x) for x in v.values())
    if isinstance(v, pd.Series):
        vals = pd.to_numeric(v, errors="coerce").dropna()
        if vals.empty:
            return pd.NA
        return float(vals.max())
    return pd.NA


def clip_max_series_for_images(df: pd.DataFrame, col: str, *, images: pd.Series) -> pd.Series:
    """Per image row: max CLIP score; same index as ``df.loc[images, col]``."""
    raw = df.loc[images, col]
    return raw.map(clip_max_score_from_cell)


def _clip_agg_per_sample_id(
    df: pd.DataFrame,
    clip_max_per_image: pd.Series,
    *,
    images: pd.Series,
    agg: Literal["max", "min"],
) -> pd.Series | None:
    """One aggregated CLIP per ``sample_id`` over image rows; index = unique ``sample_id``."""
    if "sample_id" not in df.columns:
        return None
    sid = df.loc[images, "sample_id"]
    gb = pd.DataFrame({"sample_id": sid.values, "clip_max": clip_max_per_image.values}).groupby(
        "sample_id", dropna=False
    )["clip_max"]
    return gb.max() if agg == "max" else gb.min()


def clip_max_per_sample_id(df: pd.DataFrame, clip_max_per_image: pd.Series, *, images: pd.Series) -> pd.Series | None:
    """One max CLIP per ``sample_id`` (max over image rows in that sample)."""
    return _clip_agg_per_sample_id(df, clip_max_per_image, images=images, agg="max")


def clip_min_per_sample_id(df: pd.DataFrame, clip_max_per_image: pd.Series, *, images: pd.Series) -> pd.Series | None:
    """One min CLIP per ``sample_id`` (min over image rows in that sample, NaNs skipped per group)."""
    return _clip_agg_per_sample_id(df, clip_max_per_image, images=images, agg="min")


def blur_absolute_histogram_counts(
    sharpness: pd.Series,
    *,
    bin_width: float = 10.0,
    n_bins: int = 11,
) -> tuple[pd.Series, int, list[str]]:
    s = sharpness.dropna().astype("float64").clip(lower=0.0)
    n = len(s)
    edges, labels = absolute_bin_edges_and_labels(bin_width=bin_width, n_bins=n_bins)
    if n == 0:
        return pd.Series(0, index=labels, dtype="int64"), 0, labels
    codes = pd.cut(s, bins=edges, right=False, labels=False, include_lowest=True)
    counts = codes.value_counts().reindex(range(len(labels)), fill_value=0).astype("int64")
    counts.index = labels
    return counts, n, labels


def blur_histogram_table(
    sharpness: pd.Series,
    *,
    bin_width: float = 10.0,
    n_bins: int = 11,
    include_missing_row: bool = True,
) -> pd.DataFrame:
    total_images = len(sharpness)
    missing_ct = int(sharpness.isna().sum())
    scored_ct = total_images - missing_ct
    counts, _n_scored, _labels = blur_absolute_histogram_counts(sharpness, bin_width=bin_width, n_bins=n_bins)
    if include_missing_row:
        denom = float(total_images) if total_images else 1.0
    else:
        denom = float(scored_ct) if scored_ct else 1.0
    pct = counts.astype("float64") / denom * 100.0
    out = pd.DataFrame({"bin": counts.index, "count": counts.values, "pct": pct.values})
    if include_missing_row:
        miss_pct = missing_ct / denom * 100.0
        out = pd.concat(
            [out, pd.DataFrame({"bin": [MISSING_BIN_LABEL_BLUR], "count": [missing_ct], "pct": [miss_pct]})],
            ignore_index=True,
        )
    return out


def clip_absolute_histogram_counts(
    clip_max: pd.Series,
) -> tuple[pd.Series, int, list[str]]:
    """Bin max CLIP scores into ``CLIP_BIN_LABELS`` (``clip_max`` = image rows, may contain NA)."""
    labels = list(CLIP_BIN_LABELS)
    s = clip_max.dropna().astype("float64").clip(lower=0.0, upper=CLIP_HIST_MAX - 1e-12)
    n = len(s)
    if n == 0:
        return pd.Series(0, index=labels, dtype="int64"), 0, labels
    codes = pd.cut(s, bins=CLIP_EDGES, right=False, labels=False, include_lowest=True)
    counts = codes.value_counts().reindex(range(len(labels)), fill_value=0).astype("int64")
    counts.index = labels
    return counts, n, labels


def image_word_ratio_histogram_counts(
    ratio: pd.Series,
) -> tuple[pd.Series, int, list[str]]:
    """Bin ``ratio`` values into ``RATIO_BIN_LABELS`` on ``[0, RATIO_HIST_MAX]`` (series may contain NA)."""
    labels = list(RATIO_BIN_LABELS)
    s = ratio.dropna().astype("float64").clip(lower=0.0, upper=RATIO_HIST_MAX - 1e-15)
    n = len(s)
    if n == 0:
        return pd.Series(0, index=labels, dtype="int64"), 0, labels
    codes = pd.cut(s, bins=RATIO_HIST_EDGES, right=False, labels=False, include_lowest=True)
    counts = codes.value_counts().reindex(range(len(labels)), fill_value=0).astype("int64")
    counts.index = labels
    return counts, n, labels


def image_word_ratio_histogram_table(
    ratio: pd.Series,
    *,
    include_missing_row: bool = True,
    zero_images_count: int = 0,
    missing_no_words_count: int = 0,
    total_samples: int | None = None,
) -> pd.DataFrame:
    """``ratio`` should be ``n_images/n_words`` only where ``n_images > 0`` and ``n_words > 0`` (else NA)."""
    n_total = int(total_samples if total_samples is not None else len(ratio))
    scored_ratio_ct = int(ratio.notna().sum())
    counts, _n_scored, _labels = image_word_ratio_histogram_counts(ratio)
    if include_missing_row:
        denom = float(n_total) if n_total else 1.0
    else:
        denom = float(scored_ratio_ct) if scored_ratio_ct else 1.0
    pct = counts.astype("float64") / denom * 100.0
    out = pd.DataFrame({"bin": counts.index, "count": counts.values, "pct": pct.values})
    if include_missing_row:
        zi_pct = float(zero_images_count) / denom * 100.0
        top = pd.DataFrame({"bin": [ZERO_IMAGES_BIN_LABEL], "count": [int(zero_images_count)], "pct": [zi_pct]})
        miss_pct = float(missing_no_words_count) / denom * 100.0
        bot = pd.DataFrame(
            {
                "bin": [MISSING_BIN_LABEL_IMAGE_WORD_RATIO],
                "count": [int(missing_no_words_count)],
                "pct": [miss_pct],
            }
        )
        out = pd.concat([top, out, bot], ignore_index=True)
    return out


def sample_image_word_ratio_payload(
    df: pd.DataFrame,
    *,
    include_missing_row: bool = True,
) -> dict[str, Any] | None:
    """Per ``sample_id``: ``image_num`` / ``text_word_num`` (filter stage) and ``images/words`` histogram."""
    if "sample_id" not in df.columns or "modality" not in df.columns:
        return None
    img_full, word_full = per_row_image_word_counts_broadcast(df)
    pack = pd.DataFrame({"sample_id": df["sample_id"], "_i": img_full, "_w": word_full})
    ni = pack.groupby("sample_id", dropna=False)["_i"].first()
    nw = pack.groupby("sample_id", dropna=False)["_w"].first()
    source = "per_row_image_word_counts_broadcast"
    img_col = image_to_text_ratio_image_num_column(df)
    word_col = image_to_text_ratio_text_word_num_column(df)
    if img_col is not None and word_col is not None:
        ni_p = pd.to_numeric(df.groupby("sample_id", dropna=False)[img_col].max(), errors="coerce")
        nw_p = pd.to_numeric(df.groupby("sample_id", dropna=False)[word_col].max(), errors="coerce")
        ni = ni_p.combine_first(ni)
        nw = nw_p.combine_first(nw)
        source = f"broadcast_with_parquet_override:{img_col},{word_col}"

    idx = ni.index.union(nw.index)
    if len(idx) == 0:
        return None
    ni = ni.reindex(idx)
    nw = nw.reindex(idx)
    ni_i = pd.to_numeric(ni, errors="coerce").fillna(0).astype("int64")
    nw_i = pd.to_numeric(nw, errors="coerce").fillna(0).astype("int64")
    ratio = (ni_i / nw_i.astype("float64")).where((ni_i > 0) & (nw_i > 0))
    zero_images_ct = int(((ni_i == 0) & (nw_i > 0)).sum())
    missing_no_words_ct = int((nw_i == 0).sum())
    n_tot = len(idx)
    ratio_table = image_word_ratio_histogram_table(
        ratio,
        include_missing_row=include_missing_row,
        zero_images_count=zero_images_ct,
        missing_no_words_count=missing_no_words_ct,
        total_samples=n_tot,
    )
    return {
        "source": source,
        "ratio": (
            "n_images/n_words when both>0; "
            f"bin {ZERO_IMAGES_BIN_LABEL!r} when n_images==0 and n_words>0; "
            "missing when n_words==0"
        ),
        "unique_samples": len(idx),
        "total_images_included_samples": int(ni_i.sum()),
        "total_words_included_samples": int(nw_i.sum()),
        "n_images_per_sample": {
            "min": int(ni_i.min()),
            "max": int(ni_i.max()),
            "mean": float(ni_i.mean()),
            "median": float(ni_i.median()),
        },
        "n_words_per_sample": {
            "min": int(nw_i.min()),
            "max": int(nw_i.max()),
            "mean": float(nw_i.mean()),
            "median": float(nw_i.median()),
        },
        "ratio_rows": ratio_table.assign(pct=ratio_table["pct"].round(4)).to_dict(orient="records"),
    }


def clip_histogram_table(
    clip_max: pd.Series,
    *,
    include_missing_row: bool = True,
) -> pd.DataFrame:
    total_images = len(clip_max)
    missing_ct = int(clip_max.isna().sum())
    scored_ct = total_images - missing_ct
    counts, _n_scored, _labels = clip_absolute_histogram_counts(clip_max)
    if include_missing_row:
        denom = float(total_images) if total_images else 1.0
    else:
        denom = float(scored_ct) if scored_ct else 1.0
    pct = counts.astype("float64") / denom * 100.0
    out = pd.DataFrame({"bin": counts.index, "count": counts.values, "pct": pct.values})
    if include_missing_row:
        miss_pct = missing_ct / denom * 100.0
        out = pd.concat(
            [out, pd.DataFrame({"bin": [MISSING_BIN_LABEL_CLIP], "count": [missing_ct], "pct": [miss_pct]})],
            ignore_index=True,
        )
    return out


def _safe_path_segment(label: str) -> str:
    parts: list[str] = []
    for ch in label:
        if ch.isalnum() or ch in "._-":
            parts.append(ch)
        else:
            parts.append("_")
    s = "".join(parts).strip("_")
    return s or "bin"


def _image_suffix_for_row(row: pd.Series) -> str:
    ct = row.get("content_type")
    if isinstance(ct, str):
        cl = ct.lower()
        if "jpeg" in cl or "jpg" in cl:
            return ".jpg"
        if "png" in cl:
            return ".png"
        if "webp" in cl:
            return ".webp"
    return ".bin"


def _row_image_bytes(row: pd.Series) -> bytes | None:
    blob = row.get("binary_content")
    if blob is None or (isinstance(blob, float) and pd.isna(blob)):
        return None
    if isinstance(blob, memoryview):
        return bytes(blob)
    if isinstance(blob, (bytes, bytearray)):
        return bytes(blob)
    return None


def blur_bin_labels(sharp: pd.Series, *, bin_width: float, n_bins: int) -> pd.Series:
    """Per image-row bin label (including ``(no sharpness)`` for null sharpness)."""
    out = pd.Series(index=sharp.index, dtype=object)
    mask = sharp.notna()
    out.loc[~mask] = MISSING_BIN_LABEL_BLUR
    if not mask.any():
        return out
    edges, labels = absolute_bin_edges_and_labels(bin_width=bin_width, n_bins=n_bins)
    s = sharp.loc[mask].astype("float64").clip(lower=0.0)
    codes = pd.cut(s, bins=edges, right=False, labels=False, include_lowest=True)
    lab_s = pd.Series([labels[int(c)] for c in codes], index=codes.index)
    out.loc[mask] = lab_s
    return out


def clip_bin_labels(clip_max: pd.Series) -> pd.Series:
    """Per image-row bin label (including ``(no clip score)`` for null CLIP max)."""
    out = pd.Series(index=clip_max.index, dtype=object)
    mask = clip_max.notna()
    out.loc[~mask] = MISSING_BIN_LABEL_CLIP
    if not mask.any():
        return out
    labels = list(CLIP_BIN_LABELS)
    s = clip_max.loc[mask].astype("float64").clip(lower=0.0, upper=CLIP_HIST_MAX - 1e-12)
    codes = pd.cut(s, bins=CLIP_EDGES, right=False, labels=False, include_lowest=True)
    lab_s = pd.Series([labels[int(c)] for c in codes], index=codes.index)
    out.loc[mask] = lab_s
    return out


def _sample_indices_per_bin(
    bin_labels: pd.Series, ordered_bins: list[str], *, max_per_bin: int
) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for lab in ordered_bins:
        idx = bin_labels.index[bin_labels == lab].tolist()
        out[lab] = idx[:max_per_bin]
    return out


def _clip_similarity_dict_from_cell(v: Any) -> dict[int, float] | None:
    """Parse CLIP cell (dict / JSON string / Series) into ``position -> similarity``."""
    if v is None or v is pd.NA or (pd.api.types.is_scalar(v) and pd.isna(v)):
        return None
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            parsed: Any = json.loads(s)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict) or not parsed:
            return None
        return {int(k): float(val) for k, val in parsed.items()}
    if isinstance(v, dict):
        if not v:
            return None
        return {int(k): float(val) for k, val in v.items()}
    if isinstance(v, pd.Series):
        out: dict[int, float] = {}
        for k, val in v.items():
            if pd.isna(val):
                continue
            out[int(k)] = float(val)
        return out or None
    return None


def _clip_text_pairs_for_sample(
    df: pd.DataFrame,
    sample_id: Any,
    sim_by_text_position: dict[int, float] | None,
) -> list[dict[str, Any]]:
    """Text rows for ``sample_id`` with optional CLIP similarity per ``text`` ``position``."""
    if sample_id is None or (isinstance(sample_id, float) and pd.isna(sample_id)):
        return []
    sub = df[(df["sample_id"] == sample_id) & (df["modality"] == "text")]
    if sub.empty:
        return []
    sub = sub.sort_values("position", kind="mergesort")
    pairs: list[dict[str, Any]] = []
    for _, r in sub.iterrows():
        pos_raw = r.get("position")
        pos = int(pos_raw) if pos_raw is not None and not (isinstance(pos_raw, float) and pd.isna(pos_raw)) else None
        raw = r.get("text_content")
        txt = "" if raw is None or (isinstance(raw, float) and pd.isna(raw)) else str(raw).strip()
        sim: float | None = None
        if sim_by_text_position is not None and pos is not None and pos in sim_by_text_position:
            sim = float(sim_by_text_position[pos])
        pairs.append({"text_position": pos, "text": txt, "clip_similarity": sim})
    return pairs


def _export_bin_images(
    df: pd.DataFrame,
    indices_by_bin: dict[str, list[Any]],
    *,
    root: Path,
    modality: str,
    score_key: str,
    score_values: pd.Series,
    clip_scores_column: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Write image bytes per sampled row; return manifest entries per bin.

    For ``modality == "clip"`` and a non-empty ``clip_scores_column``, also writes ``<stem>_pair.json``
    next to each image: image metadata plus ``pairs`` (text rows in the same ``sample_id`` with
    per-position CLIP scores).
    """
    manifest: dict[str, list[dict[str, Any]]] = {}
    for bin_label, idx_list in indices_by_bin.items():
        sub = root / modality / _safe_path_segment(bin_label)
        sub.mkdir(parents=True, exist_ok=True)
        entries: list[dict[str, Any]] = []
        for rank, idx in enumerate(idx_list):
            row = df.loc[idx]
            sid = row.get("sample_id")
            pos = row.get("position")
            score = score_values.loc[idx] if idx in score_values.index else None
            score_py: Any
            if score is None or (pd.api.types.is_scalar(score) and pd.isna(score)):
                score_py = None
            else:
                try:
                    score_py = float(score)
                except (TypeError, ValueError):
                    score_py = str(score)
            stem = _safe_path_segment(f"{rank:02d}_{sid}_{pos}")
            blob = _row_image_bytes(row)
            rel: str | None = None
            had_bytes = False
            if blob is not None:
                suf = _image_suffix_for_row(row)
                fn = stem + suf
                dest = sub / fn
                dest.write_bytes(blob)
                rel = str(Path(modality) / _safe_path_segment(bin_label) / fn)
                had_bytes = True
            pair_rel: str | None = None
            text_pairs: list[dict[str, Any]] = []
            if modality == "clip" and clip_scores_column and clip_scores_column in df.columns:
                cell = row.get(clip_scores_column)
                sims = _clip_similarity_dict_from_cell(cell)
                text_pairs = _clip_text_pairs_for_sample(df, sid, sims)
                pair_payload = {
                    "image": {
                        "df_index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                        "sample_id": None if sid is None or pd.isna(sid) else str(sid),
                        "image_position": None if pos is None or pd.isna(pos) else int(pos),
                        "clip_max": score_py,
                    },
                    "pairs": text_pairs,
                }
                pair_name = stem + "_pair.json"
                pair_path = sub / pair_name
                pair_path.write_text(json.dumps(pair_payload, indent=2, ensure_ascii=False), encoding="utf-8")
                pair_rel = str(Path(modality) / _safe_path_segment(bin_label) / pair_name)
            entries.append(
                {
                    "df_index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                    "sample_id": None if sid is None or pd.isna(sid) else str(sid),
                    "position": None if pos is None or pd.isna(pos) else int(pos),
                    score_key: score_py,
                    "saved_relative": rel,
                    "had_binary_content": had_bytes,
                    "pair_json_relative": pair_rel,
                    "text_pair_count": len(text_pairs),
                }
            )
        manifest[bin_label] = entries
    return manifest


def export_distribution_with_samples(
    *,
    export_dir: Path,
    df: pd.DataFrame,
    payload: dict[str, Any],
    do_blur: bool,
    do_clip: bool,
    sharp: pd.Series | None,
    clip_max: pd.Series | None,
    clip_scores_column: str | None,
    bin_width: float,
    n_bins: int,
    samples_per_bin: int,
    export_overwrite: bool = False,
) -> None:
    export_dir = Path(export_dir).resolve()
    if export_overwrite and export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    out_payload = dict(payload)
    _, blur_labs = absolute_bin_edges_and_labels(bin_width=bin_width, n_bins=n_bins)
    blur_order = list(blur_labs) + [MISSING_BIN_LABEL_BLUR]
    clip_order = list(CLIP_BIN_LABELS) + [MISSING_BIN_LABEL_CLIP]
    blur_order_samples = [b for b in blur_order if b != MISSING_BIN_LABEL_BLUR]
    clip_order_samples = [b for b in clip_order if b != MISSING_BIN_LABEL_CLIP]

    if do_blur and sharp is not None:
        bl = blur_bin_labels(sharp, bin_width=bin_width, n_bins=n_bins)
        blur_idx = _sample_indices_per_bin(bl, blur_order_samples, max_per_bin=samples_per_bin)
        out_payload["blur_samples"] = _export_bin_images(
            df,
            blur_idx,
            root=export_dir,
            modality="blur",
            score_key="sharpness",
            score_values=sharp,
            clip_scores_column=None,
        )
    if do_clip and clip_max is not None:
        cl = clip_bin_labels(clip_max)
        clip_idx = _sample_indices_per_bin(cl, clip_order_samples, max_per_bin=samples_per_bin)
        out_payload["clip_samples"] = _export_bin_images(
            df,
            clip_idx,
            root=export_dir,
            modality="clip",
            score_key="clip_max",
            score_values=clip_max,
            clip_scores_column=clip_scores_column,
        )

    dist_path = export_dir / "distribution.json"
    dist_path.write_text(json.dumps(out_payload, indent=2, default=str), encoding="utf-8")


def expand_parquet_path_inputs(
    paths: list[str],
    *,
    read_kwargs: dict[str, Any] | None = None,
) -> list[str]:
    """Expand each path: directories yield all nested ``*.parquet`` files; files pass through."""
    storage = resolve_storage_options(io_kwargs=read_kwargs or {}) or {}
    expanded: list[str] = []
    for raw in paths:
        s = str(Path(raw).expanduser()) if "://" not in raw else raw
        if "://" not in s:
            lp = Path(s)
            if lp.is_dir():
                found = sorted(p.resolve() for p in lp.rglob("*") if p.is_file() and p.suffix.lower() == ".parquet")
                if not found:
                    raise SystemExit(f"No .parquet files under directory: {lp}")
                expanded.extend(str(p) for p in found)
            else:
                expanded.append(str(lp))
        else:
            fs, root = url_to_fs(s, **storage)
            try:
                is_dir = bool(fs.isdir(root))
            except (FileNotFoundError, OSError, TypeError, ValueError):
                is_dir = False
            if is_dir:
                nested = fs.find(root, detail=False)
                keys = sorted(k for k in nested if str(k).lower().endswith(".parquet"))
                if not keys:
                    raise SystemExit(f"No .parquet files under directory: {s!r}")
                unstrip = getattr(fs, "unstrip_protocol", None)
                for k in keys:
                    expanded.append(unstrip(k) if unstrip is not None else str(k))
            else:
                expanded.append(s)
    seen: set[str] = set()
    deduped: list[str] = []
    for p in expanded:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped


def _read_single_scored_parquet(path: str, read_kwargs: dict[str, Any] | None) -> pd.DataFrame:
    storage = resolve_storage_options(io_kwargs=read_kwargs or {})
    fs, _ = url_to_fs(path, **(storage or {}))
    pa_fs = PyFileSystem(FSSpecHandler(fs))
    return pq.read_table(path, filesystem=pa_fs).to_pandas()


def _effective_read_workers(requested: int, n_paths: int) -> int:
    if n_paths <= 1:
        return 1
    if requested <= 0:
        return min(n_paths, max(16, (os.cpu_count() or 4) * 4))
    return min(max(1, requested), n_paths)


def load_scored_parquet_frames(
    paths: list[str],
    *,
    read_kwargs: dict[str, Any] | None = None,
    max_workers: int = 1,
) -> pd.DataFrame:
    if not paths:
        msg = "paths must be non-empty"
        raise ValueError(msg)
    if len(paths) == 1 or max_workers <= 1:
        frames = [_read_single_scored_parquet(ps, read_kwargs) for ps in paths]
    else:
        workers = min(max_workers, len(paths))
        frames = [None] * len(paths)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_i = {
                executor.submit(_read_single_scored_parquet, ps, read_kwargs): i for i, ps in enumerate(paths)
            }
            for fut in as_completed(future_to_i):
                frames[future_to_i[fut]] = fut.result()
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Histograms: blur sharpness (default 11 bins + no-sharpness row) and "
            "CLIP max score (10 bins on [0,0.5], width 0.05 + no-score row) on Parquet. "
            "Optional --export-dir writes distribution.json and sample images per bin; "
            "--export-overwrite clears the directory first."
        )
    )
    parser.add_argument(
        "parquet",
        nargs="+",
        help="Scored Parquet file(s) or URI(s), and/or directories (recursive *.parquet)",
    )
    parser.add_argument(
        "--blur-only",
        action="store_true",
        help="Only print blur statistics (requires a blur sharpness column).",
    )
    parser.add_argument(
        "--clip-only",
        action="store_true",
        help="Only print CLIP statistics (requires a CLIP score column).",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=10.0,
        help="Blur: width of each finite bin (default: 10).",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=11,
        metavar="N",
        help="Blur: number of bins (default: 11).",
    )
    parser.add_argument(
        "--no-missing-row",
        action="store_true",
        help="Omit (no sharpness) / (no clip score) rows; pct sums to 100 over scored images only.",
    )
    parser.add_argument("--json", action="store_true", help="Print one JSON object instead of tables")
    parser.add_argument(
        "--storage-options-json",
        default=None,
        help="JSON-encoded fsspec storage_options for remote paths",
    )
    parser.add_argument(
        "--read-workers",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Thread workers for reading multiple Parquet files in parallel (default: 0 = "
            "min(num_files, max(8, 4*cpu_count))); 1 forces sequential read."
        ),
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default=None,
        help=(
            "If set, write distribution.json and up to --samples-per-bin images per scored bin "
            "(not for no-sharpness / no-clip-score rows). See --export-overwrite."
        ),
    )
    parser.add_argument(
        "--samples-per-bin",
        type=int,
        default=10,
        metavar="K",
        help=(
            "Max image rows saved per scored bin when --export-dir is set (default: 10). "
            "CLIP also writes *_pair.json (image + text rows + similarities) per sample."
        ),
    )
    parser.add_argument(
        "--export-overwrite",
        action="store_true",
        help="With --export-dir: delete the export directory first if it exists, then write fresh output.",
    )
    args = parser.parse_args()
    if args.blur_only and args.clip_only:
        msg = "Use at most one of --blur-only and --clip-only"
        raise SystemExit(msg)
    if args.export_overwrite and not args.export_dir:
        raise SystemExit("--export-overwrite requires --export-dir.")

    read_kwargs: dict[str, Any] = {}
    if args.storage_options_json:
        read_kwargs["storage_options"] = json.loads(args.storage_options_json)

    parquet_paths = expand_parquet_path_inputs(
        [str(p) for p in args.parquet],
        read_kwargs=read_kwargs or None,
    )
    read_workers = _effective_read_workers(int(args.read_workers), len(parquet_paths))
    df = load_scored_parquet_frames(
        parquet_paths,
        read_kwargs=read_kwargs or None,
        max_workers=read_workers,
    )
    images = df["modality"] == "image"
    total_img = int(images.sum())

    blur_col = blur_sharpness_column(df)
    clip_col = clip_score_column(df)
    do_blur = not args.clip_only and blur_col is not None
    do_clip = not args.blur_only and clip_col is not None

    if args.blur_only and blur_col is None:
        raise SystemExit("No blur sharpness column found for --blur-only.")
    if args.clip_only and clip_col is None:
        raise SystemExit("No CLIP score column found for --clip-only.")
    if not do_blur and not do_clip:
        raise SystemExit("No blur sharpness column and no CLIP score column found.")

    include_missing = not args.no_missing_row
    payload: dict[str, Any] = {"image_rows": total_img}
    sharp: pd.Series | None = None
    clip_max: pd.Series | None = None

    if do_blur:
        sharp = df.loc[images, blur_col]
        blur_table = blur_histogram_table(
            sharp,
            bin_width=args.bin_width,
            n_bins=args.n_bins,
            include_missing_row=include_missing,
        )
        payload["blur"] = {
            "sharpness": blur_col,
            "bin_width": args.bin_width,
            "n_bins": args.n_bins,
            "scored_images": int(sharp.notna().sum()),
            "missing_sharpness_images": int(sharp.isna().sum()),
            "rows": blur_table.assign(pct=blur_table["pct"].round(4)).to_dict(orient="records"),
        }

    if do_clip:
        clip_max = clip_max_series_for_images(df, clip_col, images=images)
        clip_table = clip_histogram_table(clip_max, include_missing_row=include_missing)
        clip_by_sample_max = clip_max_per_sample_id(df, clip_max, images=images)
        clip_by_sample_min = clip_min_per_sample_id(df, clip_max, images=images)
        clip_entry: dict[str, Any] = {
            "clip_scores": clip_col,
            "statistic": "max_similarity_over_text_positions",
            "scored_images": int(clip_max.notna().sum()),
            "missing_clip_score_images": int(clip_max.isna().sum()),
            "rows": clip_table.assign(pct=clip_table["pct"].round(4)).to_dict(orient="records"),
        }
        if clip_by_sample_max is not None:
            by_sample_max_table = clip_histogram_table(clip_by_sample_max, include_missing_row=include_missing)
            by_sample_min_table = clip_histogram_table(clip_by_sample_min, include_missing_row=include_missing)
            clip_entry["by_sample_id"] = {
                "unique_samples_with_image_rows": len(clip_by_sample_max),
                "scored_samples": int(clip_by_sample_max.notna().sum()),
                "missing_clip_score_samples": int(clip_by_sample_max.isna().sum()),
                "max": {
                    "statistic": "max_over_image_rows_of_per_image_max_similarity",
                    "rows": by_sample_max_table.assign(pct=by_sample_max_table["pct"].round(4)).to_dict(
                        orient="records"
                    ),
                },
                "min": {
                    "statistic": "min_over_image_rows_of_per_image_max_similarity",
                    "rows": by_sample_min_table.assign(pct=by_sample_min_table["pct"].round(4)).to_dict(
                        orient="records"
                    ),
                },
            }
        payload["clip"] = clip_entry

    iw = sample_image_word_ratio_payload(df, include_missing_row=include_missing)
    if iw is not None:
        payload["sample_image_word_ratio"] = iw

    if args.export_dir:
        export_distribution_with_samples(
            export_dir=Path(args.export_dir),
            df=df,
            payload=payload,
            do_blur=do_blur,
            do_clip=do_clip,
            sharp=sharp if do_blur else None,
            clip_max=clip_max if do_clip else None,
            clip_scores_column=clip_col if do_clip else None,
            bin_width=args.bin_width,
            n_bins=args.n_bins,
            samples_per_bin=max(1, int(args.samples_per_bin)),
            export_overwrite=args.export_overwrite,
        )
        print(f"Wrote export: {Path(args.export_dir) / 'distribution.json'}")

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"image_rows={total_img}")
        if do_blur:
            b = payload["blur"]
            print()
            print("=== blur (Laplacian sharpness) ===")
            print(
                f"sharpness={b['sharpness']!r} bin_width={b['bin_width']} n_bins={b['n_bins']} "
                f"scored_images={b['scored_images']} missing_sharpness_images={b['missing_sharpness_images']}"
            )
            blur_df = pd.DataFrame(b["rows"])
            print(blur_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        if do_clip:
            c = payload["clip"]
            print()
            print("=== CLIP (max similarity per image) ===")
            print(
                f"clip_scores={c['clip_scores']!r} {c['statistic']} "
                f"scored_images={c['scored_images']} missing_clip_score_images={c['missing_clip_score_images']}"
            )
            clip_df = pd.DataFrame(c["rows"])
            print(clip_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            bs = c.get("by_sample_id")
            if bs:
                print()
                print("=== CLIP by sample_id (over images in sample) ===")
                print(
                    f"unique_samples_with_image_rows={bs['unique_samples_with_image_rows']} "
                    f"scored_samples={bs['scored_samples']} "
                    f"missing_clip_score_samples={bs['missing_clip_score_samples']}"
                )
                bmax = bs["max"]
                print()
                print(f"--- max: {bmax['statistic']} ---")
                print(pd.DataFrame(bmax["rows"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
                bmin = bs["min"]
                print()
                print(f"--- min: {bmin['statistic']} ---")
                print(pd.DataFrame(bmin["rows"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        iw = payload.get("sample_image_word_ratio")
        if iw:
            print()
            print(f"=== image / words per sample_id (histogram [0, {RATIO_HIST_MAX}], {RATIO_HIST_N_BINS} bins) ===")
            print(
                f"source={iw['source']} "
                f"unique_samples={iw['unique_samples']} "
                f"total_images_included_samples={iw['total_images_included_samples']} "
                f"total_words_included_samples={iw['total_words_included_samples']}"
            )
            print(f"n_images: {iw['n_images_per_sample']}")
            print(f"n_words: {iw['n_words_per_sample']}")
            print(f"ratio = {iw['ratio']}")
            print(pd.DataFrame(iw["ratio_rows"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
