"""Helpers for reading the per-doc lineage we stash in the metadata-row
``source_ref`` JSON (Option A of the lineage design).

Quick usage::

    import pandas as pd
    from nemo_curator.stages.nemotron_cc_mm.lineage import lineage_view

    df = pd.read_parquet("data/out/")
    lin = lineage_view(df)
    # one row per sample_id, with flat columns:
    #   sample_id  source_url  warc_id  warc_file  warc_segment
    #   snapshot   extractor

    # Example: how many docs per WARC file?
    lin.groupby("warc_file").size().sort_values(ascending=False).head()
"""
from __future__ import annotations

import json

import pandas as pd


_LINEAGE_FIELDS: tuple[str, ...] = (
    "source_url",
    "warc_id",
    "warc_file",
    "warc_segment",
    "snapshot",
    "extractor",
)


def _parse_ref(ref: str | None) -> dict:
    if not isinstance(ref, str) or not ref:
        return {}
    try:
        out = json.loads(ref)
    except (TypeError, ValueError):
        return {}
    return out if isinstance(out, dict) else {}


def lineage_view(df: pd.DataFrame) -> pd.DataFrame:
    """One row per ``sample_id`` with lineage columns unpacked from the
    metadata row's ``source_ref`` JSON.

    Pure function — does not mutate ``df``.
    """
    md = df[df["modality"] == "metadata"].copy()
    if md.empty:
        return pd.DataFrame(columns=("sample_id", *_LINEAGE_FIELDS))
    parsed = md["source_ref"].apply(_parse_ref)
    for f in _LINEAGE_FIELDS:
        md[f] = parsed.apply(lambda d, k=f: d.get(k))
    keep = ["sample_id", *_LINEAGE_FIELDS]
    return md[keep].reset_index(drop=True)


def attach_lineage(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with lineage columns broadcast to every row of each
    doc (joined by ``sample_id``).  Handy when you want to ``df.groupby
    ('snapshot')`` on a flat per-row dataframe.
    """
    lin = lineage_view(df).set_index("sample_id")
    return df.join(lin, on="sample_id")
