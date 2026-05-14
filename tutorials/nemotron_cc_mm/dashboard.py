"""Streamlit dashboard for inspecting Nemotron-CC-MM Parquet outputs.

Run with::

    PATH=Curator/.venv/bin:$PATH \\
    streamlit run Curator/tutorials/nemotron_cc_mm/dashboard.py

Then open http://localhost:8501 in a browser.  Use the sidebar to point at
a Parquet directory or file (or two, for side-by-side compare).
"""
from __future__ import annotations

import io
import json
import random
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
import streamlit as st

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Nemotron-CC-MM Inspector",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _gather_parquet_files(path: str) -> list[Path]:
    p = Path(path).expanduser()
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(p.glob("*.parquet"))
    return []


def _path_signature(path: str) -> str:
    """Stable signature for (path, file set, mtimes).  When the set of
    parquet files or any mtime changes, the signature changes — which
    invalidates ``@st.cache_data`` automatically."""
    files = _gather_parquet_files(path)
    parts = [(str(f), f.stat().st_mtime_ns, f.stat().st_size) for f in files]
    return repr(parts)


def _url_from_ref(sr) -> str:
    if not isinstance(sr, str) or not sr:
        return ""
    try:
        d = json.loads(sr)
        return d.get("source_url") or d.get("url") or ""
    except (TypeError, ValueError):
        return ""


def _parse_image_ref(sr) -> tuple[str, str]:
    if not isinstance(sr, str) or not sr:
        return "", ""
    try:
        d = json.loads(sr)
        return d.get("url", ""), d.get("alt", "") or ""
    except (TypeError, ValueError):
        return "", ""


@st.cache_data(show_spinner=False)
def list_docs(parquet_path: str, _sig: str) -> pd.DataFrame:
    """One row per document: sample_id, source_url, n_text, n_image."""
    files = _gather_parquet_files(parquet_path)
    if not files:
        return pd.DataFrame(columns=["sample_id", "source_url", "n_text", "n_image"])

    dataset = ds.dataset([str(f) for f in files])
    tbl = dataset.to_table(columns=["sample_id", "modality", "source_ref"])
    df = tbl.to_pandas()

    md = df[df["modality"] == "metadata"].copy()
    md["source_url"] = md["source_ref"].apply(_url_from_ref)
    md = md[["sample_id", "source_url"]]

    counts = (
        df[df["modality"] != "metadata"]
        .groupby(["sample_id", "modality"])
        .size()
        .unstack(fill_value=0)
    )
    counts.columns = [f"n_{c}" for c in counts.columns]
    counts = counts.reset_index()

    out = md.merge(counts, on="sample_id", how="left").fillna(0)
    for col in ("n_text", "n_image"):
        if col not in out:
            out[col] = 0
        out[col] = out[col].astype(int)
    return out.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_doc(parquet_path: str, sample_id: str, _sig: str) -> pd.DataFrame:
    """Load all rows for a single sample_id."""
    files = _gather_parquet_files(parquet_path)
    if not files:
        return pd.DataFrame()
    dataset = ds.dataset([str(f) for f in files])
    tbl = dataset.to_table(filter=ds.field("sample_id") == sample_id)
    return tbl.to_pandas().sort_values("position").reset_index(drop=True)


def _img_dims(b: bytes) -> str:
    if Image is None:
        return ""
    try:
        with Image.open(io.BytesIO(b)) as im:
            return f"{im.size[0]}×{im.size[1]} px"
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Doc rendering
# ---------------------------------------------------------------------------
def render_doc(df: pd.DataFrame, *, max_text_chars: int = 600) -> None:
    if df.empty:
        st.info("Document not present in this Parquet.")
        return

    md = df[df["modality"] == "metadata"]
    src_url = ""
    if not md.empty:
        src_url = _url_from_ref(md.iloc[0]["source_ref"])

    n_text = int((df["modality"] == "text").sum())
    n_image = int((df["modality"] == "image").sum())
    sample_id = df.iloc[0]["sample_id"]

    if src_url:
        st.markdown(f"##### [{src_url}]({src_url})")
    st.caption(f"`{sample_id}` · {n_text} text · {n_image} image")
    st.divider()

    content = df[df["modality"] != "metadata"].sort_values("position")
    for _, row in content.iterrows():
        pos = row["position"]
        if row["modality"] == "text":
            text = row.get("text_content") or ""
            if not isinstance(text, str):
                text = ""
            if len(text) > max_text_chars:
                text = text[: max_text_chars - 1] + "…"
            st.markdown(
                f"<div style='margin:8px 0'>"
                f"<span style='color:#bbb;font-size:0.78em'>#{pos}</span> "
                f"<span style='line-height:1.45'>{text}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        elif row["modality"] == "image":
            url, alt = _parse_image_ref(row.get("source_ref"))
            b = row.get("binary_content")
            has_bytes = isinstance(b, (bytes, bytearray, memoryview))
            cols = st.columns([2, 5])
            with cols[0]:
                if has_bytes:
                    try:
                        st.image(bytes(b), width=240)
                    except Exception as e:  # noqa: BLE001
                        st.caption(f"⚠ render error: {e}")
                else:
                    st.markdown(
                        "<div style='border:1px dashed #ccc;border-radius:6px;"
                        "padding:30px;text-align:center;color:#888;font-size:0.85em'>"
                        "no image bytes</div>",
                        unsafe_allow_html=True,
                    )
            with cols[1]:
                dims = _img_dims(bytes(b)) if has_bytes else ""
                size_str = f"{len(b) / 1024:.1f} KB" if has_bytes else ""
                meta_parts = [p for p in [dims, size_str] if p]
                meta_line = " · ".join(meta_parts) if meta_parts else ""
                st.markdown(
                    f"<span style='color:#bbb;font-size:0.78em'>#{pos}</span> "
                    f"**image**"
                    + (f" · <span style='color:#666'>{meta_line}</span>" if meta_line else ""),
                    unsafe_allow_html=True,
                )
                if url:
                    st.markdown(
                        f"<a href='{url}' target='_blank' style='font-size:0.8em;"
                        f"color:#1670c8;word-break:break-all'>{url[:120]}</a>",
                        unsafe_allow_html=True,
                    )
                if alt:
                    st.markdown(
                        f"<i style='color:#666;font-size:0.85em'>alt: {alt[:200]}</i>",
                        unsafe_allow_html=True,
                    )


# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------
st.sidebar.title("Nemotron-CC-MM")
st.sidebar.caption("Output inspector")

parquet_path = st.sidebar.text_input(
    "Parquet path",
    value="/home/aot/codebase/nemotron_cc_mm/data/out/",
    help="Directory containing *.parquet shards, or a single .parquet file",
)
compare_path = st.sidebar.text_input(
    "Compare against (optional)",
    value="",
    help="Second Parquet path — when set, the selected doc renders side-by-side",
)
if st.sidebar.button("🔄 Refresh data", help="Re-read Parquet files (clears cache)"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.subheader("Filters")
min_images = st.sidebar.number_input("Min images per doc", min_value=0, value=0, step=1)
min_text = st.sidebar.number_input("Min text rows per doc", min_value=0, value=0, step=1)
url_filter = st.sidebar.text_input("URL contains", value="")
max_text_chars = st.sidebar.slider(
    "Truncate text rows at (chars)", min_value=80, max_value=4000, value=600, step=40
)


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
st.title("Nemotron-CC-MM Output Inspector")

if not parquet_path:
    st.info("Enter a Parquet path in the sidebar.")
    st.stop()

primary_sig = _path_signature(parquet_path)
with st.spinner("Indexing documents…"):
    docs = list_docs(parquet_path, primary_sig)

if docs.empty:
    st.error(f"No documents found at `{parquet_path}`")
    st.stop()

# Stats banner
n_total = len(docs)
total_text = int(docs["n_text"].sum())
total_img = int(docs["n_image"].sum())
docs_with_img = int((docs["n_image"] > 0).sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Documents", f"{n_total:,}")
c2.metric("Text rows", f"{total_text:,}")
c3.metric("Image rows", f"{total_img:,}")
c4.metric("Docs w/ image", f"{docs_with_img:,}")

# Apply filters
filtered = docs.copy()
if min_images > 0:
    filtered = filtered[filtered["n_image"] >= min_images]
if min_text > 0:
    filtered = filtered[filtered["n_text"] >= min_text]
if url_filter:
    filtered = filtered[
        filtered["source_url"].str.contains(url_filter, case=False, na=False)
    ]

filtered = filtered.reset_index(drop=True)
st.caption(f"**{len(filtered):,}** of {n_total:,} documents match filters")

if filtered.empty:
    st.warning("No documents match the current filters.")
    st.stop()

# Document selector + random pick
sel_col, btn_col = st.columns([5, 1])
with sel_col:
    idx = st.selectbox(
        "Select document",
        options=range(len(filtered)),
        format_func=lambda i: (
            f"[{i + 1}] {filtered.iloc[i]['n_text']}T / {filtered.iloc[i]['n_image']}I  "
            f"·  {(filtered.iloc[i]['source_url'] or '(no url)')[:90]}"
        ),
        key="doc_idx",
    )
with btn_col:
    st.write("")  # spacer
    if st.button("🎲 Random"):
        st.session_state.doc_idx = random.randrange(len(filtered))
        st.rerun()

selected_sid = filtered.iloc[idx]["sample_id"]
st.markdown(f"### Document `{selected_sid}`")

# Render single or side-by-side
if compare_path.strip():
    compare_sig = _path_signature(compare_path)
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.caption(f"**A** — `{parquet_path}`")
        render_doc(
            load_doc(parquet_path, selected_sid, primary_sig),
            max_text_chars=max_text_chars,
        )
    with col_b:
        st.caption(f"**B** — `{compare_path}`")
        try:
            cmp_df = load_doc(compare_path, selected_sid, compare_sig)
            render_doc(cmp_df, max_text_chars=max_text_chars)
        except Exception as e:  # noqa: BLE001
            st.error(f"Compare path error: {e}")
else:
    render_doc(
        load_doc(parquet_path, selected_sid, primary_sig),
        max_text_chars=max_text_chars,
    )
