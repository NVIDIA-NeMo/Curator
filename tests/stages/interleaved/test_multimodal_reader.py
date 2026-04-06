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

import json
import logging
import tarfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.interleaved.io.reader import InterleavedParquetReader
from nemo_curator.stages.interleaved.io.readers.base import BaseInterleavedReader
from nemo_curator.stages.interleaved.io.readers.parquet import InterleavedParquetReaderStage
from nemo_curator.stages.interleaved.io.readers.webdataset import InterleavedWebdatasetReaderStage
from nemo_curator.stages.interleaved.io.writers.tabular import InterleavedParquetWriterStage
from nemo_curator.stages.interleaved.io.writers.webdataset import InterleavedWebdatasetWriterStage
from nemo_curator.tasks import FileGroupTask, InterleavedBatch
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA

from .conftest import build_multi_frame_tiff, make_interleaved_batch, make_row, task_for_tar, write_tar

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _as_df(task_or_tasks: InterleavedBatch | list[InterleavedBatch]) -> pd.DataFrame:
    task = task_or_tasks[0] if isinstance(task_or_tasks, list) else task_or_tasks
    return task.to_pandas()


def _write_tar_sample(
    tar_path: Path,
    payload: dict[str, object],
    *,
    json_name: str = "sample.json",
    image_name: str = "image.jpg",
    image_bytes: bytes = b"abc",
) -> None:
    write_tar(tar_path, {json_name: json.dumps(payload).encode("utf-8"), image_name: image_bytes})


def _write_parquet_task(batch: InterleavedBatch, out_dir: Path) -> str:
    """Write *batch* to parquet and return the written file path."""
    writer = InterleavedParquetWriterStage(path=str(out_dir), materialize_on_write=False, mode="overwrite")
    write_task = writer.process(batch)
    return write_task.data[0]


def _make_aligned_rows(fake_jpg: bytes, num_images: int = 2) -> list[dict]:
    """Build a standard s1 sample with sample_metadata/text_metadata/image_metadata on the correct rows.

    sample_metadata lives on the metadata row, text_metadata on the text row, image_metadata on each
    image row — three distinct classes used to verify per-row field alignment in round-trip tests.
    """
    rows = [
        make_row("s1", -1, "metadata", sample_metadata="doc A", text_metadata=None, image_metadata=None),
        make_row(
            "s1",
            0,
            "text",
            text_content="hello world",
            sample_metadata=None,
            text_metadata="conf:0.9",
            image_metadata=None,
        ),
    ]
    for i in range(num_images):
        rows.append(
            make_row(
                "s1",
                i + 1,
                "image",
                content_type="image/jpeg",
                binary_content=fake_jpg,
                sample_metadata=None,
                text_metadata=None,
                image_metadata=f'{{"page": {i}}}',
            )
        )
    return rows


def _assert_field_alignment(df: pd.DataFrame, *, image_vals: list, text_vals: list, sample_val: object) -> None:
    """Verify sample/text/image extra fields land on the correct rows and are null everywhere else."""
    meta = df[df["modality"] == "metadata"].sort_values("position")
    text = df[df["modality"] == "text"].sort_values("position")
    imgs = df[df["modality"] == "image"].sort_values("position")

    assert len(meta) == 1, f"expected 1 metadata row, got {len(meta)}"
    assert len(text) == len(text_vals), f"expected {len(text_vals)} text rows, got {len(text)}"
    assert len(imgs) == len(image_vals), f"expected {len(image_vals)} image rows, got {len(imgs)}"

    assert meta["sample_metadata"].iloc[0] == sample_val
    assert text["sample_metadata"].isna().all()
    assert imgs["sample_metadata"].isna().all()

    assert text["text_metadata"].tolist() == text_vals
    assert meta["text_metadata"].isna().all()
    assert imgs["text_metadata"].isna().all()

    assert imgs["image_metadata"].tolist() == image_vals
    assert meta["image_metadata"].isna().all()
    assert text["image_metadata"].isna().all()


# ---------------------------------------------------------------------------
# InterleavedWebdatasetReaderStage
# ---------------------------------------------------------------------------


def test_reader_supports_custom_field_mapping(tmp_path: Path) -> None:
    tar_path = tmp_path / "alt-shard-00000.tar"
    payload = {
        "doc_id": "doc-custom",
        "source_doc": "custom.pdf",
        "captions": ["a", "b"],
        "frames": ["custom-image.jpg"],
        "primary_image": "custom-image.jpg",
        "p_hash": "abc123",
    }
    image_bytes = b"custom-image-bytes"
    _write_tar_sample(
        tar_path,
        payload,
        json_name="sample-xyz.meta.json",
        image_name="custom-image.jpg",
        image_bytes=image_bytes,
    )
    task = task_for_tar(str(tar_path), "file_group_custom")
    reader = InterleavedWebdatasetReaderStage(
        sample_id_field="doc_id",
        texts_field="captions",
        images_field="frames",
        image_member_field="primary_image",
        json_extensions=(".meta.json",),
        materialize_on_read=True,
        fields=("p_hash",),
    )
    df = _as_df(reader.process(task))
    assert ((df["sample_id"] == "doc-custom") & (df["modality"] == "metadata")).any()
    text_rows = df[df["modality"] == "text"]
    assert text_rows["text_content"].tolist() == ["a", "b"]
    image_rows = df[df["modality"] == "image"]
    assert len(image_rows) == 1
    assert image_rows.iloc[0]["binary_content"] == image_bytes
    assert "p_hash" in df.columns
    meta_row = df[df["modality"] == "metadata"].iloc[0]
    assert meta_row["p_hash"] == "abc123"
    assert pd.isna(image_rows.iloc[0]["p_hash"])


def test_reader_reads_all_fields_by_default(tmp_path: Path) -> None:
    tar_path = tmp_path / "all-fields.tar"
    payload = {
        "doc_id": "doc-all",
        "source_doc": "all.pdf",
        "captions": ["hello"],
        "frames": ["image.jpg"],
        "primary_image": "image.jpg",
        "p_hash": "phash-1",
        "score": 0.91,
        "aux": {"page": 3},
    }
    _write_tar_sample(tar_path, payload, json_name="sample.meta.json")
    task = task_for_tar(str(tar_path), "all_fields")
    reader = InterleavedWebdatasetReaderStage(
        sample_id_field="doc_id",
        texts_field="captions",
        images_field="frames",
        image_member_field="primary_image",
        json_extensions=(".meta.json",),
    )
    df = _as_df(reader.process(task))
    meta_row = df[df["modality"] == "metadata"].iloc[0]
    assert meta_row["p_hash"] == "phash-1"
    assert meta_row["score"] == 0.91
    assert meta_row["aux"] == json.dumps({"page": 3}, ensure_ascii=False)
    image_row = df[df["modality"] == "image"].iloc[0]
    assert pd.isna(image_row["p_hash"])
    assert "captions" not in df.columns
    assert "frames" not in df.columns


def test_reader_uses_resolved_content_key_for_content_type(tmp_path: Path) -> None:
    tar_path = tmp_path / "content-type-resolve.tar"
    payload = {
        "doc_id": "doc-ct",
        "source_doc": "ct.pdf",
        "captions": ["hello"],
        "frames": ["token.png"],
        "primary_image": "fallback.jpg",
    }
    with tarfile.open(tar_path, "w") as tf:
        json_blob = json.dumps(payload).encode("utf-8")
        json_info = tarfile.TarInfo(name="sample.meta.json")
        json_info.size = len(json_blob)
        tf.addfile(json_info, BytesIO(json_blob))
        png_info = tarfile.TarInfo(name="token.png")
        png_info.size = 3
        tf.addfile(png_info, BytesIO(b"png"))
        jpg_info = tarfile.TarInfo(name="fallback.jpg")
        jpg_info.size = 3
        tf.addfile(jpg_info, BytesIO(b"jpg"))

    task = task_for_tar(str(tar_path), "content_type_resolve")
    reader = InterleavedWebdatasetReaderStage(
        sample_id_field="doc_id",
        texts_field="captions",
        images_field="frames",
        image_member_field="primary_image",
        json_extensions=(".meta.json",),
    )
    df = _as_df(reader.process(task))
    image_row = df[df["modality"] == "image"].iloc[0]
    assert image_row["content_type"] == "image/png"


def test_reader_image_tokens_with_frame_index(tmp_path: Path) -> None:
    """Non-None tokens get frame_index and resolve to default TIFF. None tokens are skipped."""
    tar_path = tmp_path / "sub-image-shard.tar"
    payload = {
        "pdf_name": "doc.pdf",
        "texts": ["text1", "text2", "text3"],
        "images": [None, "page_0_image_15", "page_1_image_22"],
    }
    _write_tar_sample(tar_path, payload, json_name="sample.json", image_name="doc.pdf.tiff", image_bytes=b"TIFF_DATA")
    task = task_for_tar(str(tar_path), "sub_image_test")
    reader = InterleavedWebdatasetReaderStage(
        sample_id_field="pdf_name",
        image_extensions=(".tiff",),
    )
    df = _as_df(reader.process(task))

    image_rows = df[df["modality"] == "image"]
    assert len(image_rows) == 2, "None image tokens should be skipped"

    assert image_rows.iloc[0]["position"] == 1, "First non-None image at interleaved position 1"
    assert image_rows.iloc[1]["position"] == 2, "Second non-None image at interleaved position 2"

    refs = [InterleavedBatch.parse_source_ref(v) for v in image_rows["source_ref"].tolist()]

    assert refs[0]["member"] == "doc.pdf.tiff", "Non-matching string should resolve to default TIFF"
    assert refs[0]["frame_index"] == 0, "First non-None token gets frame_index=0"

    assert refs[1]["member"] == "doc.pdf.tiff"
    assert refs[1]["frame_index"] == 1, "Second non-None token gets frame_index=1"

    text_rows = df[df["modality"] == "text"]
    assert len(text_rows) == 3
    assert text_rows["position"].tolist() == [0, 1, 2], "All text entries are strings so positions 0,1,2"


def test_reader_interleaved_positions_do_not_overlap(tmp_path: Path) -> None:
    """Parallel texts/images arrays with None placeholders produce non-overlapping positions."""
    tar_path = tmp_path / "interleaved-shard.tar"
    payload = {
        "pdf_name": "interleaved.pdf",
        "texts": ["intro text", None, "middle text", None, "conclusion"],
        "images": [None, "page_img", None, "chart_img", None],
    }
    _write_tar_sample(tar_path, payload, image_name="interleaved.pdf.jpg", image_bytes=b"\xff\xd8\xff")
    task = task_for_tar(str(tar_path), "interleaved_test")
    reader = InterleavedWebdatasetReaderStage(sample_id_field="pdf_name")
    df = _as_df(reader.process(task))

    text_rows = df[df["modality"] == "text"].sort_values("position")
    image_rows = df[df["modality"] == "image"].sort_values("position")

    assert text_rows["position"].tolist() == [0, 2, 4]
    assert text_rows["text_content"].tolist() == ["intro text", "middle text", "conclusion"]

    assert image_rows["position"].tolist() == [1, 3]

    text_positions = set(text_rows["position"].tolist())
    image_positions = set(image_rows["position"].tolist())
    assert text_positions.isdisjoint(image_positions), "Text and image positions must not overlap"

    all_positions = sorted(text_positions | image_positions)
    assert all_positions == [0, 1, 2, 3, 4], "Interleaved positions should cover 0..N-1 without gaps"


def test_reader_empty_output_schema_includes_requested_passthrough_fields(tmp_path: Path) -> None:
    tar_path = tmp_path / "empty-no-json.tar"
    with tarfile.open(tar_path, "w") as tf:
        img_info = tarfile.TarInfo(name="image.jpg")
        img_info.size = 3
        tf.addfile(img_info, BytesIO(b"abc"))

    task = task_for_tar(str(tar_path), "empty_schema")
    reader = InterleavedWebdatasetReaderStage(fields=("p_hash",))
    df = _as_df(reader.process(task))
    assert "p_hash" in df.columns


def test_reader_fields_reserved_key_raises(tmp_path: Path) -> None:
    tar_path = tmp_path / "reserved_key.tar"
    payload = {"pdf_name": "doc.pdf", "texts": ["t"], "images": []}
    _write_tar_sample(tar_path, payload)
    task = task_for_tar(str(tar_path), "reserved_key")
    reader = InterleavedWebdatasetReaderStage(fields=("sample_id",))
    with pytest.raises(ValueError, match="fields contains reserved keys"):
        _ = reader.process(task)


def test_reader_fields_missing_key_warns_and_fills_none(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    tar_path = tmp_path / "missing_key.tar"
    payload = {"pdf_name": "doc.pdf", "texts": ["t"], "images": []}
    _write_tar_sample(tar_path, payload)
    task = task_for_tar(str(tar_path), "missing_key")
    reader = InterleavedWebdatasetReaderStage(fields=("p_hash",))
    with caplog.at_level("WARNING"):
        result = reader.process(task)
    df = _as_df(result)
    assert "p_hash" in df.columns
    meta_row = df[df["modality"] == "metadata"].iloc[0]
    assert meta_row["p_hash"] is None or pd.isna(meta_row["p_hash"])
    assert "Requested fields not found in source sample" in caplog.text


def test_reader_per_image_fields_distributed_to_image_rows(tmp_path: Path) -> None:
    """per_image_fields lists are distributed 1:1 to non-None image rows."""
    tar_path = tmp_path / "per-image.tar"
    payload = {
        "pdf_name": "doc.pdf",
        "texts": ["hello", None, "world"],
        "images": [None, "img_token", None],
        "image_metadata": [{"height": 100, "width": 200}],
    }
    _write_tar_sample(tar_path, payload)
    task = task_for_tar(str(tar_path), "per_image")
    reader = InterleavedWebdatasetReaderStage(
        per_image_fields=("image_metadata",),
    )
    df = _as_df(reader.process(task))

    assert "image_metadata" in df.columns

    image_rows = df[df["modality"] == "image"]
    assert len(image_rows) == 1
    assert image_rows.iloc[0]["image_metadata"] == json.dumps({"height": 100, "width": 200}, ensure_ascii=False)

    text_rows = df[df["modality"] == "text"]
    assert all(pd.isna(v) for v in text_rows["image_metadata"])

    meta_rows = df[df["modality"] == "metadata"]
    assert all(pd.isna(v) for v in meta_rows["image_metadata"])


def test_reader_per_text_fields_distributed_to_text_rows(tmp_path: Path) -> None:
    """per_text_fields lists are distributed 1:1 to non-None text rows."""
    tar_path = tmp_path / "per-text.tar"
    payload = {
        "pdf_name": "doc.pdf",
        "texts": ["hello", None, "world"],
        "images": [None, "img_token", None],
        "text_scores": [0.95, 0.42],
    }
    _write_tar_sample(tar_path, payload)
    task = task_for_tar(str(tar_path), "per_text")
    reader = InterleavedWebdatasetReaderStage(
        per_text_fields=("text_scores",),
    )
    df = _as_df(reader.process(task))

    assert "text_scores" in df.columns

    text_rows = df[df["modality"] == "text"].sort_values("position")
    assert len(text_rows) == 2
    assert text_rows.iloc[0]["text_scores"] == 0.95
    assert text_rows.iloc[1]["text_scores"] == 0.42

    image_rows = df[df["modality"] == "image"]
    assert all(pd.isna(v) for v in image_rows["text_scores"])

    meta_rows = df[df["modality"] == "metadata"]
    assert all(pd.isna(v) for v in meta_rows["text_scores"])


def test_reader_per_image_and_per_text_fields_together(tmp_path: Path) -> None:
    """Both per_image_fields and per_text_fields work correctly in the same reader."""
    tar_path = tmp_path / "both-per-modality.tar"
    payload = {
        "pdf_name": "doc.pdf",
        "texts": ["intro", None, "conclusion"],
        "images": [None, "page_img", None],
        "image_metadata": [{"page": 1, "width": 640}],
        "text_lang": ["en", "fr"],
        "url": "https://example.com",
    }
    _write_tar_sample(tar_path, payload)
    task = task_for_tar(str(tar_path), "both_per_modality")
    reader = InterleavedWebdatasetReaderStage(
        per_image_fields=("image_metadata",),
        per_text_fields=("text_lang",),
    )
    df = _as_df(reader.process(task))

    image_rows = df[df["modality"] == "image"]
    assert image_rows.iloc[0]["image_metadata"] == json.dumps({"page": 1, "width": 640}, ensure_ascii=False)
    assert pd.isna(image_rows.iloc[0]["text_lang"])

    text_rows = df[df["modality"] == "text"].sort_values("position")
    assert text_rows.iloc[0]["text_lang"] == "en"
    assert text_rows.iloc[1]["text_lang"] == "fr"
    assert all(pd.isna(v) for v in text_rows["image_metadata"])

    meta_row = df[df["modality"] == "metadata"].iloc[0]
    assert meta_row["url"] == "https://example.com"
    assert pd.isna(meta_row["image_metadata"])
    assert pd.isna(meta_row["text_lang"])


def test_reader_per_modality_fields_excluded_from_sample_passthrough(tmp_path: Path) -> None:
    """Fields in per_image_fields/per_text_fields must not appear on the metadata row."""
    tar_path = tmp_path / "exclude-passthrough.tar"
    payload = {
        "pdf_name": "doc.pdf",
        "texts": ["text"],
        "images": [],
        "image_metadata": [],
        "text_scores": [],
        "url": "https://example.com",
    }
    _write_tar_sample(tar_path, payload)
    task = task_for_tar(str(tar_path), "exclude_pt")
    reader = InterleavedWebdatasetReaderStage(
        per_image_fields=("image_metadata",),
        per_text_fields=("text_scores",),
    )
    df = _as_df(reader.process(task))

    meta_row = df[df["modality"] == "metadata"].iloc[0]
    assert meta_row["url"] == "https://example.com"
    assert pd.isna(meta_row.get("image_metadata"))
    assert pd.isna(meta_row.get("text_scores"))


def test_reader_per_modality_field_missing_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A per-modality field absent from the source sample should warn, not crash."""
    tar_path = tmp_path / "missing-per-field.tar"
    payload = {
        "pdf_name": "doc.pdf",
        "texts": ["hello"],
        "images": [],
    }
    _write_tar_sample(tar_path, payload)
    task = task_for_tar(str(tar_path), "missing_per_field")
    reader = InterleavedWebdatasetReaderStage(
        per_image_fields=("image_metadata",),
    )
    with caplog.at_level("WARNING"):
        result = reader.process(task)
    df = _as_df(result)
    assert len(df) > 0
    assert "per-modality field 'image_metadata' not found in source sample" in caplog.text


def test_reader_raises_on_non_list_per_modality_field(tmp_path: Path) -> None:
    """A per-modality field that is not a list in the source sample must raise ValueError."""
    tar_path = tmp_path / "non-list-field.tar"
    payload = {
        "pdf_name": "doc.pdf",
        "texts": ["hello"],
        "images": [],
        "image_metadata": "not-a-list",
    }
    _write_tar_sample(tar_path, payload)
    task = task_for_tar(str(tar_path), "non_list_field")
    reader = InterleavedWebdatasetReaderStage(
        per_image_fields=("image_metadata",),
    )
    with pytest.raises(TypeError, match="must be a list"):
        reader.process(task)


# --- materialize_on_read: TIFF frame extraction ---


def test_reader_materialize_on_read_extracts_individual_tiff_frames(tmp_path: Path) -> None:
    """materialize_on_read must extract individual frames from multi-frame TIFFs.

    Real MINT-1T data has multi-frame TIFFs (9/10 TIFFs have >1 frame).
    Each image row's source_ref carries a frame_index; the read path must
    return a single-frame TIFF for each row, not the full multi-frame blob.
    """
    n_frames = 3
    tiff_bytes = build_multi_frame_tiff(n_frames)
    payload = {
        "pdf_name": "doc.pdf",
        "texts": ["text_0", None, None],
        "images": [None, "page_1_img", "page_2_img"],
    }
    tar_path = write_tar(
        tmp_path / "tiff-frames.tar",
        {"sample.json": json.dumps(payload).encode(), "doc.pdf.tiff": tiff_bytes},
    )
    task = task_for_tar(tar_path, "tiff_frame_test")
    reader = InterleavedWebdatasetReaderStage(
        sample_id_field="pdf_name",
        image_extensions=(".tiff",),
        materialize_on_read=True,
    )
    df = _as_df(reader.process(task))
    image_rows = df[df["modality"] == "image"].sort_values("position")
    assert len(image_rows) == 2

    full_tiff = Image.open(BytesIO(tiff_bytes))
    assert full_tiff.n_frames == n_frames

    seen_sizes = set()
    for _, row in image_rows.iterrows():
        bc = row["binary_content"]
        assert bc is not None, "binary_content must not be None for materialized image"
        assert pd.isna(row["materialize_error"]) or row["materialize_error"] is None

        frame_img = Image.open(BytesIO(bc))
        assert frame_img.n_frames == 1, "Each row must contain a single-frame TIFF"
        seen_sizes.add(frame_img.size)
        assert len(bc) < len(tiff_bytes), "Single frame must be smaller than full multi-frame TIFF"

    assert len(seen_sizes) == 2, "Distinct frames must have distinct dimensions"


def test_reader_materialize_on_read_records_error_for_missing_member(tmp_path: Path) -> None:
    """When materialize_on_read=True and _extract_tar_member returns None
    (corrupt/unreadable member), materialize_error must be set.

    The reader validates content_key against member_names, so a truly absent
    member can't be referenced.  We simulate the edge case (e.g. corrupt tar)
    by patching _extract_tar_member to return None.
    """
    from unittest.mock import patch

    tiff_bytes = build_multi_frame_tiff(1)
    payload = {
        "pdf_name": "doc.pdf",
        "texts": ["hello"],
        "images": ["page_0_img"],
    }
    tar_path = write_tar(
        tmp_path / "corrupt-member.tar",
        {"sample.json": json.dumps(payload).encode(), "doc.pdf.tiff": tiff_bytes},
    )
    task = task_for_tar(tar_path, "corrupt_member_test")
    reader = InterleavedWebdatasetReaderStage(
        sample_id_field="pdf_name",
        image_extensions=(".tiff",),
        materialize_on_read=True,
    )
    with patch.object(InterleavedWebdatasetReaderStage, "_extract_tar_member", return_value=None):
        df = _as_df(reader.process(task))

    image_rows = df[df["modality"] == "image"]
    assert len(image_rows) == 1

    row = image_rows.iloc[0]
    assert row["binary_content"] is None or pd.isna(row["binary_content"])
    assert isinstance(row["materialize_error"], str), "materialize_error must be set when extraction fails"


def test_reader_frame_counter_resets_per_content_key(tmp_path: Path) -> None:
    """When images resolve to different TIFF member files, each file must get independent 0-based frame indices."""
    tiff_a = build_multi_frame_tiff(2, width=30, height=20)
    tiff_b = build_multi_frame_tiff(3, width=50, height=40)
    payload = {
        "pdf_name": "doc.pdf",
        "texts": [None, None, None, None],
        "images": ["a.tiff", "a.tiff", "b.tiff", "b.tiff"],
    }
    tar_path = write_tar(
        tmp_path / "multi-tiff.tar",
        {"sample.json": json.dumps(payload).encode(), "a.tiff": tiff_a, "b.tiff": tiff_b},
    )
    task = task_for_tar(tar_path, "multi_tiff_test")
    reader = InterleavedWebdatasetReaderStage(
        sample_id_field="pdf_name",
        image_extensions=(".tiff",),
    )
    df = _as_df(reader.process(task))
    image_rows = df[df["modality"] == "image"].sort_values("position")
    assert len(image_rows) == 4

    refs = [InterleavedBatch.parse_source_ref(v) for v in image_rows["source_ref"].tolist()]
    assert refs[0]["member"] == "a.tiff"
    assert refs[0]["frame_index"] == 0
    assert refs[1]["member"] == "a.tiff"
    assert refs[1]["frame_index"] == 1
    assert refs[2]["member"] == "b.tiff"
    assert refs[2]["frame_index"] == 0, "frame_index must reset to 0 for a different TIFF file"
    assert refs[3]["member"] == "b.tiff"
    assert refs[3]["frame_index"] == 1


def test_reader_materialize_preserves_raw_bytes_on_frame_extraction_failure(tmp_path: Path) -> None:
    """When frame extraction fails (frame_index out of range), binary_content
    must still contain the original full TIFF bytes, not None."""
    tiff_bytes = build_multi_frame_tiff(1)
    payload = {
        "pdf_name": "doc.pdf",
        "texts": [None, None],
        "images": ["frame_0", "frame_1_oob"],
    }
    tar_path = write_tar(
        tmp_path / "oob-frame.tar",
        {"sample.json": json.dumps(payload).encode(), "doc.pdf.tiff": tiff_bytes},
    )
    task = task_for_tar(tar_path, "oob_frame_test")
    reader = InterleavedWebdatasetReaderStage(
        sample_id_field="pdf_name",
        image_extensions=(".tiff",),
        materialize_on_read=True,
    )
    df = _as_df(reader.process(task))
    image_rows = df[df["modality"] == "image"].sort_values("position")
    assert len(image_rows) == 2

    good_row = image_rows.iloc[0]
    assert good_row["binary_content"] is not None
    assert pd.isna(good_row["materialize_error"]) or good_row["materialize_error"] is None

    bad_row = image_rows.iloc[1]
    assert bad_row["binary_content"] is not None, "Original TIFF bytes must be preserved on extraction failure"
    assert bad_row["binary_content"] == tiff_bytes, "binary_content must be the full original TIFF"
    assert isinstance(bad_row["materialize_error"], str), "materialize_error must be set"
    assert "frame" in bad_row["materialize_error"]


# --- materialize_on_read: JPEG / PNG (non-TIFF formats) ---


def _make_image_bytes(fmt: str) -> bytes:
    """Return minimal valid image bytes for the given PIL format name."""
    buf = BytesIO()
    img = Image.new("RGB", (8, 8), color=(128, 64, 32))
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.mark.parametrize(
    ("ext", "fmt", "expected_content_type"),
    [
        (".jpg", "JPEG", "image/jpeg"),
        (".png", "PNG", "image/png"),
    ],
    ids=["jpeg", "png"],
)
def test_reader_materialize_on_read_jpeg_png_bytes_preserved(
    tmp_path: Path,
    ext: str,
    fmt: str,
    expected_content_type: str,
) -> None:
    """JPEG and PNG images must pass through materialize_on_read byte-for-byte.

    Unlike multi-frame TIFFs, these formats are single-frame and must never
    enter the _extract_tiff_frame path.  Verifies:
      - content_type is detected correctly
      - frame_index is absent from source_ref (not treated as TIFF)
      - binary_content exactly matches the original bytes
      - materialize_error is null
    """
    image_bytes = _make_image_bytes(fmt)
    image_member = f"sample{ext}"
    # Use the member name directly as the image token so the reader resolves it
    payload = {"texts": ["hello"], "images": [image_member]}
    tar_path = write_tar(
        tmp_path / f"wds{ext}.tar",
        {
            "sample.json": json.dumps(payload).encode(),
            image_member: image_bytes,
        },
    )
    task = task_for_tar(tar_path, f"{fmt.lower()}_test")
    reader = InterleavedWebdatasetReaderStage(materialize_on_read=True)
    df = _as_df(reader.process(task))

    image_rows = df[df["modality"] == "image"]
    assert len(image_rows) == 1, f"Expected 1 image row for {fmt}"

    row = image_rows.iloc[0]
    assert row["content_type"] == expected_content_type, f"content_type mismatch for {fmt}"
    assert row["binary_content"] == image_bytes, f"{fmt} bytes must be preserved verbatim"
    assert pd.isna(row["materialize_error"]) or row["materialize_error"] is None

    ref = InterleavedBatch.parse_source_ref(row["source_ref"])
    assert ref["frame_index"] is None, f"{fmt} must not have a frame_index in source_ref"

    # Confirm PIL can decode the round-tripped bytes
    decoded = Image.open(BytesIO(row["binary_content"]))
    assert decoded.format == fmt, f"Round-tripped bytes must decode as {fmt}"


# --- BaseInterleavedReader ---


def test_base_reader_inputs_outputs() -> None:
    reader = InterleavedWebdatasetReaderStage()
    assert reader.inputs() == (["data"], [])
    assert reader.outputs() == (["data"], ["sample_id", "position", "modality"])


# --- InterleavedWebdatasetReaderStage edge cases ---


def test_reader_empty_tar(tmp_path: Path) -> None:
    """Tar with no JSON members produces an empty batch with correct schema."""
    tar_path = tmp_path / "empty.tar"
    with tarfile.open(tar_path, "w") as tf:
        img_info = tarfile.TarInfo(name="image.jpg")
        img_info.size = 3
        tf.addfile(img_info, BytesIO(b"abc"))
    task = FileGroupTask(
        task_id="empty",
        dataset_name="d",
        data=[str(tar_path)],
        _metadata={"source_files": [str(tar_path)]},
    )
    reader = InterleavedWebdatasetReaderStage()
    result = reader.process(task)
    assert isinstance(result, InterleavedBatch)
    assert len(result.to_pandas()) == 0
    assert "sample_id" in result.get_columns()


def test_reader_multi_tar(tmp_path: Path) -> None:
    """Multiple tar paths in a single FileGroupTask combine rows from all tars."""
    for name, sample_id in [("shard1.tar", "doc1"), ("shard2.tar", "doc2")]:
        payload = {"pdf_name": f"{sample_id}.pdf", "texts": ["hello"], "images": []}
        write_tar(
            tmp_path / name,
            {f"{sample_id}.json": json.dumps(payload).encode(), f"{sample_id}.jpg": b"img"},
        )
    task = FileGroupTask(
        task_id="multi",
        dataset_name="d",
        data=[str(tmp_path / "shard1.tar"), str(tmp_path / "shard2.tar")],
        _metadata={"source_files": ["shard1.tar", "shard2.tar"]},
    )
    reader = InterleavedWebdatasetReaderStage()
    result = reader.process(task)
    if isinstance(result, list):
        all_dfs = [b.to_pandas() for b in result]
        df = pd.concat(all_dfs, ignore_index=True)
    else:
        df = result.to_pandas()
    assert df["sample_id"].nunique() == 2


def test_reader_max_batch_bytes_splits(tmp_path: Path) -> None:
    """Very small max_batch_bytes splits output into multiple InterleavedBatch."""
    for sample_id in ["doc1", "doc2"]:
        payload = {"pdf_name": f"{sample_id}.pdf", "texts": ["text"], "images": []}
        write_tar(
            tmp_path / f"{sample_id}.tar",
            {f"{sample_id}.json": json.dumps(payload).encode()},
        )
    task = FileGroupTask(
        task_id="split",
        dataset_name="d",
        data=[str(tmp_path / "doc1.tar"), str(tmp_path / "doc2.tar")],
        _metadata={"source_files": ["doc1.tar", "doc2.tar"]},
    )
    reader = InterleavedWebdatasetReaderStage(max_batch_bytes=1)
    result = reader.process(task)
    assert isinstance(result, list)
    assert len(result) >= 2
    for batch in result:
        assert "_processed_" in batch.task_id


def test_reader_source_files_per_split_only_contributing_tars(tmp_path: Path) -> None:
    """Each split's source_files lists only the tars that contributed rows to that split."""
    # doc1.tar has sample "doc1", doc2.tar has sample "doc2" — one sample per tar so
    # with max_batch_bytes=1 each split should contain exactly one sample from one tar.
    tar1 = str(tmp_path / "doc1.tar")
    tar2 = str(tmp_path / "doc2.tar")
    for sample_id, tar_path in [("doc1", tar1), ("doc2", tar2)]:
        payload = {"pdf_name": f"{sample_id}.pdf", "texts": ["hello"], "images": []}
        write_tar(Path(tar_path), {f"{sample_id}.json": json.dumps(payload).encode()})

    task = FileGroupTask(
        task_id="sf_split",
        dataset_name="d",
        data=[tar1, tar2],
        _metadata={"source_files": [tar1, tar2]},
    )
    reader = InterleavedWebdatasetReaderStage(max_batch_bytes=1)
    result = reader.process(task)

    assert isinstance(result, list), "expected multiple batches"
    assert len(result) == 2, f"expected exactly 2 splits, got {len(result)}"

    for batch in result:
        df = batch.to_pandas()
        sample_ids = set(df["sample_id"].tolist())
        src = batch._metadata["source_files"]
        assert len(src) == 1, f"expected 1 source file for split, got {src}"
        if "doc1" in sample_ids:
            assert src[0].startswith(tar1), f"doc1 split should point to {tar1}, got {src}"
            assert "::split_" in src[0], f"doc1 split should have ::split_ suffix, got {src}"
        elif "doc2" in sample_ids:
            assert src[0].startswith(tar2), f"doc2 split should point to {tar2}, got {src}"
            assert "::split_" in src[0], f"doc2 split should have ::split_ suffix, got {src}"
        else:
            pytest.fail(f"unexpected sample_ids in split: {sample_ids}")


@pytest.mark.parametrize(
    ("payload", "modality", "expected_count"),
    [
        pytest.param(
            {"pdf_name": "doc.pdf", "texts": "not a list", "images": []},
            "text",
            0,
            id="non_list_texts",
        ),
        pytest.param(
            {"pdf_name": "doc.pdf", "texts": ["hello"], "images": None},
            "image",
            0,
            id="non_list_images",
        ),
    ],
)
def test_reader_non_list_field(
    tmp_path: Path,
    payload: dict,
    modality: str,
    expected_count: int,
) -> None:
    """Non-list texts/images field produces no rows for that modality."""
    tar_path = write_tar(tmp_path / "shard.tar", {"sample.json": json.dumps(payload).encode()})
    df = _as_df(InterleavedWebdatasetReaderStage().process(task_for_tar(tar_path)))
    assert (df["modality"] == modality).sum() == expected_count


@pytest.mark.parametrize(
    ("image_token", "default_member", "member_names", "expected"),
    [
        pytest.param(None, "default.jpg", {"default.jpg"}, None, id="none_token"),
        pytest.param("explicit.jpg", "default.jpg", {"explicit.jpg", "default.jpg"}, "explicit.jpg", id="in_members"),
        pytest.param("unknown", "default.jpg", {"default.jpg"}, "default.jpg", id="fallback_to_default"),
    ],
)
def test_resolve_image_content_key(
    image_token: object,
    default_member: str | None,
    member_names: set[str],
    expected: str | None,
) -> None:
    result = InterleavedWebdatasetReaderStage._resolve_image_content_key(image_token, default_member, member_names)
    assert result == expected


def test_reader_uses_stem_as_sample_id(tmp_path: Path) -> None:
    """When sample_id_field is None, uses Path(member.name).stem as sample_id."""
    payload = {"pdf_name": "doc.pdf", "texts": ["hello"], "images": []}
    tar_path = write_tar(
        tmp_path / "stem.tar",
        {"my_custom_name.json": json.dumps(payload).encode()},
    )
    task = task_for_tar(tar_path)
    reader = InterleavedWebdatasetReaderStage()
    df = _as_df(reader.process(task))
    assert (df["sample_id"] == "my_custom_name").all()


def test_reader_unknown_fields_pass_through_by_default(tmp_path: Path) -> None:
    """All non-reserved JSON fields flow through as passthrough columns when fields= is not set."""
    payload = {"pdf_name": "doc.pdf", "url": "https://example.com", "texts": ["hi"], "images": []}
    tar_path = write_tar(
        tmp_path / "passthrough.tar",
        {"sample1.json": json.dumps(payload).encode()},
    )
    task = task_for_tar(tar_path)
    reader = InterleavedWebdatasetReaderStage()
    df = _as_df(reader.process(task))
    meta = df[df["modality"] == "metadata"].iloc[0]
    assert meta["pdf_name"] == "doc.pdf"
    assert meta["url"] == "https://example.com"


# ---------------------------------------------------------------------------
# InterleavedParquetReaderStage
# ---------------------------------------------------------------------------


def test_parquet_reader_roundtrip(tmp_path: Path) -> None:
    """Write a batch with the parquet writer, read it back; data matches."""
    batch = make_interleaved_batch(num_samples=2, include_images=False)
    pq_path = _write_parquet_task(batch, tmp_path / "out")

    task = FileGroupTask(task_id="pq_rt", dataset_name="d", data=[pq_path])
    reader = InterleavedParquetReaderStage()
    result = reader.process(task)
    assert isinstance(result, InterleavedBatch)
    df = result.to_pandas()

    assert set(df["sample_id"].tolist()) == {"sample_0", "sample_1"}
    text_rows = df[df["modality"] == "text"]
    assert set(text_rows["text_content"].tolist()) == {"Hello 0", "Hello 1"}
    assert result._metadata.get("source_files") == [pq_path]


def test_parquet_reader_missing_columns_filled_with_null(tmp_path: Path) -> None:
    """A parquet file with only 3 columns; all other schema cols become null."""
    minimal = pa.Table.from_pylist(
        [{"sample_id": "s1", "position": 0, "modality": "text"}],
        schema=pa.schema(
            [
                pa.field("sample_id", pa.string()),
                pa.field("position", pa.int32()),
                pa.field("modality", pa.string()),
            ]
        ),
    )
    pq_path = tmp_path / "minimal.parquet"
    pq.write_table(minimal, pq_path)

    task = FileGroupTask(task_id="minimal", dataset_name="d", data=[str(pq_path)])
    result = InterleavedParquetReaderStage().process(task)
    assert isinstance(result, InterleavedBatch)
    df = result.to_pandas()
    assert len(df) == 1
    assert pd.isna(df.loc[0, "text_content"])
    assert pd.isna(df.loc[0, "binary_content"])


def test_parquet_reader_fields_subset(tmp_path: Path) -> None:
    """fields=(...) reads only reserved cols + requested extras; others absent."""
    batch = make_interleaved_batch(num_samples=1, include_images=False)
    pq_path = _write_parquet_task(batch, tmp_path / "out")

    task = FileGroupTask(task_id="fields_sub", dataset_name="d", data=[pq_path])
    result = InterleavedParquetReaderStage(fields=("text_content",)).process(task)
    assert isinstance(result, InterleavedBatch)
    df = result.to_pandas()
    assert "text_content" in df.columns
    assert "binary_content" in df.columns  # reserved — always present


def test_parquet_reader_fields_null_fill_missing(tmp_path: Path) -> None:
    """A field in fields= that is absent from disk is null-filled, not errored."""
    batch = make_interleaved_batch(num_samples=1, include_images=False)
    pq_path = _write_parquet_task(batch, tmp_path / "out")

    task = FileGroupTask(task_id="null_fill", dataset_name="d", data=[pq_path])
    result = InterleavedParquetReaderStage(fields=("nonexistent_field",)).process(task)
    assert isinstance(result, InterleavedBatch)
    df = result.to_pandas()
    assert "nonexistent_field" in df.columns
    assert df["nonexistent_field"].isna().all()


def test_parquet_reader_extra_column_passthrough_by_default(tmp_path: Path) -> None:
    """fields=None reads ALL columns in the file — sample/text/image extra fields are preserved
    and land on the correct rows after read."""
    fake_jpg = b"\xff\xd8\xff\xe0fake"
    pq_path = str(tmp_path / "extra.parquet")
    pq.write_table(pa.Table.from_pylist(_make_aligned_rows(fake_jpg)), pq_path)

    result = InterleavedParquetReaderStage().process(
        FileGroupTask(task_id="extra_col", dataset_name="d", data=[pq_path])
    )
    assert isinstance(result, InterleavedBatch)
    _assert_field_alignment(
        result.to_pandas(),
        image_vals=['{"page": 0}', '{"page": 1}'],
        text_vals=["conf:0.9"],
        sample_val="doc A",
    )


def test_parquet_reader_extra_column_excluded_when_fields_set(tmp_path: Path) -> None:
    """When fields= is explicit, only listed extras are read; unlisted ones are dropped."""
    fake_jpg = b"\xff\xd8\xff\xe0fake"
    pq_path = str(tmp_path / "extra.parquet")
    pq.write_table(pa.Table.from_pylist(_make_aligned_rows(fake_jpg, num_images=1)), pq_path)

    # fields=("text_metadata",) → only text_metadata + reserved cols; others are NOT read
    result = InterleavedParquetReaderStage(fields=("text_metadata",)).process(
        FileGroupTask(task_id="fields_excl", dataset_name="d", data=[pq_path])
    )
    assert isinstance(result, InterleavedBatch)
    df = result.to_pandas()
    assert "text_metadata" in df.columns
    assert "image_metadata" not in df.columns
    assert "sample_metadata" not in df.columns


def test_parquet_reader_wds_to_pq_to_wds_roundtrip(tmp_path: Path) -> None:
    """WDS→PQ→WDS: sample/text/image extra fields survive the full round-trip through parquet.

    Primary regression test for the image_metadata loss bug: InterleavedParquetReaderStage
    with fields=None must read ALL columns, not just RESERVED_COLUMNS.
    """
    fake_jpg = b"\xff\xd8\xff\xe0fake"
    payload = {
        "sample_id": "s1",
        "sample_metadata": "doc A",
        "texts": ["hello world", None, None],
        "text_metadata": ["conf:0.9"],
        "images": [None, "s1.1.jpg", "s1.2.jpg"],
        "image_metadata": [{"page": 1}, {"page": 2}],
    }
    tar_path = write_tar(
        tmp_path / "input.tar",
        {"s1.json": json.dumps(payload).encode(), "s1.1.jpg": fake_jpg, "s1.2.jpg": fake_jpg},
    )

    wds_reader = InterleavedWebdatasetReaderStage(
        sample_id_field="sample_id",
        per_image_fields=("image_metadata",),
        per_text_fields=("text_metadata",),
    )

    batch_wds = wds_reader.process(task_for_tar(tar_path))
    assert isinstance(batch_wds, InterleavedBatch)
    _assert_field_alignment(
        batch_wds.to_pandas(),
        image_vals=['{"page": 1}', '{"page": 2}'],
        text_vals=["conf:0.9"],
        sample_val="doc A",
    )

    pq_task = InterleavedParquetWriterStage(
        path=str(tmp_path / "pq_out"), materialize_on_write=False, mode="overwrite"
    ).process(batch_wds)

    batch_pq = InterleavedParquetReaderStage().process(
        FileGroupTask(task_id="pq_rt", dataset_name="d", data=[pq_task.data[0]])
    )
    assert isinstance(batch_pq, InterleavedBatch)
    df_pq = batch_pq.to_pandas()
    for col in ("sample_metadata", "text_metadata", "image_metadata"):
        assert col in df_pq.columns, f"{col} lost in PQ read — bug regression"
    _assert_field_alignment(
        df_pq, image_vals=['{"page": 1}', '{"page": 2}'], text_vals=["conf:0.9"], sample_val="doc A"
    )

    wds2_task = InterleavedWebdatasetWriterStage(
        path=str(tmp_path / "wds_out"), materialize_on_write=False, mode="overwrite"
    ).process(batch_pq)
    batch_final = wds_reader.process(FileGroupTask(task_id="final", dataset_name="d", data=wds2_task.data))
    assert isinstance(batch_final, InterleavedBatch)
    _assert_field_alignment(
        batch_final.to_pandas(),
        image_vals=['{"page": 1}', '{"page": 2}'],
        text_vals=["conf:0.9"],
        sample_val="doc A",
    )


def test_parquet_reader_pq_to_wds_to_pq_roundtrip(tmp_path: Path) -> None:
    """PQ→WDS→PQ: sample/text/image extra columns survive a full write-to-WDS and back."""
    fake_jpg = b"\xff\xd8\xff\xe0fake"
    batch0 = InterleavedBatch(
        task_id="pq0",
        dataset_name="d",
        data=pa.Table.from_pylist(_make_aligned_rows(fake_jpg)),
        _metadata={"source_files": ["source.parquet"]},
    )

    pq_path0 = (
        InterleavedParquetWriterStage(path=str(tmp_path / "pq0_out"), materialize_on_write=False, mode="overwrite")
        .process(batch0)
        .data[0]
    )

    batch1 = InterleavedParquetReaderStage().process(FileGroupTask(task_id="pq1", dataset_name="d", data=[pq_path0]))
    assert isinstance(batch1, InterleavedBatch)
    _assert_field_alignment(
        batch1.to_pandas(),
        image_vals=['{"page": 0}', '{"page": 1}'],
        text_vals=["conf:0.9"],
        sample_val="doc A",
    )

    wds_task = InterleavedWebdatasetWriterStage(
        path=str(tmp_path / "wds1_out"), materialize_on_write=False, mode="overwrite"
    ).process(batch1)

    wds_reader = InterleavedWebdatasetReaderStage(
        sample_id_field="sample_id",
        per_image_fields=("image_metadata",),
        per_text_fields=("text_metadata",),
    )
    batch2 = wds_reader.process(FileGroupTask(task_id="wds1", dataset_name="d", data=wds_task.data))
    assert isinstance(batch2, InterleavedBatch)
    _assert_field_alignment(
        batch2.to_pandas(),
        image_vals=['{"page": 0}', '{"page": 1}'],
        text_vals=["conf:0.9"],
        sample_val="doc A",
    )

    pq_path1 = (
        InterleavedParquetWriterStage(path=str(tmp_path / "pq1_out"), materialize_on_write=False, mode="overwrite")
        .process(batch2)
        .data[0]
    )
    batch_final = InterleavedParquetReaderStage().process(
        FileGroupTask(task_id="pq_final", dataset_name="d", data=[pq_path1])
    )
    assert isinstance(batch_final, InterleavedBatch)
    _assert_field_alignment(
        batch_final.to_pandas(),
        image_vals=['{"page": 0}', '{"page": 1}'],
        text_vals=["conf:0.9"],
        sample_val="doc A",
    )


def test_parquet_reader_max_batch_bytes_splits(tmp_path: Path) -> None:
    """Two parquet files, one sample each; max_batch_bytes=1 → 2 splits,
    each split's source_files lists only its contributing file."""
    batch_a = make_interleaved_batch(num_samples=1, task_id="a", include_images=False)
    batch_b = make_interleaved_batch(num_samples=1, task_id="b", include_images=False)
    # Give distinct sample_ids
    rows_a = batch_a.to_pandas().copy()
    rows_a["sample_id"] = "doc_a"
    rows_b = batch_b.to_pandas().copy()
    rows_b["sample_id"] = "doc_b"

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    writer = InterleavedParquetWriterStage(path=str(out_a), materialize_on_write=False, mode="overwrite")
    pq_a = writer.process(InterleavedBatch(task_id="a", dataset_name="d", data=rows_a)).data[0]
    writer2 = InterleavedParquetWriterStage(path=str(out_b), materialize_on_write=False, mode="overwrite")
    pq_b = writer2.process(InterleavedBatch(task_id="b", dataset_name="d", data=rows_b)).data[0]

    task = FileGroupTask(task_id="split_test", dataset_name="d", data=[pq_a, pq_b])
    result = InterleavedParquetReaderStage(max_batch_bytes=1).process(task)

    assert isinstance(result, list)
    assert len(result) == 2
    for batch in result:
        sample_ids = set(batch.to_pandas()["sample_id"].tolist())
        src = batch._metadata["source_files"]
        assert len(src) == 1
        if "doc_a" in sample_ids:
            assert pq_a in src[0]
        elif "doc_b" in sample_ids:
            assert pq_b in src[0]


def test_parquet_reader_empty_file(tmp_path: Path) -> None:
    """An empty parquet file produces an empty InterleavedBatch with correct schema."""
    empty = pa.Table.from_pylist([], schema=INTERLEAVED_SCHEMA)
    pq_path = tmp_path / "empty.parquet"
    pq.write_table(empty, pq_path)

    task = FileGroupTask(task_id="empty", dataset_name="d", data=[str(pq_path)])
    result = InterleavedParquetReaderStage().process(task)
    assert isinstance(result, InterleavedBatch)
    assert len(result.to_pandas()) == 0


def test_parquet_reader_composite_decompose(tmp_path: Path) -> None:
    """InterleavedParquetReader.decompose() returns [FilePartitioningStage, InterleavedParquetReaderStage]."""
    reader = InterleavedParquetReader(file_paths=str(tmp_path))
    stages = reader.decompose()
    assert len(stages) == 2
    assert isinstance(stages[0], FilePartitioningStage)
    assert isinstance(stages[1], InterleavedParquetReaderStage)


def test_parquet_reader_empty_file_list_returns_empty_batch() -> None:
    result = InterleavedParquetReaderStage().process(FileGroupTask(task_id="t", dataset_name="d", data=[]))
    assert isinstance(result, InterleavedBatch)
    df = result.to_pandas()
    assert len(df) == 0
    assert set(INTERLEAVED_SCHEMA.names) <= set(df.columns)


def test_source_files_for_split_null_sample_ids_fallback(caplog) -> None:  # noqa: ANN001
    split = pa.table({"sample_id": pa.array([None, None], type=pa.string())})
    with caplog.at_level(logging.WARNING, logger="nemo_curator"):
        result = BaseInterleavedReader._source_files_for_split(split, 2, {}, ["/a.parquet", "/b.parquet"])
    assert result == ["/a.parquet::split_00002", "/b.parquet::split_00002"]
    assert any("falling back" in r.message for r in caplog.records)
