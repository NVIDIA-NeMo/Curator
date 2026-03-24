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
import tarfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from nemo_curator.stages.interleaved.io.readers.webdataset import WebdatasetReaderStage
from nemo_curator.tasks import FileGroupTask, InterleavedBatch

from .conftest import build_multi_frame_tiff, task_for_tar, write_tar


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
    with tarfile.open(tar_path, "w") as tf:
        json_blob = json.dumps(payload).encode("utf-8")
        json_info = tarfile.TarInfo(name=json_name)
        json_info.size = len(json_blob)
        tf.addfile(json_info, BytesIO(json_blob))
        img_info = tarfile.TarInfo(name=image_name)
        img_info.size = len(image_bytes)
        tf.addfile(img_info, BytesIO(image_bytes))


def _task_for_tar(tar_path: Path, task_id: str) -> FileGroupTask:
    return FileGroupTask(
        task_id=task_id,
        dataset_name="custom_dataset",
        data=[str(tar_path)],
        _metadata={"source_files": [str(tar_path)]},
    )


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
    task = _task_for_tar(tar_path, "file_group_custom")
    reader = WebdatasetReaderStage(
        sample_id_field="doc_id",
        source_id_field="source_doc",
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
    task = _task_for_tar(tar_path, "all_fields")
    reader = WebdatasetReaderStage(
        sample_id_field="doc_id",
        source_id_field="source_doc",
        texts_field="captions",
        images_field="frames",
        image_member_field="primary_image",
        json_extensions=(".meta.json",),
    )
    df = _as_df(reader.process(task))
    meta_row = df[df["modality"] == "metadata"].iloc[0]
    assert meta_row["p_hash"] == "phash-1"
    assert meta_row["score"] == 0.91
    assert meta_row["aux"] == json.dumps({"page": 3}, ensure_ascii=True)
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

    task = _task_for_tar(tar_path, "content_type_resolve")
    reader = WebdatasetReaderStage(
        sample_id_field="doc_id",
        source_id_field="source_doc",
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
    task = _task_for_tar(tar_path, "sub_image_test")
    reader = WebdatasetReaderStage(
        source_id_field="pdf_name",
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
    task = _task_for_tar(tar_path, "interleaved_test")
    reader = WebdatasetReaderStage(source_id_field="pdf_name", sample_id_field="pdf_name")
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

    task = _task_for_tar(tar_path, "empty_schema")
    reader = WebdatasetReaderStage(source_id_field="pdf_name", fields=("p_hash",))
    df = _as_df(reader.process(task))
    assert "p_hash" in df.columns


def test_reader_fields_reserved_key_raises(tmp_path: Path) -> None:
    tar_path = tmp_path / "reserved_key.tar"
    payload = {"pdf_name": "doc.pdf", "texts": ["t"], "images": []}
    _write_tar_sample(tar_path, payload)
    task = _task_for_tar(tar_path, "reserved_key")
    reader = WebdatasetReaderStage(source_id_field="pdf_name", fields=("sample_id",))
    with pytest.raises(ValueError, match="fields contains reserved keys"):
        _ = reader.process(task)


def test_reader_fields_missing_key_warns_and_fills_none(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    tar_path = tmp_path / "missing_key.tar"
    payload = {"pdf_name": "doc.pdf", "texts": ["t"], "images": []}
    _write_tar_sample(tar_path, payload)
    task = _task_for_tar(tar_path, "missing_key")
    reader = WebdatasetReaderStage(source_id_field="pdf_name", fields=("p_hash",))
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
    task = _task_for_tar(tar_path, "per_image")
    reader = WebdatasetReaderStage(
        source_id_field="pdf_name",
        per_image_fields=("image_metadata",),
    )
    df = _as_df(reader.process(task))

    assert "image_metadata" in df.columns

    image_rows = df[df["modality"] == "image"]
    assert len(image_rows) == 1
    assert image_rows.iloc[0]["image_metadata"] == json.dumps({"height": 100, "width": 200})

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
    task = _task_for_tar(tar_path, "per_text")
    reader = WebdatasetReaderStage(
        source_id_field="pdf_name",
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
    task = _task_for_tar(tar_path, "both_per_modality")
    reader = WebdatasetReaderStage(
        source_id_field="pdf_name",
        per_image_fields=("image_metadata",),
        per_text_fields=("text_lang",),
    )
    df = _as_df(reader.process(task))

    image_rows = df[df["modality"] == "image"]
    assert image_rows.iloc[0]["image_metadata"] == json.dumps({"page": 1, "width": 640})
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
    task = _task_for_tar(tar_path, "exclude_pt")
    reader = WebdatasetReaderStage(
        source_id_field="pdf_name",
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
    task = _task_for_tar(tar_path, "missing_per_field")
    reader = WebdatasetReaderStage(
        source_id_field="pdf_name",
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
    task = _task_for_tar(tar_path, "non_list_field")
    reader = WebdatasetReaderStage(
        source_id_field="pdf_name",
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
    reader = WebdatasetReaderStage(
        source_id_field="pdf_name",
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
    reader = WebdatasetReaderStage(
        source_id_field="pdf_name",
        sample_id_field="pdf_name",
        image_extensions=(".tiff",),
        materialize_on_read=True,
    )
    with patch.object(WebdatasetReaderStage, "_extract_tar_member", return_value=None):
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
    reader = WebdatasetReaderStage(
        source_id_field="pdf_name",
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
    reader = WebdatasetReaderStage(
        source_id_field="pdf_name",
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


# --- BaseInterleavedReader ---


def test_base_reader_inputs_outputs() -> None:
    reader = WebdatasetReaderStage(source_id_field="pdf_name")
    assert reader.inputs() == (["data"], [])
    assert reader.outputs() == (["data"], ["sample_id", "position", "modality"])


# --- WebdatasetReaderStage edge cases ---


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
    reader = WebdatasetReaderStage(source_id_field="pdf_name")
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
    reader = WebdatasetReaderStage(source_id_field="pdf_name")
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
    reader = WebdatasetReaderStage(source_id_field="pdf_name", max_batch_bytes=1)
    result = reader.process(task)
    assert isinstance(result, list)
    assert len(result) >= 2
    for batch in result:
        assert "_processed_" in batch.task_id


def test_reader_non_list_texts_field(tmp_path: Path) -> None:
    """Non-list texts field produces no text rows."""
    payload = {"pdf_name": "doc.pdf", "texts": "not a list", "images": []}
    tar_path = write_tar(
        tmp_path / "non_list.tar",
        {"sample.json": json.dumps(payload).encode()},
    )
    task = task_for_tar(tar_path)
    reader = WebdatasetReaderStage(source_id_field="pdf_name")
    df = _as_df(reader.process(task))
    assert (df["modality"] == "text").sum() == 0


def test_reader_non_list_images_field(tmp_path: Path) -> None:
    """Non-list images field produces no image rows."""
    payload = {"pdf_name": "doc.pdf", "texts": ["hello"], "images": None}
    tar_path = write_tar(
        tmp_path / "no_images.tar",
        {"sample.json": json.dumps(payload).encode()},
    )
    task = task_for_tar(tar_path)
    reader = WebdatasetReaderStage(source_id_field="pdf_name")
    df = _as_df(reader.process(task))
    assert (df["modality"] == "image").sum() == 0
    assert (df["modality"] == "text").sum() == 1


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
    result = WebdatasetReaderStage._resolve_image_content_key(image_token, default_member, member_names)
    assert result == expected


def test_reader_uses_stem_as_sample_id(tmp_path: Path) -> None:
    """When sample_id_field is None, uses Path(member.name).stem as sample_id."""
    payload = {"pdf_name": "doc.pdf", "texts": ["hello"], "images": []}
    tar_path = write_tar(
        tmp_path / "stem.tar",
        {"my_custom_name.json": json.dumps(payload).encode()},
    )
    task = task_for_tar(tar_path)
    reader = WebdatasetReaderStage(source_id_field="pdf_name", sample_id_field=None)
    df = _as_df(reader.process(task))
    assert (df["sample_id"] == "my_custom_name").all()
