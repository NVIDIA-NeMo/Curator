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
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pytest

from nemo_curator.core.utils import split_table_by_group_max_bytes
from nemo_curator.stages.interleaved.io.reader import WebdatasetReader
from nemo_curator.stages.interleaved.stages import (
    BaseInterleavedAnnotatorStage,
    BaseInterleavedFilterStage,
    InterleavedAspectRatioFilterStage,
)
from nemo_curator.stages.interleaved.utils.materialization import (
    _classify_rows,
    _read_direct_file,
    materialize_task_binary_content,
)
from nemo_curator.tasks import InterleavedBatch
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA

from .conftest import build_multi_frame_tiff, write_tar

# --- helpers ---


def _make_tar(tmp_path: Path, members: dict[str, bytes], name: str = "shard.tar") -> str:
    tar_path = tmp_path / name
    with tarfile.open(tar_path, "w") as tf:
        for member_name, payload in members.items():
            info = tarfile.TarInfo(name=member_name)
            info.size = len(payload)
            tf.addfile(info, BytesIO(payload))
    return str(tar_path)


def _image_task(rows: list[dict], metadata: dict | None = None) -> InterleavedBatch:
    table = pa.Table.from_pylist(rows, schema=INTERLEAVED_SCHEMA)
    return InterleavedBatch(task_id="test", dataset_name="d", data=table, _metadata=metadata or {})


def _image_row(
    path: str | None,
    member: str | None = None,
    byte_offset: int | None = None,
    byte_size: int | None = None,
) -> dict:
    return {
        "sample_id": "s1",
        "position": 0,
        "modality": "image",
        "content_type": "image/jpeg",
        "text_content": None,
        "binary_content": None,
        "source_ref": InterleavedBatch.build_source_ref(
            path=path, member=member, byte_offset=byte_offset, byte_size=byte_size
        ),
        "materialize_error": None,
    }


# --- source_ref parsing tests ---


@pytest.fixture
def single_row_table() -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "sample_id": "s1",
                "position": 0,
                "modality": "text",
                "content_type": "text/plain",
                "text_content": "hello",
                "binary_content": None,
                "source_ref": json.dumps(
                    {"path": "/dataset/shard.tar", "member": "s1.json", "byte_offset": 10, "byte_size": 20}
                ),
                "materialize_error": None,
            }
        ],
        schema=INTERLEAVED_SCHEMA,
    )


@pytest.fixture
def single_row_task(single_row_table: pa.Table) -> InterleavedBatch:
    return InterleavedBatch(task_id="t1", dataset_name="d1", data=single_row_table)


def test_with_parsed_source_ref_columns(single_row_task: InterleavedBatch) -> None:
    df = single_row_task.with_parsed_source_ref_columns()
    assert df.loc[0, "_src_path"] == "/dataset/shard.tar"
    assert df.loc[0, "_src_member"] == "s1.json"
    assert df.loc[0, "_src_byte_offset"] == 10
    assert df.loc[0, "_src_byte_size"] == 20


def test_parse_source_ref_ignores_legacy_keys() -> None:
    legacy_format = json.dumps({"content_path": "/old/path.tar", "content_key": "old.json"})
    parsed = InterleavedBatch.parse_source_ref(legacy_format)
    assert parsed["path"] is None
    assert parsed["member"] is None
    assert parsed["byte_offset"] is None
    assert parsed["byte_size"] is None


def test_parse_source_ref_empty_values() -> None:
    assert InterleavedBatch.parse_source_ref(None)["path"] is None
    assert InterleavedBatch.parse_source_ref("")["path"] is None


# --- classify_rows tests ---


def test_classify_rows_direct_read() -> None:
    df = pd.DataFrame(
        {"_src_path": ["/img.jpg"], "_src_member": [None], "_src_byte_offset": [None], "_src_byte_size": [None]}
    )
    mask = pd.Series([True])
    result = _classify_rows(df, mask)
    assert "/img.jpg" in result.direct_read
    assert not result.tar_extract
    assert not result.range_read


def test_classify_rows_tar_extract() -> None:
    df = pd.DataFrame(
        {"_src_path": ["/shard.tar"], "_src_member": ["img.jpg"], "_src_byte_offset": [None], "_src_byte_size": [None]}
    )
    mask = pd.Series([True])
    result = _classify_rows(df, mask)
    assert "/shard.tar" in result.tar_extract
    assert not result.range_read
    assert not result.direct_read


def test_classify_rows_range_read() -> None:
    df = pd.DataFrame(
        {"_src_path": ["/shard.tar"], "_src_member": ["img.jpg"], "_src_byte_offset": [512], "_src_byte_size": [1024]}
    )
    mask = pd.Series([True])
    result = _classify_rows(df, mask)
    assert "/shard.tar" in result.range_read
    assert not result.tar_extract
    assert not result.direct_read
    entry = result.range_read["/shard.tar"][0]
    assert entry == (0, "img.jpg", 512, 1024, None)


def test_classify_rows_missing_path() -> None:
    df = pd.DataFrame(
        {"_src_path": [None], "_src_member": [None], "_src_byte_offset": [None], "_src_byte_size": [None]}
    )
    mask = pd.Series([True])
    result = _classify_rows(df, mask)
    assert result.missing == [0]


def test_classify_rows_mixed_batch() -> None:
    df = pd.DataFrame(
        {
            "_src_path": ["/img.jpg", "/shard.tar", "/shard.tar", None],
            "_src_member": [None, "a.jpg", "b.jpg", None],
            "_src_byte_offset": [None, None, 100, None],
            "_src_byte_size": [None, None, 200, None],
        }
    )
    mask = pd.Series([True, True, True, True])
    result = _classify_rows(df, mask)
    assert len(result.direct_read["/img.jpg"]) == 1
    assert len(result.tar_extract["/shard.tar"]) == 1
    assert len(result.range_read["/shard.tar"]) == 1
    assert result.missing == [3]


# --- materialize: direct read ---


def test_materialize_fills_binary_from_direct_path(tmp_path: Path) -> None:
    image_bytes = b"test-image-content"
    img_path = tmp_path / "test.jpg"
    img_path.write_bytes(image_bytes)

    task = _image_task([_image_row(path=str(img_path))])
    result = materialize_task_binary_content(task)
    df = result.to_pandas()
    assert df.loc[0, "binary_content"] == image_bytes
    assert pd.isna(df.loc[0, "materialize_error"])


# --- materialize: tar extract (no byte_offset) ---


def test_materialize_fills_binary_from_tar_extract(tmp_path: Path) -> None:
    payload = b"tar-image-bytes"
    tar_path = _make_tar(tmp_path, {"img.jpg": payload})

    task = _image_task([_image_row(path=tar_path, member="img.jpg")])
    result = materialize_task_binary_content(task)
    df = result.to_pandas()
    assert df.loc[0, "binary_content"] == payload
    assert pd.isna(df.loc[0, "materialize_error"])


def test_materialize_tar_extract_missing_member(tmp_path: Path) -> None:
    tar_path = _make_tar(tmp_path, {"other.jpg": b"data"})

    task = _image_task([_image_row(path=tar_path, member="missing.jpg")])
    result = materialize_task_binary_content(task)
    df = result.to_pandas()
    assert pd.isna(df.loc[0, "binary_content"]) or df.loc[0, "binary_content"] is None
    assert "missing member" in str(df.loc[0, "materialize_error"])


# --- materialize: range read (with byte_offset/byte_size) ---


def test_materialize_fills_binary_from_range_read(tmp_path: Path) -> None:
    payload = b"range-read-image-bytes"
    raw_file = tmp_path / "data.bin"
    raw_file.write_bytes(b"HEADER" + payload + b"FOOTER")

    task = _image_task([_image_row(path=str(raw_file), member="data.bin", byte_offset=6, byte_size=len(payload))])
    result = materialize_task_binary_content(task)
    df = result.to_pandas()
    assert df.loc[0, "binary_content"] == payload
    assert pd.isna(df.loc[0, "materialize_error"])


def test_materialize_range_read_bad_path(tmp_path: Path) -> None:
    task = _image_task([_image_row(path=str(tmp_path / "nonexistent.bin"), member="x", byte_offset=0, byte_size=10)])
    result = materialize_task_binary_content(task)
    df = result.to_pandas()
    assert isinstance(df.loc[0, "materialize_error"], str)


# --- materialize: range read deduplication ---


def test_materialize_range_read_deduplicates_identical_ranges(tmp_path: Path) -> None:
    payload = b"shared-image-bytes"
    raw_file = tmp_path / "data.bin"
    raw_file.write_bytes(b"HDR" + payload + b"TRL")

    rows = [
        _image_row(path=str(raw_file), member="img.tiff", byte_offset=3, byte_size=len(payload)),
        _image_row(path=str(raw_file), member="img.tiff", byte_offset=3, byte_size=len(payload)),
        _image_row(path=str(raw_file), member="img.tiff", byte_offset=3, byte_size=len(payload)),
    ]
    task = _image_task(rows)
    result = materialize_task_binary_content(task)
    df = result.to_pandas()
    for i in range(3):
        assert df.loc[i, "binary_content"] == payload
        assert pd.isna(df.loc[i, "materialize_error"])


# --- materialize: mixed batch ---


def test_materialize_mixed_strategies(tmp_path: Path) -> None:
    direct_bytes = b"direct-img"
    direct_path = tmp_path / "direct.jpg"
    direct_path.write_bytes(direct_bytes)

    tar_bytes = b"tar-img"
    tar_path = _make_tar(tmp_path, {"member.jpg": tar_bytes})

    range_bytes = b"range-img"
    range_file = tmp_path / "range.bin"
    range_file.write_bytes(b"XX" + range_bytes + b"YY")

    rows = [
        _image_row(path=str(direct_path)),
        _image_row(path=tar_path, member="member.jpg"),
        _image_row(path=str(range_file), member="range.bin", byte_offset=2, byte_size=len(range_bytes)),
    ]
    for i, row in enumerate(rows):
        row["position"] = i
    task = _image_task(rows)
    result = materialize_task_binary_content(task)
    df = result.to_pandas()
    assert df.loc[0, "binary_content"] == direct_bytes
    assert df.loc[1, "binary_content"] == tar_bytes
    assert df.loc[2, "binary_content"] == range_bytes


# --- materialize: edge cases ---


def test_materialize_empty_task() -> None:
    task = InterleavedBatch(
        task_id="empty",
        dataset_name="d",
        data=pa.table(
            {
                "sample_id": pa.array([], type=pa.string()),
                "position": pa.array([], type=pa.int32()),
                "modality": pa.array([], type=pa.string()),
                "content_type": pa.array([], type=pa.string()),
                "text_content": pa.array([], type=pa.string()),
                "binary_content": pa.array([], type=pa.large_binary()),
                "source_ref": pa.array([], type=pa.string()),
                "materialize_error": pa.array([], type=pa.string()),
            }
        ),
    )
    result = materialize_task_binary_content(task)
    assert result.num_items == 0


def test_materialize_no_image_rows() -> None:
    table = pa.Table.from_pylist(
        [
            {
                "sample_id": "s1",
                "position": 0,
                "modality": "text",
                "content_type": "text/plain",
                "text_content": "hello",
                "binary_content": None,
                "source_ref": None,
                "materialize_error": None,
            }
        ],
        schema=INTERLEAVED_SCHEMA,
    )
    task = InterleavedBatch(task_id="no_img", dataset_name="d", data=table)
    result = materialize_task_binary_content(task)
    assert result.num_items == 1


def test_materialize_missing_path_sets_error() -> None:
    task = _image_task([_image_row(path=None)])
    result = materialize_task_binary_content(task)
    df = result.to_pandas()
    assert "missing path" in str(df.loc[0, "materialize_error"])


# --- aspect ratio filter ---


def test_aspect_ratio_filter_handles_non_default_dataframe_index() -> None:
    df = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "position": 0,
                "modality": "text",
                "content_type": "text/plain",
                "text_content": "ok",
                "binary_content": None,
                "source_ref": None,
                "materialize_error": None,
            },
            {
                "sample_id": "s1",
                "position": 1,
                "modality": "image",
                "content_type": "image/jpeg",
                "text_content": None,
                "binary_content": b"not-a-valid-jpeg",
                "source_ref": None,
                "materialize_error": None,
            },
        ]
    )
    df.index = pd.Index([10, 42])
    task = InterleavedBatch(task_id="non_default_index", dataset_name="d1", data=df)
    stage = InterleavedAspectRatioFilterStage(drop_invalid_rows=False)
    out = stage.process(task).to_pandas()
    assert len(out) == 1
    assert out.iloc[0]["modality"] == "text"


def test_aspect_ratio_filter_works_on_png_images() -> None:
    """The filter must apply to all image formats, not just JPEG."""
    from PIL import Image as PILImage

    buf = BytesIO()
    PILImage.new("RGB", (200, 100)).save(buf, format="PNG")
    valid_png = buf.getvalue()

    narrow_buf = BytesIO()
    PILImage.new("RGB", (10, 100)).save(narrow_buf, format="PNG")
    narrow_png = narrow_buf.getvalue()

    df = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "position": 0,
                "modality": "text",
                "content_type": "text/plain",
                "text_content": "ok",
                "binary_content": None,
                "source_ref": None,
                "materialize_error": None,
            },
            {
                "sample_id": "s1",
                "position": 1,
                "modality": "image",
                "content_type": "image/png",
                "text_content": None,
                "binary_content": valid_png,
                "source_ref": None,
                "materialize_error": None,
            },
            {
                "sample_id": "s1",
                "position": 2,
                "modality": "image",
                "content_type": "image/png",
                "text_content": None,
                "binary_content": narrow_png,
                "source_ref": None,
                "materialize_error": None,
            },
        ]
    )
    task = InterleavedBatch(task_id="png_test", dataset_name="d1", data=df)
    stage = InterleavedAspectRatioFilterStage(min_aspect_ratio=0.2, max_aspect_ratio=5.0, drop_invalid_rows=False)
    out = stage.process(task).to_pandas()
    assert len(out) == 2
    assert out["modality"].tolist() == ["text", "image"]
    assert out["position"].tolist() == [0, 1]


# --- split_table_by_group_max_bytes tests ---


def test_split_table_none_max_bytes() -> None:
    table = pa.table({"g": ["a", "a", "b"], "v": [1, 2, 3]})
    result = split_table_by_group_max_bytes(table, "g", None)
    assert len(result) == 1
    assert result[0].num_rows == 3


def test_split_table_empty_table() -> None:
    table = pa.table({"g": pa.array([], type=pa.string()), "v": pa.array([], type=pa.int64())})
    result = split_table_by_group_max_bytes(table, "g", 100)
    assert len(result) == 1
    assert result[0].num_rows == 0


def test_split_table_invalid_max_bytes() -> None:
    table = pa.table({"g": ["a"], "v": [1]})
    with pytest.raises(ValueError, match="max_batch_bytes must be > 0"):
        split_table_by_group_max_bytes(table, "g", 0)


def test_split_table_missing_column() -> None:
    table = pa.table({"g": ["a"], "v": [1]})
    with pytest.raises(ValueError, match="not found in table"):
        split_table_by_group_max_bytes(table, "missing", 100)


def test_split_table_single_large_group() -> None:
    table = pa.table({"g": ["a"] * 100, "v": list(range(100))})
    result = split_table_by_group_max_bytes(table, "g", 1)
    assert len(result) == 1
    assert result[0].num_rows == 100


def test_split_table_multiple_groups_split() -> None:
    table = pa.table({"g": ["a", "a", "b", "b", "c", "c"], "v": [1, 2, 3, 4, 5, 6]})
    small_limit = table.slice(0, 2).nbytes + 1
    result = split_table_by_group_max_bytes(table, "g", small_limit)
    assert len(result) >= 2
    total_rows = sum(t.num_rows for t in result)
    assert total_rows == 6


def test_split_table_preserves_group_integrity() -> None:
    table = pa.table({"g": ["a", "b", "a", "b"], "v": [1, 2, 3, 4]})
    result = split_table_by_group_max_bytes(table, "g", 1)
    for chunk in result:
        groups = chunk["g"].to_pylist()
        assert len(set(groups)) == 1 or all(g == groups[0] for g in groups)


# --- basic_row_validity_mask tests ---


def test_basic_row_validity_mask_filters_bad_modality() -> None:
    df = pd.DataFrame({"modality": ["text", "image", "video", "metadata"], "position": [0, 1, 2, -1]})
    mask = BaseInterleavedFilterStage._basic_row_validity_mask(df)
    assert mask.tolist() == [True, True, False, True]


def test_basic_row_validity_mask_enforces_position_rules() -> None:
    df = pd.DataFrame({"modality": ["metadata", "metadata", "text", "text"], "position": [-1, 0, 0, -1]})
    mask = BaseInterleavedFilterStage._basic_row_validity_mask(df)
    assert mask.tolist() == [True, False, True, False]


# --- filter position preservation test ---


def test_filter_recomputes_positions_after_drop() -> None:
    """Filtering must recompute content positions to close gaps; metadata stays at -1."""

    class _DropOddPositions(BaseInterleavedFilterStage):
        name: str = "drop_odd"

        def content_keep_mask(self, task: InterleavedBatch, df: pd.DataFrame) -> pd.Series:
            pos = df["position"].astype(int)
            return ~((df["modality"] != "metadata") & (pos % 2 == 1))

    rows = [
        {
            "sample_id": "s1",
            "position": i,
            "modality": "text",
            "content_type": "text/plain",
            "text_content": f"t{i}",
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        }
        for i in range(4)
    ] + [
        {
            "sample_id": "s1",
            "position": -1,
            "modality": "metadata",
            "content_type": "application/json",
            "text_content": None,
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        },
    ]
    task = InterleavedBatch(
        task_id="pos_test",
        dataset_name="d",
        data=pa.Table.from_pylist(rows, schema=INTERLEAVED_SCHEMA),
    )
    stage = _DropOddPositions(drop_invalid_rows=False)
    result = stage.process(task)
    out_df = result.to_pandas()
    assert out_df["position"].tolist() == [-1, 0, 1]
    assert pd.isna(out_df["text_content"].iloc[0])
    assert out_df["text_content"].iloc[1] == "t0"
    assert out_df["text_content"].iloc[2] == "t2"


def test_filter_preserves_interleaved_ordering_across_modalities() -> None:
    """When text and image rows are interleaved, filtering must preserve relative order."""

    class _DropSecondImage(BaseInterleavedFilterStage):
        name: str = "drop_second_image"

        def content_keep_mask(self, task: InterleavedBatch, df: pd.DataFrame) -> pd.Series:
            keep = pd.Series(True, index=df.index, dtype=bool)
            image_indices = df.index[df["modality"] == "image"].tolist()
            if len(image_indices) > 1:
                keep.loc[image_indices[1]] = False
            return keep

    def _row(sample_id: str, position: int, modality: str, text: str | None = None) -> dict:
        return {
            "sample_id": sample_id,
            "position": position,
            "modality": modality,
            "content_type": "text/plain" if modality == "text" else "image/jpeg",
            "text_content": text,
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        }

    rows = [
        {
            "sample_id": "s1",
            "position": -1,
            "modality": "metadata",
            "content_type": "application/json",
            "text_content": None,
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        },
        _row("s1", 0, "text", "intro"),
        _row("s1", 1, "image"),
        _row("s1", 2, "text", "middle"),
        _row("s1", 3, "image"),
        _row("s1", 4, "text", "end"),
    ]
    task = InterleavedBatch(
        task_id="interleave_test",
        dataset_name="d",
        data=pa.Table.from_pylist(rows, schema=INTERLEAVED_SCHEMA),
    )
    stage = _DropSecondImage(drop_invalid_rows=False)
    result = stage.process(task)
    out_df = result.to_pandas()

    assert out_df["modality"].tolist() == ["metadata", "text", "image", "text", "text"]
    assert out_df["position"].tolist() == [-1, 0, 1, 2, 3]
    content = out_df[out_df["modality"] != "metadata"]
    assert content["text_content"].tolist()[0] == "intro"
    assert content["text_content"].tolist()[2] == "middle"
    assert content["text_content"].tolist()[3] == "end"


def test_filter_preserves_interleaved_ordering_with_noninterleaved_row_order() -> None:
    """Even when DataFrame rows are grouped by modality (not position order), filter must preserve interleaving."""

    class _KeepAll(BaseInterleavedFilterStage):
        name: str = "keep_all"

        def content_keep_mask(self, task: InterleavedBatch, df: pd.DataFrame) -> pd.Series:
            return pd.Series(True, index=df.index, dtype=bool)

    def _row(sample_id: str, position: int, modality: str, text: str | None = None) -> dict:
        return {
            "sample_id": sample_id,
            "position": position,
            "modality": modality,
            "content_type": "text/plain" if modality == "text" else "image/jpeg",
            "text_content": text,
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        }

    rows = [
        {
            "sample_id": "s1",
            "position": -1,
            "modality": "metadata",
            "content_type": "application/json",
            "text_content": None,
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        },
        _row("s1", 0, "text", "intro"),
        _row("s1", 2, "text", "middle"),
        _row("s1", 4, "text", "end"),
        _row("s1", 1, "image"),
        _row("s1", 3, "image"),
    ]
    task = InterleavedBatch(
        task_id="noninterleaved_row_order",
        dataset_name="d",
        data=pa.Table.from_pylist(rows, schema=INTERLEAVED_SCHEMA),
    )
    stage = _KeepAll(drop_invalid_rows=False)
    result = stage.process(task)
    out_df = result.to_pandas()

    assert out_df["modality"].tolist() == ["metadata", "text", "image", "text", "image", "text"]
    assert out_df["position"].tolist() == [-1, 0, 1, 2, 3, 4]


def test_filter_drops_orphaned_metadata_rows() -> None:
    """When all content rows for a sample are filtered out, the metadata row must also be removed."""

    class _DropAllSample2Content(BaseInterleavedFilterStage):
        name: str = "drop_s2"

        def content_keep_mask(self, task: InterleavedBatch, df: pd.DataFrame) -> pd.Series:
            keep = pd.Series(True, index=df.index, dtype=bool)
            keep &= ~((df["sample_id"] == "s2") & (df["modality"] != "metadata"))
            return keep

    def _row(sample_id: str, position: int, modality: str, text: str | None = None) -> dict:
        return {
            "sample_id": sample_id,
            "position": position,
            "modality": modality,
            "content_type": "text/plain" if modality == "text" else "application/json",
            "text_content": text,
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        }

    rows = [
        _row("s1", -1, "metadata"),
        _row("s1", 0, "text", "hello"),
        _row("s1", 1, "text", "world"),
        _row("s2", -1, "metadata"),
        _row("s2", 0, "text", "dropped1"),
        _row("s2", 1, "text", "dropped2"),
    ]
    task = InterleavedBatch(
        task_id="orphan_test",
        dataset_name="d",
        data=pa.Table.from_pylist(rows, schema=INTERLEAVED_SCHEMA),
    )
    stage = _DropAllSample2Content(drop_invalid_rows=False)
    result = stage.process(task)
    out_df = result.to_pandas()

    assert set(out_df["sample_id"]) == {"s1"}, "s2 must be fully removed (including metadata)"
    assert len(out_df) == 3
    assert out_df["modality"].tolist() == ["metadata", "text", "text"]
    assert out_df["position"].tolist() == [-1, 0, 1]


# --- count / num_samples tests ---


def test_count_and_num_items() -> None:
    table = pa.Table.from_pylist(
        [
            {
                "sample_id": "s1",
                "position": 0,
                "modality": "text",
                "content_type": None,
                "text_content": "a",
                "binary_content": None,
                "source_ref": None,
                "materialize_error": None,
            },
            {
                "sample_id": "s1",
                "position": 1,
                "modality": "image",
                "content_type": None,
                "text_content": None,
                "binary_content": None,
                "source_ref": None,
                "materialize_error": None,
            },
            {
                "sample_id": "s2",
                "position": 0,
                "modality": "text",
                "content_type": None,
                "text_content": "b",
                "binary_content": None,
                "source_ref": None,
                "materialize_error": None,
            },
        ],
        schema=INTERLEAVED_SCHEMA,
    )
    task = InterleavedBatch(task_id="cnt", dataset_name="d", data=table)
    assert task.num_items == 2
    assert task.count() == 3
    assert task.count(modality="text") == 2
    assert task.count(modality="image") == 1
    assert task.count(modality="metadata") == 0


def test_count_with_pandas_data() -> None:
    table = pa.Table.from_pylist(
        [
            {
                "sample_id": "s1",
                "position": 0,
                "modality": "text",
                "content_type": None,
                "text_content": "a",
                "binary_content": None,
                "source_ref": None,
                "materialize_error": None,
            },
            {
                "sample_id": "s1",
                "position": 1,
                "modality": "image",
                "content_type": None,
                "text_content": None,
                "binary_content": None,
                "source_ref": None,
                "materialize_error": None,
            },
        ],
        schema=INTERLEAVED_SCHEMA,
    )
    task = InterleavedBatch(task_id="pd_cnt", dataset_name="d", data=table.to_pandas())
    assert task.num_items == 1
    assert task.count() == 2
    assert task.count(modality="image") == 1


# --- CompositeStage decomposition test ---


def test_webdataset_reader_composite_decompose(tmp_path: Path) -> None:
    reader = WebdatasetReader(file_paths=str(tmp_path), source_id_field="pdf_name")
    stages = reader.decompose()
    assert len(stages) == 2
    assert stages[0].name == "file_partitioning"
    assert stages[1].name == "webdataset_reader"


# --- exception broadening in materialization ---


def test_read_direct_file_handles_non_oserror_exceptions() -> None:
    """_read_direct_file must gracefully return None for non-OSError exceptions
    (e.g. RuntimeError from fsspec plugins) instead of crashing.
    """
    with patch(
        "nemo_curator.stages.interleaved.utils.materialization.fsspec.open", side_effect=RuntimeError("plugin error")
    ):
        result = _read_direct_file("/some/path.jpg", {})
    assert result is None


def test_materialize_records_error_for_non_oserror_on_direct_read() -> None:
    """Non-OSError exceptions during direct-read materialization must be
    recorded as materialize_error, not crash the pipeline.
    """
    task = _image_task([_image_row(path="/fake/path.jpg")])
    with patch("nemo_curator.stages.interleaved.utils.materialization.fsspec.open", side_effect=RuntimeError("boom")):
        result = materialize_task_binary_content(task)
    df = result.to_pandas()
    assert isinstance(df.loc[0, "materialize_error"], str)


# --- iter_materialized_bytes: only materializes masked rows ---


def test_iter_materialized_bytes_only_yields_masked_rows(tmp_path: Path) -> None:
    """iter_materialized_bytes must only materialize and yield bytes for
    the subset of rows selected by row_mask.
    """
    image_bytes_a = b"image-a-bytes"
    image_bytes_b = b"image-b-bytes"
    file_a = tmp_path / "a.jpg"
    file_b = tmp_path / "b.jpg"
    file_a.write_bytes(image_bytes_a)
    file_b.write_bytes(image_bytes_b)

    rows = [
        {
            "sample_id": "s1",
            "position": -1,
            "modality": "metadata",
            "content_type": "application/json",
            "text_content": None,
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        },
        {
            "sample_id": "s1",
            "position": 0,
            "modality": "text",
            "content_type": "text/plain",
            "text_content": "hello",
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        },
        {
            "sample_id": "s1",
            "position": 1,
            "modality": "image",
            "content_type": "image/jpeg",
            "text_content": None,
            "binary_content": None,
            "source_ref": InterleavedBatch.build_source_ref(path=str(file_a), member=None),
            "materialize_error": None,
        },
        {
            "sample_id": "s1",
            "position": 2,
            "modality": "image",
            "content_type": "image/jpeg",
            "text_content": None,
            "binary_content": None,
            "source_ref": InterleavedBatch.build_source_ref(path=str(file_b), member=None),
            "materialize_error": None,
        },
    ]
    task = InterleavedBatch(
        task_id="iter_test",
        dataset_name="d",
        data=pa.Table.from_pylist(rows, schema=INTERLEAVED_SCHEMA),
    )
    df = task.to_pandas().copy()

    image_mask = df["modality"] == "image"
    only_first_image = image_mask & (df["position"] == 1)

    stage = InterleavedAspectRatioFilterStage()
    yielded = list(stage.iter_materialized_bytes(task, df, only_first_image))
    assert len(yielded) == 1, "Must only yield for the single masked row"

    idx, raw_bytes = yielded[0]
    assert idx == df[only_first_image].index[0], "Must yield the original df index"
    assert raw_bytes == image_bytes_a


def test_iter_materialized_bytes_preserves_original_indices(tmp_path: Path) -> None:
    """Yielded indices must be the original DataFrame indices, even when
    the DataFrame has a non-default index.
    """
    img_bytes = b"test-bytes"
    img_path = tmp_path / "img.jpg"
    img_path.write_bytes(img_bytes)

    rows = [
        {
            "sample_id": "s1",
            "position": 0,
            "modality": "image",
            "content_type": "image/jpeg",
            "text_content": None,
            "binary_content": None,
            "source_ref": InterleavedBatch.build_source_ref(path=str(img_path), member=None),
            "materialize_error": None,
        },
    ]
    df = pd.DataFrame(rows)
    df.index = pd.Index([99])
    task = InterleavedBatch(task_id="idx_test", dataset_name="d", data=df)

    stage = InterleavedAspectRatioFilterStage()
    mask = pd.Series([True], index=df.index)
    yielded = list(stage.iter_materialized_bytes(task, df, mask))
    assert len(yielded) == 1
    assert yielded[0][0] == 99, "Must yield the original non-default index"
    assert yielded[0][1] == img_bytes


# --- TIFF frame materialization via write path ---


def test_materialize_extracts_individual_tiff_frames(tmp_path: Path) -> None:
    """materialize_task_binary_content must extract individual frames from
    multi-frame TIFFs when frame_index is present in source_ref.
    """
    n_frames = 3
    tiff_bytes = build_multi_frame_tiff(n_frames)
    tar_path = write_tar(tmp_path / "tiff.tar", {"doc.tiff": tiff_bytes})

    rows = []
    for i in range(n_frames):
        rows.append(
            {
                "sample_id": "s1",
                "position": i,
                "modality": "image",
                "content_type": "image/tiff",
                "text_content": None,
                "binary_content": None,
                "source_ref": InterleavedBatch.build_source_ref(
                    path=tar_path,
                    member="doc.tiff",
                    frame_index=i,
                ),
                "materialize_error": None,
            }
        )
    task = InterleavedBatch(
        task_id="tiff_mat",
        dataset_name="d",
        data=pa.Table.from_pylist(rows, schema=INTERLEAVED_SCHEMA),
    )
    result = materialize_task_binary_content(task)
    df = result.to_pandas()

    from PIL import Image

    for i in range(n_frames):
        bc = df.loc[i, "binary_content"]
        assert bc is not None
        frame_img = Image.open(BytesIO(bc))
        assert frame_img.n_frames == 1, f"Frame {i} must be a single-frame TIFF"
        assert len(bc) < len(tiff_bytes), "Single frame must be smaller than full multi-frame TIFF"


# --- annotator / filter stage edge cases ---


def test_annotator_process_empty_batch() -> None:
    """BaseInterleavedAnnotatorStage.process returns task unchanged for empty data."""

    class _Passthrough(BaseInterleavedAnnotatorStage):
        name: str = "passthrough"

        def annotate(self, task: InterleavedBatch, df: pd.DataFrame) -> pd.DataFrame:
            return df

    empty_table = pa.Table.from_pylist([], schema=INTERLEAVED_SCHEMA)
    task = InterleavedBatch(task_id="empty", dataset_name="d", data=empty_table)
    result = _Passthrough().process(task)
    assert result is task


def test_filter_drop_invalid_rows_true() -> None:
    """drop_invalid_rows=True (default) filters rows with bad modality or invalid position."""

    class _KeepAllContent(BaseInterleavedFilterStage):
        name: str = "keep_all"

        def content_keep_mask(self, task: InterleavedBatch, df: pd.DataFrame) -> pd.Series:
            return pd.Series(True, index=df.index, dtype=bool)

    rows = [
        {
            "sample_id": "s1",
            "position": -1,
            "modality": "metadata",
            "content_type": "application/json",
            "text_content": None,
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        },
        {
            "sample_id": "s1",
            "position": 0,
            "modality": "text",
            "content_type": "text/plain",
            "text_content": "ok",
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        },
        {
            "sample_id": "s1",
            "position": 1,
            "modality": "video",
            "content_type": "video/mp4",
            "text_content": None,
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        },
        {
            "sample_id": "s1",
            "position": -1,
            "modality": "text",
            "content_type": "text/plain",
            "text_content": "bad",
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        },
    ]
    task = InterleavedBatch(
        task_id="drop_test",
        dataset_name="d",
        data=pa.Table.from_pylist(rows, schema=INTERLEAVED_SCHEMA),
    )
    stage = _KeepAllContent(drop_invalid_rows=True)
    out_df = stage.process(task).to_pandas()
    assert len(out_df) == 2
    assert out_df["modality"].tolist() == ["metadata", "text"]


def test_iter_materialized_bytes_empty_mask() -> None:
    """iter_materialized_bytes yields nothing when row_mask selects no rows."""
    rows = [
        {
            "sample_id": "s1",
            "position": 0,
            "modality": "text",
            "content_type": "text/plain",
            "text_content": "hello",
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        },
    ]
    task = InterleavedBatch(
        task_id="empty_mask",
        dataset_name="d",
        data=pa.Table.from_pylist(rows, schema=INTERLEAVED_SCHEMA),
    )
    df = task.to_pandas()
    stage = InterleavedAspectRatioFilterStage()
    empty_mask = pd.Series(False, index=df.index, dtype=bool)
    assert list(stage.iter_materialized_bytes(task, df, empty_mask)) == []


def test_annotate_metadata_only_rows() -> None:
    """Metadata-only rows are orphans and must all be dropped."""

    class _KeepAllContent(BaseInterleavedFilterStage):
        name: str = "keep_all"

        def content_keep_mask(self, task: InterleavedBatch, df: pd.DataFrame) -> pd.Series:
            return pd.Series(True, index=df.index, dtype=bool)

    rows = [
        {
            "sample_id": "s1",
            "position": -1,
            "modality": "metadata",
            "content_type": "application/json",
            "text_content": None,
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        },
        {
            "sample_id": "s2",
            "position": -1,
            "modality": "metadata",
            "content_type": "application/json",
            "text_content": None,
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        },
    ]
    task = InterleavedBatch(
        task_id="meta_only",
        dataset_name="d",
        data=pa.Table.from_pylist(rows, schema=INTERLEAVED_SCHEMA),
    )
    stage = _KeepAllContent(drop_invalid_rows=False)
    out_df = stage.process(task).to_pandas()
    assert len(out_df) == 0


def test_aspect_ratio_filter_no_image_rows() -> None:
    """Filter is a no-op when there are no image rows."""
    rows = [
        {
            "sample_id": "s1",
            "position": -1,
            "modality": "metadata",
            "content_type": "application/json",
            "text_content": None,
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        },
        {
            "sample_id": "s1",
            "position": 0,
            "modality": "text",
            "content_type": "text/plain",
            "text_content": "hello",
            "binary_content": None,
            "source_ref": None,
            "materialize_error": None,
        },
    ]
    task = InterleavedBatch(
        task_id="no_img",
        dataset_name="d",
        data=pa.Table.from_pylist(rows, schema=INTERLEAVED_SCHEMA),
    )
    stage = InterleavedAspectRatioFilterStage(drop_invalid_rows=False)
    out_df = stage.process(task).to_pandas()
    assert len(out_df) == 2
    assert out_df["modality"].tolist() == ["metadata", "text"]
