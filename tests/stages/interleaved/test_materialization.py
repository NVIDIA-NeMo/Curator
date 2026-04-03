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

from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from PIL import Image

from nemo_curator.stages.interleaved.utils.materialization import (
    _build_global_range_index,
    _build_image_mask,
    _classify_rows,
    _extract_tiff_frame,
    _fill_range_read_rows,
    _fill_tar_extract_rows,
    _get_frame_index,
    _init_materialization_buffers,
    _scatter_range_blobs,
    materialize_task_binary_content,
)
from nemo_curator.tasks import InterleavedBatch
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA

from .conftest import build_jpeg_in_tiff, build_multi_frame_tiff, make_image_row, make_image_task, write_tar

# --- _get_frame_index ---


@pytest.mark.parametrize(
    ("val", "expected"),
    [
        pytest.param(None, None, id="none_value"),
        pytest.param(float("nan"), None, id="nan_value"),
    ],
)
def test_get_frame_index_returns_none_for_missing_values(val: object, expected: None) -> None:
    df = pd.DataFrame({"_src_frame_index": [val], "other": [1]})
    assert _get_frame_index(df, 0) is expected


# --- _classify_rows edge cases ---


@pytest.mark.parametrize(
    "path_val",
    [pytest.param(float("nan"), id="nan_path"), pytest.param("", id="empty_path")],
)
def test_classify_rows_missing_path_variants(path_val: object) -> None:
    df = pd.DataFrame(
        {
            "_src_path": [path_val],
            "_src_member": [None],
            "_src_byte_offset": [None],
            "_src_byte_size": [None],
        }
    )
    result = _classify_rows(df, pd.Series([True]))
    assert result.missing == [0]


def test_classify_rows_range_with_zero_size() -> None:
    df = pd.DataFrame(
        {
            "_src_path": ["/shard.tar"],
            "_src_member": ["img.jpg"],
            "_src_byte_offset": [100],
            "_src_byte_size": [0],
        }
    )
    result = _classify_rows(df, pd.Series([True]))
    assert "/shard.tar" in result.tar_extract
    assert not result.range_read


# --- _extract_tiff_frame ---


def _make_jpeg_bytes() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (10, 10)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.mark.parametrize(
    ("image_bytes", "frame_index"),
    [
        pytest.param(None, 0, id="non_tiff_passthrough"),
        pytest.param(None, 99, id="oob_frame_returns_none"),
        pytest.param(b"not-an-image", 0, id="corrupt_returns_none"),
    ],
)
def test_extract_tiff_frame_variants(image_bytes: bytes | None, frame_index: int) -> None:
    if image_bytes is None and frame_index == 0:
        jpeg_bytes = _make_jpeg_bytes()
        result = _extract_tiff_frame(jpeg_bytes, frame_index)
        assert result == jpeg_bytes
    elif image_bytes is None and frame_index == 99:
        tiff_bytes = build_multi_frame_tiff(1)
        result = _extract_tiff_frame(tiff_bytes, frame_index)
        assert result is None
    else:
        result = _extract_tiff_frame(image_bytes, frame_index)
        assert result is None


def test_extract_tiff_frame_jpeg_in_tiff_preserves_pixels() -> None:
    """Regression: JPEG-compressed TIFF frames must not corrupt pixel values.

    MINT-1T PDFs are stored as JPEG-in-TIFF multi-frame files.  Previously,
    _extract_tiff_frame reused the source JPEG compression when saving the
    extracted frame; JPEG does not support alpha channels, which caused wrong
    pixel values and a corrupted alpha channel on RGBA frames.
    """
    tiff_bytes = build_jpeg_in_tiff(n_frames=2)
    orig = Image.open(BytesIO(tiff_bytes))
    for frame_index in range(2):
        orig.seek(frame_index)
        orig_arr = np.array(orig)

        result = _extract_tiff_frame(tiff_bytes, frame_index)
        assert result is not None
        result_img = Image.open(BytesIO(result))
        assert result_img.mode == orig.mode
        assert result_img.size == orig.size
        # All pixels must be identical (lossless round-trip after JPEG decode)
        assert np.array_equal(np.array(result_img), orig_arr), f"Frame {frame_index}: pixel mismatch after extraction"


# --- _fill_tar_extract_rows ---


@pytest.mark.parametrize(
    ("tar_path_factory", "member", "frame_index", "expected_error"),
    [
        pytest.param(lambda _: "/nonexistent/path.tar", "img.jpg", None, "failed to read path", id="bad_tar_path"),
        pytest.param(
            lambda tmp: write_tar(tmp / "oob.tar", {"doc.tiff": build_multi_frame_tiff(1)}),
            "doc.tiff",
            99,
            "failed to extract frame",
            id="oob_frame",
        ),
    ],
)
def test_fill_tar_extract_rows_errors(
    tmp_path: Path,
    tar_path_factory: object,
    member: str,
    frame_index: int | None,
    expected_error: str,
) -> None:
    tar_path = tar_path_factory(tmp_path)
    groups = {tar_path: [(0, member, frame_index)]}
    binary_values: list[object] = [None]
    error_values: list[str | None] = [None]
    _fill_tar_extract_rows(groups, {}, binary_values, error_values)
    assert error_values[0] is not None
    assert expected_error in error_values[0]


# --- _scatter_range_blobs ---


@pytest.mark.parametrize(
    ("blob", "expected_error_substr"),
    [
        pytest.param(RuntimeError("fail"), "range read error", id="exception_blob"),
        pytest.param(None, "empty range read", id="none_blob"),
        pytest.param(b"", "empty range read", id="empty_blob"),
    ],
)
def test_scatter_range_blobs_error_cases(blob: object, expected_error_substr: str) -> None:
    range_keys = [(0, 10)]
    unique_ranges: dict[tuple[int, int], list[tuple[int, str, int | None]]] = {
        (0, 10): [(0, "img.jpg", None)],
    }
    binary_values: list[object] = [None]
    error_values: list[str | None] = [None]
    _scatter_range_blobs([blob], range_keys, unique_ranges, binary_values, error_values)
    assert error_values[0] is not None
    assert expected_error_substr in error_values[0]


def test_scatter_range_blobs_bytearray_conversion() -> None:
    range_keys = [(0, 10)]
    unique_ranges: dict[tuple[int, int], list[tuple[int, str, int | None]]] = {
        (0, 10): [(0, "img.jpg", None)],
    }
    binary_values: list[object] = [None]
    error_values: list[str | None] = [None]
    _scatter_range_blobs([bytearray(b"image-data")], range_keys, unique_ranges, binary_values, error_values)
    assert binary_values[0] == b"image-data"
    assert isinstance(binary_values[0], bytes)
    assert error_values[0] is None


@pytest.mark.parametrize(
    ("n_frames", "frame_index", "expect_success", "expected_error_substr"),
    [
        pytest.param(3, 1, True, None, id="valid_frame"),
        pytest.param(1, 99, False, "failed to extract frame", id="oob_frame"),
    ],
)
def test_scatter_range_blobs_tiff_frame(
    n_frames: int,
    frame_index: int,
    expect_success: bool,
    expected_error_substr: str | None,
) -> None:
    tiff_bytes = build_multi_frame_tiff(n_frames)
    range_keys = [(0, len(tiff_bytes))]
    unique_ranges: dict[tuple[int, int], list[tuple[int, str, int | None]]] = {
        (0, len(tiff_bytes)): [(0, "doc.tiff", frame_index)],
    }
    binary_values: list[object] = [None]
    error_values: list[str | None] = [None]
    _scatter_range_blobs([tiff_bytes], range_keys, unique_ranges, binary_values, error_values)
    if expect_success:
        assert binary_values[0] is not None
        assert error_values[0] is None
        assert Image.open(BytesIO(binary_values[0])).n_frames == 1
    else:
        assert error_values[0] is not None
        assert expected_error_substr in error_values[0]


# --- _fill_range_read_rows ---


def test_fill_range_read_rows_url_to_fs_failure() -> None:
    groups = {"bad://path": [(0, "img.jpg", 100, 200, None)]}
    binary_values: list[object] = [None]
    error_values: list[str | None] = [None]
    with patch(
        "nemo_curator.stages.interleaved.utils.materialization.url_to_fs",
        side_effect=ValueError("bad"),
    ):
        _fill_range_read_rows(groups, {}, binary_values, error_values)
    assert error_values[0] == "failed to resolve filesystem"


def test_build_global_range_index_groups_by_filesystem(tmp_path: Path) -> None:
    """Paths on different filesystems produce separate (fs, unique_ranges) groups."""
    import fsspec

    local_path = str(tmp_path / "a.bin")
    Path(local_path).write_bytes(b"x" * 64)

    mem_path = "memory://test_group_index/b.bin"
    with fsspec.open(mem_path, "wb") as f:
        f.write(b"y" * 64)

    groups = {
        local_path: [(0, "m0", 0, 8, None)],
        mem_path: [(1, "m1", 0, 8, None)],
    }
    error_values: list[str | None] = [None, None]
    result = _build_global_range_index(groups, {}, error_values)

    # Two distinct filesystem backends → two groups
    assert len(result) == 2
    fs_types = {type(fs).__name__ for fs, _ in result}
    assert "LocalFileSystem" in fs_types
    assert "MemoryFileSystem" in fs_types
    assert all(e is None for e in error_values)


def test_fill_range_read_rows_mixed_filesystems(tmp_path: Path) -> None:
    """Range reads work correctly when groups span local and in-memory filesystems."""
    import fsspec

    # Local file: embed target bytes at a known offset
    local_payload = b"LOCAL_PAYLOAD"
    local_prefix = b"HEADER_PREFIX_"
    local_content = local_prefix + local_payload + b"_SUFFIX"
    local_file = tmp_path / "local.bin"
    local_file.write_bytes(local_content)
    local_path = str(local_file)

    # Memory file: embed target bytes at a known offset
    mem_payload = b"MEMORY_PAYLOAD"
    mem_prefix = b"MEM_PREFIX_"
    mem_content = mem_prefix + mem_payload + b"_SUFFIX"
    mem_path = "memory://test_mixed_fs/data.bin"
    with fsspec.open(mem_path, "wb") as f:
        f.write(mem_content)

    groups = {
        local_path: [(0, "local_member", len(local_prefix), len(local_payload), None)],
        mem_path: [(1, "mem_member", len(mem_prefix), len(mem_payload), None)],
    }
    binary_values: list[object] = [None, None]
    error_values: list[str | None] = [None, None]

    _fill_range_read_rows(groups, {}, binary_values, error_values)

    assert binary_values[0] == local_payload
    assert binary_values[1] == mem_payload
    assert error_values[0] is None
    assert error_values[1] is None


# --- _init_materialization_buffers ---


@pytest.mark.parametrize(
    "drop_col",
    [pytest.param("materialize_error", id="no_materialize_error"), pytest.param("binary_content", id="no_binary")],
)
def test_init_materialization_buffers_missing_column(drop_col: str) -> None:
    df = pd.DataFrame({"modality": ["image"], "binary_content": [None], "materialize_error": [None]})
    df = df.drop(columns=[drop_col])
    binary_values, error_values = _init_materialization_buffers(df)
    assert len(binary_values) == 1
    assert len(error_values) == 1


# --- _build_image_mask ---


@pytest.mark.parametrize(
    ("df_data", "kwargs", "expected"),
    [
        pytest.param(
            {"other": [1, 2]},
            {"only_missing_binary": True, "image_content_types": None},
            [False, False],
            id="no_modality_column",
        ),
        pytest.param(
            {
                "modality": ["image", "image"],
                "content_type": ["image/jpeg", "image/png"],
                "binary_content": [None, None],
            },
            {"only_missing_binary": True, "image_content_types": ("image/jpeg",)},
            [True, False],
            id="content_type_filter",
        ),
        pytest.param(
            {
                "modality": ["image", "image"],
                "content_type": ["image/jpeg", "image/jpeg"],
                "binary_content": [b"existing", None],
            },
            {"only_missing_binary": False, "image_content_types": None},
            [True, True],
            id="only_missing_binary_false",
        ),
    ],
)
def test_build_image_mask(df_data: dict, kwargs: dict, expected: list[bool]) -> None:
    mask = _build_image_mask(pd.DataFrame(df_data), **kwargs)
    assert mask.tolist() == expected


# --- materialize_task_binary_content with content_type filter ---


def test_materialize_with_content_type_filter(tmp_path: Path) -> None:
    jpeg_bytes = b"jpeg-data"
    png_bytes = b"png-data"
    jpeg_path = tmp_path / "img.jpg"
    png_path = tmp_path / "img.png"
    jpeg_path.write_bytes(jpeg_bytes)
    png_path.write_bytes(png_bytes)

    rows = [
        make_image_row(path=str(jpeg_path), content_type="image/jpeg"),
        {**make_image_row(path=str(png_path), content_type="image/png"), "position": 1},
    ]
    task = make_image_task(rows)
    result = materialize_task_binary_content(task, image_content_types=("image/jpeg",))
    df = result.to_pandas()
    assert df.loc[0, "binary_content"] == jpeg_bytes
    assert df.loc[1, "binary_content"] is None or pd.isna(df.loc[1, "binary_content"])


def test_materialize_with_only_missing_binary_false(tmp_path: Path) -> None:
    new_bytes = b"fresh-image"
    img_path = tmp_path / "img.jpg"
    img_path.write_bytes(new_bytes)

    rows = [
        {
            "sample_id": "s1",
            "position": 0,
            "modality": "image",
            "content_type": "image/jpeg",
            "text_content": None,
            "binary_content": b"old-bytes",
            "source_ref": InterleavedBatch.build_source_ref(path=str(img_path), member=None),
            "materialize_error": None,
        }
    ]
    task = InterleavedBatch(
        task_id="re_mat",
        dataset_name="d",
        data=pa.Table.from_pylist(rows, schema=INTERLEAVED_SCHEMA),
    )
    result = materialize_task_binary_content(task, only_missing_binary=False)
    df = result.to_pandas()
    assert df.loc[0, "binary_content"] == new_bytes


def test_materialize_preserves_passthrough_columns_with_src_prefix(tmp_path: Path) -> None:
    """User passthrough columns starting with '_src_' must survive materialization unchanged.

    Previously, the cleanup step used startswith('_src_') which silently dropped any
    user column whose JSON key happened to start with that prefix (e.g. '_src_html').
    """
    img_path = tmp_path / "img.jpg"
    img_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)  # minimal JPEG header

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
            "_src_html": "keep-me",
            "_src_metadata": "also-keep-me",
        }
    ]
    schema_with_passthrough = INTERLEAVED_SCHEMA.append(pa.field("_src_html", pa.string())).append(
        pa.field("_src_metadata", pa.string())
    )
    task = InterleavedBatch(
        task_id="passthrough_test",
        dataset_name="d",
        data=pa.Table.from_pylist(rows, schema=schema_with_passthrough),
    )
    result = materialize_task_binary_content(task)
    df = result.to_pandas()

    assert "_src_html" in df.columns, "_src_html passthrough column was dropped"
    assert "_src_metadata" in df.columns, "_src_metadata passthrough column was dropped"
    assert df.loc[0, "_src_html"] == "keep-me"
    assert df.loc[0, "_src_metadata"] == "also-keep-me"
