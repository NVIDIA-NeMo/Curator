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

import pandas as pd
import pyarrow as pa
import pytest
from PIL import Image

from nemo_curator.stages.interleaved.utils.materialization import (
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

from .conftest import build_multi_frame_tiff, write_tar


def _image_task(rows: list[dict], metadata: dict | None = None) -> InterleavedBatch:
    table = pa.Table.from_pylist(rows, schema=INTERLEAVED_SCHEMA)
    return InterleavedBatch(task_id="test", dataset_name="d", data=table, _metadata=metadata or {})


def _image_row(
    path: str | None,
    member: str | None = None,
    byte_offset: int | None = None,
    byte_size: int | None = None,
    content_type: str = "image/jpeg",
) -> dict:
    return {
        "sample_id": "s1",
        "position": 0,
        "modality": "image",
        "content_type": content_type,
        "text_content": None,
        "binary_content": None,
        "source_ref": InterleavedBatch.build_source_ref(
            path=path,
            member=member,
            byte_offset=byte_offset,
            byte_size=byte_size,
        ),
        "materialize_error": None,
    }


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


# --- _fill_tar_extract_rows ---


def test_fill_tar_extract_rows_bad_tar_path() -> None:
    groups = {"/nonexistent/path.tar": [(0, "img.jpg", None)]}
    binary_values: list[object] = [None]
    error_values: list[str | None] = [None]
    _fill_tar_extract_rows(groups, {}, binary_values, error_values)
    assert error_values[0] == "failed to read path"


def test_fill_tar_extract_rows_frame_extraction_failure(tmp_path: Path) -> None:
    tiff_bytes = build_multi_frame_tiff(1)
    tar_path = write_tar(tmp_path / "oob.tar", {"doc.tiff": tiff_bytes})
    groups = {tar_path: [(0, "doc.tiff", 99)]}
    binary_values: list[object] = [None]
    error_values: list[str | None] = [None]
    _fill_tar_extract_rows(groups, {}, binary_values, error_values)
    assert error_values[0] is not None
    assert "failed to extract frame" in error_values[0]


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


def test_scatter_range_blobs_with_tiff_frame() -> None:
    tiff_bytes = build_multi_frame_tiff(3)
    range_keys = [(0, len(tiff_bytes))]
    unique_ranges: dict[tuple[int, int], list[tuple[int, str, int | None]]] = {
        (0, len(tiff_bytes)): [(0, "doc.tiff", 1)],
    }
    binary_values: list[object] = [None]
    error_values: list[str | None] = [None]
    _scatter_range_blobs([tiff_bytes], range_keys, unique_ranges, binary_values, error_values)
    assert binary_values[0] is not None
    assert error_values[0] is None
    img = Image.open(BytesIO(binary_values[0]))
    assert img.n_frames == 1


def test_scatter_range_blobs_tiff_frame_extraction_failure() -> None:
    tiff_bytes = build_multi_frame_tiff(1)
    range_keys = [(0, len(tiff_bytes))]
    unique_ranges: dict[tuple[int, int], list[tuple[int, str, int | None]]] = {
        (0, len(tiff_bytes)): [(0, "doc.tiff", 99)],
    }
    binary_values: list[object] = [None]
    error_values: list[str | None] = [None]
    _scatter_range_blobs([tiff_bytes], range_keys, unique_ranges, binary_values, error_values)
    assert error_values[0] is not None
    assert "failed to extract frame" in error_values[0]


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
        _image_row(path=str(jpeg_path), content_type="image/jpeg"),
        {**_image_row(path=str(png_path), content_type="image/png"), "position": 1},
    ]
    task = _image_task(rows)
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
