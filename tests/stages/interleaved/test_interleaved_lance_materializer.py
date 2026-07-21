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
from typing import TYPE_CHECKING, Any, ClassVar

import pyarrow as pa
import pytest

from nemo_curator.stages.interleaved.lance import LanceRowIdImageMaterializationStage, LanceTableConfig
from nemo_curator.stages.interleaved.lance.fetch import (
    _LanceFetchTimeoutError,
    _restore_fetched_original_order,
    _RowIdFetchResult,
)
from nemo_curator.tasks import InterleavedBatch
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA

if TYPE_CHECKING:
    from pathlib import Path


class _FakeRowIdFetcher:
    source_types: ClassVar[dict[str, pa.DataType]] = {"image": pa.large_binary(), "mime_type": pa.string()}

    def __init__(self, rows_by_id: dict[int, dict[str, object]]) -> None:
        self.rows_by_id = rows_by_id
        self.calls: list[list[int]] = []
        self.closed = False

    def fetch(self, row_ids: list[int]) -> _RowIdFetchResult:
        self.calls.append(list(row_ids))
        rows = [self.rows_by_id[row_id] for row_id in row_ids]
        table = pa.table(
            {
                "image": pa.array([row["image"] for row in rows], type=pa.large_binary()),
                "mime_type": pa.array([row["mime_type"] for row in rows], type=pa.string()),
            }
        )
        return _RowIdFetchResult(
            tables=[table],
            fetch_seconds=0.5,
            fetched_bytes_by_column={"image": sum(len(row["image"]) for row in rows), "mime_type": 0},
        )

    def close(self, *, wait_for_fetches: bool = True) -> None:
        del wait_for_fetches
        self.closed = True


class _TimeoutRowIdFetcher(_FakeRowIdFetcher):
    def __init__(self) -> None:
        super().__init__({})

    def fetch(self, row_ids: list[int]) -> _RowIdFetchResult:
        self.calls.append(list(row_ids))
        msg = "timed out"
        raise _LanceFetchTimeoutError(msg)


class _RetryMaterializationStage(LanceRowIdImageMaterializationStage):
    def __init__(self, fetchers: list[_FakeRowIdFetcher], **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.fetchers = fetchers

    def _ensure_fetcher(self) -> _FakeRowIdFetcher:
        if self._fetcher is None:
            self._fetcher = self.fetchers.pop(0)  # type: ignore[assignment]
        return self._fetcher  # type: ignore[return-value]


class _FallbackMaterializationStage(_RetryMaterializationStage):
    def __init__(self, fallback_fetcher: _FakeRowIdFetcher, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.fallback_fetcher = fallback_fetcher
        self.fallback_calls: list[list[int]] = []

    def _fetch_requested_images_subprocess_fallback(self, requested_row_ids: list[int]) -> _RowIdFetchResult:
        self.fallback_calls.append(list(requested_row_ids))
        return self.fallback_fetcher.fetch(requested_row_ids)


def _table_config(uri: str = "memory://images") -> LanceTableConfig:
    return LanceTableConfig(uri=uri, version=1)


def _interleaved_task(rows: list[dict[str, Any]]) -> InterleavedBatch:
    return InterleavedBatch(dataset_name="docs", data=pa.Table.from_pylist(rows, schema=INTERLEAVED_SCHEMA))


def _interleaved_rowid_task(rows: list[dict[str, Any]], row_ids: list[int | None]) -> InterleavedBatch:
    table = pa.Table.from_pylist(rows, schema=INTERLEAVED_SCHEMA)
    table = table.append_column("lance_row_id", pa.array(row_ids, type=pa.uint64(), from_pandas=True))
    return InterleavedBatch(dataset_name="docs", data=table)


def _image_row(
    source_ref: str | None, *, binary_content: bytes | None = None, content_type: str | None = None
) -> dict:
    return {
        "sample_id": "s1",
        "position": 0,
        "modality": "image",
        "content_type": content_type,
        "text_content": None,
        "binary_content": binary_content,
        "source_ref": source_ref,
        "materialize_error": None,
    }


def test_lance_table_config_requires_uri() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        LanceTableConfig(uri="")


def test_lance_rowid_image_materializer_fills_bytes_without_url_lookup() -> None:
    stage = LanceRowIdImageMaterializationStage(
        dataset=_table_config(),
        presence_column="lance_image_present",
    )
    fake = _FakeRowIdFetcher(
        {
            10: {"image": b"jpeg-a", "mime_type": "image/jpeg"},
            20: {"image": b"png-b", "mime_type": "image/png"},
        }
    )
    stage._fetcher = fake
    task = _interleaved_rowid_task(
        [
            _image_row("https://a.example/img.jpg"),
            _image_row("https://b.example/img.png"),
            {
                "sample_id": "s1",
                "position": 1,
                "modality": "text",
                "content_type": "text/plain",
                "text_content": "caption",
                "binary_content": None,
                "source_ref": None,
                "materialize_error": None,
            },
        ],
        [10, 20, None],
    )

    result = stage.process(task)
    table = result.to_pyarrow()

    assert fake.calls == [[10, 20]]
    assert table["binary_content"].combine_chunks().to_pylist() == [b"jpeg-a", b"png-b", None]
    assert table["content_type"].combine_chunks().to_pylist() == ["image/jpeg", "image/png", "text/plain"]
    assert table["lance_image_present"].combine_chunks().to_pylist() == [True, True, None]


def test_lance_rowid_image_materializer_fill_null_skips_populated_rows() -> None:
    stage = LanceRowIdImageMaterializationStage(
        dataset=_table_config(),
        presence_column="lance_image_present",
    )
    fake = _FakeRowIdFetcher({20: {"image": b"png-b", "mime_type": "image/png"}})
    stage._fetcher = fake
    task = _interleaved_rowid_task(
        [
            _image_row("https://a.example/img.jpg", binary_content=b"existing", content_type="image/jpeg"),
            _image_row("https://b.example/img.png"),
        ],
        [10, 20],
    )

    result = stage.process(task)

    assert fake.calls == [[20]]
    assert result.to_pyarrow()["binary_content"].combine_chunks().to_pylist() == [b"existing", b"png-b"]


def test_lance_rowid_image_materializer_existing_policy_error() -> None:
    stage = LanceRowIdImageMaterializationStage(
        dataset=_table_config(),
        existing_column_policy="error",
    )
    stage._fetcher = _FakeRowIdFetcher({10: {"image": b"jpeg-a", "mime_type": "image/jpeg"}})
    task = _interleaved_rowid_task([_image_row("https://a.example/img.jpg")], [10])

    with pytest.raises(ValueError, match="already exist"):
        stage.process(task)


def test_lance_rowid_image_materializer_retries_timed_out_fetcher() -> None:
    timed_out = _TimeoutRowIdFetcher()
    success = _FakeRowIdFetcher({10: {"image": b"jpeg-a", "mime_type": "image/jpeg"}})
    stage = _RetryMaterializationStage(
        fetchers=[timed_out, success],
        dataset=_table_config(),
        presence_column="lance_image_present",
        fetch_timeout_seconds=0.1,
        fetch_retries=1,
    )
    task = _interleaved_rowid_task([_image_row("https://a.example/img.jpg")], [10])

    result = stage.process(task)

    assert timed_out.calls == [[10]]
    assert timed_out.closed
    assert success.calls == [[10]]
    assert result.to_pyarrow()["binary_content"].combine_chunks().to_pylist() == [b"jpeg-a"]


def test_lance_rowid_image_materializer_can_fallback_to_subprocess_after_timeout() -> None:
    timed_out = _TimeoutRowIdFetcher()
    fallback = _FakeRowIdFetcher({10: {"image": b"jpeg-a", "mime_type": "image/jpeg"}})
    stage = _FallbackMaterializationStage(
        fetchers=[timed_out],
        fallback_fetcher=fallback,
        dataset=_table_config(),
        presence_column="lance_image_present",
        fetch_mode="subprocess_on_timeout",
        fetch_timeout_seconds=0.1,
        fetch_retries=0,
    )
    task = _interleaved_rowid_task([_image_row("https://a.example/img.jpg")], [10])

    result = stage.process(task)

    assert timed_out.calls == [[10]]
    assert timed_out.closed
    assert stage.fallback_calls == [[10]]
    assert fallback.calls == [[10]]
    assert result.to_pyarrow()["binary_content"].combine_chunks().to_pylist() == [b"jpeg-a"]


def test_lance_rowid_image_materializer_can_parse_json_source_ref() -> None:
    stage = LanceRowIdImageMaterializationStage(
        dataset=_table_config(),
        input_row_id_column="source_ref",
        input_row_id_json_field="row_id",
        presence_column="lance_image_present",
    )
    fake = _FakeRowIdFetcher({33: {"image": b"jpeg-a", "mime_type": "image/jpeg"}})
    stage._fetcher = fake
    task = _interleaved_task([_image_row(json.dumps({"row_id": 33}))])

    result = stage.process(task)

    assert fake.calls == [[33]]
    assert result.to_pyarrow()["binary_content"].combine_chunks().to_pylist() == [b"jpeg-a"]


def test_restore_fetched_original_order() -> None:
    sorted_table = pa.table(
        {
            "image": pa.array([b"row-10", b"row-20", b"row-30"], type=pa.large_binary()),
            "mime_type": pa.array(["a", "b", "c"], type=pa.string()),
        }
    )

    restored = _restore_fetched_original_order([sorted_table], [1, 2, 0])

    assert len(restored) == 1
    assert restored[0]["image"].to_pylist() == [b"row-30", b"row-10", b"row-20"]
    assert restored[0]["mime_type"].to_pylist() == ["c", "a", "b"]


@pytest.mark.parametrize("fetch_mode", ["in_process", "subprocess"])
def test_lance_rowid_image_materializer_real_local_dataset(tmp_path: Path, fetch_mode: str) -> None:
    lance = pytest.importorskip("lance")

    dataset_path = tmp_path / "rowid-images.lance"
    table = pa.table(
        {
            "image": [b"jpeg-a", b"jpeg-b"],
            "mime_type": ["image/jpeg", "image/jpeg"],
        },
        schema=pa.schema(
            [
                pa.field("image", pa.large_binary()),
                pa.field("mime_type", pa.string()),
            ]
        ),
    )
    lance.write_dataset(
        table,
        str(dataset_path),
        mode="create",
        max_rows_per_file=1,
        max_rows_per_group=1,
        enable_stable_row_ids=True,
    )
    dataset = lance.dataset(str(dataset_path))
    row_ids = dataset.scanner(columns=[], with_row_id=True, limit=2).to_table()["_rowid"].combine_chunks().to_pylist()
    row_id = int(row_ids[1])

    stage = LanceRowIdImageMaterializationStage(
        dataset=LanceTableConfig(uri=str(dataset_path), version=dataset.version),
        presence_column="lance_image_present",
        fetch_batch_size=1,
        fetch_mode=fetch_mode,
        io_threads=1,
    )
    task = _interleaved_rowid_task([_image_row("rowid://b")], [row_id])

    try:
        result = stage.process(task)
    finally:
        stage.teardown()

    assert result.to_pyarrow()["binary_content"].combine_chunks().to_pylist() == [b"jpeg-b"]
    assert result.to_pyarrow()["lance_image_present"].combine_chunks().to_pylist() == [True]
