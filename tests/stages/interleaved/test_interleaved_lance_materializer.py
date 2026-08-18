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

from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, ClassVar

import pyarrow as pa
import pytest

from nemo_curator.stages.interleaved.lance import InterleavedLanceMaterializerStage
from nemo_curator.stages.interleaved.lance.fetch import (
    _as_table,
    _LanceFetcher,
    _LanceFetchResult,
    _LanceFetchTimeoutError,
    _LanceRowAddress,
    _restore_fetched_original_order,
)
from nemo_curator.tasks import InterleavedBatch
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA

if TYPE_CHECKING:
    from pathlib import Path

_TEST_PATH = "memory://images"


class _FakeRowIdFetcher:
    source_types: ClassVar[dict[str, pa.DataType]] = {"image": pa.large_binary(), "mime_type": pa.string()}

    def __init__(self, rows_by_id: dict[Any, dict[str, object]]) -> None:
        self.rows_by_id = rows_by_id
        self.calls: list[list[Any]] = []
        self.closed = False

    def fetch(self, row_ids: list[Any]) -> _LanceFetchResult:
        self.calls.append(list(row_ids))
        rows = [self.rows_by_id[row_id] for row_id in row_ids]
        table = pa.table(
            {
                "image": pa.array([row["image"] for row in rows], type=pa.large_binary()),
                "mime_type": pa.array([row["mime_type"] for row in rows], type=pa.string()),
            }
        )
        return _LanceFetchResult(
            table=table,
            fetch_seconds=0.5,
            fetched_bytes_by_column={"image": sum(len(row["image"]) for row in rows), "mime_type": 0},
        )

    def close(self, *, wait_for_fetches: bool = True) -> None:
        del wait_for_fetches
        self.closed = True


class _TimeoutRowIdFetcher(_FakeRowIdFetcher):
    def __init__(self) -> None:
        super().__init__({})

    def fetch(self, row_ids: list[int]) -> _LanceFetchResult:
        self.calls.append(list(row_ids))
        msg = "timed out"
        raise _LanceFetchTimeoutError(msg)


class _RetryMaterializationStage(InterleavedLanceMaterializerStage):
    def __init__(self, fetchers: list[_FakeRowIdFetcher], **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.fetchers = fetchers

    def _ensure_fetcher(self) -> _FakeRowIdFetcher:
        if self._fetcher is None:
            self._fetcher = self.fetchers.pop(0)  # type: ignore[assignment]
        return self._fetcher  # type: ignore[return-value]


def _interleaved_task(rows: list[dict[str, Any]]) -> InterleavedBatch:
    return InterleavedBatch(dataset_name="docs", data=pa.Table.from_pylist(rows, schema=INTERLEAVED_SCHEMA))


def _interleaved_rowid_task(rows: list[dict[str, Any]], row_ids: list[int | None]) -> InterleavedBatch:
    table = pa.Table.from_pylist(rows, schema=INTERLEAVED_SCHEMA)
    table = table.append_column("lance_row_id", pa.array(row_ids, type=pa.uint64(), from_pandas=True))
    return InterleavedBatch(dataset_name="docs", data=table)


def _interleaved_rowaddr_task(
    rows: list[dict[str, Any]],
    fragment_ids: list[int | None],
    row_offsets: list[int | None],
) -> InterleavedBatch:
    table = pa.Table.from_pylist(rows, schema=INTERLEAVED_SCHEMA)
    table = table.append_column("lance_fragment_id", pa.array(fragment_ids, type=pa.uint32(), from_pandas=True))
    table = table.append_column("lance_row_offset", pa.array(row_offsets, type=pa.uint32(), from_pandas=True))
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


def test_lance_materializer_requires_path() -> None:
    with pytest.raises(ValueError, match="path must not be empty"):
        InterleavedLanceMaterializerStage(path="")


def test_interleaved_lazy_exports_resolve_lance_symbols() -> None:
    from nemo_curator.stages import interleaved
    from nemo_curator.stages.interleaved import io as interleaved_io
    from nemo_curator.stages.interleaved.lance import InterleavedLanceReader, InterleavedLanceReaderStage

    assert interleaved.InterleavedLanceMaterializerStage is InterleavedLanceMaterializerStage
    assert interleaved.InterleavedLanceReader is InterleavedLanceReader
    assert interleaved.InterleavedLanceReaderStage is InterleavedLanceReaderStage
    assert interleaved_io.InterleavedLanceReader is InterleavedLanceReader
    with pytest.raises(AttributeError, match="no attribute"):
        interleaved.__getattr__("MissingStage")
    with pytest.raises(AttributeError, match="no attribute"):
        interleaved_io.__getattr__("MissingReader")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"address_mode": "url"}, "Unsupported address_mode"),
        ({"input_row_id_column": ""}, "input_row_id_column"),
        ({"address_mode": "row_address", "input_fragment_id_column": ""}, "input_fragment_id_column"),
        ({"address_mode": "row_address", "input_row_offset_column": ""}, "input_row_offset_column"),
        ({"columns": {}}, "columns must not be empty"),
        ({"columns": {"image": "binary_content", "mime_type": "binary_content"}}, "distinct destination"),
        ({"presence_column": "binary_content"}, "presence_column"),
        ({"fetch_batch_size": 0}, "fetch_batch_size"),
        ({"io_threads": 0}, "io_threads"),
        ({"metadata_cache_size_bytes": 0}, "metadata_cache_size_bytes"),
        ({"fetch_timeout_seconds": -1}, "fetch_timeout_seconds"),
        ({"fetch_retries": -1}, "fetch_retries"),
    ],
)
def test_lance_rowid_image_materializer_rejects_invalid_config(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        InterleavedLanceMaterializerStage(path=_TEST_PATH, **kwargs)


def test_lance_rowid_image_materializer_fills_bytes_without_url_lookup() -> None:
    stage = InterleavedLanceMaterializerStage(
        path=_TEST_PATH,
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
    stage = InterleavedLanceMaterializerStage(
        path=_TEST_PATH,
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


def test_lance_rowid_image_materializer_fill_null_preserves_each_populated_value() -> None:
    stage = InterleavedLanceMaterializerStage(path=_TEST_PATH)
    fake = _FakeRowIdFetcher({10: {"image": b"replacement", "mime_type": "image/jpeg"}})
    stage._fetcher = fake
    task = _interleaved_rowid_task(
        [_image_row("https://a.example/img.jpg", binary_content=b"existing")],
        [10],
    )

    result = stage.process(task).to_pyarrow()

    assert fake.calls == [[10]]
    assert result["binary_content"].to_pylist() == [b"existing"]
    assert result["content_type"].to_pylist() == ["image/jpeg"]


def test_lance_rowid_image_materializer_overwrites_existing_values() -> None:
    stage = InterleavedLanceMaterializerStage(
        path=_TEST_PATH,
        overwrite_existing=True,
    )
    fake = _FakeRowIdFetcher({10: {"image": b"replacement", "mime_type": "image/png"}})
    stage._fetcher = fake
    task = _interleaved_rowid_task(
        [_image_row("https://a.example/img.jpg", binary_content=b"existing", content_type="image/jpeg")],
        [10],
    )

    result = stage.process(task).to_pyarrow()

    assert result["binary_content"].to_pylist() == [b"replacement"]
    assert result["content_type"].to_pylist() == ["image/png"]


def test_lance_materializer_projects_configured_non_image_column() -> None:
    stage = InterleavedLanceMaterializerStage(
        path=_TEST_PATH,
        columns={"mime_type": "lance_mime_type"},
    )
    stage._fetcher = _FakeRowIdFetcher({10: {"image": b"unused", "mime_type": "image/jpeg"}})
    task = _interleaved_rowid_task([_image_row("https://a.example/img.jpg")], [10])

    result = stage.process(task).to_pyarrow()

    assert result["lance_mime_type"].to_pylist() == ["image/jpeg"]


@pytest.mark.parametrize(
    ("stage", "task", "error_type", "match"),
    [
        (
            InterleavedLanceMaterializerStage(path=_TEST_PATH, input_row_id_column="missing"),
            _interleaved_task([_image_row("https://a.example/img.jpg")]),
            ValueError,
            "does not exist",
        ),
        (
            InterleavedLanceMaterializerStage(path=_TEST_PATH, input_row_id_column="source_ref"),
            _interleaved_task([_image_row("https://a.example/img.jpg")]),
            TypeError,
            "expected an integer",
        ),
        (
            InterleavedLanceMaterializerStage(path=_TEST_PATH, address_mode="row_address"),
            _interleaved_task([_image_row("https://a.example/img.jpg")]),
            ValueError,
            "row-address columns do not exist",
        ),
        (
            InterleavedLanceMaterializerStage(path=_TEST_PATH, columns={"image": "text_content"}),
            _interleaved_rowid_task([_image_row("https://a.example/img.jpg")], [10]),
            TypeError,
            "Destination column",
        ),
        (
            InterleavedLanceMaterializerStage(path=_TEST_PATH, presence_column="source_ref"),
            _interleaved_rowid_task([_image_row("https://a.example/img.jpg")], [10]),
            TypeError,
            "Presence column",
        ),
    ],
)
def test_lance_rowid_image_materializer_validates_input_table(
    stage: InterleavedLanceMaterializerStage,
    task: InterleavedBatch,
    error_type: type[Exception],
    match: str,
) -> None:
    stage._fetcher = _FakeRowIdFetcher({10: {"image": b"jpeg-a", "mime_type": "image/jpeg"}})

    with pytest.raises(error_type, match=match):
        stage.process(task)


def test_lance_rowid_image_materializer_retries_timed_out_fetcher() -> None:
    timed_out = _TimeoutRowIdFetcher()
    success = _FakeRowIdFetcher({10: {"image": b"jpeg-a", "mime_type": "image/jpeg"}})
    stage = _RetryMaterializationStage(
        fetchers=[timed_out, success],
        path=_TEST_PATH,
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


def test_lance_rowid_image_materializer_raises_after_retry_exhaustion() -> None:
    first_timeout = _TimeoutRowIdFetcher()
    second_timeout = _TimeoutRowIdFetcher()
    stage = _RetryMaterializationStage(
        fetchers=[first_timeout, second_timeout],
        path=_TEST_PATH,
        fetch_timeout_seconds=0.1,
        fetch_retries=1,
    )
    task = _interleaved_rowid_task([_image_row("https://a.example/img.jpg")], [10])

    with pytest.raises(RuntimeError, match="timed out after 2 attempts"):
        stage.process(task)

    assert first_timeout.closed
    assert second_timeout.closed


def test_lance_row_address_image_materializer_fills_bytes_without_url_lookup() -> None:
    stage = InterleavedLanceMaterializerStage(
        path=_TEST_PATH,
        address_mode="row_address",
        presence_column="lance_image_present",
    )
    fake = _FakeRowIdFetcher(
        {
            _LanceRowAddress(fragment_id=3, row_offset=1): {"image": b"jpeg-a", "mime_type": "image/jpeg"},
            _LanceRowAddress(fragment_id=1, row_offset=7): {"image": b"png-b", "mime_type": "image/png"},
        }
    )
    stage._fetcher = fake
    task = _interleaved_rowaddr_task(
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
        [3, 1, None],
        [1, 7, None],
    )

    result = stage.process(task)
    table = result.to_pyarrow()

    assert fake.calls == [
        [_LanceRowAddress(fragment_id=3, row_offset=1), _LanceRowAddress(fragment_id=1, row_offset=7)]
    ]
    assert table["binary_content"].combine_chunks().to_pylist() == [b"jpeg-a", b"png-b", None]
    assert table["content_type"].combine_chunks().to_pylist() == ["image/jpeg", "image/png", "text/plain"]
    assert table["lance_image_present"].combine_chunks().to_pylist() == [True, True, None]


def test_restore_fetched_original_order() -> None:
    sorted_table = pa.table(
        {
            "image": pa.array([b"row-10", b"row-20", b"row-30"], type=pa.large_binary()),
            "mime_type": pa.array(["a", "b", "c"], type=pa.string()),
        }
    )

    restored = _restore_fetched_original_order(sorted_table, [1, 2, 0])

    assert restored["image"].to_pylist() == [b"row-30", b"row-10", b"row-20"]
    assert restored["mime_type"].to_pylist() == ["c", "a", "b"]


def test_lance_fetch_helpers_handle_edge_cases() -> None:
    table = pa.table({"image": [b"a", b"b"], "mime_type": ["image/jpeg", "image/png"]})

    assert _as_table([]).num_rows == 0
    assert _as_table([table]).equals(table)

    fetcher = object.__new__(_LanceFetcher)
    fetcher.fetch_timeout_seconds = 0.001
    assert fetcher._submit_fetches([], operation="noop") == []
    pending: Future[int] = Future()
    with pytest.raises(_LanceFetchTimeoutError, match="Timed out"):
        fetcher._submit_fetches([pending], operation="blocked fetch")

    closed_fetcher = object.__new__(_LanceFetcher)
    closed_fetcher.executor = None
    with pytest.raises(RuntimeError, match="fetcher is closed"):
        closed_fetcher._take_rows([1])
    with pytest.raises(RuntimeError, match="fetcher is closed"):
        closed_fetcher._take_row_addresses([_LanceRowAddress(fragment_id=1, row_offset=0)])


def test_lance_rowid_image_materializer_real_local_dataset(tmp_path: Path) -> None:
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

    stage = InterleavedLanceMaterializerStage(
        path=str(dataset_path),
        version=dataset.version,
        presence_column="lance_image_present",
        fetch_batch_size=1,
        io_threads=1,
    )
    task = _interleaved_rowid_task([_image_row("rowid://b")], [row_id])

    try:
        result = stage.process(task)
    finally:
        stage.teardown()

    assert result.to_pyarrow()["binary_content"].combine_chunks().to_pylist() == [b"jpeg-b"]
    assert result.to_pyarrow()["lance_image_present"].combine_chunks().to_pylist() == [True]


def test_lance_row_address_image_materializer_real_local_dataset(tmp_path: Path) -> None:
    lance = pytest.importorskip("lance")

    dataset_path = tmp_path / "rowaddr-images.lance"
    table = pa.table(
        {
            "image": [b"jpeg-a", b"jpeg-b", b"jpeg-c"],
            "mime_type": ["image/jpeg", "image/jpeg", "image/jpeg"],
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
    )
    dataset = lance.dataset(str(dataset_path))
    fragments = sorted(dataset.get_fragments(), key=lambda fragment: fragment.fragment_id)

    stage = InterleavedLanceMaterializerStage(
        path=str(dataset_path),
        version=dataset.version,
        address_mode="row_address",
        presence_column="lance_image_present",
        fetch_batch_size=1,
        io_threads=2,
    )
    task = _interleaved_rowaddr_task(
        [_image_row("rowaddr://c"), _image_row("rowaddr://a"), _image_row("rowaddr://b")],
        [fragments[2].fragment_id, fragments[0].fragment_id, fragments[1].fragment_id],
        [0, 0, 0],
    )

    try:
        result = stage.process(task)
    finally:
        stage.teardown()

    assert result.to_pyarrow()["binary_content"].combine_chunks().to_pylist() == [b"jpeg-c", b"jpeg-a", b"jpeg-b"]
    assert result.to_pyarrow()["lance_image_present"].combine_chunks().to_pylist() == [True, True, True]


def test_lance_materializer_rejects_blob_v2_columns(tmp_path: Path) -> None:
    lance = pytest.importorskip("lance")

    dataset_path = tmp_path / "blob-images.lance"
    schema = pa.schema([lance.blob_field("payload")])
    table = pa.Table.from_arrays([lance.blob_array([b"jpeg-a"])], schema=schema)
    lance.write_dataset(
        table,
        str(dataset_path),
        data_storage_version="2.2",
        enable_stable_row_ids=True,
    )
    dataset = lance.dataset(str(dataset_path))
    row_id = int(dataset.scanner(columns=[], with_row_id=True).to_table()["_rowid"][0].as_py())
    stage = InterleavedLanceMaterializerStage(
        path=str(dataset_path),
        columns={"payload": "binary_content"},
    )
    task = _interleaved_rowid_task([_image_row("rowid://blob")], [row_id])

    with pytest.raises(TypeError, match="Blob v2 columns are not supported"):
        stage.process(task)
