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

import io
import tarfile
from typing import TYPE_CHECKING
from unittest.mock import patch

import pandas as pd
import pytest

from nemo_curator.stages.image.io.tar_range_reader import ImageTarRangeReader
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from pathlib import Path


def _write_tar(path: Path, members: dict[str, bytes]) -> list[dict]:
    """Write a TAR and return one locator row per member."""
    with tarfile.open(path, "w") as tar:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    with tarfile.open(path) as tar:
        return [
            {"tar_path": str(path), "byte_offset": m.offset_data, "byte_size": m.size, "image_id": m.name}
            for m in tar.getmembers()
        ]


def _run(reader: ImageTarRangeReader, rows: list[dict]) -> list:
    split, fetch = reader.decompose()
    batch = DocumentBatch(dataset_name="test", data=pd.DataFrame(rows))
    return [fetch.process(group) for group in split.process(batch)]


@pytest.fixture
def rows(tmp_path: Path) -> list[dict]:
    first = _write_tar(tmp_path / "a.tar", {f"a{i}.jpg": bytes([i]) * (100 + i) for i in range(5)})
    second = _write_tar(tmp_path / "b.tar", {"b0.jpg": b"x" * 300, "b1.jpg": b"y" * 7})
    # Interleave TARs and reverse offsets: the reader must sort and group them itself.
    return [*reversed(first), *second]


def test_reads_exact_payloads_grouped_by_tar(rows: list[dict]) -> None:
    batches = _run(ImageTarRangeReader(images_per_batch=2), rows)
    images = [image for batch in batches for image in batch.data]
    assert sorted(image.image_id for image in images) == sorted(row["image_id"] for row in rows)
    for batch in batches:
        assert len(batch.data) <= 2
        assert len({image.metadata["tar_path"] for image in batch.data}) == 1
    by_id = {image.image_id: image for image in images}
    assert by_id["a3.jpg"].image_bytes == bytes([3]) * 103
    assert by_id["b1.jpg"].image_bytes == b"y" * 7
    assert by_id["b1.jpg"].metadata["byte_size"] == 7


def test_one_ranged_read_per_group(rows: list[dict]) -> None:
    from fsspec.implementations import local

    with patch.object(
        local.LocalFileSystem, "cat_file", autospec=True, side_effect=local.LocalFileSystem.cat_file
    ) as cat:
        batches = _run(ImageTarRangeReader(images_per_batch=128), rows)
    assert len(batches) == 2
    assert cat.call_count == 2


def test_request_size_limit_splits_groups(rows: list[dict]) -> None:
    batches = _run(ImageTarRangeReader(max_request_bytes=600), rows)
    assert all(
        max(i.metadata["byte_offset"] + i.metadata["byte_size"] for i in b.data) - b.data[0].metadata["byte_offset"]
        <= 600
        for b in batches
    )
    assert sum(len(b.data) for b in batches) == len(rows)


def test_missing_tar_keeps_rows_without_bytes(rows: list[dict], tmp_path: Path) -> None:
    missing = [{**row, "tar_path": str(tmp_path / "gone.tar")} for row in rows[:2]]
    (batch,) = _run(ImageTarRangeReader(), missing)
    assert [image.image_bytes for image in batch.data] == [None, None]


def test_rejects_non_positive_limits() -> None:
    with pytest.raises(ValueError, match="positive"):
        ImageTarRangeReader(images_per_batch=0)


def test_rows_without_locator_pass_through(rows: list[dict]) -> None:
    rows = [*rows, {"tar_path": None, "byte_offset": None, "byte_size": None, "image_id": "nowhere.jpg"}]
    batches = _run(ImageTarRangeReader(), rows)
    images = {image.image_id: image for batch in batches for image in batch.data}
    assert len(images) == len(rows)
    assert images["nowhere.jpg"].image_bytes is None
    assert images["a0.jpg"].image_bytes == bytes([0]) * 100


def test_split_children_do_not_share_task_state(rows: list[dict]) -> None:
    split, _ = ImageTarRangeReader(images_per_batch=1).decompose()
    children = split.process(DocumentBatch(dataset_name="test", data=pd.DataFrame(rows)))
    children[0]._stage_perf.append("perf")
    children[0]._metadata["key"] = "value"
    assert children[1]._stage_perf == []
    assert "key" not in children[1]._metadata


def test_rows_larger_than_one_request_are_not_fetched(rows: list[dict]) -> None:
    oversized = {**rows[0], "image_id": "huge.jpg", "byte_size": 10**12}  # e.g. a corrupt catalog size
    with patch("fsspec.implementations.local.LocalFileSystem.cat_file", autospec=True, side_effect=AssertionError):
        (batch,) = _run(ImageTarRangeReader(max_request_bytes=1000), [oversized])
    assert batch.data[0].image_bytes is None
