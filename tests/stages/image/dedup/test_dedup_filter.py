# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nemo_curator.stages.image.deduplication.removal import ImageDuplicatesRemovalStage
from nemo_curator.tasks import ImageBatch, ImageObject


def _write_parquet_ids(tmpdir: Path, filename: str, ids: list[str], id_column: str = "id") -> str:
    table = pa.Table.from_pydict({id_column: ids})
    out_path = tmpdir / filename
    pq.write_table(table, out_path.as_posix())
    return out_path.as_posix()


def _make_batch(ids: list[str]) -> ImageBatch:
    images = [ImageObject(image_id=i) for i in ids]
    return ImageBatch(task_id="t0", dataset_name="ds", data=images)


def test_setup_raises_when_no_parquet(tmp_path: Path) -> None:
    stage = ImageDuplicatesRemovalStage(removal_parquets_dir=tmp_path.as_posix())
    with pytest.raises(FileNotFoundError):
        stage.setup()


def test_filters_with_default_id_column(tmp_path: Path) -> None:
    _write_parquet_ids(tmp_path, "a.parquet", ["img2", "img4", "img5"], id_column="id")
    stage = ImageDuplicatesRemovalStage(removal_parquets_dir=tmp_path.as_posix())
    stage.setup()

    batch = _make_batch(["img1", "img2", "img3", "img4"])  # expect remove img2, img4
    out = stage.process(batch)

    kept_ids = [img.image_id for img in out.data]
    assert kept_ids == ["img1", "img3"]
    assert out.task_id.endswith(stage._name)
    assert out.dataset_name == batch.dataset_name


def test_filters_with_custom_id_column(tmp_path: Path) -> None:
    _write_parquet_ids(tmp_path, "b.parquet", ["x2"], id_column="image_id")
    stage = ImageDuplicatesRemovalStage(removal_parquets_dir=tmp_path.as_posix(), duplicate_id_field="image_id")
    stage.setup()

    batch = _make_batch(["x1", "x2", "x3"])  # expect remove x2
    out = stage.process(batch)
    kept_ids = [img.image_id for img in out.data]
    assert kept_ids == ["x1", "x3"]


def test_aggregates_ids_across_multiple_parquets(tmp_path: Path) -> None:
    _write_parquet_ids(tmp_path, "p1.parquet", ["a", "b", "c"])  # default id column
    _write_parquet_ids(tmp_path, "p2.parquet", ["d", "e"])  # default id column

    stage = ImageDuplicatesRemovalStage(removal_parquets_dir=tmp_path.as_posix())
    stage.setup()

    batch = _make_batch(["a", "z", "e", "y", "x"])  # expect remove a, e
    out = stage.process(batch)
    kept_ids = [img.image_id for img in out.data]
    assert kept_ids == ["z", "y", "x"]
