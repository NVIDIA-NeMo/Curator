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

from __future__ import annotations

import importlib
import pathlib
import sys
import tarfile
import types
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import io

import numpy as np
import pytest

from nemo_curator.tasks.image import ImageBatch, ImageObject


def _import_writer_with_stubbed_pyarrow() -> tuple[types.ModuleType, type]:
    """Import ImageWriterStage ensuring pyarrow is stubbed if not installed."""
    if "pyarrow" not in sys.modules:
        sys.modules["pyarrow"] = types.SimpleNamespace(Table=types.SimpleNamespace(from_pylist=lambda rows: rows))
    if "pyarrow.parquet" not in sys.modules:
        sys.modules["pyarrow.parquet"] = types.SimpleNamespace(write_table=lambda _table, _path: None)

    module = importlib.import_module("nemo_curator.stages.image.io.image_writer")
    return module, module.ImageWriterStage


def test_inputs_outputs_and_name(tmp_path: pathlib.Path) -> None:
    module, image_writer_stage_cls = _import_writer_with_stubbed_pyarrow()

    stage = image_writer_stage_cls(output_dir=str(tmp_path), images_per_tar=3)
    assert stage.inputs() == (["data"], [])
    assert stage.outputs() == (["data"], [])
    assert stage.name == "image_writer"


def test_setup_no_actor_id(tmp_path: pathlib.Path) -> None:
    _module, image_writer_stage_cls = _import_writer_with_stubbed_pyarrow()

    stage = image_writer_stage_cls(output_dir=str(tmp_path), images_per_tar=2)

    class _Worker:
        def __init__(self, worker_id: str) -> None:
            self.worker_id = worker_id

    # Should not create any _actor_id attribute
    stage.setup(worker_metadata=_Worker(worker_id="worker1"))
    assert not hasattr(stage, "_actor_id")


def test_process_writes_tars_and_parquet_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    _module, image_writer_stage_cls = _import_writer_with_stubbed_pyarrow()

    stage = image_writer_stage_cls(output_dir=str(tmp_path), images_per_tar=2)
    stage.setup(worker_metadata=types.SimpleNamespace(worker_id="w"))

    # Avoid PIL dependency by stubbing encoder to return fixed bytes and extension
    monkeypatch.setattr(
        image_writer_stage_cls,
        "_encode_image_to_bytes",
        lambda _self, _arr: (b"imgbytes", ".jpg"),
    )

    # Capture parquet rows per base_name without touching filesystem
    captured_rows: dict[str, list[dict]] = {}

    def _capture_parquet(_self: object, base_name: str, rows: list[dict]) -> str:
        captured_rows[base_name] = rows
        return str(tmp_path / f"{base_name}.parquet")

    monkeypatch.setattr(image_writer_stage_cls, "_write_parquet", _capture_parquet)

    # Build 5 images, force split into 3 tars (2,2,1)
    images = [
        ImageObject(image_id=f"img{i}", image_path=f"/p/{i}.jpg", image_data=np.zeros((4, 4, 3), np.uint8))
        for i in range(5)
    ]

    batch = ImageBatch(task_id="t1", dataset_name="ds", data=images)
    out = stage.process(batch)

    # Validate output task
    assert out.task_id == batch.task_id
    assert out.dataset_name == batch.dataset_name

    tar_paths = [p for p in out.data if p.endswith(".tar")]
    parquet_paths = [p for p in out.data if p.endswith(".parquet")]
    assert len(tar_paths) == 3
    assert len(parquet_paths) == 3

    # Check tar contents and match with captured parquet rows per base
    all_member_names: set[str] = set()
    for tar_path in tar_paths:
        with tarfile.open(tar_path, "r") as tf:
            members = tf.getmembers()
            # Each tar has at most images_per_tar members
            assert 1 <= len(members) <= 2
            for m in members:
                assert m.name.endswith(".jpg")
                all_member_names.add(m.name)

        base = tar_path.split("/")[-1].rsplit(".", 1)[0]
        assert base in captured_rows
        rows = captured_rows[base]
        # Member names in parquet rows should correspond to files in tar
        assert {r["member_name"] for r in rows}.issubset(all_member_names)

    # All expected image ids should appear once
    expected = {f"img{i}.jpg" for i in range(5)}
    assert expected.issubset(all_member_names)

    # Metadata propagated
    assert out._metadata["num_images"] == 5
    assert out._metadata["images_per_tar"] == 2
    assert out._metadata["output_dir"] == str(tmp_path)


def test_process_raises_on_missing_image_data(tmp_path: pathlib.Path) -> None:
    _module, image_writer_stage_cls = _import_writer_with_stubbed_pyarrow()
    stage = image_writer_stage_cls(output_dir=str(tmp_path), images_per_tar=2)
    stage.setup()

    bad = ImageBatch(
        task_id="bad",
        dataset_name="ds",
        data=[ImageObject(image_id="x", image_path="/p/x.jpg", image_data=None)],
    )

    with pytest.raises(ValueError, match="image_data is None"):
        stage.process(bad)


def test_process_handles_empty_batch(tmp_path: pathlib.Path) -> None:
    _module, image_writer_stage_cls = _import_writer_with_stubbed_pyarrow()
    stage = image_writer_stage_cls(output_dir=str(tmp_path), images_per_tar=3)
    stage.setup()

    empty = ImageBatch(task_id="e", dataset_name="ds", data=[])
    out = stage.process(empty)

    assert out.data == []
    assert out._metadata["num_images"] == 0
    assert out._metadata["output_dir"] == str(tmp_path)


def test_construct_base_name_deterministic_and_random(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    module, image_writer_stage_cls = _import_writer_with_stubbed_pyarrow()

    # Deterministic name should depend only on sorted image paths + task id
    stage_det = image_writer_stage_cls(output_dir=str(tmp_path), deterministic_name=True)
    imgs1 = [
        ImageObject(image_path="/p/b.jpg"),
        ImageObject(image_path="/p/a.jpg"),
    ]
    imgs2 = list(reversed(imgs1))
    b1 = stage_det.construct_base_name(ImageBatch(task_id="T", dataset_name="ds", data=imgs1))
    b2 = stage_det.construct_base_name(ImageBatch(task_id="T", dataset_name="ds", data=imgs2))
    assert b1 == b2
    assert b1.startswith("images-")

    # Random name path uses uuid4; make it deterministic for the test
    class _FakeUUID:
        def __init__(self, hex_image_id: str) -> None:
            self.hex = hex_image_id

    monkeypatch.setattr(module.uuid, "uuid4", lambda: _FakeUUID("deadbeefcafebabe0123456789abcdef"))
    stage_rand = image_writer_stage_cls(output_dir=str(tmp_path), deterministic_name=False)
    b3 = stage_rand.construct_base_name(ImageBatch(task_id="T2", dataset_name="ds", data=imgs1))
    assert b3 == "images-deadbeefcafebabe"


def test_encode_image_to_bytes_modes(tmp_path: pathlib.Path) -> None:
    _module, image_writer_stage_cls = _import_writer_with_stubbed_pyarrow()
    stage = image_writer_stage_cls(output_dir=str(tmp_path))

    # Stub PIL.Image to avoid real dependency and to capture mode/shape
    captured: list[tuple[tuple[int, ...], str, np.dtype]] = []

    class _StubImageModule:
        @staticmethod
        def fromarray(arr: np.ndarray, mode: str | None = None) -> object:
            captured.append((tuple(arr.shape), "" if mode is None else mode, arr.dtype))

            class _Img:
                def save(self, buffer: io.BytesIO, image_format: str | None = None, **kwargs) -> None:  # noqa: ARG002
                    buffer.write(b"ok")

            return _Img()

    pil_pkg = types.ModuleType("PIL")
    pil_pkg.Image = _StubImageModule  # type: ignore[attr-defined]
    sys.modules["PIL"] = pil_pkg

    # Grayscale float -> converts to uint8 and mode L
    gray = (np.random.rand(3, 4) * 300).astype(np.float32)  # noqa: NPY002
    b, ext = stage._encode_image_to_bytes(gray)
    assert ext == ".jpg"
    assert isinstance(b, (bytes, bytearray))
    assert captured[-1][1] == "L"
    assert captured[-1][2] == np.uint8

    # RGB uint8 -> mode RGB
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    _b, ext = stage._encode_image_to_bytes(rgb)
    assert ext == ".jpg"
    assert captured[-1][1] == "RGB"

    # RGBA uint8 -> mode RGBA
    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    _b, ext = stage._encode_image_to_bytes(rgba)
    assert ext == ".jpg"
    assert captured[-1][1] == "RGBA"

    # >3 channels -> cropped to 3 and mode RGB, dtype coerced to uint8
    five = np.zeros((2, 2, 5), dtype=np.uint16)
    _b, ext = stage._encode_image_to_bytes(five)
    assert ext == ".jpg"
    shape, mode, dtype = captured[-1]
    assert shape == (2, 2, 3)
    assert mode == "RGB"
    assert dtype == np.uint8


def test_write_tar_collision_and_content(tmp_path: pathlib.Path) -> None:
    _module, image_writer_stage_cls = _import_writer_with_stubbed_pyarrow()
    stage = image_writer_stage_cls(output_dir=str(tmp_path))

    base = "images-abc123"
    path = stage._write_tar(base, [("a.jpg", b"a"), ("b.jpg", b"bb")])
    assert pathlib.Path(path).exists()

    with tarfile.open(path, "r") as tf:
        names = {m.name for m in tf.getmembers()}
        assert names == {"a.jpg", "b.jpg"}

    with pytest.raises(AssertionError, match="Collision detected"):
        stage._write_tar(base, [("c.jpg", b"c")])


def test_write_parquet_collision_and_path(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    import types

    # Track writes without stubbing global sys.modules
    written: dict[str, object] = {}

    module = importlib.import_module("nemo_curator.stages.image.io.image_writer")

    # Patch module-local pyarrow bindings
    stub_pa = types.SimpleNamespace(Table=types.SimpleNamespace(from_pylist=lambda rows: ("T", rows)))
    stub_pq = types.SimpleNamespace(write_table=lambda tbl, p: written.update({"tbl": tbl, "path": p}))
    monkeypatch.setattr(module, "pa", stub_pa, raising=False)
    monkeypatch.setattr(module, "pq", stub_pq, raising=False)

    stage = module.ImageWriterStage(output_dir=str(tmp_path))

    base = "images-parq"
    # Pre-create to trigger collision
    (tmp_path / f"{base}.parquet").write_bytes(b"")
    with pytest.raises(AssertionError, match="Collision detected"):
        stage._write_parquet(base, [{"k": 1}])

    # Remove and write successfully
    (tmp_path / f"{base}.parquet").unlink()
    out_path = stage._write_parquet(base, [{"k": 2}])
    assert out_path == str(tmp_path / f"{base}.parquet")
    assert written.get("path") == out_path


@pytest.mark.parametrize("remove", [True, False])
def test_process_respects_remove_image_data_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, remove: bool
) -> None:
    _module, image_writer_stage_cls = _import_writer_with_stubbed_pyarrow()

    stage = image_writer_stage_cls(output_dir=str(tmp_path), images_per_tar=2, remove_image_data=remove)
    stage.setup()

    # Avoid PIL dependency by stubbing encoder to return fixed bytes and extension
    monkeypatch.setattr(
        image_writer_stage_cls,
        "_encode_image_to_bytes",
        lambda _self, _arr: (b"imgbytes", ".jpg"),
    )

    # Build a few images
    images = [
        ImageObject(image_id=f"img{i}", image_path=f"/p/{i}.jpg", image_data=np.zeros((2, 2, 3), np.uint8))
        for i in range(4)
    ]

    batch = ImageBatch(task_id="t1", dataset_name="ds", data=images)
    _out = stage.process(batch)

    # Image data should be removed only when the flag is True
    for img in images:
        if remove:
            assert img.image_data is None
        else:
            assert img.image_data is not None
