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
import json
import pathlib
import sys
import tarfile
import types

import numpy as np
import pytest

from nemo_curator.tasks.webdataset import WebDatasetBatch, WebDatasetSample


def _import_writer_with_stubbed_pyarrow() -> tuple[types.ModuleType, type]:
    """Import WebDatasetWriterStage ensuring pyarrow is stubbed if not installed."""
    if "pyarrow" not in sys.modules:
        sys.modules["pyarrow"] = types.SimpleNamespace(Table=types.SimpleNamespace(from_pylist=lambda rows: rows))
    if "pyarrow.parquet" not in sys.modules:
        sys.modules["pyarrow.parquet"] = types.SimpleNamespace(write_table=lambda _table, _path: None)

    module = importlib.import_module("nemo_curator.stages.image.io.webdataset_writer")
    return module, module.WebDatasetWriterStage


def test_inputs_outputs_and_name(tmp_path: pathlib.Path) -> None:
    _module, writer_cls = _import_writer_with_stubbed_pyarrow()
    stage = writer_cls(output_dir=str(tmp_path), samples_per_shard=3)
    assert stage.inputs() == (["data"], [])
    assert stage.outputs() == (["data"], [])
    assert stage.name == "webdataset_writer"


def test_process_writes_tars_and_parquet(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    _module, writer_cls = _import_writer_with_stubbed_pyarrow()
    stage = writer_cls(output_dir=str(tmp_path), samples_per_shard=2)

    # Stub image encoding to avoid PIL dependency
    monkeypatch.setattr(writer_cls, "_encode_image", staticmethod(lambda _arr: b"imgbytes"))

    # Capture parquet rows
    captured_rows: dict[str, list[dict]] = {}

    def _capture_parquet(_self: object, base_name: str, rows: list[dict]) -> str:
        captured_rows[base_name] = rows
        return str(tmp_path / f"{base_name}.parquet")

    monkeypatch.setattr(writer_cls, "_write_parquet", _capture_parquet)

    # Build 5 samples with image + text components
    samples = [
        WebDatasetSample(
            key=f"sample_{i:04d}",
            components={"jpg": np.zeros((4, 4, 3), dtype=np.uint8), "txt": f"caption {i}"},
            metadata={"score": float(i)},
        )
        for i in range(5)
    ]

    batch = WebDatasetBatch(task_id="t1", dataset_name="ds", data=samples)
    out = stage.process(batch)

    # Validate output task
    assert out.task_id == batch.task_id
    assert out.dataset_name == batch.dataset_name

    tar_paths = [p for p in out.data if p.endswith(".tar")]
    parquet_paths = [p for p in out.data if p.endswith(".parquet")]
    assert len(tar_paths) == 3  # 5 samples / 2 per shard = 3 shards
    assert len(parquet_paths) == 3

    # Check tar contents
    all_member_names: set[str] = set()
    for tar_path in tar_paths:
        with tarfile.open(tar_path, "r") as tf:
            members = tf.getmembers()
            for m in members:
                all_member_names.add(m.name)

    # Each sample produces 2 members (jpg + txt)
    assert len(all_member_names) == 10  # 5 samples * 2 components

    # Verify member naming convention
    assert "sample_0000.jpg" in all_member_names
    assert "sample_0000.txt" in all_member_names

    # Metadata propagated
    assert out._metadata["num_samples"] == 5
    assert out._metadata["samples_per_shard"] == 2
    assert out._metadata["output_dir"] == str(tmp_path)


def test_process_empty_batch(tmp_path: pathlib.Path) -> None:
    _module, writer_cls = _import_writer_with_stubbed_pyarrow()
    stage = writer_cls(output_dir=str(tmp_path), samples_per_shard=3)

    empty = WebDatasetBatch(task_id="e", dataset_name="ds", data=[])
    out = stage.process(empty)

    assert out.data == []
    assert out._metadata["num_samples"] == 0


def test_construct_base_name_deterministic(tmp_path: pathlib.Path) -> None:
    _module, writer_cls = _import_writer_with_stubbed_pyarrow()
    stage = writer_cls(output_dir=str(tmp_path), deterministic_name=True)

    samples1 = [WebDatasetSample(key="b"), WebDatasetSample(key="a")]
    samples2 = [WebDatasetSample(key="a"), WebDatasetSample(key="b")]

    b1 = stage._construct_base_name(WebDatasetBatch(task_id="T", dataset_name="ds", data=samples1))
    b2 = stage._construct_base_name(WebDatasetBatch(task_id="T", dataset_name="ds", data=samples2))
    assert b1 == b2
    assert b1.startswith("shard-")


def test_construct_base_name_random(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    module, writer_cls = _import_writer_with_stubbed_pyarrow()

    class _FakeUUID:
        def __init__(self, hex_str: str) -> None:
            self.hex = hex_str

    monkeypatch.setattr(module.uuid, "uuid4", lambda: _FakeUUID("deadbeefcafebabe0123456789abcdef"))
    stage = writer_cls(output_dir=str(tmp_path), deterministic_name=False)
    b = stage._construct_base_name(WebDatasetBatch(task_id="T2", dataset_name="ds", data=[]))
    assert b == "shard-deadbeefcafebabe"


def test_encode_component_text() -> None:
    _module, writer_cls = _import_writer_with_stubbed_pyarrow()
    payload = writer_cls._encode_component("txt", "hello world")
    assert payload == b"hello world"


def test_encode_component_json() -> None:
    _module, writer_cls = _import_writer_with_stubbed_pyarrow()
    payload = writer_cls._encode_component("json", {"key": "val"})
    decoded = json.loads(payload)
    assert decoded == {"key": "val"}


def test_encode_component_bytes() -> None:
    _module, writer_cls = _import_writer_with_stubbed_pyarrow()
    raw = b"\x00\x01\x02"
    payload = writer_cls._encode_component("bin", raw)
    assert payload == raw


def test_encode_component_numpy() -> None:
    _module, writer_cls = _import_writer_with_stubbed_pyarrow()
    arr = np.array([1.0, 2.0, 3.0])
    payload = writer_cls._encode_component("npy", arr)
    # Should be numpy .npy format
    assert len(payload) > 0


def test_metadata_in_parquet_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    _module, writer_cls = _import_writer_with_stubbed_pyarrow()
    stage = writer_cls(output_dir=str(tmp_path), samples_per_shard=10)

    # Stub image encoding
    monkeypatch.setattr(writer_cls, "_encode_image", staticmethod(lambda _arr: b"imgbytes"))

    # Capture parquet rows
    captured_rows: list[dict] = []

    def _capture_parquet(_self: object, base_name: str, rows: list[dict]) -> str:
        captured_rows.extend(rows)
        return str(tmp_path / f"{base_name}.parquet")

    monkeypatch.setattr(writer_cls, "_write_parquet", _capture_parquet)

    samples = [
        WebDatasetSample(
            key="s0",
            components={"jpg": np.zeros((2, 2, 3), dtype=np.uint8)},
            metadata={"aesthetic_score": 0.8, "source": "web"},
        )
    ]

    batch = WebDatasetBatch(task_id="t1", dataset_name="ds", data=samples)
    stage.process(batch)

    assert len(captured_rows) == 1
    row = captured_rows[0]
    assert row["key"] == "s0"
    assert row["aesthetic_score"] == 0.8
    assert row["source"] == "web"
    assert row["extensions"] == "jpg"
