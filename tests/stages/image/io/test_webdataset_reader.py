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

import sys
import types
from collections.abc import Callable
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest
import torch

from nemo_curator.tasks.file_group import FileGroupTask
from nemo_curator.tasks.webdataset import WebDatasetBatch, WebDatasetSample


class _FakeTensorList:
    """Minimal stand-in for a DALI TensorList returned by Pipeline.run()."""

    def __init__(self, batch_size: int, height: int = 8, width: int = 8) -> None:
        self._arrays: list[np.ndarray] = [
            np.zeros((height, width, 3), dtype=np.uint8) for _ in range(batch_size)
        ]

    def as_cpu(self) -> _FakeTensorList:
        return self

    def __len__(self) -> int:
        return len(self._arrays)

    def at(self, index: int) -> np.ndarray:
        return self._arrays[index]


@dataclass
class _FakePipeline:
    """A fake DALI pipeline that yields a fixed batch size until a total is reached."""

    total_samples: int
    batch_size: int
    num_extensions: int = 1

    def build(self) -> None:
        return None

    def epoch_size(self) -> dict[int, int]:
        return {0: self.total_samples}

    def run(self) -> tuple[_FakeTensorList, ...] | _FakeTensorList:
        tls = tuple(_FakeTensorList(self.batch_size) for _ in range(self.num_extensions))
        return tls if self.num_extensions > 1 else tls[0]


def _fake_create_pipeline_factory(
    per_tar_total: int, batch: int, num_extensions: int = 1
) -> Callable[[list[str]], _FakePipeline]:
    def _factory(tar_paths: list[str] | tuple[str, ...]) -> _FakePipeline:
        num_paths = len(tar_paths) if isinstance(tar_paths, (list, tuple)) else 1
        return _FakePipeline(
            total_samples=per_tar_total * num_paths,
            batch_size=batch,
            num_extensions=num_extensions,
        )

    return _factory


@pytest.fixture(autouse=True)
def _stub_dali_modules() -> None:
    """Stub nvidia.dali only on CPU-only environments without real DALI."""
    import importlib.util

    if torch.cuda.is_available():
        return
    try:
        dali_spec = importlib.util.find_spec("nvidia.dali")
    except (ValueError, ModuleNotFoundError, ImportError):
        dali_spec = None
    if dali_spec is not None:
        return

    nvidia = types.ModuleType("nvidia")
    dali = types.ModuleType("nvidia.dali")
    pipeline = types.ModuleType("nvidia.dali.pipeline")

    def pipeline_def(*_args: object, **_kwargs: object) -> Callable[[Callable[..., object]], Callable[..., object]]:
        def _decorator(func: Callable[..., object]) -> Callable[..., object]:
            return func

        return _decorator

    class _Types:
        RGB = None

    dali.pipeline_def = pipeline_def
    dali.types = _Types
    dali.fn = types.SimpleNamespace(
        readers=types.SimpleNamespace(webdataset=lambda **_kwargs: None),
        decoders=types.SimpleNamespace(image=lambda *_a, **_k: None),
    )
    pipeline.Pipeline = type("Pipeline", (), {})

    sys.modules["nvidia"] = nvidia
    sys.modules["nvidia.dali"] = dali
    sys.modules["nvidia.dali.pipeline"] = pipeline


def test_inputs_outputs_and_name() -> None:
    from nemo_curator.stages.image.io.webdataset_reader import WebDatasetReaderStage

    with patch("torch.cuda.is_available", return_value=True):
        stage = WebDatasetReaderStage(extensions=["jpg"], dali_batch_size=3, verbose=False)
    assert stage.inputs() == ([], [])
    assert stage.outputs() == (["data"], ["key", "components", "shard_path"])
    assert stage.name == "webdataset_reader"


def test_init_allows_cpu_when_no_cuda() -> None:
    from nemo_curator.stages.image.io.webdataset_reader import WebDatasetReaderStage

    with patch("torch.cuda.is_available", return_value=False):
        stage = WebDatasetReaderStage(extensions=["jpg"], dali_batch_size=2, verbose=False)
    assert stage is not None


def test_process_single_extension() -> None:
    from nemo_curator.stages.image.io.webdataset_reader import WebDatasetReaderStage

    task = FileGroupTask(
        task_id="t1",
        dataset_name="ds",
        data=["/data/shard_000.tar", "/data/shard_001.tar"],
    )

    with patch("torch.cuda.is_available", return_value=True):
        stage = WebDatasetReaderStage(extensions=["jpg"], dali_batch_size=2, verbose=False)

    with patch.object(
        WebDatasetReaderStage,
        "_create_dali_pipeline",
        side_effect=_fake_create_pipeline_factory(per_tar_total=5, batch=2, num_extensions=1),
    ):
        batches = stage.process(task)

    assert isinstance(batches, list)
    assert all(isinstance(b, WebDatasetBatch) for b in batches)

    total_samples = sum(b.num_items for b in batches)
    assert total_samples == 10  # 2 tars * 5 samples each

    for batch in batches:
        for sample in batch.data:
            assert isinstance(sample, WebDatasetSample)
            assert sample.key != ""
            assert "jpg" in sample.components
            assert isinstance(sample.components["jpg"], np.ndarray)


def test_process_multiple_extensions() -> None:
    from nemo_curator.stages.image.io.webdataset_reader import WebDatasetReaderStage

    task = FileGroupTask(
        task_id="t2",
        dataset_name="ds",
        data=["/data/shard_000.tar"],
    )

    with patch("torch.cuda.is_available", return_value=True):
        stage = WebDatasetReaderStage(extensions=["jpg", "json"], dali_batch_size=3, verbose=False)

    with patch.object(
        WebDatasetReaderStage,
        "_create_dali_pipeline",
        side_effect=_fake_create_pipeline_factory(per_tar_total=4, batch=3, num_extensions=2),
    ):
        batches = stage.process(task)

    total = sum(b.num_items for b in batches)
    assert total == 4

    for batch in batches:
        for sample in batch.data:
            assert "jpg" in sample.components
            assert "json" in sample.components


def test_process_raises_on_empty_task() -> None:
    from nemo_curator.stages.image.io.webdataset_reader import WebDatasetReaderStage

    empty = FileGroupTask(task_id="e1", dataset_name="ds", data=[])

    with patch("torch.cuda.is_available", return_value=True):
        stage = WebDatasetReaderStage(extensions=["jpg"], dali_batch_size=2, verbose=False)

    with pytest.raises(ValueError, match="No tar file paths"):
        stage.process(empty)


def test_resources_with_cuda() -> None:
    from nemo_curator.stages.image.io.webdataset_reader import WebDatasetReaderStage

    with patch("torch.cuda.is_available", return_value=True):
        stage = WebDatasetReaderStage(dali_batch_size=2, verbose=False)
        res = stage.resources

    assert res.gpus == stage.num_gpus_per_worker
    assert res.requires_gpu is True


def test_resources_without_cuda() -> None:
    from nemo_curator.stages.image.io.webdataset_reader import WebDatasetReaderStage

    with patch("torch.cuda.is_available", return_value=False):
        stage = WebDatasetReaderStage(dali_batch_size=2, verbose=False)
        res = stage.resources

    assert res.gpus == 0
    assert res.requires_gpu is False


def test_shard_path_populated() -> None:
    from nemo_curator.stages.image.io.webdataset_reader import WebDatasetReaderStage

    task = FileGroupTask(
        task_id="t3",
        dataset_name="ds",
        data=["/data/shard_000.tar"],
    )

    with patch("torch.cuda.is_available", return_value=True):
        stage = WebDatasetReaderStage(extensions=["jpg"], dali_batch_size=2, verbose=False)

    with patch.object(
        WebDatasetReaderStage,
        "_create_dali_pipeline",
        side_effect=_fake_create_pipeline_factory(per_tar_total=2, batch=2, num_extensions=1),
    ):
        batches = stage.process(task)

    for batch in batches:
        for sample in batch.data:
            assert sample.shard_path == "/data/shard_000.tar"
