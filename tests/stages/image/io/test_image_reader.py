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

import pathlib
import sys
import types
from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
import torch

if TYPE_CHECKING:
    from collections.abc import Callable
from nemo_curator.tasks.file_group import FileGroupTask
from nemo_curator.tasks.image import ImageBatch, ImageObject


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

    def build(self) -> None:
        return None

    def epoch_size(self) -> dict[int, int]:
        return {0: self.total_samples}

    def run(self) -> _FakeTensorList:
        return _FakeTensorList(self.batch_size)


def _fake_create_pipeline_factory(per_tar_total: int, batch: int) -> Callable[[list[str]], _FakePipeline]:
    def _factory(tar_paths: list[str] | tuple[str, ...]) -> _FakePipeline:
        num_paths = len(tar_paths) if isinstance(tar_paths, (list, tuple)) else 1
        return _FakePipeline(total_samples=per_tar_total * num_paths, batch_size=batch)

    return _factory


@pytest.fixture(autouse=True)
def _stub_dali_modules() -> None:
    """Stub nvidia.dali only on CPU-only environments without real DALI.

    We avoid stubbing when CUDA is available so the GPU test can either use
    the real DALI (if installed) or skip cleanly if it's not.
    """
    import importlib.util

    if torch.cuda.is_available():
        return
    # Some environments may have a broken/partial installation where
    # nvidia.dali is present in sys.modules with __spec__ = None.
    # importlib.util.find_spec raises ValueError in that case. Treat this as
    # "not installed" so we provide our stub.
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
    from nemo_curator.stages.image.io.image_reader import ImageReaderStage
    with patch("torch.cuda.is_available", return_value=True):
        stage = ImageReaderStage(task_batch_size=3, verbose=False)
    assert stage.inputs() == ([], [])
    assert stage.outputs() == (["data"], ["image_data", "image_path", "image_id"])
    assert stage.name == "image_reader"


def test_init_allows_cpu_when_no_cuda() -> None:
    from nemo_curator.stages.image.io.image_reader import ImageReaderStage
    # When CUDA is unavailable, the stage should initialize and use CPU DALI
    with patch("torch.cuda.is_available", return_value=False):
        stage = ImageReaderStage(task_batch_size=2, verbose=False)
    assert stage is not None


def test_process_streams_batches_from_dali() -> None:
    from nemo_curator.stages.image.io.image_reader import ImageReaderStage
    # Two tar files; each has 5 total samples, emitted in batches of 2 (2,2,1)
    task = FileGroupTask(
        task_id="t1",
        dataset_name="ds",
        data=["/data/a.tar", "/data/b.tar"],
    )

    with patch("torch.cuda.is_available", return_value=True):
        stage = ImageReaderStage(task_batch_size=2, verbose=False)

    with patch.object(
        ImageReaderStage,
        "_create_dali_pipeline",
        side_effect=_fake_create_pipeline_factory(per_tar_total=5, batch=2),
    ):
        batches = stage.process(task)

    assert isinstance(batches, list)
    assert all(isinstance(b, ImageBatch) for b in batches)

    total_images = sum(len(b.data) for b in batches)
    assert total_images == 10  # 2 tars * 5 images each
    # Spot-check a couple of ImageObject fields
    assert all(isinstance(img, ImageObject) for b in batches for img in b.data)


def test_process_raises_on_empty_task() -> None:
    from nemo_curator.stages.image.io.image_reader import ImageReaderStage
    empty = FileGroupTask(task_id="e1", dataset_name="ds", data=[])

    with patch("torch.cuda.is_available", return_value=True):
        stage = ImageReaderStage(task_batch_size=2, verbose=False)

    with pytest.raises(ValueError, match="No tar file paths"):
        stage.process(empty)



def test_resources_with_cuda_available() -> None:
    from nemo_curator.stages.image.io.image_reader import ImageReaderStage
    # Instantiate with CUDA available so __post_init__ passes
    with patch("torch.cuda.is_available", return_value=True):
        stage = ImageReaderStage(task_batch_size=2, verbose=False)
        res = stage.resources

    assert res.gpus == stage.num_gpus_per_worker
    assert res.requires_gpu is True


def test_resources_without_cuda() -> None:
    from nemo_curator.stages.image.io.image_reader import ImageReaderStage
    # Create the stage without CUDA available
    with patch("torch.cuda.is_available", return_value=False):
        stage = ImageReaderStage(task_batch_size=2, verbose=False)
        res = stage.resources

    assert res.gpus == 0
    assert res.requires_gpu is False


# GPU integration test using real DALI if available
@pytest.mark.gpu
def test_dali_image_reader_on_gpu() -> None:
    if not torch.cuda.is_available():  # pragma: no cover - CPU CI
        pytest.skip("CUDA not available; GPU test skipped")

    try:
        import nvidia.dali  # noqa: F401
    except (ModuleNotFoundError, ImportError):  # pragma: no cover - environment without DALI
        pytest.skip("nvidia.dali not available; skipping GPU reader test")

    # Reuse sample webdataset tar from repository-level tests assets
    # Project root is parents[5] from this file (ray-curator/tests/stages/image/io)
    tar_path = pathlib.Path(__file__).resolve().parents[5] / "tests" / "image_data" / "00000.tar"
    if not tar_path.exists():  # pragma: no cover - missing asset
        pytest.skip(f"Sample dataset not found at {tar_path}")

    from nemo_curator.stages.image.io.image_reader import ImageReaderStage
    from nemo_curator.tasks import FileGroupTask

    stage = ImageReaderStage(task_batch_size=2, num_threads=2, verbose=False)
    task = FileGroupTask(task_id="t0", dataset_name="ds", data=[str(tar_path)])

    batches = stage.process(task)

    # Should yield at least one batch with decoded images
    assert isinstance(batches, list)
    assert len(batches) >= 1
    total_images = 0
    for batch in batches:
        assert len(batch.data) >= 1
        for img in batch.data:
            # Validate decoded image
            assert img.image_data is not None
            assert img.image_data.ndim == 3  # H, W, C
            assert img.image_data.shape[2] == 3
            assert img.image_id != ""
            assert img.image_path.endswith(".jpg")
            total_images += 1

    assert total_images >= 1
