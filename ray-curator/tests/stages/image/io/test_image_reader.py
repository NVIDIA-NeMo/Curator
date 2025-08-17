"""Unit tests for the DALI-based ImageReaderStage (no-resize)."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable
from ray_curator.tasks.file_group import FileGroupTask
from ray_curator.tasks.image import ImageBatch, ImageObject


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


def _fake_create_pipeline_factory(total: int, batch: int) -> Callable[[str], _FakePipeline]:
    def _factory(_tar_path: str) -> _FakePipeline:
        return _FakePipeline(total_samples=total, batch_size=batch)

    return _factory


@pytest.fixture(autouse=True)
def _stub_dali_modules() -> None:
    """Stub nvidia.dali imports so tests run without real DALI installed."""
    nvidia = types.ModuleType("nvidia")
    dali = types.ModuleType("nvidia.dali")
    pipeline = types.ModuleType("nvidia.dali.pipeline")

    def pipeline_def(*_args, **_kwargs):  # noqa: ANN001
        def _decorator(func):  # noqa: ANN001
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
    from ray_curator.stages.image.io.image_reader import ImageReaderStage
    with patch("torch.cuda.is_available", return_value=True):
        stage = ImageReaderStage(task_batch_size=3, verbose=False)
    assert stage.inputs() == ([], [])
    assert stage.outputs() == (["data"], ["image_data", "image_path", "image_id"])
    assert stage.name == "image_reader"


def test_init_requires_cuda() -> None:
    from ray_curator.stages.image.io.image_reader import ImageReaderStage
    with patch("torch.cuda.is_available", return_value=False), pytest.raises(
        RuntimeError, match="requires CUDA"
    ):
        ImageReaderStage(task_batch_size=2, verbose=False)


def test_process_streams_batches_from_dali() -> None:
    from ray_curator.stages.image.io.image_reader import ImageReaderStage
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
        side_effect=_fake_create_pipeline_factory(total=5, batch=2),
    ):
        batches = stage.process(task)

    assert isinstance(batches, list)
    assert all(isinstance(b, ImageBatch) for b in batches)

    total_images = sum(len(b.data) for b in batches)
    assert total_images == 10  # 2 tars * 5 images each
    # Spot-check a couple of ImageObject fields
    assert all(isinstance(img, ImageObject) for b in batches for img in b.data)


def test_process_raises_on_empty_task() -> None:
    from ray_curator.stages.image.io.image_reader import ImageReaderStage
    empty = FileGroupTask(task_id="e1", dataset_name="ds", data=[])

    with patch("torch.cuda.is_available", return_value=True):
        stage = ImageReaderStage(task_batch_size=2, verbose=False)

    with pytest.raises(ValueError, match="No tar file paths"):
        stage.process(empty)



def test_resources_with_cuda_available() -> None:
    from ray_curator.stages.image.io.image_reader import ImageReaderStage
    # Instantiate with CUDA available so __post_init__ passes
    with patch("torch.cuda.is_available", return_value=True):
        stage = ImageReaderStage(task_batch_size=2, verbose=False)
        res = stage.resources

    assert res.gpus == stage.num_gpus_per_worker
    assert res.requires_gpu is True


def test_resources_without_cuda() -> None:
    from ray_curator.stages.image.io.image_reader import ImageReaderStage
    # Create the stage with CUDA available to bypass the init guard
    with patch("torch.cuda.is_available", return_value=True):
        stage = ImageReaderStage(task_batch_size=2, verbose=False)

    # Now simulate CUDA being unavailable when accessing the property
    with patch("torch.cuda.is_available", return_value=False):
        res = stage.resources

    assert res.gpus == 0
    assert res.requires_gpu is False
