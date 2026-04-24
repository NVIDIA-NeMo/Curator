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

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from collections.abc import Generator


def _make_task(text: str = "hello", sr: int = 16000, dur_s: float = 1.0) -> AudioTask:
    waveform = np.zeros(int(sr * dur_s), dtype=np.float32)
    return AudioTask(
        task_id=f"utt_{text}",
        dataset_name="test",
        data={"waveform": waveform, "sample_rate": sr},
    )


@pytest.fixture
def patch_qwen_model() -> Generator[tuple[MagicMock, MagicMock]]:
    """Patch the QwenOmni import so vLLM/torch are not required."""
    mock_model_cls = MagicMock()
    mock_model_instance = MagicMock()
    mock_model_cls.return_value = mock_model_instance

    mock_model_instance.generate.return_value = (
        ["transcribed text"],
        ["refined text"],
        {"batch_size": 1.0, "t1_prep_s": 0.01, "t1_inference_s": 0.5},
    )

    with patch(
        "nemo_curator.stages.audio.inference.qwen_omni.QwenOmni",
        mock_model_cls,
    ):
        yield mock_model_cls, mock_model_instance


def _make_stage(**kwargs: object) -> object:
    from nemo_curator.stages.audio.inference.qwen_omni import InferenceQwenOmniStage

    defaults: dict[str, object] = {
        "model_id": "mock/model",
        "followup_prompt": "Refine.",
    }
    defaults.update(kwargs)
    return InferenceQwenOmniStage(**defaults)


# ---------------------------------------------------------------------------
# I/O contract
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("patch_qwen_model")
def test_inputs_declares_waveform_and_sample_rate() -> None:
    stage = _make_stage()
    _, required = stage.inputs()
    assert "waveform" in required
    assert "sample_rate" in required


@pytest.mark.usefixtures("patch_qwen_model")
def test_outputs_includes_pred_text() -> None:
    stage = _make_stage(followup_prompt=None)
    _, data_attrs = stage.outputs()
    assert "qwen3_prediction_s1" in data_attrs


@pytest.mark.usefixtures("patch_qwen_model")
def test_outputs_includes_disfluency_when_followup() -> None:
    stage = _make_stage(followup_prompt="Refine this.")
    _, data_attrs = stage.outputs()
    assert "qwen3_prediction_s2" in data_attrs


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_setup_creates_model(
    patch_qwen_model: tuple[MagicMock, MagicMock],
) -> None:
    mock_cls, mock_instance = patch_qwen_model
    stage = _make_stage()
    stage.setup(None)
    mock_cls.assert_called_once()
    mock_instance.setup.assert_called_once()


def test_teardown_cleans_up(
    patch_qwen_model: tuple[MagicMock, MagicMock],
) -> None:
    _, mock_instance = patch_qwen_model
    stage = _make_stage()
    stage.setup(None)
    stage.teardown()
    mock_instance.teardown.assert_called_once()
    assert stage._model is None


@pytest.mark.usefixtures("patch_qwen_model")
def test_teardown_noop_when_no_model() -> None:
    stage = _make_stage()
    stage.teardown()


# ---------------------------------------------------------------------------
# process() raises NotImplementedError
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("patch_qwen_model")
def test_process_raises() -> None:
    stage = _make_stage()
    with pytest.raises(NotImplementedError, match="only supports process_batch"):
        stage.process(_make_task())


# ---------------------------------------------------------------------------
# process_batch
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("patch_qwen_model")
def test_process_batch_empty_returns_empty() -> None:
    stage = _make_stage()
    stage.setup(None)
    assert stage.process_batch([]) == []


def test_process_batch_populates_predictions(
    patch_qwen_model: tuple[MagicMock, MagicMock],
) -> None:
    _, mock_instance = patch_qwen_model
    mock_instance.generate.return_value = (
        ["hello world"],
        ["hello world refined"],
        {"batch_size": 1.0, "t1_inference_s": 0.1},
    )

    stage = _make_stage(followup_prompt="Refine.")
    stage.setup(None)
    tasks = [_make_task()]
    results = stage.process_batch(tasks)

    assert len(results) == 1
    assert results[0].data["qwen3_prediction_s1"] == "hello world"
    assert results[0].data["qwen3_prediction_s2"] == "hello world refined"
    assert "waveform" not in results[0].data


def test_process_batch_without_followup(
    patch_qwen_model: tuple[MagicMock, MagicMock],
) -> None:
    _, mock_instance = patch_qwen_model
    mock_instance.generate.return_value = (["text"], [None], {"batch_size": 1.0})

    stage = _make_stage(followup_prompt=None)
    stage.setup(None)
    results = stage.process_batch([_make_task()])

    assert results[0].data["qwen3_prediction_s1"] == "text"
    assert "qwen3_prediction_s2" not in results[0].data


@pytest.mark.usefixtures("patch_qwen_model")
def test_process_batch_raises_without_setup() -> None:
    stage = _make_stage()
    with pytest.raises(RuntimeError, match="Model not initialized"):
        stage.process_batch([_make_task()])


@pytest.mark.usefixtures("patch_qwen_model")
def test_process_batch_validates_input() -> None:
    stage = _make_stage()
    stage.setup(None)
    bad_task = AudioTask(task_id="bad", dataset_name="test", data={"wrong_key": 1})
    with pytest.raises(ValueError, match="missing required columns"):
        stage.process_batch([bad_task])


def test_process_batch_multi_task(
    patch_qwen_model: tuple[MagicMock, MagicMock],
) -> None:
    _, mock_instance = patch_qwen_model
    mock_instance.generate.return_value = (
        ["a", "b", "c"],
        ["a2", "b2", "c2"],
        {"batch_size": 3.0, "t1_inference_s": 0.3, "t2_inference_s": 0.2},
    )

    stage = _make_stage(followup_prompt="Refine.")
    stage.setup(None)
    tasks = [_make_task(text=t) for t in ["a", "b", "c"]]
    results = stage.process_batch(tasks)

    assert len(results) == 3
    assert [r.data["qwen3_prediction_s1"] for r in results] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# GPU resource declaration
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("patch_qwen_model")
def test_default_resources_request_gpu() -> None:
    stage = _make_stage()
    assert stage.resources.gpus == 1.0


@pytest.mark.usefixtures("patch_qwen_model")
def test_tensor_parallel_updates_resources() -> None:
    stage = _make_stage(tensor_parallel_size=4)
    assert stage.resources.gpus == 4.0


# ---------------------------------------------------------------------------
# Field order: name first, resources/batch_size last
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("patch_qwen_model")
def test_field_order_compliance() -> None:
    from dataclasses import fields

    from nemo_curator.stages.audio.inference.qwen_omni import InferenceQwenOmniStage

    names = [f.name for f in fields(InferenceQwenOmniStage)]
    assert names[0] == "name"
    assert names[-1] == "batch_size"
    assert names[-2] == "resources"


# ---------------------------------------------------------------------------
# New performance parameters passthrough
# ---------------------------------------------------------------------------


def test_fp8_passthrough(
    patch_qwen_model: tuple[MagicMock, MagicMock],
) -> None:
    mock_cls, _ = patch_qwen_model
    stage = _make_stage(fp8=True)
    stage.setup(None)
    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["fp8"] is True


def test_enforce_eager_passthrough(
    patch_qwen_model: tuple[MagicMock, MagicMock],
) -> None:
    mock_cls, _ = patch_qwen_model
    stage = _make_stage(enforce_eager=True)
    stage.setup(None)
    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["enforce_eager"] is True


def test_max_retries_passthrough(
    patch_qwen_model: tuple[MagicMock, MagicMock],
) -> None:
    mock_cls, _ = patch_qwen_model
    stage = _make_stage(max_retries=5)
    stage.setup(None)
    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["max_retries"] == 5


def test_inference_chunk_size_passthrough(
    patch_qwen_model: tuple[MagicMock, MagicMock],
) -> None:
    mock_cls, _ = patch_qwen_model
    stage = _make_stage(inference_chunk_size=16)
    stage.setup(None)
    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["inference_chunk_size"] == 16


def test_mm_cache_gb_passthrough(
    patch_qwen_model: tuple[MagicMock, MagicMock],
) -> None:
    mock_cls, _ = patch_qwen_model
    stage = _make_stage(mm_cache_gb=8.0)
    stage.setup(None)
    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["mm_cache_gb"] == 8.0


def test_default_disable_log_stats(
    patch_qwen_model: tuple[MagicMock, MagicMock],
) -> None:
    mock_cls, _ = patch_qwen_model
    stage = _make_stage()
    stage.setup(None)
    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["disable_log_stats"] is True


# ---------------------------------------------------------------------------
# Metrics are recorded via _log_metrics
# ---------------------------------------------------------------------------


def test_process_batch_records_metrics(
    patch_qwen_model: tuple[MagicMock, MagicMock],
) -> None:
    _, mock_instance = patch_qwen_model
    mock_instance.generate.return_value = (
        ["text"],
        ["refined"],
        {"batch_size": 1.0, "t1_prep_s": 0.01, "t1_inference_s": 0.5, "t2_inference_s": 0.3},
    )

    stage = _make_stage(followup_prompt="Refine.")
    stage.setup(None)
    stage.process_batch([_make_task()])

    assert hasattr(stage, "_custom_metrics")
    assert stage._custom_metrics["t1_inference_s"] == 0.5
