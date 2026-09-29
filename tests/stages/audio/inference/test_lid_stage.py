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

"""Tests for task I/O and resume behavior in the generic audio-LID stage."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile
import torch

from nemo_curator.models.audio.lid.base import AudioLIDResult
from nemo_curator.stages.audio.inference.lid.stage import AudioLIDInferenceStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from pathlib import Path

_SAMPLE_RATE = 16000
_MODEL_ID = "SpeechBrainLangID"
_ADAPTER_TARGET = "package.AudioLIDAdapter"


class _FakeAudioLIDAdapter:
    def __init__(self, results: list[AudioLIDResult] | None = None) -> None:
        self.results = results
        self.calls: list[list[dict[str, Any]]] = []

    def identify_batch(self, items: list[dict[str, Any]]) -> list[AudioLIDResult]:
        self.calls.append(items)
        if self.results is not None:
            return self.results
        return [AudioLIDResult(language="ta", confidence=0.75) for _ in items]


def _stage(
    *, results: list[AudioLIDResult] | None = None, **kwargs: object
) -> tuple[AudioLIDInferenceStage, _FakeAudioLIDAdapter]:
    config: dict[str, object] = {
        "adapter_target": _ADAPTER_TARGET,
        "model_id": _MODEL_ID,
        "tag": "primary",
        "waveform_key": "waveform",
        "sample_rate": _SAMPLE_RATE,
        "min_duration_sec": 0.5,
    }
    config.update(kwargs)
    stage = AudioLIDInferenceStage(**config)  # type: ignore[arg-type]
    adapter = _FakeAudioLIDAdapter(results)
    stage._adapter = adapter
    return stage, adapter


def _task(seconds: float = 1.0, sample_rate: int = _SAMPLE_RATE, **extra: object) -> AudioTask:
    return AudioTask(
        data={
            "waveform": np.zeros(int(seconds * sample_rate), dtype=np.float32),
            "sample_rate": sample_rate,
            **extra,
        }
    )


def test_an_empty_batch_returns_an_empty_list() -> None:
    stage, adapter = _stage()
    assert stage.process_batch([]) == []
    assert adapter.calls == []


def test_a_ray_data_numpy_batch_is_processed() -> None:
    stage, adapter = _stage()
    tasks: Any = np.asarray([_task(), _task()], dtype=object)

    results = stage.process_batch(tasks)

    assert len(results) == 2
    assert len(adapter.calls[0]) == 2
    assert all(_MODEL_ID in task.data["lid"] for task in results)


def test_process_rejects_non_batch_execution() -> None:
    stage, adapter = _stage()
    with pytest.raises(NotImplementedError, match="only supports process_batch"):
        stage.process(_task())
    assert adapter.calls == []


def test_results_are_json_safe_and_preserve_other_models() -> None:
    stage, _ = _stage(results=[AudioLIDResult(language="TA", confidence=np.float32(0.625))])
    task = _task(lid={"WhisperLangID": {"language": "ta", "confidence": 0.7, "tag": "tertiary"}})

    (result,) = stage.process_batch([task])

    assert result.data["lid"] == {
        "WhisperLangID": {"language": "ta", "confidence": 0.7, "tag": "tertiary"},
        _MODEL_ID: {"language": "ta", "confidence": 0.625, "tag": "primary"},
    }
    json.dumps(result.data["lid"])


def test_a_ragged_batch_becomes_one_adapter_call_and_long_audio_is_truncated() -> None:
    stage, adapter = _stage(max_duration_sec=2.0)

    stage.process_batch([_task(1.0), _task(3.0)])

    assert len(adapter.calls) == 1
    assert [len(item["waveform"]) for item in adapter.calls[0]] == [_SAMPLE_RATE, 2 * _SAMPLE_RATE]


@pytest.mark.parametrize(
    "waveform",
    [
        np.zeros(_SAMPLE_RATE, dtype=np.float32),
        torch.zeros(_SAMPLE_RATE, dtype=torch.float32),
        np.zeros((2, _SAMPLE_RATE), dtype=np.float32),
        torch.zeros((2, _SAMPLE_RATE), dtype=torch.float32),
    ],
)
def test_waveforms_are_normalized_to_contiguous_mono(waveform: object) -> None:
    stage, adapter = _stage()
    stage.process_batch([AudioTask(data={"waveform": waveform, "sample_rate": _SAMPLE_RATE})])

    prepared = adapter.calls[0][0]["waveform"]
    assert prepared.shape == (_SAMPLE_RATE,)
    assert prepared.dtype == np.float32
    assert prepared.flags.c_contiguous


def test_a_mismatched_sample_rate_is_resampled_before_inference() -> None:
    librosa = MagicMock()
    librosa.resample.side_effect = lambda waveform, *, orig_sr, target_sr: np.repeat(waveform, target_sr // orig_sr)
    stage, adapter = _stage()
    with patch.dict("sys.modules", {"librosa": librosa}):
        stage.process_batch([_task(sample_rate=8000)])

    assert len(adapter.calls[0][0]["waveform"]) == _SAMPLE_RATE
    librosa.resample.assert_called_once()


def test_file_mode_loads_audio_for_the_adapter(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    soundfile.write(audio_path, np.zeros(_SAMPLE_RATE, dtype=np.float32), _SAMPLE_RATE)
    stage, adapter = _stage(waveform_key=None)

    (task,) = stage.process_batch([AudioTask(data={"audio_filepath": str(audio_path)})])

    assert len(adapter.calls[0][0]["waveform"]) == _SAMPLE_RATE
    assert task.data["lid"][_MODEL_ID]["language"] == "ta"


def test_a_short_clip_gets_an_empty_completed_result_without_a_skip_flag() -> None:
    stage, adapter = _stage(min_duration_sec=1.0)

    (task,) = stage.process_batch([_task(0.25)])

    assert task.data["lid"][_MODEL_ID] == {"language": "", "confidence": 0.0, "tag": "primary"}
    assert "_skipme" not in task.data
    assert adapter.calls == []


def test_an_adapter_model_miss_is_stored_as_a_completed_result() -> None:
    stage, _ = _stage(results=[AudioLIDResult(language="", confidence=0.0)])
    (task,) = stage.process_batch([_task()])
    assert task.data["lid"][_MODEL_ID] == {"language": "", "confidence": 0.0, "tag": "primary"}


def test_an_audio_error_is_non_terminal_by_default() -> None:
    stage, adapter = _stage()

    (task,) = stage.process_batch([AudioTask(data={"sample_rate": _SAMPLE_RATE})])

    assert task.data["lid"][_MODEL_ID] == {"language": "", "confidence": 0.0, "tag": "primary"}
    assert "_skipme" not in task.data
    assert adapter.calls == []


def test_fail_on_audio_error_marks_the_empty_result_terminal() -> None:
    stage, adapter = _stage(fail_on_audio_error=True)

    (task,) = stage.process_batch([AudioTask(data={"sample_rate": _SAMPLE_RATE})])

    assert task.data["lid"][_MODEL_ID]["confidence"] == 0.0
    assert task.data["_skipme"] == "audio_load_error"
    assert adapter.calls == []


def test_an_existing_shared_skip_is_passed_through_untouched() -> None:
    stage, adapter = _stage()
    task = _task(_skipme="rejected upstream")

    (result,) = stage.process_batch([task])

    assert result.data["_skipme"] == "rejected upstream"
    assert "lid" not in result.data
    assert adapter.calls == []


def test_resume_uses_own_model_key_membership_even_at_zero_confidence() -> None:
    stage, adapter = _stage(skip_if_output_exists=True)
    existing = {"language": "", "confidence": 0.0, "tag": "primary"}
    done = _task(lid={_MODEL_ID: existing})
    unfinished = _task(lid={"WhisperLangID": {"language": "en", "confidence": 0.9, "tag": "tertiary"}})

    stage.process_batch([done, unfinished])

    assert len(adapter.calls) == 1
    assert len(adapter.calls[0]) == 1
    assert done.data["lid"][_MODEL_ID] is existing
    assert unfinished.data["lid"][_MODEL_ID]["language"] == "ta"


def test_adapter_result_count_must_match_the_inference_items() -> None:
    stage, _ = _stage(results=[])
    with pytest.raises(RuntimeError, match="must match 1:1"):
        stage.process_batch([_task()])


def test_a_non_mapping_results_field_is_rejected_without_data_loss() -> None:
    stage, _ = _stage()
    task = _task(0.25, lid=["legacy"])
    with pytest.raises(TypeError, match="must be a mapping"):
        stage.process_batch([task])
    assert task.data["lid"] == ["legacy"]


def test_adapter_construction_uses_provider_kwargs_not_the_stable_result_id() -> None:
    seen: dict[str, object] = {}

    class _LifecycleAdapter:
        def __init__(self, *, sample_rate: int, source: str) -> None:
            seen.update(sample_rate=sample_rate, source=source)

        def load_model(self, *, num_gpus: int) -> None:
            seen["num_gpus"] = num_gpus

        def unload_model(self) -> None:
            seen["unloaded"] = True

    stage = AudioLIDInferenceStage(
        adapter_target=_ADAPTER_TARGET,
        model_id=_MODEL_ID,
        tag="primary",
        adapter_kwargs={"source": "speechbrain/lang-id-voxlingua107-ecapa"},
        resources=Resources(cpus=1.0),
    )
    with patch.object(stage, "_adapter_class", return_value=_LifecycleAdapter):
        stage.setup()
    stage.teardown()

    assert seen == {
        "sample_rate": _SAMPLE_RATE,
        "source": "speechbrain/lang-id-voxlingua107-ecapa",
        "num_gpus": 0,
        "unloaded": True,
    }


def test_stage_declares_file_or_waveform_inputs_and_shared_outputs() -> None:
    file_stage = AudioLIDInferenceStage(adapter_target=_ADAPTER_TARGET, model_id=_MODEL_ID, tag="primary")
    waveform_stage = AudioLIDInferenceStage(
        adapter_target=_ADAPTER_TARGET,
        model_id=_MODEL_ID,
        tag="primary",
        waveform_key="samples",
        sample_rate_key="sr",
    )

    assert file_stage.inputs() == ([], ["audio_filepath"])
    assert waveform_stage.inputs() == ([], ["samples", "sr"])
    assert file_stage.outputs() == ([], ["lid", "_skipme"])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"model_id": ""}, "model_id"),
        ({"sample_rate": 0}, "sample_rate"),
        ({"batch_size": 0}, "batch_size"),
        ({"min_duration_sec": -1.0}, "min_duration_sec"),
        ({"max_duration_sec": float("nan")}, "max_duration_sec"),
    ],
)
def test_invalid_stage_configuration_is_rejected(kwargs: dict[str, object], match: str) -> None:
    base: dict[str, object] = {"adapter_target": _ADAPTER_TARGET, "model_id": _MODEL_ID, "tag": "primary"}
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        AudioLIDInferenceStage(**base)  # type: ignore[arg-type]
