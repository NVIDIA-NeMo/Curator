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

"""Unit tests for the generic VADStage.

Stage is tested with a fake adapter -- no WhisperX import, no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest

from nemo_curator.adapters.vad import VADInterval, VADResult
from nemo_curator.stages.audio.inference.vad import VADStage
from nemo_curator.tasks import AudioTask


@dataclass
class _FakeVADAdapter:
    model_id: str = "fake/vad"
    revision: str | None = None
    device: str = "cpu"
    fixed_result: VADResult | None = None
    setup_called: int = 0
    teardown_called: int = 0
    last_batch: list[dict[str, Any]] | None = None
    last_metrics: dict[str, float] = field(default_factory=dict)

    @classmethod
    def prefetch_weights(cls, model_id: str, revision: str | None = None) -> None:
        del model_id, revision

    def setup(self) -> None:
        self.setup_called += 1

    def teardown(self) -> None:
        self.teardown_called += 1

    def detect_batch(self, items: list[dict[str, Any]]) -> list[VADResult]:
        self.last_batch = list(items)
        self.last_metrics = {"batch_size": float(len(items))}
        if self.fixed_result is not None:
            return [self.fixed_result for _ in items]
        return [
            VADResult(
                intervals=[VADInterval(start=0.0, end=1.0)],
                model_id=self.model_id,
                extras={"duration_s": 5.0},
            )
            for _ in items
        ]


_ADAPTER_TARGET = f"{__name__}._FakeVADAdapter"


def _task(**data: Any) -> AudioTask:  # noqa: ANN401
    base = {"resampled_audio_filepath": "/tmp/x.wav", "duration": 5.0}
    base.update(data)
    return AudioTask(data=base)


class TestVADStageConstruction:
    def test_requires_adapter_target(self) -> None:
        with pytest.raises(ValueError, match="adapter_target is required"):
            VADStage()

    def test_defaults(self) -> None:
        s = VADStage(adapter_target=_ADAPTER_TARGET)
        assert s.audio_filepath_key == "resampled_audio_filepath"
        assert s.segments_key == "vad_segments"

    def test_inputs_outputs(self) -> None:
        s = VADStage(adapter_target=_ADAPTER_TARGET)
        assert s.inputs() == ([], ["resampled_audio_filepath"])
        assert s.outputs() == ([], ["resampled_audio_filepath", "vad_segments"])


class TestVADStageLifecycle:
    def test_setup_instantiates_adapter(self) -> None:
        s = VADStage(adapter_target=_ADAPTER_TARGET, model_id="m", revision="r")
        s.setup()
        assert isinstance(s._adapter, _FakeVADAdapter)
        assert s._adapter.model_id == "m"
        assert s._adapter.revision == "r"
        assert s._adapter.setup_called == 1

    def test_setup_forwards_device(self) -> None:
        s = VADStage(adapter_target=_ADAPTER_TARGET)
        s.setup()
        assert s._adapter.device == "cuda"

    def test_teardown_clears_adapter(self) -> None:
        s = VADStage(adapter_target=_ADAPTER_TARGET)
        s.setup()
        adapter = s._adapter
        s.teardown()
        assert s._adapter is None
        assert adapter.teardown_called == 1

    def test_setup_on_node_calls_prefetch(self) -> None:
        with patch.object(_FakeVADAdapter, "prefetch_weights") as mock_pf:
            s = VADStage(adapter_target=_ADAPTER_TARGET, model_id="m")
            s.setup_on_node()
        mock_pf.assert_called_once_with("m", None)

    def test_setup_on_node_swallows_when_disabled(self) -> None:
        with patch.object(_FakeVADAdapter, "prefetch_weights", side_effect=RuntimeError("boom")):
            s = VADStage(adapter_target=_ADAPTER_TARGET, prefetch_fail_on_error=False)
            s.setup_on_node()  # must not raise

    def test_xenna_stage_spec(self) -> None:
        s = VADStage(adapter_target=_ADAPTER_TARGET, xenna_num_workers=3)
        assert s.xenna_stage_spec() == {"num_workers": 3}


class TestVADStageProcess:
    def test_process_requires_setup(self) -> None:
        s = VADStage(adapter_target=_ADAPTER_TARGET)
        with pytest.raises(RuntimeError, match="setup\\(\\) was not called"):
            s.process(_task())

    def test_process_requires_audio_filepath(self) -> None:
        s = VADStage(adapter_target=_ADAPTER_TARGET)
        s.setup()
        with pytest.raises(ValueError, match="Missing key 'resampled_audio_filepath'"):
            s.process(AudioTask(data={"duration": 5.0}))

    def test_process_writes_canonical_intervals(self) -> None:
        s = VADStage(adapter_target=_ADAPTER_TARGET)
        s.setup()
        s._adapter.fixed_result = VADResult(
            intervals=[
                VADInterval(start=0.0, end=1.5),
                VADInterval(start=2.0, end=3.5),
            ],
            model_id="x",
            extras={"duration_s": 5.0},
        )
        out = s.process(_task())
        assert out.data["vad_segments"] == [
            {"start": 0.0, "end": 1.5},
            {"start": 2.0, "end": 3.5},
        ]

    def test_process_empty_intervals_skipped_short(self) -> None:
        s = VADStage(adapter_target=_ADAPTER_TARGET)
        s.setup()
        s._adapter.fixed_result = VADResult(intervals=[], model_id="x", extras={"duration_s": 0.2})
        out = s.process(_task(duration=0.2))
        assert out.data["vad_segments"] == []

    def test_process_forwards_item_dict_to_adapter(self) -> None:
        s = VADStage(
            adapter_target=_ADAPTER_TARGET,
            waveform_key="waveform",
            sample_rate_key="sample_rate",
        )
        s.setup()
        s.process(_task(waveform=[1.0], sample_rate=16000))
        assert s._adapter.last_batch is not None
        item = s._adapter.last_batch[0]
        assert item["audio_filepath"] == "/tmp/x.wav"
        assert item["duration"] == 5.0
        assert item["waveform"] == [1.0]
        assert item["sample_rate"] == 16000


class TestVADStageMetrics:
    def test_log_metrics_includes_adapter_aliases(self) -> None:
        s = VADStage(adapter_target=_ADAPTER_TARGET)
        s.setup()
        observed: list[dict[str, float]] = []
        s._log_metrics = observed.append  # type: ignore[assignment]
        s._adapter.fixed_result = VADResult(
            intervals=[VADInterval(start=0.0, end=1.0)],
            model_id="x",
            extras={"duration_s": 5.0},
        )
        s.process(_task())
        assert observed
        m = observed[-1]
        assert m["vad_segments_detected"] == 1.0
        assert m["audio_duration"] == 5.0
        assert m["model_batch_size"] == 1.0
        assert "process_time" in m
