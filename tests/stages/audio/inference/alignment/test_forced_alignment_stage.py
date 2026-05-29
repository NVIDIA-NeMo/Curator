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

"""Unit tests for the generic ForcedAlignmentStage.

Stage is tested with a fake adapter -- no NeMo / torch / GPU needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from nemo_curator.adapters.alignment import AlignmentResult, WordAlignment
from nemo_curator.stages.audio.inference.alignment import ForcedAlignmentStage
from nemo_curator.tasks import AudioTask


@dataclass
class _FakeAlignAdapter:
    model_id: str = "fake/align"
    revision: str | None = None
    device: str = "cpu"
    setup_called: int = 0
    teardown_called: int = 0
    last_batch: list[dict[str, Any]] | None = None
    last_metrics: dict[str, float] = field(default_factory=dict)
    fixed_results: list[AlignmentResult] | None = None

    @classmethod
    def prefetch_weights(cls, model_id: str, revision: str | None = None) -> None:
        del model_id, revision

    def setup(self) -> None:
        self.setup_called += 1

    def teardown(self) -> None:
        self.teardown_called += 1

    def align_batch(self, items: list[dict[str, Any]]) -> list[AlignmentResult]:
        self.last_batch = list(items)
        self.last_metrics = {"batch_size": float(len(items))}
        if self.fixed_results is not None:
            return list(self.fixed_results)
        return [
            AlignmentResult(
                alignments=[WordAlignment(word="hello", start=0.0, end=0.5, confidence=0.9)],
                text="hello",
                model_id=self.model_id,
            )
            for _ in items
        ]


_ADAPTER_TARGET = f"{__name__}._FakeAlignAdapter"


class TestConstruction:
    def test_requires_adapter_target(self) -> None:
        with pytest.raises(ValueError, match="adapter_target is required"):
            ForcedAlignmentStage()

    def test_default_io(self) -> None:
        s = ForcedAlignmentStage(adapter_target=_ADAPTER_TARGET)
        ins, outs = s.inputs()
        assert ins == ["data"]
        assert "split_filepaths" in outs
        assert "split_metadata" in outs


class TestLifecycle:
    def test_setup_instantiates_with_tier1_tier2(self) -> None:
        s = ForcedAlignmentStage(
            adapter_target=_ADAPTER_TARGET,
            model_id="m",
            revision="r",
            adapter_kwargs={"setup_called": 0},
        )
        s.setup()
        assert isinstance(s._adapter, _FakeAlignAdapter)
        assert s._adapter.model_id == "m"
        assert s._adapter.revision == "r"
        assert s._adapter.setup_called == 1

    def test_teardown_clears(self) -> None:
        s = ForcedAlignmentStage(adapter_target=_ADAPTER_TARGET)
        s.setup()
        adapter = s._adapter
        s.teardown()
        assert s._adapter is None
        assert adapter.teardown_called == 1

    def test_prefetch_called_setup_on_node(self) -> None:
        with patch.object(_FakeAlignAdapter, "prefetch_weights") as mock_pf:
            s = ForcedAlignmentStage(adapter_target=_ADAPTER_TARGET, model_id="m", revision="r")
            s.setup_on_node()
        mock_pf.assert_called_once_with("m", "r")

    def test_prefetch_failure_swallowed(self) -> None:
        with patch.object(_FakeAlignAdapter, "prefetch_weights", side_effect=RuntimeError("x")):
            s = ForcedAlignmentStage(adapter_target=_ADAPTER_TARGET, prefetch_fail_on_error=False)
            s.setup_on_node()


class TestProcessBatchFullAudio:
    def test_requires_setup(self) -> None:
        s = ForcedAlignmentStage(adapter_target=_ADAPTER_TARGET)
        with pytest.raises(RuntimeError, match="setup\\(\\) was not called"):
            s.process_batch([AudioTask(data={})])

    def test_empty_batch_returns_empty(self) -> None:
        s = ForcedAlignmentStage(adapter_target=_ADAPTER_TARGET)
        s.setup()
        assert s.process_batch([]) == []

    def test_scatters_results_into_split_metadata(self) -> None:
        s = ForcedAlignmentStage(adapter_target=_ADAPTER_TARGET)
        s.setup()
        s._adapter.fixed_results = [
            AlignmentResult(
                alignments=[WordAlignment(word="a", start=0.0, end=0.3, confidence=0.8)],
                text="a",
                model_id="m",
            ),
            AlignmentResult(
                alignments=[WordAlignment(word="b", start=0.0, end=0.4, confidence=0.7)],
                text="b",
                model_id="m",
            ),
        ]
        task = AudioTask(
            data={
                "split_filepaths": ["/p/1.wav", "/p/2.wav"],
                "split_metadata": [
                    {"resampled_audio_filepath": "/p/1.wav"},
                    {"resampled_audio_filepath": "/p/2.wav"},
                ],
            }
        )
        out = s.process_batch([task])
        assert len(out) == 1
        md = out[0].data["split_metadata"]
        assert md[0]["text"] == "a"
        assert md[0]["alignment"][0]["word"] == "a"
        assert md[1]["text"] == "b"
        assert md[1]["alignment"][0]["word"] == "b"
        # Adapter received homogeneous path-mode items.
        batch = s._adapter.last_batch
        assert batch is not None and len(batch) == 2
        assert all("audio_path" in item for item in batch)
        assert all("audio_segment" not in item for item in batch)

    def test_writes_top_level_when_no_split_metadata(self) -> None:
        s = ForcedAlignmentStage(adapter_target=_ADAPTER_TARGET)
        s.setup()
        s._adapter.fixed_results = [
            AlignmentResult(alignments=[], text="hello", model_id="m"),
        ]
        task = AudioTask(data={"split_filepaths": ["/p/x.wav"], "split_metadata": []})
        out = s.process_batch([task])
        assert out[0].data["text"] == "hello"
        assert out[0].data["alignment"] == []

    def test_sentinel_split_filepaths_empty_string(self) -> None:
        s = ForcedAlignmentStage(adapter_target=_ADAPTER_TARGET)
        s.setup()
        task = AudioTask(data={"split_filepaths": "skip-me"})
        out = s.process_batch([task])
        assert out[0].data["text"] == ""
        assert out[0].data["alignment"] == []

    def test_missing_split_filepaths_passes_through(self) -> None:
        s = ForcedAlignmentStage(adapter_target=_ADAPTER_TARGET)
        s.setup()
        # split_filepaths key absent -> meta entry mode; adapter still called with 0 items.
        # Stage handles by emitting empty results -- nothing to scatter.
        task = AudioTask(data={"split_metadata": []})
        out = s.process_batch([task])
        # No exception, no text written (no splits -> nothing to do).
        assert "text" not in out[0].data


class TestProcessBatchSegmentMode:
    @patch("nemo_curator.stages.audio.inference.alignment.stage.torchaudio.load")
    def test_segments_cut_and_scattered_with_time_offset(self, mock_load: MagicMock) -> None:
        import torch

        mock_load.return_value = (torch.zeros(1, 16000 * 10), 16000)
        s = ForcedAlignmentStage(
            adapter_target=_ADAPTER_TARGET,
            infer_segment_only=True,
            min_len=0.5,
        )
        s.setup()
        s._adapter.fixed_results = [
            AlignmentResult(
                alignments=[WordAlignment(word="hi", start=0.0, end=0.2, confidence=0.9)],
                text="hi",
                model_id="m",
            ),
            AlignmentResult(
                alignments=[WordAlignment(word="bye", start=0.0, end=0.3, confidence=0.95)],
                text="bye",
                model_id="m",
            ),
        ]
        task = AudioTask(
            data={
                "resampled_audio_filepath": "/p/x.wav",
                "segments": [
                    {"start": 1.0, "end": 2.5},  # 1.5s -> included
                    {"start": 3.0, "end": 4.6},  # 1.6s -> included
                    {"start": 5.0, "end": 5.2},  # 0.2s -> skipped (< min_len)
                ],
            }
        )
        out = s.process_batch([task])
        segs = out[0].data["segments"]
        # First segment cut + transcribed + words[start] offset by seg start (1.0).
        assert segs[0]["text"] == "hi"
        assert segs[0]["words"][0]["start"] == 1.0
        assert segs[0]["words"][0]["end"] == 1.2
        # Second segment offset by 3.0.
        assert segs[1]["text"] == "bye"
        assert segs[1]["words"][0]["start"] == 3.0
        assert segs[1]["words"][0]["end"] == 3.3
        # Third segment skipped (too short) -> no text added.
        assert "text" not in segs[2]
        # Adapter received homogeneous segment-mode items.
        batch = s._adapter.last_batch
        assert batch is not None and len(batch) == 2
        assert all("audio_segment" in item for item in batch)
        assert all("audio_path" not in item for item in batch)

    @patch("nemo_curator.stages.audio.inference.alignment.stage.torchaudio.load")
    def test_no_eligible_segments_does_not_call_adapter(self, mock_load: MagicMock) -> None:
        import torch

        mock_load.return_value = (torch.zeros(1, 16000 * 10), 16000)
        s = ForcedAlignmentStage(adapter_target=_ADAPTER_TARGET, infer_segment_only=True, min_len=2.0)
        s.setup()
        task = AudioTask(
            data={
                "resampled_audio_filepath": "/p/x.wav",
                "segments": [{"start": 0.0, "end": 0.5}],
            }
        )
        s.process_batch([task])
        assert s._adapter.last_batch is None


class TestMetrics:
    def test_logs_entries_processed_and_adapter_aliases(self) -> None:
        s = ForcedAlignmentStage(adapter_target=_ADAPTER_TARGET)
        s.setup()
        observed: list[dict[str, float]] = []
        s._log_metrics = observed.append  # type: ignore[assignment]
        task = AudioTask(
            data={"split_filepaths": ["/p/1.wav"], "split_metadata": [{"resampled_audio_filepath": "/p/1.wav"}]}
        )
        s.process_batch([task])
        assert observed
        m = observed[-1]
        assert m["entries_processed"] == 1.0
        assert m["model_batch_size"] == 1.0
        assert "process_time" in m
