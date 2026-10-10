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

import os
from itertools import combinations
from pathlib import Path
from typing import Any

import pytest
import torch

from nemo_curator.stages.audio.preprocessing.concatenation import SegmentConcatenationStage
from nemo_curator.tasks import AudioTask


def _make_segment_dict(duration_ms: int = 1000, sample_rate: int = 48000, segment_num: int = 0) -> dict:
    num_samples = int(sample_rate * duration_ms / 1000)
    return {
        "waveform": torch.randn(1, num_samples),
        "sample_rate": sample_rate,
        "original_file": "test.wav",
        "start_ms": 0,
        "end_ms": duration_ms,
        "segment_num": segment_num,
    }


def _make_nested_task(segments: list[dict]) -> AudioTask:
    return AudioTask(
        data={"segments": segments, "original_file": "test.wav"},
        dataset_name="ds",
    )


class TestSegmentConcatenationStage:
    @pytest.mark.parametrize("write_to_disk", [False, True])
    @pytest.mark.parametrize(
        ("left", "right"),
        list(
            combinations(
                [
                    "waveform_key",
                    "sample_rate_key",
                    "original_file_key",
                    "num_segments_key",
                    "total_duration_sec_key",
                    "audio_filepath_key",
                ],
                2,
            )
        ),
    )
    def test_every_active_output_pair_must_be_distinct(self, left: str, right: str, write_to_disk: bool) -> None:
        if not write_to_disk and "audio_filepath_key" in {left, right}:
            return
        with pytest.raises(ValueError, match="must be distinct"):
            SegmentConcatenationStage(
                **{left: "collision", right: "collision"}, write_to_disk=write_to_disk, output_dir="/unused"
            )

    @pytest.mark.parametrize(
        "params",
        [
            {"num_segments_key": "waveform"},
            {"total_duration_sec_key": "sample_rate"},
            {"original_file_key": "num_segments"},
            {"waveform_key": "sample_rate"},
            {"write_to_disk": True, "output_dir": "/unused", "audio_filepath_key": "waveform"},
            {"num_segments_key": ""},
            {"segments_key": ""},
            {
                "keep_waveform_in_task": False,
                "write_to_disk": True,
                "output_dir": "/unused",
                "waveform_key": "sample_rate",
            },
        ],
    )
    def test_rejects_destructive_output_keys(self, params: dict) -> None:
        with pytest.raises(ValueError, match=r"must be distinct|must be a non-empty"):
            SegmentConcatenationStage(**params)

    def test_inactive_output_key_can_alias_active_metadata(self) -> None:
        stage = SegmentConcatenationStage(audio_filepath_key="num_segments")
        result = stage.process(_make_nested_task([_make_segment_dict()]))
        assert result.data["num_segments"] == 1
        assert torch.is_tensor(result.data["waveform"])

    def test_process_batch_concatenates_segments(self) -> None:
        segments = [
            _make_segment_dict(duration_ms=2000, segment_num=0),
            _make_segment_dict(duration_ms=3000, segment_num=1),
        ]
        task = _make_nested_task(segments)

        stage = SegmentConcatenationStage(silence_duration_sec=1.0)
        result = stage.process(task)

        assert isinstance(result, AudioTask)
        out = result.data
        assert out["num_segments"] == 2
        expected_duration = (2000 + 1000 + 3000) / 1000.0
        assert abs(out["total_duration_sec"] - expected_duration) < 0.1

    def test_process_batch_empty_input(self) -> None:
        stage = SegmentConcatenationStage()
        result = stage.process_batch([])
        assert result == []

    def test_process_batch_single_segment(self) -> None:
        segments = [_make_segment_dict(duration_ms=5000)]
        task = _make_nested_task(segments)

        stage = SegmentConcatenationStage(silence_duration_sec=0.5)
        result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert result.data["num_segments"] == 1
        assert abs(result.data["total_duration_sec"] - 5.0) < 0.1

    def test_silence_duration_in_output(self) -> None:
        segments = [
            _make_segment_dict(duration_ms=1000, segment_num=0),
            _make_segment_dict(duration_ms=1000, segment_num=1),
        ]
        task = _make_nested_task(segments)

        stage = SegmentConcatenationStage(silence_duration_sec=2.0)
        result = stage.process(task)

        assert isinstance(result, AudioTask)
        combined = result.data["waveform"]
        sample_rate = result.data["sample_rate"]
        combined_duration_sec = combined.shape[-1] / sample_rate
        expected = 1.0 + 2.0 + 1.0
        assert abs(combined_duration_sec - expected) < 0.1

    def test_no_waveform_in_tasks(self) -> None:
        task = AudioTask(
            data={"segments": [{"other_key": "value"}]},
            dataset_name="ds",
        )
        stage = SegmentConcatenationStage()
        result = stage.process(task)
        assert result == []

    def test_missing_segments_key_raises(self) -> None:
        task = AudioTask(data={"other_key": "value"}, dataset_name="ds")
        stage = SegmentConcatenationStage()
        with pytest.raises(ValueError):  # noqa: PT011
            stage.process(task)

    def test_empty_segments_returns_empty(self) -> None:
        task = _make_nested_task([])
        stage = SegmentConcatenationStage()
        result = stage.process(task)
        assert result == []

    # --- output residency (write-to-disk extension) ---

    def test_default_output_is_in_memory_only(self) -> None:
        """Regression: default config keeps the in-memory waveform and writes no path."""
        result = SegmentConcatenationStage().process(_make_nested_task([_make_segment_dict(duration_ms=1000)]))
        assert "waveform" in result.data
        assert "audio_filepath" not in result.data

    def test_write_to_disk_persists_and_sets_path(self, tmp_path) -> None:  # noqa: ANN001
        out_dir = tmp_path / "concat"
        stage = SegmentConcatenationStage(write_to_disk=True, output_dir=str(out_dir))
        segments = [
            _make_segment_dict(duration_ms=1000, segment_num=0),
            _make_segment_dict(duration_ms=1000, segment_num=1),
        ]
        result = stage.process(_make_nested_task(segments))
        assert isinstance(result, AudioTask)
        # default keep_waveform_in_task=True -> both the waveform AND a written path
        assert "waveform" in result.data
        assert "audio_filepath" in result.data
        assert os.path.exists(result.data["audio_filepath"])

    def test_write_to_disk_only_drops_waveform(self, tmp_path) -> None:  # noqa: ANN001
        stage = SegmentConcatenationStage(
            write_to_disk=True, output_dir=str(tmp_path / "c"), keep_waveform_in_task=False
        )
        result = stage.process(_make_nested_task([_make_segment_dict(duration_ms=1000)]))
        assert "waveform" not in result.data
        assert os.path.exists(result.data["audio_filepath"])

    def test_requires_output_dir_when_write_to_disk(self) -> None:
        with pytest.raises(ValueError, match="output_dir"):
            SegmentConcatenationStage(write_to_disk=True)

    def test_requires_at_least_one_output_sink(self) -> None:
        with pytest.raises(ValueError, match="keep_waveform_in_task or write_to_disk"):
            SegmentConcatenationStage(keep_waveform_in_task=False)

    # --- positional compatibility (KW_ONLY sentinel) ---

    def test_legacy_positional_call_matches_legacy_order(self) -> None:
        """Pre-agent positionals were (silence_duration_sec, name, batch_size, resources)."""
        from nemo_curator.stages.resources import Resources

        stage = SegmentConcatenationStage(1.5)
        assert stage.silence_duration_sec == 1.5
        # Agent-added fields stayed keyword-only, so they keep their defaults.
        assert stage.segments_key == "segments"
        assert stage.waveform_key == "waveform"

        # name/batch_size/resources remain the legacy positional slots after silence_duration_sec.
        stage2 = SegmentConcatenationStage(1.5, "Concat", 4, Resources(cpus=1.0))
        assert (stage2.name, stage2.batch_size) == ("Concat", 4)

        # A 5th positional would be a keyword-only agent field -> TypeError.
        with pytest.raises(TypeError):
            SegmentConcatenationStage(1.5, "Concat", 4, Resources(cpus=1.0), "segs")

    # --- output key insertion order (legacy layout) ---

    def test_output_key_order_matches_legacy(self) -> None:
        segments = [
            _make_segment_dict(duration_ms=1000, segment_num=0),
            _make_segment_dict(duration_ms=1000, segment_num=1),
        ]
        result = SegmentConcatenationStage().process(_make_nested_task(segments))
        assert list(result.data.keys()) == [
            "waveform",
            "sample_rate",
            "original_file",
            "num_segments",
            "total_duration_sec",
        ]

    def test_output_key_order_places_disk_path_last(self, tmp_path) -> None:  # noqa: ANN001
        stage = SegmentConcatenationStage(write_to_disk=True, output_dir=str(tmp_path / "c"))
        result = stage.process(_make_nested_task([_make_segment_dict(duration_ms=1000)]))
        assert list(result.data.keys()) == [
            "waveform",
            "sample_rate",
            "original_file",
            "num_segments",
            "total_duration_sec",
            "audio_filepath",
        ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_segments_key": "waveform"},
        {"total_duration_sec_key": "sample_rate"},
        {"original_file_key": "num_segments"},
        {"waveform_key": "sample_rate"},
        {"write_to_disk": True, "audio_filepath_key": "waveform"},
    ],
)
def test_output_aliases_are_rejected_before_writing(tmp_path: Path, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="keys must be distinct"):
        SegmentConcatenationStage(output_dir=str(tmp_path / "output"), **kwargs)
    assert not (tmp_path / "output").exists()


def test_cross_scope_segment_container_alias_remains_supported() -> None:
    stage = SegmentConcatenationStage(segments_key="waveform")
    result = stage.process(AudioTask(data={"waveform": [_make_segment_dict()]}))
    assert torch.is_tensor(result.data["waveform"])


def test_concatenation_cardinality_matches_independent_parent_batch() -> None:
    stage = SegmentConcatenationStage(silence_duration_sec=0)
    parents = []
    for name in ("one.wav", "two.wav"):
        segment = _make_segment_dict(duration_ms=10)
        segment["original_file"] = name
        parents.append(_make_nested_task([segment]))
    parents.append(_make_nested_task([]))
    outputs = stage.process_batch(parents)
    assert stage.describe().cardinality == "filter"
    assert len(outputs) == 2
    assert [output.data["original_file"] for output in outputs] == ["one.wav", "two.wav"]


def test_parent_filter_contract_passes_real_conformance() -> None:
    from nemo_curator.stages.audio._agent._conformance import assert_agent_ready

    assert_agent_ready(
        SegmentConcatenationStage(),
        fixture_factory=lambda: _make_nested_task([_make_segment_dict(duration_ms=10)]),
        expected_cardinality="filter",
    )
