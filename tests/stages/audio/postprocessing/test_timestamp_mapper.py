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

from typing import ClassVar

import torch

from nemo_curator.stages.audio.postprocessing.timestamp_mapper import (
    _NEVER_PASS_KEYS,
    TimestampMapperStage,
    _translate_to_original,
)
from nemo_curator.tasks import AudioTask


def _make_task(data: dict, metadata: dict | None = None) -> AudioTask:
    t = AudioTask(data=data, dataset_name="test_ds")
    if metadata:
        t._metadata = metadata
    return t


class TestTranslateToOriginal:
    """Unit tests for the pure _translate_to_original() function."""

    MAPPINGS: ClassVar[list[dict]] = [
        {
            "concat_start_ms": 0,
            "concat_end_ms": 2000,
            "original_file": "a.wav",
            "original_start_ms": 5000,
            "original_end_ms": 7000,
        },
        {
            "concat_start_ms": 2000,
            "concat_end_ms": 5000,
            "original_file": "b.wav",
            "original_start_ms": 0,
            "original_end_ms": 3000,
        },
        {
            "concat_start_ms": 5000,
            "concat_end_ms": 8000,
            "original_file": "c.wav",
            "original_start_ms": 10000,
            "original_end_ms": 13000,
        },
    ]

    def test_single_mapping_exact_match(self) -> None:
        """Segment exactly matches one mapping."""
        results = _translate_to_original(self.MAPPINGS, 0, 2000)
        assert len(results) == 1
        assert results[0]["original_file"] == "a.wav"
        assert results[0]["original_start_ms"] == 5000
        assert results[0]["original_end_ms"] == 7000
        assert results[0]["duration_ms"] == 2000

    def test_single_mapping_partial_overlap(self) -> None:
        """Segment partially overlaps one mapping."""
        results = _translate_to_original(self.MAPPINGS, 500, 1500)
        assert len(results) == 1
        assert results[0]["original_file"] == "a.wav"
        assert results[0]["original_start_ms"] == 5500
        assert results[0]["original_end_ms"] == 6500
        assert results[0]["duration_ms"] == 1000

    def test_cross_boundary_span(self) -> None:
        """Segment spans two mappings — returns both."""
        results = _translate_to_original(self.MAPPINGS, 1500, 3000)
        assert len(results) == 2
        assert results[0]["original_file"] == "a.wav"
        assert results[0]["original_start_ms"] == 6500
        assert results[0]["original_end_ms"] == 7000
        assert results[0]["duration_ms"] == 500
        assert results[1]["original_file"] == "b.wav"
        assert results[1]["original_start_ms"] == 0
        assert results[1]["original_end_ms"] == 1000
        assert results[1]["duration_ms"] == 1000

    def test_silence_gap_no_overlap(self) -> None:
        """Segment falls entirely in a gap between mappings."""
        mappings = [
            {
                "concat_start_ms": 0,
                "concat_end_ms": 1000,
                "original_file": "a.wav",
                "original_start_ms": 0,
                "original_end_ms": 1000,
            },
            {
                "concat_start_ms": 3000,
                "concat_end_ms": 5000,
                "original_file": "b.wav",
                "original_start_ms": 0,
                "original_end_ms": 2000,
            },
        ]
        results = _translate_to_original(mappings, 1000, 3000)
        assert len(results) == 0

    def test_malformed_mapping_missing_key(self) -> None:
        """Malformed mapping (missing key) is skipped gracefully."""
        mappings = [
            {"concat_start_ms": 0, "concat_end_ms": 2000},
            {
                "concat_start_ms": 2000,
                "concat_end_ms": 4000,
                "original_file": "b.wav",
                "original_start_ms": 0,
                "original_end_ms": 2000,
            },
        ]
        results = _translate_to_original(mappings, 0, 4000)
        assert len(results) == 1
        assert results[0]["original_file"] == "b.wav"

    def test_empty_mappings(self) -> None:
        """Empty mappings list returns empty results."""
        results = _translate_to_original([], 0, 1000)
        assert results == []

    def test_no_overlap_before_all_mappings(self) -> None:
        """Segment ends before any mapping starts."""
        mappings = [
            {
                "concat_start_ms": 5000,
                "concat_end_ms": 8000,
                "original_file": "a.wav",
                "original_start_ms": 0,
                "original_end_ms": 3000,
            },
        ]
        results = _translate_to_original(mappings, 0, 1000)
        assert results == []


def test_combo4_with_segment_mappings() -> None:
    """Full pipeline: remaps concat-space timestamps to original file positions."""
    mappings = [
        {
            "concat_start_ms": 0,
            "concat_end_ms": 2000,
            "original_file": "test.wav",
            "original_start_ms": 5000,
            "original_end_ms": 7000,
        },
    ]
    task = _make_task(
        {"waveform": torch.randn(1, 48000), "sample_rate": 48000, "start_ms": 100, "end_ms": 1500, "utmos_mos": 4.2},
        metadata={"segment_mappings": mappings},
    )
    stage = TimestampMapperStage()
    result = stage.process(task)

    assert result.data["original_file"] == "test.wav"
    assert result.data["original_start_ms"] == 5100
    assert result.data["original_end_ms"] == 6500
    assert result.data["duration_ms"] == 1400
    assert result.data["utmos_mos"] == 4.2
    assert result.data["sample_rate"] == 48000
    assert "waveform" not in result.data
    assert "start_ms" not in result.data


def test_combo2_vad_fanout_start_end() -> None:
    """VAD fan-out: uses start_ms/end_ms directly."""
    task = _make_task(
        {
            "waveform": torch.randn(1, 48000),
            "sample_rate": 48000,
            "start_ms": 5200,
            "end_ms": 15400,
            "segment_num": 0,
            "duration": 10.2,
            "original_file": "/a.wav",
            "utmos_mos": 4.2,
        }
    )
    stage = TimestampMapperStage()
    result = stage.process(task)

    assert result.data["original_file"] == "/a.wav"
    assert result.data["original_start_ms"] == 5200
    assert result.data["original_end_ms"] == 15400
    assert result.data["duration_ms"] == 10200
    assert abs(result.data["duration"] - 10.2) < 0.01
    assert result.data["utmos_mos"] == 4.2
    assert "waveform" not in result.data
    assert "start_ms" not in result.data
    assert "segment_num" not in result.data


def test_combo3_diar_segments() -> None:
    """Speaker-only: computes span from diar_segments."""
    task = _make_task(
        {
            "waveform": torch.randn(1, 48000),
            "sample_rate": 48000,
            "speaker_id": "speaker_0",
            "num_speakers": 3,
            "duration": 42.6,
            "diar_segments": [(5.2, 15.4), (30.1, 42.8), (100.0, 120.5)],
            "audio_filepath": "/a.wav",
            "sigmos_noise": 4.5,
        }
    )
    stage = TimestampMapperStage()
    result = stage.process(task)

    assert result.data["original_file"] == "/a.wav"
    assert result.data["original_start_ms"] == 5200
    assert result.data["original_end_ms"] == 120500
    assert result.data["duration_ms"] == 115300
    assert abs(result.data["speaking_duration"] - 43.4) < 0.01
    assert len(result.data["diar_segments"]) == 3
    assert result.data["speaker_id"] == "speaker_0"
    assert result.data["num_speakers"] == 3
    assert result.data["sigmos_noise"] == 4.5
    assert "waveform" not in result.data


def test_combo1_duration_fallback() -> None:
    """Filters-only: uses duration from MonoConversion."""
    task = _make_task(
        {
            "audio_filepath": "/a.wav",
            "waveform": torch.randn(1, 48000),
            "sample_rate": 48000,
            "duration": 10.5,
            "is_mono": True,
            "num_samples": 504000,
            "sigmos_ovrl": 3.5,
        }
    )
    stage = TimestampMapperStage()
    result = stage.process(task)

    assert result.data["original_file"] == "/a.wav"
    assert result.data["original_start_ms"] == 0
    assert result.data["original_end_ms"] == 10500
    assert result.data["duration"] == 10.5
    assert result.data["sigmos_ovrl"] == 3.5
    assert result.data["sample_rate"] == 48000
    assert "waveform" not in result.data
    assert "is_mono" not in result.data
    assert "num_samples" not in result.data


def test_never_pass_keys_blocked() -> None:
    """Non-serializable keys are blocked even if in passthrough_keys."""
    task = _make_task(
        {
            "audio_filepath": "/a.wav",
            "waveform": torch.randn(1, 48000),
            "segments": [{"waveform": torch.randn(1, 100)}],
            "duration": 1.0,
            "sigmos_ovrl": 3.0,
        }
    )
    stage = TimestampMapperStage(passthrough_keys=["waveform", "segments", "sigmos_ovrl"])
    result = stage.process(task)

    for key in _NEVER_PASS_KEYS:
        assert key not in result.data, f"{key!r} must never pass through"
    assert result.data["sigmos_ovrl"] == 3.0


def test_default_passthrough_covers_all_filters() -> None:
    """Default passthrough_keys includes all built-in filter scores."""
    task = _make_task(
        {
            "audio_filepath": "/a.wav",
            "duration": 1.0,
            "utmos_mos": 4.2,
            "sigmos_noise": 4.0,
            "sigmos_ovrl": 3.5,
            "sigmos_sig": 3.8,
            "sigmos_col": 4.0,
            "sigmos_disc": 4.2,
            "sigmos_loud": 3.7,
            "sigmos_reverb": 4.9,
            "band_prediction": "full_band",
            "sample_rate": 48000,
        }
    )
    stage = TimestampMapperStage()
    result = stage.process(task)

    assert result.data["utmos_mos"] == 4.2
    assert result.data["sigmos_noise"] == 4.0
    assert result.data["sigmos_ovrl"] == 3.5
    assert result.data["band_prediction"] == "full_band"
    assert result.data["sample_rate"] == 48000


def test_custom_passthrough_keys() -> None:
    """User can restrict output to only specific keys."""
    task = _make_task(
        {
            "audio_filepath": "/a.wav",
            "duration": 1.0,
            "sigmos_ovrl": 3.0,
            "sigmos_noise": 4.0,
            "utmos_mos": 4.2,
            "book_id": "123",
        }
    )
    stage = TimestampMapperStage(passthrough_keys=["sigmos_ovrl", "book_id"])
    result = stage.process(task)

    assert result.data["sigmos_ovrl"] == 3.0
    assert result.data["book_id"] == "123"
    assert "sigmos_noise" not in result.data
    assert "utmos_mos" not in result.data


def test_dataset_metadata_not_in_default_output() -> None:
    """Dataset-specific keys (text, book_id) are excluded by default passthrough."""
    task = _make_task(
        {
            "audio_filepath": "/a.wav",
            "duration": 1.0,
            "text": "hello world",
            "book_id": "123",
            "reader_id": "456",
            "sigmos_ovrl": 3.0,
        }
    )
    stage = TimestampMapperStage()
    result = stage.process(task)

    assert result.data["sigmos_ovrl"] == 3.0
    assert "text" not in result.data
    assert "book_id" not in result.data
    assert "reader_id" not in result.data


class TestAcceptsEitherSegmentShape:
    """Diarizers emit ``{start, end, speaker}`` dicts; VAD emits ``[start, end]`` pairs.

    Reading only pairs raised ``KeyError: 0`` on real diarizer output, so a
    diarize -> map pipeline died on the shape its own upstream stage is documented to
    produce (``InferenceSortformerStage.diarize`` is typed ``list[list[dict[str, Any]]]``).
    """

    DICTS: ClassVar[list[dict]] = [
        {"start": 0.0, "end": 1.5, "speaker": "speaker_0"},
        {"start": 1.5, "end": 3.0, "speaker": "speaker_1"},
    ]
    PAIRS: ClassVar[list[list[float]]] = [[0.0, 1.5], [1.5, 3.0]]

    def _run(self, segments: list) -> dict:
        stage = TimestampMapperStage()
        stage.setup()
        out = stage.process(_make_task({"audio_filepath": "clip.wav", "diar_segments": segments}))
        rows = out if isinstance(out, list) else [out]
        assert len(rows) == 1
        return rows[0].data

    def test_dict_segments_from_a_diarizer_do_not_crash(self) -> None:
        data = self._run(self.DICTS)
        assert data["original_start_ms"] == 0
        assert data["original_end_ms"] == 3000
        assert data["speaking_duration"] == 3.0

    def test_pair_segments_still_work_unchanged(self) -> None:
        data = self._run(self.PAIRS)
        assert data["original_start_ms"] == 0
        assert data["original_end_ms"] == 3000
        assert data["diar_segments"] == [[0.0, 1.5], [1.5, 3.0]]

    def test_speaker_labels_survive_the_round_trip(self) -> None:
        # Rewriting a diarizer's segments as bare pairs would discard the speaker -- the one
        # thing a diarization pipeline is run to learn.
        data = self._run(self.DICTS)
        assert [s["speaker"] for s in data["diar_segments"]] == ["speaker_0", "speaker_1"]

    def test_overlapping_speech_spans_to_the_latest_end(self) -> None:
        # Diarized speech overlaps, so the segment that STARTS last need not FINISH last.
        data = self._run(
            [
                {"start": 0.0, "end": 9.0, "speaker": "a"},
                {"start": 1.0, "end": 2.0, "speaker": "b"},
            ]
        )
        assert data["original_end_ms"] == 9000

    def test_one_malformed_segment_is_skipped_not_fatal(self) -> None:
        data = self._run([{"start": 0.0, "end": 1.5, "speaker": "a"}, {"no": "bounds"}, "garbage"])
        assert data["original_end_ms"] == 1500

    def test_the_contract_admits_that_it_sanitizes(self) -> None:
        # It builds output from an allowlist and hard-blocks _NEVER_PASS_KEYS, so no waveform
        # can leave it. Declaring otherwise made the validator report tensor_into_sink against
        # a JSON sink placed after this stage -- refusing a pipeline that was already safe.
        assert TimestampMapperStage().describe().gates.sanitizes_output is True

    def test_no_waveform_key_can_escape(self) -> None:
        stage = TimestampMapperStage()
        stage.setup()
        data = {"audio_filepath": "clip.wav", "diar_segments": self.PAIRS}
        for k in _NEVER_PASS_KEYS:
            data[k] = torch.zeros(4) if k != "segments" else [[0.0, 1.0]]
        out = stage.process(_make_task(data))
        rows = out if isinstance(out, list) else [out]
        assert not (set(rows[0].data) & set(_NEVER_PASS_KEYS) - {"segments"})


# Lifted from tests/stages/audio/test_agent_simulation_pipelines.py: it drives only
# TimestampMapperStage, so it belongs with the rest of that stage's regressions.
def test_timestamp_mapper_multispeaker_maps_distinct_windows() -> None:
    """Regression: SpeakerSep->VAD per-speaker segments map to DISTINCT original windows.

    After SpeakerSeparation the separator emits full-length stems (each speaker's
    audio overlaid on a silent track at its concat-time position), so VAD_Speaker's
    start_ms/end_ms are concat-time and must be mapped directly through the
    concat->original mappings. A removed guard used to discard start_ms/end_ms
    whenever diar_segments were also present and span the diar UNION instead.

    The fixture is DISCRIMINATING: each speaker's VAD start_ms/end_ms is a strict
    sub-interval of its diar segment, so the two branches produce different output.
    Direct mapping (the fix) yields the narrow refined window; the removed
    diar-union guard would yield the wider whole-segment window. Asserting the
    narrow windows therefore fails on the guarded code and pins the fix.
    """
    # Two concat segments; original coords differ from concat so translation is visible.
    mappings = [
        {"original_file": "/audio.wav", "original_start_ms": 0, "concat_start_ms": 0, "concat_end_ms": 400},
        {"original_file": "/audio.wav", "original_start_ms": 1000, "concat_start_ms": 400, "concat_end_ms": 800},
    ]
    mapper = TimestampMapperStage(passthrough_keys=["speaker_id"])
    spans = {}
    for speaker_id, (start_ms, end_ms), diar in [
        # VAD window (100,300) sits inside diar seg [0.0,0.4]=concat[0,400] -> fix maps to original (100,300)
        ("speaker_0", (100, 300), [[0.0, 0.4]]),
        # VAD window (500,700) sits inside diar seg [0.4,0.8]=concat[400,800] -> fix maps to original (1100,1300)
        ("speaker_1", (500, 700), [[0.4, 0.8]]),
    ]:
        task = AudioTask(
            dataset_name="multispeaker",
            data={
                "start_ms": start_ms,
                "end_ms": end_ms,
                "diar_segments": diar,  # present alongside start/end — must NOT trigger union collapse
                "speaker_id": speaker_id,
                "original_file": "/audio.wav",
            },
            _metadata={"segment_mappings": mappings},
        )
        out = mapper.process(task)
        assert isinstance(out, AudioTask)  # not dropped
        spans[speaker_id] = (out.data["original_start_ms"], out.data["original_end_ms"])
    # narrow refined windows; the removed guard would give (0,400) and (1000,1400)
    assert spans == {"speaker_0": (100, 300), "speaker_1": (1100, 1300)}
