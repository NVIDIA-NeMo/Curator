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

"""Channel policy and rate policy as two independent stages.

Each does one job and leaves the other alone, so a pipeline sets a channel policy and a
rate policy separately -- or uses only the one it needs. Each also says which job it is
doing: recording a value, selecting on it, or changing it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from nemo_curator.stages.audio.preprocessing import ChannelCountStage, SampleRateFilterStage
from nemo_curator.stages.audio.preprocessing import channel_count as cc
from nemo_curator.stages.audio.preprocessing import sample_rate_filter as srf
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from pathlib import Path


def _wav(tmp_path: Path, channels: int = 1, rate: int = 16000, name: str = "a.wav") -> str:
    path = tmp_path / name
    data = np.zeros(rate, dtype="float32") if channels == 1 else np.zeros((rate, channels), dtype="float32")
    sf.write(str(path), data, rate)
    return str(path)


def _task(path: str) -> AudioTask:
    return AudioTask(task_id="t", dataset_name="d", data={"audio_filepath": path})


def _convert(**kwargs: object) -> ChannelCountStage:
    return ChannelCountStage(action="convert", **kwargs)


class TestConvertingTheChannelCount:
    @pytest.mark.parametrize(
        ("source", "target"),
        [(1, 1), (2, 1), (6, 1), (2, 2), (6, 6)],
    )
    def test_downmix_and_passthrough_produce_the_requested_count(
        self, tmp_path: Path, source: int, target: int
    ) -> None:
        result = _convert(target_channels=target).process(
            _task(_wav(tmp_path, channels=source, name=f"{source}to{target}.wav"))
        )
        assert result != []
        assert result.data["num_channels"] == target

    def test_mono_downmix_averages_rather_than_selecting_a_channel(self, tmp_path: Path) -> None:
        """Averaging is what mono means; taking channel 0 would discard half the signal."""
        path = str(tmp_path / "stereo.wav")
        left = np.ones(16000, dtype="float32")
        right = -np.ones(16000, dtype="float32")
        sf.write(path, np.stack([left, right], axis=1), 16000)

        result = _convert(target_channels=1).process(_task(path))

        assert result != []
        # +1 and -1 average to ~0; selecting either channel would give ~1. The residual is
        # 16-bit quantization (sf.write stores PCM_16, so +/-1.0 becomes 32767/-32768),
        # which is ~3e-5 -- three orders of magnitude below what a channel-select yields.
        assert float(result.data["waveform"].abs().max()) < 0.01

    def test_mono_upmixes_by_duplication(self, tmp_path: Path) -> None:
        """Matches ``ffmpeg -ac``. It adds no information -- the channels are identical."""
        result = _convert(target_channels=2).process(_task(_wav(tmp_path, channels=1)))

        assert result != []
        waveform = result.data["waveform"]
        assert waveform.shape[0] == 2
        assert waveform[0].equal(waveform[1]), "upmix duplicates, so both channels are the same signal"

    def test_an_unset_target_means_mono(self, tmp_path: Path) -> None:
        """``target_channels`` defaults to None so the constructor can tell "make it mono" from
        "channels were never mentioned" and refuse the first under the wrong action. Mono is
        still what an unset target converts to."""
        result = _convert().process(_task(_wav(tmp_path, channels=2)))

        assert result != []
        assert result.data["num_channels"] == 1

    @pytest.mark.parametrize(("source", "target"), [(6, 2), (4, 3), (6, 5)])
    def test_a_surround_downmix_is_refused_rather_than_approximated(
        self, tmp_path: Path, source: int, target: int
    ) -> None:
        """Correct downmix to >1 channel needs BS.775 coefficients AND the file's channel
        order, and a (channels, samples) tensor carries neither -- the WAV channel mask is
        gone by then. Averaging 5.1 into two channels sounds plausible and is wrong, so the
        row is dropped and ResampleAudioStage (ffmpeg, layout-aware) is named instead."""
        result = _convert(target_channels=target).process(
            _task(_wav(tmp_path, channels=source, name=f"{source}to{target}.wav"))
        )
        assert result == []

    def test_the_sample_rate_is_never_changed(self, tmp_path: Path) -> None:
        """This stage converts channels only; rate policy belongs to another stage."""
        result = _convert(target_channels=1).process(_task(_wav(tmp_path, channels=2, rate=44100)))
        assert result != []
        assert result.data["sample_rate"] == 44100

    def test_a_nonsensical_target_is_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="target_channels"):
            _convert(target_channels=0)

    @pytest.mark.parametrize("bad", [2.0, "2"])
    def test_a_non_integer_channel_count_is_rejected_at_construction(self, bad: object) -> None:
        """YAML reads ``target_channels: 2.0`` as a float. It used to construct fine and then
        die inside a worker at ``waveform.repeat(2.0, 1)`` with a TypeError, which is not one
        of the errors this stage drops rows for -- so it propagated and took the run down
        partway through the corpus rather than being caught at the recipe."""
        with pytest.raises(ValueError, match="whole number of channels"):
            _convert(target_channels=bad)


class TestRecordingTheChannelCount:
    """``action="annotate"``: measure and keep. The default, because a neutral name should not
    rewrite audio until asked."""

    @pytest.mark.parametrize("channels", [1, 2, 6])
    def test_every_row_is_kept_and_stamped_with_what_it_actually_has(self, tmp_path: Path, channels: int) -> None:
        result = ChannelCountStage().process(_task(_wav(tmp_path, channels=channels, name=f"{channels}ch.wav")))

        assert result != []
        assert result.data["num_channels"] == channels

    def test_nothing_is_rewritten(self, tmp_path: Path) -> None:
        """Recording a count is not a conversion: no waveform is produced and no file written."""
        result = ChannelCountStage().process(_task(_wav(tmp_path, channels=2)))

        assert result != []
        assert "waveform" not in result.data
        assert "converted_audio_filepath" not in result.data

    def test_it_reads_the_header_and_never_decodes(self, tmp_path: Path) -> None:
        """The count sits in the first bytes of the file, so putting this in front of a
        decoding stage costs almost nothing. A decode here would forfeit exactly that."""

        def explode(*_args: object, **_kwargs: object) -> None:
            msg = "decoded the audio to read a header value"
            raise AssertionError(msg)

        with patch.object(cc.sf, "read", explode), patch.object(cc, "load_audio_file", explode):
            result = ChannelCountStage().process(_task(_wav(tmp_path, channels=2)))

        assert result != []
        assert result.data["num_channels"] == 2

    def test_resident_audio_answers_before_the_file_does(self, tmp_path: Path) -> None:
        """After a conversion the file on disk still has its original channels while the
        waveform in the task has the converted ones. Reading the header there would report on
        audio nobody is using any more, so a resident waveform wins."""
        task = _task(_wav(tmp_path, channels=6))
        task.data["waveform"] = np.zeros((1, 16000), dtype="float32")

        result = ChannelCountStage().process(task)

        assert result != []
        assert result.data["num_channels"] == 1, "the resident mono waveform, not the 6-channel file"

    def test_a_stale_manifest_column_is_corrected_rather_than_believed(self, tmp_path: Path) -> None:
        """``num_channels`` standing alone is metadata about a file nobody re-opened. Trusting
        it would let a wrong column decide a filter AND be re-stamped as if measured."""
        task = _task(_wav(tmp_path, channels=2))
        task.data["num_channels"] = 1

        result = ChannelCountStage().process(task)

        assert result != []
        assert result.data["num_channels"] == 2

    def test_an_unreadable_row_is_dropped_not_crashed(self, tmp_path: Path) -> None:
        assert ChannelCountStage().process(_task(str(tmp_path / "missing.wav"))) == []

    def test_an_unverifiable_count_is_used_rather_than_dropping_the_row(self) -> None:
        """No resident audio and no path leaves nothing to check against. Using the declared
        count beats discarding a row that may well be fine."""
        task = AudioTask(task_id="t", dataset_name="d", data={"num_channels": 2})

        result = ChannelCountStage().process(task)

        assert result != []
        assert result.data["num_channels"] == 2

    def test_waveform_residency_never_falls_back_to_disk(self, tmp_path: Path) -> None:
        """``input_residency="waveform"`` is a promise not to touch the filesystem. Silently
        reading the header instead would break it for a caller who set it to avoid exactly
        that."""
        assert ChannelCountStage(input_residency="waveform").process(_task(_wav(tmp_path, channels=2))) == []


class TestSelectingByChannelCount:
    """``action="filter"``: keep the rows that already comply, rewrite nothing."""

    @pytest.mark.parametrize(
        ("kwargs", "keeps"),
        [
            ({"allowed_channels": [1]}, False),
            ({"allowed_channels": [2]}, True),
            ({"allowed_channels": [1, 2]}, True),
            ({"min_channels": 2}, True),
            ({"min_channels": 3}, False),
            ({"max_channels": 2}, True),
            ({"max_channels": 1}, False),
            ({"min_channels": 1, "max_channels": 6}, True),
            ({"allowed_channels": [2], "min_channels": 3}, False),
        ],
    )
    def test_a_list_and_a_range_are_separate_constraints(
        self, tmp_path: Path, kwargs: dict[str, object], keeps: bool
    ) -> None:
        """Separate parameters for the same reason the rate side has them: ``[1, 6]`` as a
        single knob is ambiguous between "these two counts" and "this range". Every constraint
        that IS set must be satisfied."""
        result = ChannelCountStage(action="filter", **kwargs).process(_task(_wav(tmp_path, channels=2)))

        assert (result != []) is keeps

    def test_the_count_is_recorded_on_rows_that_pass(self, tmp_path: Path) -> None:
        result = ChannelCountStage(action="filter", allowed_channels=[2]).process(_task(_wav(tmp_path, channels=2)))

        assert result != []
        assert result.data["num_channels"] == 2

    def test_selection_does_not_convert_the_rows_it_keeps(self, tmp_path: Path) -> None:
        """A kept row is untouched: this is the drop-only path, so no audio is rewritten."""
        result = ChannelCountStage(action="filter", min_channels=1).process(_task(_wav(tmp_path, channels=2)))

        assert result != []
        assert "waveform" not in result.data

    def test_a_constraint_free_filter_is_rejected_at_construction(self) -> None:
        """It would declare a filter that drops nothing, and the stage already has an action
        for that."""
        with pytest.raises(ValueError, match="action='annotate'"):
            ChannelCountStage(action="filter")

    def test_an_empty_allow_list_is_rejected_at_construction(self) -> None:
        """``[]`` would silently discard the entire corpus; None means "no constraint"."""
        with pytest.raises(ValueError, match="at least one count"):
            ChannelCountStage(action="filter", allowed_channels=[])

    def test_an_inverted_range_is_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="nothing can pass"):
            ChannelCountStage(action="filter", min_channels=6, max_channels=2)

    @pytest.mark.parametrize("bad", [[0], [1.5], [True], 0])
    def test_a_nonsensical_count_is_rejected_at_construction(self, bad: object) -> None:
        key = "min_channels" if isinstance(bad, int) else "allowed_channels"
        with pytest.raises(ValueError, match="whole channel counts"):
            ChannelCountStage(action="filter", **{key: bad})


class TestSelectingAndConvertingAreNotTheSameKnob:
    """The footgun this stage exists to remove: ``target_channels=1`` must never quietly mean
    "drop everything that is not mono", and ``allowed_channels=[1]`` must never quietly mean
    "make it mono". They do opposite things to a corpus -- one keeps every row and changes it,
    the other changes no row and keeps a subset -- so a parameter naming the action you did
    not choose is refused instead of ignored.
    """

    @pytest.mark.parametrize(
        ("kwargs", "unusable"),
        [
            ({"action": "convert", "allowed_channels": [1]}, "allowed_channels"),
            ({"action": "convert", "min_channels": 1}, "min_channels"),
            ({"action": "filter", "allowed_channels": [1], "target_channels": 1}, "target_channels"),
            ({"action": "filter", "allowed_channels": [1], "write_to_disk": True}, "write_to_disk"),
            ({"action": "annotate", "allowed_channels": [1]}, "allowed_channels"),
            ({"action": "annotate", "target_channels": 1}, "target_channels"),
        ],
    )
    def test_the_other_actions_parameters_are_refused(self, kwargs: dict[str, object], unusable: str) -> None:
        with pytest.raises(ValueError, match=unusable):
            ChannelCountStage(**kwargs)

    def test_the_refusal_names_the_action_that_would_use_it(self) -> None:
        """A message that only says "unused" leaves the caller to guess which half of their
        intent was dropped."""
        with pytest.raises(ValueError, match="action='filter'"):
            ChannelCountStage(action="convert", allowed_channels=[1])

    def test_an_unknown_action_is_rejected_at_construction(self) -> None:
        """A recipe is free text before it is a stage; a typo must not fall through to a
        default behaviour the caller did not name."""
        with pytest.raises(ValueError, match="action must be one of"):
            ChannelCountStage(action="downmix")

    def test_converting_keeps_the_rows_selection_would_have_dropped(self, tmp_path: Path) -> None:
        """Same corpus, same "mono" intent, opposite outcomes -- which is why they cannot share
        a parameter."""
        path = _wav(tmp_path, channels=2)

        converted = _convert(target_channels=1).process(_task(path))
        selected = ChannelCountStage(action="filter", allowed_channels=[1]).process(_task(path))

        assert converted != []
        assert converted.data["num_channels"] == 1
        assert selected == []


class TestSampleRateFilter:
    @pytest.mark.parametrize(
        ("kwargs", "keeps"),
        [
            ({"allowed_sample_rates": [16000]}, True),
            ({"allowed_sample_rates": [22050, 44100]}, False),
            ({"min_sample_rate": 16000}, True),
            ({"min_sample_rate": 22050}, False),
            ({"max_sample_rate": 16000}, True),
            ({"max_sample_rate": 8000}, False),
            ({"min_sample_rate": 8000, "max_sample_rate": 48000}, True),
            ({"allowed_sample_rates": [16000], "min_sample_rate": 22050}, False),
            ({}, True),
        ],
    )
    def test_a_list_and_a_range_are_separate_constraints(
        self, tmp_path: Path, kwargs: dict[str, object], keeps: bool
    ) -> None:
        """Separate parameters on purpose: ``[16000, 48000]`` as a single knob is ambiguous
        between "these two rates" and "this range", and the readings filter very different
        corpora. Every constraint that IS set must be satisfied."""
        result = SampleRateFilterStage(**kwargs).process(_task(_wav(tmp_path, rate=16000)))
        assert (result != []) is keeps

    def test_the_rate_is_recorded_on_rows_that_pass(self, tmp_path: Path) -> None:
        result = SampleRateFilterStage().process(_task(_wav(tmp_path, rate=44100)))
        assert result != []
        assert result.data["sample_rate"] == 44100

    def test_it_reads_the_header_and_never_decodes(self, tmp_path: Path) -> None:
        """The rate is metadata sitting in the first bytes. Decoding to read it costs ~186x
        more per file, and placing this stage before any decoding is what keeps rejected
        rows from ever being decoded -- a decode here would forfeit exactly that."""
        stage = SampleRateFilterStage(allowed_sample_rates=[16000])
        path = _wav(tmp_path, rate=16000)

        def explode(*_args: object, **_kwargs: object) -> None:
            msg = "decoded the audio to read a header value"
            raise AssertionError(msg)

        with patch.object(srf.sf, "read", explode):
            result = stage.process(_task(path))

        assert result != []
        assert result.data["sample_rate"] == 16000

    def test_a_resident_rate_avoids_touching_disk_entirely(self) -> None:
        """A rate carried alongside resident audio describes audio this pipeline is holding,
        so it is reused and the file is never opened."""
        stage = SampleRateFilterStage(allowed_sample_rates=[16000])
        task = AudioTask(
            task_id="t",
            dataset_name="d",
            data={
                "audio_filepath": "/nonexistent/never-opened.wav",
                "sample_rate": 16000,
                "waveform": object(),
            },
        )

        result = stage.process(task)

        assert result != []
        assert result.data["sample_rate"] == 16000

    def test_a_manifest_rate_with_no_resident_audio_is_verified_against_the_file(self, tmp_path: Path) -> None:
        """``sample_rate`` is a standard manifest column, and a stale one used to decide the
        filter outright: a genuinely 48 kHz file labelled 16000 was KEPT for a 16 kHz-only
        corpus and then re-stamped with the wrong rate, so the model downstream silently got
        pitch-shifted audio. With nothing resident to back the number, the header wins."""
        path = _wav(tmp_path, rate=48000, name="mislabelled.wav")
        task = AudioTask(
            task_id="t",
            dataset_name="d",
            data={"audio_filepath": path, "sample_rate": 16000},
        )

        assert SampleRateFilterStage(allowed_sample_rates=[16000]).process(task) == []

        task = AudioTask(
            task_id="t",
            dataset_name="d",
            data={"audio_filepath": path, "sample_rate": 16000},
        )
        kept = SampleRateFilterStage(allowed_sample_rates=[48000]).process(task)
        assert kept != []
        assert kept.data["sample_rate"] == 48000, "the recorded rate is the measured one"

    def test_an_unverifiable_rate_is_used_rather_than_dropping_the_row(self) -> None:
        """No resident audio and no path leaves nothing to check against. Filtering on the
        declared rate beats discarding a row that may well be fine."""
        task = AudioTask(task_id="t", dataset_name="d", data={"sample_rate": 16000})
        result = SampleRateFilterStage(allowed_sample_rates=[16000]).process(task)
        assert result != []
        assert result.data["sample_rate"] == 16000

    def test_an_unreadable_row_is_dropped_not_crashed(self, tmp_path: Path) -> None:
        result = SampleRateFilterStage().process(_task(str(tmp_path / "missing.wav")))
        assert result == []

    def test_a_home_relative_path_is_read_rather_than_dropped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A manifest written with ``~/audio/x.wav`` must not vanish through this stage.

        ``sf.info`` does not expand ``~``, and the read failure above drops the row rather than
        raising -- so before ``expanduser`` was applied a corpus addressed that way was silently
        emptied here, while ChannelCountStage beside it read the same paths fine.
        """
        monkeypatch.setenv("HOME", str(tmp_path))
        _wav(tmp_path, rate=16000, name="home.wav")
        out = SampleRateFilterStage(allowed_sample_rates=[16000]).process(_task("~/home.wav"))
        assert out != [], "row dropped: the '~' path was never expanded"
        assert out.data["sample_rate"] == 16000

    def test_an_empty_allow_list_is_rejected_at_construction(self) -> None:
        """``[]`` would silently discard the entire corpus; None means "no constraint"."""
        with pytest.raises(ValueError, match="at least one rate"):
            SampleRateFilterStage(allowed_sample_rates=[])

    def test_an_inverted_range_is_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="nothing can pass"):
            SampleRateFilterStage(min_sample_rate=48000, max_sample_rate=16000)


class TestRowDroppingIsDeclared:
    """A stage that drops rows has to say ``cardinality="filter"``, because that is the only
    thing that puts a filter seam in the semantic review packet. Left at the ``1:1`` default,
    a reviewer is never told the corpus can shrink here and nobody asks how much of it
    survives -- a run over a 90%-telephony corpus then reports success on 10% of the data.
    """

    def test_sample_rate_selection_declares_itself_a_filter(self) -> None:
        from nemo_curator.stages.audio import agent as foundation

        contract = foundation.build_contract(SampleRateFilterStage(min_sample_rate=16000))
        assert contract.cardinality == "filter"

    def test_channel_selection_declares_itself_a_filter(self) -> None:
        from nemo_curator.stages.audio import agent as foundation

        contract = foundation.build_contract(ChannelCountStage(action="filter", allowed_channels=[1]))
        assert contract.cardinality == "filter"

    def test_recording_a_count_declares_that_it_drops_nothing(self) -> None:
        from nemo_curator.stages.audio import agent as foundation

        assert foundation.build_contract(ChannelCountStage()).cardinality == "1:1"

    def test_channel_conversion_declares_a_filter_only_when_it_can_refuse(self) -> None:
        """Downmixing to mono always succeeds. Any other target refuses the conversions it
        cannot do correctly (N > target > 1) and drops those rows."""
        from nemo_curator.stages.audio import agent as foundation

        assert foundation.build_contract(_convert(target_channels=1)).cardinality == "1:1"
        assert foundation.build_contract(_convert(target_channels=2)).cardinality == "filter"

    def test_both_row_cardinalities_are_advertised_whichever_action_is_set(self) -> None:
        """One stage, one card, one resolved cardinality -- so the resolved contract alone
        cannot say the stage is capable of the other behaviour. ``cardinality_options`` is what
        tells a planner reading an annotating instance that this stage can also drop rows.
        """
        from nemo_curator.stages.audio import agent as foundation

        for stage in (ChannelCountStage(), ChannelCountStage(action="filter", allowed_channels=[1]), _convert()):
            assert foundation.build_contract(stage).cardinality_options == ["filter", "annotate"]


class TestTheyCompose:
    def test_rate_selection_then_channel_conversion(self, tmp_path: Path) -> None:
        """Independent policies: accept a range of rates, and separately require mono.

        Neither stage constrains the other, so a 22.05 kHz corpus can be taken to mono
        without also having to declare 22050 the only acceptable rate.
        """
        path = _wav(tmp_path, channels=2, rate=22050)

        selected = SampleRateFilterStage(min_sample_rate=16000).process(_task(path))
        assert selected != []

        converted = _convert(target_channels=1).process(selected)
        assert converted != []
        assert converted.data["num_channels"] == 1
        assert converted.data["sample_rate"] == 22050, "selection does not resample"

    def test_selecting_48k_mono_rewrites_nothing(self, tmp_path: Path) -> None:
        """Drop-only on both axes, which needs no conversion stage at all: the rate side
        selects, the channel side selects, and every surviving row is the original file.
        """
        kept = _wav(tmp_path, channels=1, rate=48000, name="keep.wav")
        wrong_channels = _wav(tmp_path, channels=2, rate=48000, name="stereo.wav")
        wrong_rate = _wav(tmp_path, channels=1, rate=16000, name="slow.wav")

        def survives(path: str) -> bool:
            row = SampleRateFilterStage(allowed_sample_rates=[48000]).process(_task(path))
            if row == []:
                return False
            return ChannelCountStage(action="filter", allowed_channels=[1]).process(row) != []

        assert survives(kept)
        assert not survives(wrong_channels)
        assert not survives(wrong_rate)
