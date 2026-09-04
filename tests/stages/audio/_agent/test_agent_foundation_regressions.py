# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Regressions for foundation behavior exposed to audio-pipeline agents."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import soundfile as sf
import torch

from nemo_curator.stages import audio
from nemo_curator.stages.audio import agent
from nemo_curator.stages.audio._agent._agent_registry import build_contract, static_contract
from nemo_curator.stages.audio._agent._catalog import unavailable_modules
from nemo_curator.stages.audio._agent._composite import expand_composites
from nemo_curator.stages.audio._agent._planning import validate_pipeline
from nemo_curator.stages.audio._agent._residency import resolve_audio, write_audio_stable
from nemo_curator.stages.audio.common import (
    CreateInitialManifestAudioFolderStage,
    ManifestReader,
    ManifestReaderStage,
    ManifestWriterStage,
    PreserveByValueStage,
    ensure_waveform_2d,
)
from nemo_curator.stages.audio.preprocessing import (
    ChannelCountStage,
    MonoConversionStage,
    SegmentConcatenationStage,
)
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from pathlib import Path


def test_stable_audio_names_include_layout_and_written_short_stereo_shape(tmp_path: Path) -> None:
    """Different channel layouts with identical samples need distinct artifacts."""
    output_dir = str(tmp_path)
    mono = torch.zeros(1, 32000)
    stereo = torch.zeros(2, 16000)

    mono_path = write_audio_stable(mono, 16000, output_dir=output_dir, stem="audio")
    stereo_path = write_audio_stable(stereo, 16000, output_dir=output_dir, stem="audio")

    assert mono_path != stereo_path
    assert sf.info(mono_path).channels == 1
    assert sf.info(stereo_path).channels == 2

    short_stereo_path = write_audio_stable(
        torch.tensor([[0.25], [0.75]]),
        16000,
        output_dir=output_dir,
        stem="short",
    )
    short_info = sf.info(short_stereo_path)
    assert (short_info.frames, short_info.channels) == (1, 2)


def test_nested_composite_is_reported_as_unrunnable(monkeypatch) -> None:  # noqa: ANN001
    """A shape rejected by Pipeline must not be downgraded to opaque."""
    stage = ManifestReader("manifest.jsonl")
    nested_children = [ManifestReader("one.jsonl"), ManifestReader("two.jsonl")]
    monkeypatch.setattr(stage, "decompose_and_apply_with", lambda: nested_children)

    expansion = expand_composites([stage])
    assert expansion.stages == []
    assert 0 not in expansion.opaque
    assert "nested composition" in expansion.unrunnable[0]

    report = validate_pipeline([stage])
    assert not report.ok
    assert any(issue.code == "composite_unrunnable" for issue in report.issues)


def test_manifest_writer_static_contract_exposes_invariant_sink_gates(tmp_path: Path) -> None:
    """Static discovery must not describe a required-path JSONL sink as pure."""
    static = static_contract(ManifestWriterStage)
    configured = build_contract(ManifestWriterStage(output_path=str(tmp_path / "out.jsonl")))

    assert static.gates == configured.gates


def test_public_facade_exposes_unavailable_modules_and_folder_source() -> None:
    """The documented public layer must expose foundation discovery features."""
    from nemo_curator.stages.audio import CreateInitialManifestAudioFolderStage
    from nemo_curator.stages.audio.common import CreateInitialManifestAudioFolderStage as FolderSource

    assert agent.unavailable_modules is unavailable_modules
    assert CreateInitialManifestAudioFolderStage is FolderSource
    assert "CreateInitialManifestAudioFolderStage" in audio.__all__


def _stereo_task(tmp_path: Path, sample_rate: int = 16000) -> tuple[AudioTask, str]:
    """A row carrying BOTH a resident stereo waveform and the file it came from."""
    path = str(tmp_path / "stereo.wav")
    waveform = torch.stack([torch.zeros(sample_rate), torch.ones(sample_rate) * 0.5])
    sf.write(path, waveform.T.numpy(), sample_rate)
    task = AudioTask(
        dataset_name="resident",
        data={"audio_filepath": path, "waveform": waveform, "sample_rate": sample_rate},
    )
    return task, path


@pytest.mark.parametrize(
    ("factory", "channels_key"),
    [
        (
            lambda out: MonoConversionStage(
                output_sample_rate=16000,
                input_residency="waveform",
                keep_waveform_in_task=False,
                write_to_disk=True,
                update_audio_filepath=True,
                output_dir=out,
            ),
            "is_mono",
        ),
        (
            lambda out: ChannelCountStage(
                action="convert",
                target_channels=1,
                input_residency="waveform",
                keep_waveform_in_task=False,
                write_to_disk=True,
                update_audio_filepath=True,
                output_dir=out,
            ),
            "num_channels",
        ),
    ],
    ids=["mono_conversion", "channel_count"],
)
def test_disk_only_conversion_does_not_leave_the_pre_conversion_waveform(
    tmp_path: Path,
    factory,  # noqa: ANN001
    channels_key: str,
) -> None:
    """Resident input -> disk-only conversion -> auto-residency consumer must not read stale audio."""
    task, original = _stereo_task(tmp_path)
    stage = factory(str(tmp_path / "out"))

    result = stage.process(task)
    assert result is not None
    assert not isinstance(result, list)

    # The converted metadata and the audio a downstream stage can reach have to agree.
    assert result.data[channels_key] in (True, 1)
    assert "waveform" not in result.data
    assert "sample_rate" not in result.data

    consumed = resolve_audio(result.data, residency="auto")
    assert consumed is not None
    assert ensure_waveform_2d(consumed[0]).shape[0] == 1
    assert result.data["audio_filepath"] != original

    # And validation knows, so a downstream waveform reader is caught before the run.
    assert set(build_contract(stage).removes_keys) == {"waveform", "sample_rate"}


@pytest.mark.parametrize(
    "cls",
    [MonoConversionStage, ChannelCountStage],
    ids=["mono_conversion", "channel_count"],
)
def test_conversion_without_an_output_sink_is_rejected(cls) -> None:  # noqa: ANN001
    """Converting into neither the task nor disk keeps the original audio under converted metadata."""
    extra = {"action": "convert", "target_channels": 1} if cls is ChannelCountStage else {}
    with pytest.raises(ValueError, match="keep_waveform_in_task or write_to_disk"):
        cls(keep_waveform_in_task=False, write_to_disk=False, **extra)
    with pytest.raises(ValueError, match="update_audio_filepath"):
        cls(write_to_disk=False, update_audio_filepath=True, **extra)


def test_task_type_mismatch_is_an_error_not_a_clean_report(tmp_path: Path) -> None:
    """A folder source feeding a FileGroupTask reader is a runtime FileNotFoundError."""
    chain = [
        CreateInitialManifestAudioFolderStage(data_dir=str(tmp_path)),
        ManifestReaderStage(),
    ]
    report = validate_pipeline(chain, initial_task_type="EmptyTask")
    assert not report.ok
    mismatches = [i for i in report.issues if i.code == "task_type_mismatch"]
    assert [i.stage_index for i in mismatches] == [1]
    assert "AudioTask" in mismatches[0].message
    assert "FileGroupTask" in mismatches[0].message

    # Two readers in a row is the same fault: the first consumes the FileGroupTask and the
    # second is handed the AudioTask it produced.
    doubled = validate_pipeline([ManifestReaderStage(), ManifestReaderStage()], initial_task_type="FileGroupTask")
    assert [i.stage_index for i in doubled.issues if i.code == "task_type_mismatch"] == [1]

    # The composite that exists to get this right stays clean -- the check must not fire on
    # the pipeline the caller is being steered towards.
    good = validate_pipeline([ManifestReader("manifest.jsonl")], initial_task_type="EmptyTask")
    assert not [i for i in good.issues if i.code == "task_type_mismatch"]


def test_concatenation_does_not_promise_upstream_keys_it_drops() -> None:
    """N:1 concatenation rebuilds the task, so a downstream text read must fail validation."""
    concat = SegmentConcatenationStage()
    assert build_contract(concat).preserves_upstream_keys is False

    report = validate_pipeline(
        [concat, PreserveByValueStage(input_value_key="text", target_value="keep")],
        initial_roles={"audio_filepath", "segments", "transcript"},
        initial_keys={"audio_filepath", "segments", "text"},
    )
    assert not report.ok
    assert any(i.code in {"unsatisfied_reads", "dangling_key"} and i.stage_index == 1 for i in report.issues)

    # The state the walk carries past the stage, rather than the report's union: the filter
    # above re-declares ``text`` as its own write (it passes the column through), so only the
    # concatenation's own output shows what survived it.
    after_concat = validate_pipeline(
        [concat],
        initial_roles={"audio_filepath", "segments", "transcript"},
        initial_keys={"audio_filepath", "segments", "text"},
    )
    assert "text" not in after_concat.produced_keys
    assert "segments" not in after_concat.produced_keys
