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

from pathlib import Path

import pytest

from nemo_curator.stages.audio._agent._agent_ready import AgentReady, ConditionalWrite, IOSpec, StageContract
from nemo_curator.stages.audio._agent._planning import validate_pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


class _ContractStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    def __init__(self, contract: StageContract) -> None:
        self.contract = contract

    def describe(self) -> StageContract:
        return self.contract

    def process(self, task: AudioTask) -> AudioTask:
        return task


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("reemit", [False, True])
def test_rebuild_discards_inherited_conditional_state(nested: bool, reemit: bool) -> None:
    scope = "segment_data_keys" if nested else "data_keys"
    conditional = ConditionalWrite(writes=IOSpec(**{scope: ["probe_key"]}), condition="runtime branch")
    producer = _ContractStage(StageContract(conditional_writes=[conditional]))
    rebuild = _ContractStage(
        StageContract(
            writes=IOSpec(data_keys=["audio_filepath"]),
            conditional_writes=[conditional] if reemit else [],
            preserves_upstream_keys=False,
        )
    )
    consumer = _ContractStage(StageContract(reads=IOSpec(**{scope: ["probe_key"]})))
    report = validate_pipeline([producer, rebuild, consumer], initial_keys={"audio_filepath"})
    assert report.ok is reemit
    expected = "conditional_read" if reemit else "unsatisfied_reads"
    assert any(issue.code == expected for issue in report.issues)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("conditional", [False, True])
def test_literal_overwrite_replaces_role(nested: bool, conditional: bool) -> None:
    scope = "segment_data_keys" if nested else "data_keys"
    producer = _ContractStage(StageContract(writes=IOSpec(**{scope: ["payload"]}), key_roles={"payload": "waveform"}))
    writes = IOSpec(**{scope: ["payload"]})
    overwrite = _ContractStage(
        StageContract(
            writes=IOSpec() if conditional else writes,
            conditional_writes=[ConditionalWrite(writes=writes, condition="runtime branch")] if conditional else [],
            key_roles={"payload": "text"},
        )
    )
    consumer = _ContractStage(StageContract(reads=IOSpec(**{scope: ["payload"]}), key_roles={"payload": "waveform"}))
    report = validate_pipeline([producer, overwrite, consumer])
    assert report.ok is conditional
    assert any(issue.code == ("conditional_role" if conditional else "key_role_conflict") for issue in report.issues)


@pytest.mark.parametrize("nested", [False, True])
def test_task_alias_removal_preserves_remaining_and_nested_roles(nested: bool) -> None:
    scope = "segment_data_keys" if nested else "data_keys"
    producer = _ContractStage(
        StageContract(writes=IOSpec(**{scope: ["left", "right"]}), key_roles={"left": "text", "right": "text"})
    )
    remove = _ContractStage(StageContract(removes_keys=["left"]))
    consumer = _ContractStage(StageContract(reads=IOSpec(**{scope: ["right"]}), key_roles={"right": "text"}))
    report = validate_pipeline([producer, remove, consumer])
    assert report.ok, report.summary()


def test_missing_alternative_does_not_hide_literal_role_conflict() -> None:
    producer = _ContractStage(StageContract(writes=IOSpec(data_keys=["payload"]), key_roles={"payload": "text"}))
    consumer = _ContractStage(
        StageContract(
            reads_one_of=[IOSpec(data_keys=["payload"]), IOSpec(data_keys=["missing"])],
            key_roles={"payload": "waveform", "missing": "waveform"},
        )
    )
    report = validate_pipeline([producer, consumer])
    assert not report.ok
    assert any(issue.code == "key_role_conflict" for issue in report.issues)


def test_overwriting_task_key_does_not_change_nested_role() -> None:
    producer = _ContractStage(
        StageContract(
            writes=IOSpec(data_keys=["payload"], segment_data_keys=["payload"]), key_roles={"payload": "waveform"}
        )
    )
    overwrite = _ContractStage(StageContract(writes=IOSpec(data_keys=["payload"]), key_roles={"payload": "text"}))
    consumer = _ContractStage(
        StageContract(reads=IOSpec(segment_data_keys=["payload"]), key_roles={"payload": "waveform"})
    )
    report = validate_pipeline([producer, overwrite, consumer])
    assert report.ok, report.summary()


@pytest.mark.parametrize("carrier", ["audio_filepath", "waveform"])
def test_file_to_waveform_in_place_alias_remains_composable(wav_filepath, carrier: str) -> None:  # noqa: ANN001
    from nemo_curator.stages.audio.common import GetAudioDurationStage
    from nemo_curator.stages.audio.preprocessing.mono_conversion import MonoConversionStage

    producer = _ContractStage(StageContract(writes=IOSpec(data_keys=[carrier]), key_roles={carrier: "audio_filepath"}))
    conversion = MonoConversionStage(audio_filepath_key=carrier, waveform_key=carrier, output_sample_rate=16000)
    waveform_consumer = GetAudioDurationStage(input_residency="waveform", waveform_key=carrier)
    report = validate_pipeline([producer, conversion, waveform_consumer], initial_keys=set(), initial_roles=set())
    assert report.ok, report.summary()
    assert report.keys_ok, report.summary()
    assert "audio_filepath" not in report.produced_roles
    result = conversion.process(AudioTask(data={carrier: str(wav_filepath)}))
    rows = waveform_consumer.process_batch([result])
    assert len(rows) == 1
    assert rows[0].data["duration"] > 0
    file_consumer = GetAudioDurationStage(audio_filepath_key=carrier)
    invalid = validate_pipeline([producer, conversion, file_consumer], initial_keys=set(), initial_roles=set())
    assert not invalid.ok
    assert any(issue.code == "key_role_conflict" for issue in invalid.issues)


@pytest.mark.parametrize("nested", [False, True])
def test_explicit_seed_role_overrides_alias_name(nested: bool) -> None:
    scope = "segment_data_keys" if nested else "data_keys"
    consumer = _ContractStage(
        StageContract(
            reads=IOSpec(**{scope: ["waveform"]}, accepts=["file"]), key_roles={"waveform": "audio_filepath"}
        )
    )
    kwargs = (
        {"initial_segment_keys": {"waveform"}, "initial_segment_roles": {"audio_filepath"}}
        if nested
        else {"initial_keys": {"waveform"}, "initial_roles": {"audio_filepath"}}
    )
    report = validate_pipeline([consumer], **kwargs)
    assert report.ok, report.summary()
    assert report.keys_ok, report.summary()


@pytest.mark.parametrize("retain_other_tensor", [False, True])
def test_scalar_overwrite_releases_only_its_tensor_carrier(
    wav_filepath: Path, tmp_path: Path, retain_other_tensor: bool
) -> None:
    from nemo_curator.stages.audio.common import GetAudioDurationStage, ManifestWriterStage
    from nemo_curator.stages.audio.preprocessing.mono_conversion import MonoConversionStage

    conversion = MonoConversionStage(output_sample_rate=16000)
    duration = GetAudioDurationStage(duration_key="waveform")
    sink = ManifestWriterStage(output_path=str(tmp_path / "rows.jsonl"))
    seed = {"audio_filepath", "other"} if retain_other_tensor else {"audio_filepath"}
    report = validate_pipeline(
        [conversion, duration, sink],
        initial_keys=seed,
        initial_tensor_keys={"other"} if retain_other_tensor else set(),
    )
    assert duration.describe().key_roles["waveform"] == "duration"
    assert report.ok is (not retain_other_tensor), report.summary()
    if retain_other_tensor:
        assert any(issue.code == "tensor_into_sink" for issue in report.issues)
    else:
        task = conversion.process(AudioTask(data={"audio_filepath": str(wav_filepath)}))
        result = duration.process(task)
        assert isinstance(result.data["waveform"], float)
        sink.setup()
        sink.process(result)
        assert (tmp_path / "rows.jsonl").is_file()
