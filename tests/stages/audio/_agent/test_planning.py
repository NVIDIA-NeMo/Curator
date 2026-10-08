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


@pytest.mark.parametrize("overwrite", [False, True])
def test_manifest_reader_checks_fresh_downstream_roles(tmp_path: Path, overwrite: bool) -> None:
    from nemo_curator.stages.audio.common import GetAudioDurationStage, ManifestReader
    from nemo_curator.stages.audio.preprocessing import MonoConversionStage

    manifest = tmp_path / "input.jsonl"
    manifest.write_text('{"audio_filepath": "clip.wav"}\n')
    stages = [ManifestReader(str(manifest)), MonoConversionStage(output_sample_rate=16000)]
    if overwrite:
        stages.append(GetAudioDurationStage(duration_key="waveform"))
    stages.append(GetAudioDurationStage(input_residency="waveform"))
    report = validate_pipeline(stages)
    conflicts = [issue for issue in report.issues if issue.code == "key_role_conflict"]
    assert bool(conflicts) is overwrite
    assert report.ok is not overwrite


@pytest.mark.parametrize("fresh_write", [False, True])
def test_unknown_child_invalidates_only_prior_role_evidence(fresh_write: bool) -> None:
    from nemo_curator.stages.base import CompositeStage

    class UnknownAudioStage(ProcessingStage[AudioTask, AudioTask]):
        def process(self, task: AudioTask) -> AudioTask:
            return task

    producer = _ContractStage(StageContract(writes=IOSpec(data_keys=["payload"]), key_roles={"payload": "duration"}))
    consumer = _ContractStage(StageContract(reads=IOSpec(data_keys=["payload"]), key_roles={"payload": "waveform"}))

    class MixedAudioComposite(CompositeStage[AudioTask, AudioTask]):
        def decompose(self) -> list[ProcessingStage]:
            return [producer, UnknownAudioStage(), *([producer] if fresh_write else []), consumer]

    report = validate_pipeline([MixedAudioComposite(), consumer])
    conflicts = [issue for issue in report.issues if issue.code == "key_role_conflict"]
    assert bool(conflicts) is fresh_write
    assert report.ok is not fresh_write


@pytest.mark.parametrize("multiple", [False, True])
def test_scalar_filter_preserves_producer_role(multiple: bool) -> None:
    from nemo_curator.stages.audio.common import (
        GetAudioDurationStage,
        PreserveByValueConditionsStage,
        PreserveByValueStage,
    )
    from nemo_curator.stages.audio.preprocessing import MonoConversionStage

    selection = (
        PreserveByValueConditionsStage({"payload": {"target_value": 0, "operator": "gt"}})
        if multiple
        else PreserveByValueStage("payload", 0, operator="gt")
    )
    stages = [MonoConversionStage(output_sample_rate=16000), GetAudioDurationStage(duration_key="payload"), selection]
    invalid = validate_pipeline([*stages, GetAudioDurationStage(input_residency="waveform", waveform_key="payload")])
    assert not invalid.ok
    assert any(issue.code == "key_role_conflict" for issue in invalid.issues)
    valid = validate_pipeline([*stages, PreserveByValueStage("payload", 1, operator="lt")])
    assert valid.ok
    assert valid.keys_ok
    assert "duration" in valid.produced_roles


@pytest.mark.parametrize("consumer_kind", ["mono", "channels", "duration"])
def test_auto_rejects_known_incompatible_preferred_input(consumer_kind: str) -> None:
    from nemo_curator.stages.audio.common import GetAudioDurationStage
    from nemo_curator.stages.audio.preprocessing import ChannelCountStage, MonoConversionStage

    consumer = {
        "mono": MonoConversionStage(input_residency="auto", output_sample_rate=16000),
        "channels": ChannelCountStage(action="convert", input_residency="auto"),
        "duration": GetAudioDurationStage(input_residency="auto"),
    }[consumer_kind]
    stages = [MonoConversionStage(output_sample_rate=16000), GetAudioDurationStage(duration_key="waveform"), consumer]
    report = validate_pipeline(stages)
    assert not report.ok
    assert any(issue.code == "key_role_conflict" for issue in report.issues)


def test_auto_invalid_rate_can_still_use_file_alternative() -> None:
    from nemo_curator.stages.audio.common import GetAudioDurationStage
    from nemo_curator.stages.audio.preprocessing import MonoConversionStage

    report = validate_pipeline(
        [
            MonoConversionStage(output_sample_rate=16000),
            GetAudioDurationStage(duration_key="sample_rate"),
            MonoConversionStage(input_residency="auto", output_sample_rate=16000),
        ]
    )
    assert report.ok


def test_composite_known_conflict_is_an_error_without_outside_consumer() -> None:
    from nemo_curator.stages.base import CompositeStage

    class IncompatibleAudioComposite(CompositeStage[AudioTask, AudioTask]):
        def decompose(self) -> list[ProcessingStage]:
            return [
                _ContractStage(StageContract(writes=IOSpec(data_keys=["payload"]), key_roles={"payload": "duration"})),
                _ContractStage(StageContract(reads=IOSpec(data_keys=["payload"]), key_roles={"payload": "waveform"})),
            ]

    report = validate_pipeline([IncompatibleAudioComposite()])
    assert not report.ok
    assert any(issue.code == "key_role_conflict" and issue.severity == "error" for issue in report.issues)


@pytest.mark.parametrize(
    ("produced_role", "read_role"),
    [
        ("diar_segments", "segments"),
        ("vad_segments", "segments"),
        ("pred_text", "text"),
        ("reference_text", "text"),
    ],
)
def test_generic_read_accepts_structurally_compatible_role_variant(produced_role: str, read_role: str) -> None:
    producer = _ContractStage(
        StageContract(writes=IOSpec(data_keys=["payload"]), key_roles={"payload": produced_role})
    )
    consumer = _ContractStage(StageContract(reads=IOSpec(data_keys=["payload"]), key_roles={"payload": read_role}))
    report = validate_pipeline([producer, consumer])
    assert report.ok
    assert report.keys_ok
    assert not report.issues


@pytest.mark.parametrize("filter_kind", ["single", "conditions"])
def test_explicit_drop_policy_allows_missing_top_level_keys(filter_kind: str) -> None:
    from nemo_curator.stages.audio.common import PreserveByValueConditionsStage, PreserveByValueStage

    def make_stage(policy: str) -> ProcessingStage:
        if filter_kind == "single":
            return PreserveByValueStage("absent", 1, missing_value_policy=policy)
        return PreserveByValueConditionsStage({"absent": 1}, missing_value_policy=policy)

    task = AudioTask(data={})
    drop = make_stage("drop")
    assert validate_pipeline([drop], initial_keys=[]).ok
    assert drop.process_batch([task]) == []
    assert not validate_pipeline([make_stage("error")], initial_keys=[]).ok


@pytest.mark.parametrize("conditional", [False, True])
@pytest.mark.parametrize("reemit", [False, True])
def test_child_replacement_preserves_only_parent_and_new_child_keys(conditional: bool, reemit: bool) -> None:
    spec = IOSpec(segment_data_keys=["old_score"], produces=["tensor"])
    producer = _ContractStage(
        StageContract(
            writes=IOSpec() if conditional else spec,
            conditional_writes=[ConditionalWrite(writes=spec, condition="scored input")] if conditional else [],
            key_roles={"old_score": "waveform"},
        )
    )
    rebuild = _ContractStage(
        StageContract(
            writes=IOSpec(data_keys=["segments"], segment_data_keys=["new_score"]),
            conditional_writes=[
                ConditionalWrite(writes=IOSpec(segment_data_keys=["old_score"]), condition="new score")
            ]
            if reemit
            else [],
            preserves_upstream_segment_keys=False,
        )
    )
    parent_consumer = _ContractStage(StageContract(reads=IOSpec(data_keys=["recording_id", "old_score"])))
    child_consumer = _ContractStage(StageContract(reads=IOSpec(segment_data_keys=["new_score"])))
    chain = [producer, rebuild, parent_consumer, child_consumer]
    report = validate_pipeline(chain, initial_keys={"recording_id", "old_score"})
    assert report.ok
    assert report.keys_ok
    old_child_consumer = _ContractStage(StageContract(reads=IOSpec(segment_data_keys=["old_score"])))
    report = validate_pipeline([*chain, old_child_consumer], initial_keys={"recording_id", "old_score"})
    assert report.ok is reemit
    expected = "conditional_read" if reemit else "unsatisfied_reads"
    assert any(issue.code == expected for issue in report.issues)
    assert rebuild.describe().to_dict()["preserves_upstream_segment_keys"] is False


@pytest.mark.parametrize("parent_tensor", [False, True])
@pytest.mark.parametrize("new_child_tensor", [False, True])
def test_child_replacement_rebuilds_tensor_residency(
    tmp_path: Path, parent_tensor: bool, new_child_tensor: bool
) -> None:
    from nemo_curator.stages.audio.common import ManifestWriterStage

    rebuild = _ContractStage(
        StageContract(
            writes=IOSpec(
                data_keys=["segments"],
                segment_data_keys=["waveform"] if new_child_tensor else [],
                produces=["tensor"] if new_child_tensor else [],
            ),
            preserves_upstream_segment_keys=False,
        )
    )
    report = validate_pipeline(
        [rebuild, ManifestWriterStage(output_path=str(tmp_path / "rows.jsonl"))],
        initial_keys={"segments", "waveform"} if parent_tensor else {"segments"},
        initial_segment_keys={"waveform"},
    )
    assert report.ok is (not parent_tensor and not new_child_tensor)
    assert any(issue.code == "tensor_into_sink" for issue in report.issues) is (parent_tensor or new_child_tensor)
