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

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import soundfile as sf
import torch

from nemo_curator.stages.audio._agent._agent_ready import (
    AgentReady,
    ConditionalRead,
    ConditionalWrite,
    IOSpec,
    StageContract,
)
from nemo_curator.stages.audio._agent._agent_registry import build_contract
from nemo_curator.stages.audio._agent._conformance import assert_agent_ready
from nemo_curator.stages.audio._agent._planning import validate_pipeline
from nemo_curator.stages.audio._agent._residency import (
    resolve_audio,
    resolve_audio_path,
)
from nemo_curator.stages.audio.common import (
    CreateInitialManifestAudioFolderStage,
    ManifestCheckpointStage,
    ManifestReader,
    ManifestReaderStage,
    ManifestWriterStage,
    PreserveByValueStage,
)
from nemo_curator.stages.audio.preprocessing import (
    ChannelCountStage,
    MonoConversionStage,
    SegmentConcatenationStage,
)
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask, FileGroupTask

if TYPE_CHECKING:
    from pathlib import Path


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
    from nemo_curator.stages.audio.preprocessing import MonoConversionStage

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


class _ConfiguredContractStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    def __init__(self, contract: StageContract) -> None:
        self.contract = contract

    def describe(self) -> StageContract:
        return self.contract

    def process(self, task: AudioTask) -> AudioTask:
        return task


def test_optional_reads_are_visible_without_blocking_fallback_paths() -> None:
    contract = StageContract(
        reads=IOSpec(data_keys=["text"]),
        optional_reads=IOSpec(data_keys=["speaker_id"]),
    )
    stage = _ConfiguredContractStage(contract)

    assert build_contract(stage).to_dict()["optional_reads"]["data_keys"] == ["speaker_id"]
    assert validate_pipeline([stage], initial_keys={"text"}).ok


def test_conditional_reads_follow_the_runtime_scope_selector() -> None:
    contract = StageContract(
        conditional_reads=[
            ConditionalRead(
                reads_one_of=[IOSpec(data_keys=["waveform", "sample_rate"])],
                condition="'segments' is absent",
                forbids_keys=["segments"],
            ),
            ConditionalRead(
                reads_one_of=[IOSpec(segment_data_keys=["waveform", "sample_rate"])],
                condition="'segments' is present",
                requires_keys=["segments"],
            ),
        ],
        key_roles={
            "segments": "segments",
            "waveform": "waveform",
            "sample_rate": "sample_rate",
        },
    )
    stage = _ConfiguredContractStage(contract)

    task_report = validate_pipeline(
        [stage],
        initial_keys={"waveform", "sample_rate"},
        initial_roles={"waveform", "sample_rate"},
    )
    incomplete_nested_report = validate_pipeline(
        [stage],
        initial_keys={"waveform", "sample_rate", "segments"},
        initial_roles={"waveform", "sample_rate", "segments"},
        initial_segment_keys={"segment_num"},
    )
    complete_nested_report = validate_pipeline(
        [stage],
        initial_keys={"waveform", "sample_rate", "segments"},
        initial_roles={"waveform", "sample_rate", "segments"},
        initial_segment_keys={"waveform", "sample_rate"},
        initial_segment_roles={"waveform", "sample_rate"},
    )

    assert task_report.ok
    assert not incomplete_nested_report.ok
    assert complete_nested_report.ok
    assert contract.to_dict()["conditional_reads"][1]["requires_keys"] == ["segments"]


def test_invalidated_provenance_key_is_retained_but_not_planner_available() -> None:
    invalidator = _ConfiguredContractStage(StageContract(invalidates_keys=["audio_filepath"]))
    consumer = _ConfiguredContractStage(StageContract(reads=IOSpec(data_keys=["audio_filepath"])))

    contract = build_contract(invalidator)
    report = validate_pipeline(
        [invalidator, consumer],
        initial_keys={"audio_filepath"},
        initial_roles={"audio_filepath"},
    )

    assert contract.to_dict()["invalidates_keys"] == ["audio_filepath"]
    assert not report.ok
    assert any(issue.code == "key_removed_upstream" for issue in report.issues)


def test_conditional_tensor_write_is_not_guaranteed_but_still_blocks_json_sink(tmp_path: Path) -> None:
    producer = _ConfiguredContractStage(
        StageContract(
            conditional_writes=[
                ConditionalWrite(
                    writes=IOSpec(data_keys=["resident_audio"], produces=["tensor"]),
                    condition="file decoding succeeds and resident audio is assigned",
                )
            ],
            key_roles={"resident_audio": "waveform"},
        )
    )
    writer = ManifestWriterStage(output_path=str(tmp_path / "out.jsonl"))

    report = validate_pipeline(
        [producer, writer],
        initial_roles=set(),
        initial_keys=set(),
    )

    assert "resident_audio" not in report.produced_keys
    assert any(issue.code == "tensor_into_sink" and issue.severity == "error" for issue in report.issues)


def test_unknown_role_selector_requires_its_exact_conditional_key() -> None:
    producer = _ConfiguredContractStage(
        StageContract(
            conditional_writes=[
                ConditionalWrite(
                    writes=IOSpec(data_keys=["row_score"]),
                    condition="valid runtime data causes score assignment",
                )
            ],
            key_roles={"row_score": "score"},
        )
    )
    selector = PreserveByValueStage("row_score", 1.0, "le")

    conditional_only = validate_pipeline(
        [producer, selector],
        initial_roles=set(),
        initial_keys=set(),
    )
    assert conditional_only.ok
    assert any(issue.code == "conditional_read" and issue.stage_index == 1 for issue in conditional_only.issues)

    seeded = validate_pipeline(
        [selector],
        initial_roles=set(),
        initial_keys={"row_score"},
    )
    assert seeded.ok
    assert seeded.keys_ok


def test_nested_input_requires_explicit_segment_seeds_and_accepts_remapped_key() -> None:
    nested_reader = _ConfiguredContractStage(
        StageContract(
            reads=IOSpec(data_keys=["segments"], segment_data_keys=["custom_text"]),
            key_roles={"segments": "segments", "custom_text": "text"},
        )
    )

    unseeded = validate_pipeline(
        [nested_reader],
        initial_roles={"segments", "text"},
        initial_keys={"segments", "custom_text"},
    )
    assert not unseeded.ok
    assert any(issue.code == "unsatisfied_reads" for issue in unseeded.issues)

    seeded = validate_pipeline(
        [nested_reader],
        initial_roles={"segments"},
        initial_keys={"segments"},
        initial_segment_roles={"text"},
        initial_segment_keys={"custom_text"},
    )
    assert seeded.ok
    assert seeded.keys_ok


@pytest.mark.parametrize(
    "sample_rate",
    [
        pytest.param(True, id="bool"),
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(16000.5, id="fractional"),
        pytest.param(torch.tensor([16000]), id="non-scalar-tensor"),
    ],
)
def test_auto_resolvers_fall_back_from_invalid_resident_rates(tmp_path: Path, sample_rate: object) -> None:
    file_path = tmp_path / "valid.wav"
    sf.write(file_path, torch.ones(16000).numpy(), 16000)
    loaded = torch.ones(1, 16000)

    def loader(_path: str, *, mono: bool) -> tuple[torch.Tensor, int]:
        assert mono
        return loaded, 16000

    item = {
        "audio_filepath": str(file_path),
        "waveform": torch.zeros(1, 8000),
        "sample_rate": sample_rate,
    }
    resolved = resolve_audio(
        item,
        residency="auto",
        loader=loader,
        file_audio_hydration="auto_partial",
    )

    assert resolved is not None
    assert resolved[0] is loaded
    assert resolved[1] == 16000
    assert item["waveform"] is loaded
    assert item["sample_rate"] == 16000

    temporary_paths: list[str] = []
    resolved_path = resolve_audio_path(
        {
            "audio_filepath": str(file_path),
            "waveform": torch.zeros(1, 8000),
            "sample_rate": sample_rate,
        },
        residency="auto",
        temp_dir=str(tmp_path),
        register_temp=temporary_paths,
    )
    assert resolved_path == str(file_path)
    assert temporary_paths == []


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


@pytest.mark.parametrize("sink", [ManifestWriterStage, ManifestCheckpointStage])
def test_an_input_that_arrives_with_a_waveform_is_blocked_from_a_json_sink(sink: type, tmp_path: Path) -> None:
    """validate_pipeline advertises a resident-waveform input; the sink gate must see it."""
    stage = sink(output_path=str(tmp_path / "out.jsonl"))
    report = validate_pipeline(
        [stage],
        initial_roles={"waveform", "sample_rate"},
        initial_keys={"waveform", "sample_rate"},
    )
    assert not report.ok
    assert any(i.code == "tensor_into_sink" and i.severity == "error" for i in report.issues)

    # The runtime failure the gate stands in for.
    stage.setup()
    with pytest.raises(TypeError, match="not JSON serializable"):
        stage.process(AudioTask(dataset_name="d", data={"waveform": torch.zeros(1, 16), "sample_rate": 16000}))


def test_nested_waveform_seed_is_inferred_as_tensor_resident(tmp_path: Path) -> None:
    writer = ManifestWriterStage(output_path=str(tmp_path / "out.jsonl"))
    report = validate_pipeline(
        [writer],
        initial_roles={"segments"},
        initial_keys={"segments"},
        initial_segment_roles={"waveform"},
        initial_segment_keys={"waveform"},
    )

    assert not report.ok
    assert any(issue.code == "tensor_into_sink" for issue in report.issues)


def test_a_tensor_under_an_uninferable_name_can_be_declared_resident(tmp_path: Path) -> None:
    """A custom carrier has no role to infer from, so the seed has to be sayable outright."""
    writer = ManifestWriterStage(output_path=str(tmp_path / "out.jsonl"))
    assert validate_pipeline([writer], initial_keys={"audio_tensor"}).ok
    assert not validate_pipeline([writer], initial_keys={"audio_tensor"}, initial_tensor_keys={"audio_tensor"}).ok


def test_an_explicit_empty_tensor_seed_overrides_waveform_name_inference(tmp_path: Path) -> None:
    """A nullable waveform-named manifest column is not automatically a resident tensor."""
    writer = ManifestWriterStage(output_path=str(tmp_path / "out.jsonl"))
    report = validate_pipeline(
        [writer],
        initial_keys={"waveform"},
        initial_tensor_keys=set(),
    )
    assert report.ok
    assert not any(issue.code == "tensor_into_sink" for issue in report.issues)


def test_a_plain_manifest_input_still_reaches_a_json_sink(tmp_path: Path) -> None:
    """The seeding must not make every pipeline look tensor-resident."""
    report = validate_pipeline([ManifestWriterStage(output_path=str(tmp_path / "out.jsonl"))])
    assert report.ok
    assert not any(i.code == "tensor_into_sink" for i in report.issues)


def test_a_custom_manifest_path_column_is_what_the_reader_declares(tmp_path: Path) -> None:
    """The reader emits the row verbatim, so its contract must name the column it was pointed at."""
    manifest = tmp_path / "m.jsonl"
    manifest.write_text('{"recording_path": "/tmp/a.wav", "text": "hi"}\n')
    reader = ManifestReaderStage(include_files_key="recording_path")

    assert build_contract(reader).writes.data_keys == ["recording_path"]
    emitted = reader.process(FileGroupTask(dataset_name="d", data=[str(manifest)]))
    assert "recording_path" in emitted[0].data
    assert "audio_filepath" not in emitted[0].data
    assert_agent_ready(
        reader,
        lambda: FileGroupTask(dataset_name="d", data=[str(manifest)]),
        expected_cardinality="1:N fan-out",
        available_keys=set(),
    )

    # Seeded empty because the input is a FileGroupTask of manifest PATHS: it carries no
    # audio columns, and the default seed would otherwise supply the very ``audio_filepath``
    # whose absence is the point.
    seed = {"initial_keys": set(), "initial_roles": set(), "initial_task_type": "FileGroupTask"}

    # A default consumer reads ``audio_filepath``, which this manifest does not carry.
    assert not validate_pipeline([reader, MonoConversionStage()], **seed).keys_ok

    # Pointed at the same column, it validates.
    assert validate_pipeline([reader, MonoConversionStage(audio_filepath_key="recording_path")], **seed).keys_ok

    # And the ordinary manifest still pairs with the ordinary consumer.
    assert validate_pipeline([ManifestReaderStage(), MonoConversionStage()], **seed).keys_ok


def test_concatenation_reads_require_nested_segment_audio_keys() -> None:
    """SegmentConcatenation reads waveform+sample_rate from EACH child, not just the container."""
    concat = SegmentConcatenationStage()

    # Only the top-level segments container is present; the per-child audio the runtime reads
    # is missing, so the stage must not validate clean.
    top_only = validate_pipeline(
        [concat],
        initial_roles={"segments"},
        initial_keys={"segments"},
    )
    assert not top_only.ok
    assert any(i.code == "unsatisfied_reads" and i.stage_index == 0 for i in top_only.issues)

    # Seeding the nested waveform/sample_rate the child carries makes it compose.
    with_nested = validate_pipeline(
        [concat],
        initial_roles={"segments"},
        initial_keys={"segments"},
        initial_segment_roles={"waveform", "sample_rate"},
        initial_segment_keys={"waveform", "sample_rate"},
    )
    assert with_nested.ok
    assert with_nested.keys_ok

    # Remapped child keys chain by role: pointed at the names the seed carries, it stays clean.
    remapped = SegmentConcatenationStage(waveform_key="seg_wav", sample_rate_key="seg_sr")
    report = validate_pipeline(
        [remapped],
        initial_roles={"segments"},
        initial_keys={"segments"},
        initial_segment_roles={"waveform", "sample_rate"},
        initial_segment_keys={"seg_wav", "seg_sr"},
    )
    assert report.ok
    assert report.keys_ok


@pytest.mark.parametrize("sink", [ManifestWriterStage, ManifestCheckpointStage])
def test_same_name_nested_tensor_survives_top_level_drop_into_sink(sink: type, tmp_path: Path) -> None:
    """A disk-only conversion drops the TOP-LEVEL waveform; a same-named nested one still blocks a sink."""
    mono = MonoConversionStage(
        output_sample_rate=16000,
        input_residency="waveform",
        keep_waveform_in_task=False,
        write_to_disk=True,
        output_dir=str(tmp_path / "out"),
    )
    assert set(build_contract(mono).removes_keys) == {"waveform", "sample_rate"}
    writer = sink(output_path=str(tmp_path / "out.jsonl"))

    # Task-level AND segment-level waveforms share the key name "waveform". The conversion
    # removes only the task-level carrier; the nested one reaches the JSON sink.
    report = validate_pipeline(
        [mono, writer],
        initial_roles={"waveform", "sample_rate", "segments"},
        initial_keys={"waveform", "sample_rate", "segments"},
        initial_segment_roles={"waveform", "sample_rate"},
        initial_segment_keys={"waveform", "sample_rate"},
    )
    assert not report.ok
    assert any(i.code == "tensor_into_sink" and i.severity == "error" for i in report.issues)

    # Single-scope behavior is unchanged: with no nested carrier, dropping the top-level one
    # clears residency and the sink is clean (no false positive from the scope split).
    clean = validate_pipeline(
        [mono, writer],
        initial_roles={"waveform", "sample_rate"},
        initial_keys={"waveform", "sample_rate"},
    )
    assert not any(i.code == "tensor_into_sink" for i in clean.issues)


def test_multi_alternative_read_dangles_when_no_literal_branch_is_complete() -> None:
    """An auto consumer whose role is met only by a renamed producer key has no complete branch."""
    renamed_role_only = _ConfiguredContractStage(
        StageContract(
            writes=IOSpec(data_keys=["resampled_audio_filepath"]),
            key_roles={"resampled_audio_filepath": "audio_filepath"},
        )
    )
    auto_consumer = MonoConversionStage(input_residency="auto")

    dangling = validate_pipeline(
        [renamed_role_only, auto_consumer],
        initial_roles=set(),
        initial_keys=set(),
    )
    # Role-level composability holds, but no reads_one_of branch is literally complete.
    assert dangling.ok
    assert not dangling.keys_ok
    assert any(i.code == "dangling_key" and i.stage_index == 1 for i in dangling.issues)

    # A complete FILE branch (literal audio_filepath) stays clean.
    file_producer = _ConfiguredContractStage(
        StageContract(
            writes=IOSpec(data_keys=["audio_filepath"]),
            key_roles={"audio_filepath": "audio_filepath"},
        )
    )
    clean_file = validate_pipeline(
        [file_producer, MonoConversionStage(input_residency="auto")],
        initial_roles=set(),
        initial_keys=set(),
    )
    assert clean_file.ok
    assert clean_file.keys_ok

    # A complete WAVEFORM-PAIR branch stays clean too.
    waveform_producer = _ConfiguredContractStage(
        StageContract(
            writes=IOSpec(data_keys=["waveform", "sample_rate"], produces=["tensor"]),
            key_roles={"waveform": "waveform", "sample_rate": "sample_rate"},
        )
    )
    clean_waveform = validate_pipeline(
        [waveform_producer, MonoConversionStage(input_residency="auto")],
        initial_roles=set(),
        initial_keys=set(),
    )
    assert clean_waveform.ok
    assert clean_waveform.keys_ok


class _ConditionalScoreProducer(AgentReady):
    """Writes ``score`` only on rows where ``source`` is non-null (a data-dependent branch)."""

    name = "ConditionalScoreProducer"

    def describe(self) -> StageContract:
        return StageContract(
            reads=IOSpec(data_keys=["audio_filepath"]),
            conditional_writes=[
                ConditionalWrite(writes=IOSpec(data_keys=["score"]), condition="'source' is non-null"),
            ],
        )

    def process(self, task: object) -> object:
        return task


class _ReachabilityGatedTensorProducer(AgentReady):
    """Hydrates a waveform only when a ``sample_rate`` column already exists upstream."""

    name = "ReachabilityGatedTensorProducer"

    def describe(self) -> StageContract:
        return StageContract(
            reads=IOSpec(data_keys=["audio_filepath"]),
            conditional_writes=[
                ConditionalWrite(
                    writes=IOSpec(data_keys=["waveform", "sample_rate"], produces=["tensor"]),
                    condition="a resident sample_rate without a waveform is completed from the file",
                    requires_keys=["sample_rate"],
                ),
            ],
        )

    def process(self, task: object) -> object:
        return task


def test_a_read_met_only_by_a_conditional_write_is_a_warning_not_an_error() -> None:
    """Conditional outputs stay non-guaranteed, but their consumers still compose."""
    report = validate_pipeline([_ConditionalScoreProducer(), PreserveByValueStage("score", 3.0, "ge")])

    assert report.ok, report.summary()
    producer_only = validate_pipeline([_ConditionalScoreProducer()])
    assert "score" not in producer_only.produced_keys, "a conditional write must not become a guaranteed key"
    assert [issue.code for issue in report.issues] == ["conditional_read"]
    assert "score" in report.issues[0].message

    # Nothing upstream even possibly writes the key: still a hard error.
    missing = validate_pipeline([PreserveByValueStage("score", 3.0, "ge")])
    assert not missing.ok
    assert any(issue.code == "unsatisfied_reads" for issue in missing.issues)


def test_conditional_metric_then_selector_chain_composes() -> None:
    """A metric followed by a threshold composes when the metric output is conditional."""
    report = validate_pipeline(
        [_ConditionalScoreProducer(), PreserveByValueStage("score", 25.0, "le")],
        initial_keys={"audio_filepath"},
        initial_roles={"audio_filepath"},
    )
    assert report.ok, report.summary()
    assert report.keys_ok
    assert any(issue.code == "conditional_read" for issue in report.issues)


def test_conditional_tensor_writes_seed_residency_only_when_reachable(tmp_path: Path) -> None:
    """``requires_keys`` decides whether a hydration branch can fire on the seeded input."""
    sink = ManifestWriterStage(output_path=str(tmp_path / "out.jsonl"))

    # Plain manifest: the branch needs ``sample_rate`` upstream, which nothing provides.
    plain = validate_pipeline([_ReachabilityGatedTensorProducer(), sink])
    assert plain.ok, plain.summary()

    # A ``sample_rate`` column makes the branch reachable, so the sink is (correctly) refused.
    with_rate = validate_pipeline(
        [_ReachabilityGatedTensorProducer(), sink],
        initial_keys={"audio_filepath", "sample_rate"},
        initial_roles={"audio_filepath", "sample_rate"},
    )
    assert not with_rate.ok
    assert any(issue.code == "tensor_into_sink" for issue in with_rate.issues)

    # A stage must not make its own branch reachable through the key that branch would write.
    self_enabling = validate_pipeline(
        [_ReachabilityGatedTensorProducer(), _ReachabilityGatedTensorProducer(), sink],
    )
    assert self_enabling.ok, self_enabling.summary()
