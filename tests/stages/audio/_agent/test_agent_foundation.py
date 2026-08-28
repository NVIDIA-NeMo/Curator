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

"""Tests for the agent-ready foundation: param derivation, semantic roles,
contract serialization, the discovery catalog, and the conformance gate.

These exercise the shared machinery (``_agent_ready``, ``_agent_registry``,
``_roles``, ``_catalog``, ``_conformance``) without GPUs. The catalog sweep
imports the audio stage modules; optional heavy deps that are absent are skipped
by the catalog rather than failing the test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import ClassVar, Literal

import pytest

from nemo_curator.stages.audio._agent._agent_ready import (
    AgentReady,
    Gates,
    IOSpec,
    ParamSpec,
    StageContract,
    StaticHints,
    to_json_schema,
)
from nemo_curator.stages.audio._agent._agent_registry import (
    _parse_args_section,
    build_contract,
    stage_params,
    static_contract,
)
from nemo_curator.stages.audio._agent._conformance import (
    assert_agent_ready,
    assert_contract_wellformed,
    produced_roles,
    reads_satisfied_by_role,
)


# --------------------------------------------------------------------------- #
# Toy stages (deterministic; no heavy imports)
# --------------------------------------------------------------------------- #
@dataclass
class _ToyDataclassStage(AgentReady):
    """A toy dataclass stage.

    Args:
        audio_filepath_key: Key for the audio path.
        score_key: Where the score is written.
        mode: Operating mode.
        threshold: Minimum score; None disables.
    """

    audio_filepath_key: str = "audio_filepath"
    score_key: str = "utmos_mos"
    mode: Literal["task", "segments", "auto"] = "auto"
    threshold: float | None = 3.5
    name: str = "ToyDataclassStage"  # framework field -> excluded from params

    def describe(self) -> StageContract:
        return StageContract(
            reads=IOSpec(data_keys=[self.audio_filepath_key], accepts=["file"]),
            writes=IOSpec(data_keys=[self.score_key]),
            cardinality="1:1",
        )


class _ToyInitStage(AgentReady):
    """A toy stage that uses ``__init__`` (not a dataclass).

    Args:
        input_value_key: Field to evaluate.
        target_value: Value to compare against.
        operator: Comparison operator.
    """

    def __init__(self, input_value_key: str, target_value: int, operator: str = "eq"):
        self.input_value_key = input_value_key
        self.target_value = target_value
        self.operator = operator

    def describe(self) -> StageContract:
        return StageContract(reads=IOSpec(data_keys=[self.input_value_key]), cardinality="filter")


# --------------------------------------------------------------------------- #
# Param derivation
# --------------------------------------------------------------------------- #
def test_stage_params_dataclass_choices_required_roles_descriptions():  # noqa: ANN202
    params = {p.name: p for p in stage_params(_ToyDataclassStage)}
    assert "name" not in params  # framework field excluded
    assert params["mode"].choices == ["task", "segments", "auto"]
    assert params["mode"].type == "str"
    assert params["threshold"].type == "float | None"
    assert params["threshold"].default == 3.5
    assert all(not p.required for p in params.values())
    assert params["audio_filepath_key"].role == "audio_filepath"
    assert params["score_key"].role == "score"
    assert params["audio_filepath_key"].description == "Key for the audio path."
    assert params["threshold"].description == "Minimum score; None disables."


def test_stage_params_init_signature_required_and_defaults():  # noqa: ANN202
    params = {p.name: p for p in stage_params(_ToyInitStage)}
    assert params["input_value_key"].required is True
    assert params["target_value"].required is True
    assert params["operator"].required is False
    assert params["operator"].default == "eq"
    assert params["input_value_key"].description == "Field to evaluate."


def test_docstring_args_parser_handles_continuations_and_sections():  # noqa: ANN202
    doc = """Summary.

    Args:
        a: first.
        b: spans
            two lines.

    Returns:
        not a param.
    """
    parsed = _parse_args_section(doc)
    assert parsed["a"] == "first."
    assert parsed["b"] == "spans two lines."
    assert "not a param" not in str(parsed)


# --------------------------------------------------------------------------- #
# Contract assembly + serialization
# --------------------------------------------------------------------------- #
def test_build_contract_fills_params_and_key_roles():  # noqa: ANN202
    contract = build_contract(_ToyDataclassStage(score_key="renamed_score"))
    assert contract.contract_resolution == "configured"
    assert contract.params, "params should be auto-derived"
    assert contract.key_roles.get("renamed_score") == "score"
    assert contract.key_roles.get("audio_filepath") == "audio_filepath"
    assert contract.stage_id == "_ToyDataclassStage"


def test_static_contract_is_instance_free_for_required_arg_stage():  # noqa: ANN202
    contract = static_contract(_ToyInitStage)  # no instantiation needed
    assert contract.contract_resolution == "static_params_and_hints"
    names = {p.name for p in contract.params}
    assert {"input_value_key", "target_value", "operator"} <= names
    assert contract.stage_id == "_ToyInitStage"


def test_to_dict_is_json_safe_even_with_nonserializable_default():  # noqa: ANN202
    class _Weird:
        pass

    contract = StageContract(params=[ParamSpec(name="x", default=_Weird(), choices=["a"], role="score")])
    payload = contract.to_dict()
    json.dumps(payload)  # must not raise
    assert payload["params"][0]["default"].startswith("<non-serializable")


def test_to_json_schema_maps_types_enums_and_required():  # noqa: ANN202
    schema = to_json_schema(
        [
            ParamSpec(name="mode", type="str", choices=["a", "b"], default="a"),
            ParamSpec(name="path", type="str", required=True),
            ParamSpec(name="n", type="int", default=1),
        ]
    )
    assert schema["type"] == "object"
    assert schema["properties"]["mode"]["enum"] == ["a", "b"]
    assert schema["properties"]["n"]["type"] == "integer"
    assert schema["required"] == ["path"]


# --------------------------------------------------------------------------- #
# By-role matching
# --------------------------------------------------------------------------- #
def test_reads_satisfied_by_role_survives_key_rename():  # noqa: ANN202
    # Producer writes a score under a renamed key value.
    producer = build_contract(_ToyDataclassStage(score_key="model_b_mos"))
    # Consumer reads a score under a *different* key value.
    consumer = StageContract(
        reads=IOSpec(data_keys=["some_other_name"]),
        key_roles={"some_other_name": "score"},
    )
    assert "score" in produced_roles(producer)
    assert reads_satisfied_by_role(consumer, produced_roles(producer)) is True
    # A consumer needing 'waveform' is not satisfied by a 'score'-only producer.
    needs_waveform = StageContract(reads=IOSpec(data_keys=["wf"]), key_roles={"wf": "waveform"})
    assert reads_satisfied_by_role(needs_waveform, produced_roles(producer)) is False


def test_assert_agent_ready_static_and_dynamic_on_toy_stage():  # noqa: ANN202
    contract = assert_agent_ready(
        _ToyDataclassStage(),
        fixture_factory=None,  # static-only (no execution)
        expected_cardinality="1:1",
        available_keys={"audio_filepath"},
    )
    assert contract.cardinality == "1:1"


# --------------------------------------------------------------------------- #
# Catalog + full-stack static conformance sweep
# --------------------------------------------------------------------------- #
def test_catalog_discovers_stages_and_round_trips_json():  # noqa: ANN202
    from nemo_curator.stages.audio._agent._catalog import audio_stage_catalog, catalog_as_json, list_agent_ready_stages

    names = list_agent_ready_stages()
    assert len(names) >= 30, f"expected the audio catalog to discover many stages, got {len(names)}"
    payload = catalog_as_json()
    parsed = json.loads(payload)  # must round-trip
    assert len(parsed) == len(names)
    # representative stages are present
    assert "MonoConversionStage" in names
    assert "UTMOSFilterStage" in names
    # every entry carries a contract dict
    assert all("contract" in e and isinstance(e["contract"], dict) for e in audio_stage_catalog())


def test_all_agent_ready_stages_pass_static_conformance():  # noqa: ANN202
    from nemo_curator.stages.audio._agent._catalog import get_agent_ready_stage_class, list_agent_ready_stages

    failures = []
    names = list_agent_ready_stages()
    for name in names:
        cls = get_agent_ready_stage_class(name)
        try:
            assert_contract_wellformed(cls)  # shape + roles + serialization, instance-free
        except Exception as e:  # noqa: BLE001
            failures.append(f"{name}: {e}")
    assert not failures, "static conformance failures:\n" + "\n".join(failures)


def test_static_hints_are_optional_and_additive():  # noqa: ANN202
    # A stage with no AGENT_STATIC still yields a valid static contract.
    assert static_contract(_ToyDataclassStage).error_policy == "unknown"

    @dataclass
    class _Hinted(_ToyDataclassStage):
        AGENT_STATIC = StaticHints(error_policy="skip", gates=Gates(requires_gpu=True))

    contract = static_contract(_Hinted)
    assert contract.error_policy == "skip"
    assert contract.gates.requires_gpu is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


class TestStagesDeclareTheirOwnAgentFacts:
    """Adding a stage must not mean editing a central table in the agent.

    Two facts used to live in agent-side tables that every new stage had to be added to:
    which params hold a writer's output paths (``verbs._SMOKE_DISK_ADAPTERS``) and whether
    a ``*_key`` field is bookkeeping (``_roles.INTERNAL_KEY_FIELDS``). Both are facts only
    the stage knows, so both are now declared by the stage.
    """

    # The exact mapping the deleted verbs._SMOKE_DISK_ADAPTERS table held. Pinned so a
    # refactor that silently drops or widens a redirect is caught, not just noticed.
    EXPECTED_OUTPUT_PATH_PARAMS: ClassVar[dict[str, list[str]]] = {
        "CreateInitialManifestReadSpeechStage": [],
        "DocumentBatchJsonlWriterStage": ["output_path"],
        "InferenceSortformerStage": ["rttm_out_dir"],
        "ManifestCheckpointStage": ["output_path"],
        "ManifestGroupExportStage": ["output_dir"],
        "ManifestWriterStage": ["output_path"],
        "MonoConversionStage": ["output_dir"],
        "PretrainMetricsAggregatorStage": ["output_path"],
        "ResampleAudioStage": ["resampled_audio_dir"],
        "SegmentConcatenationStage": ["output_dir"],
        "SegmentExtractionStage": ["output_dir"],
        "SnippetExtractionStage": ["output_dir", "output_audio_tar_path"],
        "SnippetManifestWriterStage": ["output_path"],
        "SpeakerSeparationStage": ["separated_audio_dir"],
        "SplitLongAudioStage": ["output_dir"],
    }

    _REQUIRED_DIRS: ClassVar[set[str]] = {"output_dir", "separated_audio_dir", "resampled_audio_dir", "rttm_out_dir"}

    def _configured(self, cls: type) -> object:
        """A disk-writing instance, supplying whatever the constructor demands."""
        from dataclasses import MISSING, fields

        names = {f.name for f in fields(cls) if f.init}
        kwargs = {
            f.name: ("/tmp/x" if ("path" in f.name or "dir" in f.name) else "x")  # noqa: S108
            for f in fields(cls)
            if f.init and f.default is MISSING and f.default_factory is MISSING
        }
        if "write_to_disk" in names:
            kwargs["write_to_disk"] = True
        for name in self._REQUIRED_DIRS & names:
            kwargs.setdefault(name, "/tmp/x")  # noqa: S108
        return cls(**kwargs)

    def test_every_writer_declares_exactly_the_params_the_table_used_to_hold(self) -> None:
        from nemo_curator.stages.audio._agent._catalog import get_agent_ready_stage_class

        declared = {
            name: list(self._configured(get_agent_ready_stage_class(name)).describe().gates.output_path_params or [])
            for name in self.EXPECTED_OUTPUT_PATH_PARAMS
        }
        assert declared == self.EXPECTED_OUTPUT_PATH_PARAMS

    def test_an_undeclared_writer_still_fails_closed(self) -> None:
        """The safety property, not the table, is what mattered.

        A stage claiming writes_to_disk without naming its outputs cannot be sandboxed.
        Guessing which params look path-like would risk a smoke writing into the caller's
        real output tree, so ``None`` must stay distinguishable from ``[]``.
        """
        from nemo_curator.stages.audio._agent._agent_ready import Gates

        assert Gates(writes_to_disk=True).output_path_params is None, "undeclared is not empty"
        assert Gates(writes_to_disk=True, output_path_params=[]).output_path_params == []

    def test_a_stage_can_declare_a_bookkeeping_key_without_touching_the_shared_table(self) -> None:
        from nemo_curator.stages.audio._agent._roles import INTERNAL_KEY_FIELDS, field_has_declared_role
        from nemo_curator.stages.audio.preprocessing import ChannelCountStage

        assert "num_channels_key" not in INTERNAL_KEY_FIELDS, "not in the central table"
        assert "num_channels_key" in ChannelCountStage.INTERNAL_KEY_FIELDS
        assert field_has_declared_role("num_channels_key", ChannelCountStage)
        # ...and it is still undeclared for a stage that did not claim it.
        assert not field_has_declared_role("num_channels_key")

    def test_a_subclass_declaring_its_own_key_does_not_disown_its_parents(self) -> None:
        """``getattr`` returns only the most-derived declaration.

        A subclass that declares one internal field of its own therefore shadowed every field
        its parent declared, and those fields -- still inherited, still bookkeeping -- began
        failing the conformance check as though a role had been forgotten for them. The fix a
        subclass author would reach for is to re-list keys they did not write, which is how a
        shared table gets copied downwards one subclass at a time.
        """
        from nemo_curator.stages.audio._agent._roles import field_has_declared_role
        from nemo_curator.stages.audio.preprocessing import ChannelCountStage

        class NarrowerChannelCount(ChannelCountStage):
            INTERNAL_KEY_FIELDS = frozenset({"my_own_key"})

        assert field_has_declared_role("my_own_key", NarrowerChannelCount)
        assert field_has_declared_role("num_channels_key", NarrowerChannelCount)

    def test_an_undeclared_key_is_still_caught(self) -> None:
        """The gate must keep catching a genuinely forgotten role mapping."""
        from nemo_curator.stages.audio._agent._roles import field_has_declared_role
        from nemo_curator.stages.audio.preprocessing import ChannelCountStage

        assert not field_has_declared_role("invented_thing_key", ChannelCountStage)


class TestEveryStageSaysWhetherItsRowsStandAlone:
    """A delta run must never be refused merely because a stage author forgot ``Gates``.

    ``audio_agent.delta.region`` ends the reusable region at the first stage whose
    ``gates.per_row_independent`` is undeclared and which could reach another row at all. It
    cannot tell a forgotten field from a deliberate "this stage reads the whole corpus", so it
    conservatively stops at both -- and a single silent omission in an ingest stage therefore
    costs every stage behind it its incremental run. These tests keep ``None`` meaning only
    that nobody has looked yet.
    """

    # The stages whose honest answer is "no", each naming the corpus-wide quantity in its
    # describe(). Pinned so that a flip to True has to be argued for, and so a newly
    # corpus-dependent stage cannot join the set unremarked.
    EXPECTED_CORPUS_DEPENDENT: ClassVar[set[str]] = {
        "ManifestGroupExportStage",
        "PretrainMetricsAggregatorStage",
        "PyAnnoteDiarizationStage",
        "SegmentExtractionStage",
        "TorchSquimQualityMetricsStage",
    }

    @staticmethod
    def _shipped_stage_names() -> list[str]:
        """The stages Curator itself ships.

        The registry is open by design -- any imported agent-ready subclass registers itself,
        which is what lets a user extend the catalog. The toy stages above would otherwise be
        asserted about as though Curator shipped them.
        """
        from nemo_curator.stages.audio._agent._catalog import get_agent_ready_stage_class, list_agent_ready_stages

        return sorted(
            name
            for name in list_agent_ready_stages()
            if getattr(get_agent_ready_stage_class(name), "__module__", "").startswith("nemo_curator.")
        )

    @staticmethod
    def _at_its_defaults(stage_cls: type) -> object:
        """The stage as shipped, supplying only what its constructor refuses to go without.

        The defaults are the point: several stages answer this gate per instance rather than per
        class -- ``SplitLongAudioStage`` is independent only while no shared ``output_dir``
        flattens every source's splits into one namespace -- so filling in optional params would
        measure a configuration nobody runs.
        """
        import inspect
        from dataclasses import MISSING, fields, is_dataclass

        try:
            return stage_cls()
        except (TypeError, ValueError):
            pass

        def placeholder(name: str) -> object:
            if name == "conditions":
                return [{"input_value_key": "score", "target_value": 0.0, "operator": "ge"}]
            return "/tmp/x" if ("dir" in name or "path" in name) else "x"  # noqa: S108

        if is_dataclass(stage_cls):
            # An empty-string default is how these stages spell "required": the field exists so
            # the dataclass stays keyword-constructible, and __post_init__ rejects the blank.
            demanded = {
                item.name: placeholder(item.name)
                for item in fields(stage_cls)
                if item.init
                and not item.name.startswith("_")
                and ((item.default is MISSING and item.default_factory is MISSING) or item.default == "")
            }
        else:
            demanded = {
                name: placeholder(name)
                for name, parameter in inspect.signature(stage_cls).parameters.items()
                if parameter.default is inspect.Parameter.empty
            }
        return stage_cls(**demanded)

    def _declared_gate(self, name: str) -> bool | None:
        from nemo_curator.stages.audio._agent._catalog import get_agent_ready_stage_class

        contract = build_contract(self._at_its_defaults(get_agent_ready_stage_class(name)))
        return contract.gates.per_row_independent

    def test_no_shipped_audio_stage_leaves_per_row_independence_undeclared(self) -> None:
        """An omission must not be what ends a delta region.

        A real session lost a one-file delta across a whole pipeline because its ingest stage
        wrote to disk and had never declared this gate, so ``region`` stopped at stage 0.
        """
        undeclared = [name for name in self._shipped_stage_names() if self._declared_gate(name) is None]

        assert not undeclared, (
            "each of these would end a delta region by omission rather than by fact; declare "
            f"gates.per_row_independent in describe(): {undeclared}"
        )

    def test_a_corpus_dependent_stage_declares_false_instead_of_staying_silent(self) -> None:
        """Leaving a genuinely corpus-dependent stage undeclared refuses for the wrong reason.

        Undeclared is only refused once ``_can_see_other_rows`` finds a channel, so silence is
        both weaker than the truth and indistinguishable from an unaudited stage -- the next
        reader would have to re-derive from ``process()`` which of the two it was.
        """
        corpus_dependent = {name for name in self._shipped_stage_names() if self._declared_gate(name) is False}

        assert corpus_dependent == self.EXPECTED_CORPUS_DEPENDENT

    def test_the_gate_is_resolved_through_describe_and_not_the_instance_free_contract(self) -> None:
        """The sweep above must not be rebuilt on ``static_contract``.

        ``static_contract`` takes its gates straight from ``AGENT_STATIC`` and never calls
        ``describe()``, so it reports ``None`` even for stages that do declare the gate. An
        invariant asserted on that view would either fail for the whole catalog or, inverted,
        pass forever while a real omission went on refusing deltas. ``delta.region`` resolves
        through ``build_contract`` on live stages, so this test does too.
        """
        from nemo_curator.stages.audio.filtering.sigmos import SIGMOSFilterStage

        assert static_contract(SIGMOSFilterStage).gates.per_row_independent is None
        assert build_contract(SIGMOSFilterStage()).gates.per_row_independent is True
