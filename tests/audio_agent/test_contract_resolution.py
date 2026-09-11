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

"""``describe`` answers what a stage reads and writes, or says why it cannot.

For the whole life of the verb it did neither. ``describe`` resolved through
``static_contract``, which documents in its own docstring that reads and writes are not
resolved there, so all 49 agent-ready stages reported ``reads: []`` and ``writes: []``.
Nothing labelled that as unknown, so it read as "this stage requires nothing".

One session shows what that costs. A pipeline produced ``diar_segments`` and fed
``SplitASRAlignJoinStage``, whose inner ``SplitLongAudioStage`` requires ``segments``.
``describe`` was called sixteen times and returned blanks sixteen times; the agent concluded
the contracts were uninformative, went to read source code, and wrote throwaway scripts to
work out key names. Validation passed, two models downloaded, a GPU diarization pass ran, and
only then did the splitter refuse to start.

These tests pin the three properties that failure needed: a contract resolves from params, an
unresolvable one is labelled rather than blank, and instantiating a stage to ask is free of
side effects.
"""

from pathlib import Path
from typing import Any

import pytest

from nemo_curator import audio_agent as aa
from nemo_curator.audio_agent import cli, verbs
from nemo_curator.audio_agent._resolve import resolve_stage_class, resolved_contract_for
from nemo_curator.stages.audio._agent._agent_registry import stage_params


def _keys(contract: dict[str, Any], side: str) -> list[str]:
    return list((contract.get(side) or {}).get("data_keys") or [])


class TestADescribeAnswersTheQuestionItIsAskedFor:
    def test_a_stage_reports_the_key_it_writes(self) -> None:
        contract = aa.describe("GetAudioDurationStage")["contract"]

        assert _keys(contract, "writes") == ["duration"]
        assert contract["contract_resolution"] == "configured"

    def test_params_change_the_answer_because_they_change_the_contract(self) -> None:
        """The exact confusion that survived validation and died on the GPU."""
        default = aa.describe("SplitLongAudioStage")["contract"]
        configured = aa.describe("SplitLongAudioStage", {"segments_key": "diar_segments"})["contract"]

        assert "segments" in _keys(default, "reads")
        assert "diar_segments" in _keys(configured, "reads")
        assert "segments" not in _keys(configured, "reads")

    def test_no_agent_ready_stage_answers_with_silence(self) -> None:
        """Either a resolved contract, an expansion, or a stated reason -- never a bare blank.

        The bug was not that some stages were unresolvable; it was that unresolvable and
        "requires nothing" looked identical to the caller. A stage converting between task types
        is the one honest blank: it maps whole tasks rather than named keys, so it has no data
        keys to report and nothing is being hidden.
        """
        mute = []
        for entry in verbs.discover()["stages"]:
            out = aa.describe(entry["stage"])
            contract = out.get("contract") or {}
            if contract.get("accepts_task_type") != contract.get("produces_task_type"):
                continue
            answered = _keys(contract, "reads") or _keys(contract, "writes")
            expanded = (out.get("expands_to") or {}).get("stages")
            explained = out.get("contract_unresolved") or out.get("contract_error")
            if not answered and not expanded and not explained:
                mute.append(entry["stage"])

        assert mute == [], f"stages reporting nothing and not saying why: {mute}"


class TestACompositeReportsWhatItsInnerStagesNeed:
    """A composite's own contract is empty by design, which is how the ALM run died."""

    def test_the_inner_requirement_is_visible_without_running_anything(self) -> None:
        out = aa.describe("SplitASRAlignJoinStage")

        assert "segments" in out["expands_to"]["requires_upstream"]
        assert _keys(out["contract"], "reads") == [], "the composite's own contract is still empty"

    def test_the_requirement_follows_the_params(self) -> None:
        """The whole point: a caller producing ``diar_segments`` can see whether it fits."""
        out = aa.describe("SplitASRAlignJoinStage", {"segments_key": "diar_segments"})
        required = out["expands_to"]["requires_upstream"]

        assert "diar_segments" in required
        assert "segments" not in required

    def test_it_names_the_stages_and_what_they_produce(self) -> None:
        expanded = aa.describe("SplitASRAlignJoinStage")["expands_to"]

        assert "SplitLongAudioStage" in [s["stage"] for s in expanded["stages"]]
        assert "text" in expanded["produces"]

    def test_a_key_produced_inside_is_not_demanded_from_upstream(self) -> None:
        """``split_filepaths`` is made by the first inner stage and read by the second."""
        expanded = aa.describe("SplitASRAlignJoinStage")["expands_to"]

        assert "split_filepaths" in expanded["produces"]
        assert "split_filepaths" not in expanded["requires_upstream"]

    def test_a_plain_stage_is_not_given_an_expansion(self) -> None:
        assert "expands_to" not in aa.describe("GetAudioDurationStage")

    def test_an_inner_stages_alternative_reads_are_not_reported_as_nothing(self) -> None:
        """Many stages put every requirement in ``reads_one_of`` and leave ``reads`` empty.

        Consulting only the flat ``reads`` made such a composite claim it needed nothing from
        upstream -- an empty list presented as an answer, which is the bug this whole change
        exists to remove, one level down.
        """
        expanded = aa.describe("AudioDataFilterStage")["expands_to"]
        one_of = expanded.get("requires_one_of") or []

        assert one_of, "an inner stage reading audio requires SOMETHING from upstream"
        options = one_of[0]["one_of"]
        assert ["audio_filepath"] in options
        # Only ``audio_filepath``: this composite's entry stage is MonoConversionStage, whose
        # ``input_residency`` defaults to "file" because that is all it read before the agent
        # work -- so a waveform alternative here would be a capability main never had. This
        # test's subject is that alternatives are REPORTED rather than collapsed to nothing,
        # which the non-empty ``one_of`` above establishes; that a single stage can offer
        # SEVERAL alternatives is covered where it belongs, on the stage itself:
        # tests/stages/audio/test_agent_planning.py::test_reads_one_of_is_satisfied_by_any_alternative
        # and tests/stages/audio/metrics/test_metrics.py::test_waveform_and_auto_residency.
        assert ["waveform", "sample_rate"] not in options

    def test_alternatives_stay_out_of_the_flat_requirement_list(self) -> None:
        """Demanding both a file path and a waveform would ask for a form nobody needs."""
        expanded = aa.describe("AudioDataFilterStage")["expands_to"]

        assert "waveform" not in expanded["requires_upstream"]


class TestTheCallerLearnsWhenADefaultIsNotTheOnlyOption:
    """A resolved contract answers for one configuration and cannot say it was the only one."""

    def test_residency_alternatives_are_reported_without_being_asked_for(self) -> None:
        varies = {v["param"]: v for v in aa.describe("GetAudioDurationStage")["contract_varies_with"]}
        residency = varies["input_residency"]

        assert residency["current"] == "file"
        by_value = {alt["value"]: alt for alt in residency["changes_it_to"]}
        assert by_value["waveform"]["reads_one_of"] == [["waveform", "sample_rate"]]

    def test_the_alternative_is_what_the_stage_really_does_when_set_that_way(self) -> None:
        """The report is derived by resolving the choice, so it cannot drift from the truth."""
        alt = next(
            item
            for entry in aa.describe("GetAudioDurationStage")["contract_varies_with"]
            if entry["param"] == "input_residency"
            for item in entry["changes_it_to"]
            if item["value"] == "waveform"
        )
        direct = aa.describe("GetAudioDurationStage", {"input_residency": "waveform"})["contract"]

        assert alt["reads_one_of"] == [list(o["data_keys"]) for o in direct["reads_one_of"]]

    def test_a_choice_that_changes_nothing_is_not_mentioned(self) -> None:
        """Restating the parameter list would bury the settings that matter."""
        for entry in aa.describe("BandFilterStage")["contract_varies_with"]:
            for alt in entry["changes_it_to"]:
                keys = (alt.get("reads"), alt.get("writes"), alt.get("reads_one_of"))
                assert any(keys), f"{entry['param']}={alt['value']} listed with no difference"

    def test_a_stage_with_nothing_enumerable_says_nothing(self, tmp_path: Path) -> None:
        out = aa.describe("ManifestWriterStage", {"output_path": str(tmp_path / "o.jsonl")})

        assert "contract_varies_with" not in out


class TestAnUnresolvedContractSaysSoAndSaysWhatWouldFixIt:
    def test_a_stage_needing_arguments_names_them(self) -> None:
        out = aa.describe("ResampleAudioStage")
        detail = out["contract_unresolved"]

        assert "resampled_audio_dir" in detail["required_params"]
        assert "resampled_audio_dir" in detail["retry_with"]
        assert detail["reads_writes_are"] == "unknown, not empty"

    def test_supplying_them_resolves_it(self, tmp_path: Path) -> None:
        out = aa.describe("ResampleAudioStage", {"resampled_audio_dir": str(tmp_path)})

        assert "contract_unresolved" not in out
        assert out["contract"]["contract_resolution"] == "configured"
        assert _keys(out["contract"], "writes")

    def test_a_param_the_stage_does_not_accept_is_reported_not_swallowed(self) -> None:
        out = aa.describe("GetAudioDurationStage", {"no_such_param": 1})
        detail = out["contract_unresolved"]

        assert "no_such_param" in detail["reason"]
        assert "audio_filepath_key" in detail["accepted_params"], "says what it would have taken"

    def test_asr_stage_reports_its_required_adapter_and_model(self) -> None:
        detail = aa.describe("ASRStage")["contract_unresolved"]

        assert detail["required_params"] == ["adapter_target", "model_id"]
        assert "adapter_target" in detail["retry_with"]
        assert "model_id" in detail["retry_with"]

    def test_an_unregistered_stage_is_still_an_error(self) -> None:
        out = aa.describe("NoSuchStageAnywhere")

        assert "not a registered agent-ready audio stage" in out["error"]
        assert "contract" not in out


class TestRecipeParamsCanBePastedInVerbatim:
    def test_execution_knobs_do_not_masquerade_as_a_broken_stage(self) -> None:
        """``resources`` configures how a stage runs, not what it reads.

        A caller holding a recipe stage passes its params as they are. Letting ``resources``
        reach the constructor would answer "this stage could not be configured", blaming the
        stage for a param the recipe layer owns.
        """
        out = aa.describe("GetAudioDurationStage", {"resources": {"gpus": 1}, "batch_size": 8})

        assert "contract_unresolved" not in out
        assert _keys(out["contract"], "writes") == ["duration"]


class TestAskingIsFree:
    def test_no_stage_does_io_when_constructed(self) -> None:
        """The invariant that makes resolution safe, pinned so a future stage cannot break it.

        Resolution constructs stages. That is only acceptable while construction stays inert --
        models load in ``setup()``, which nothing here calls. A stage that opened a file or hit
        the network in ``__init__`` would turn a read-only question into an action, and
        ``describe`` is called during planning, before any approval gate.
        """
        import inspect
        import re

        risky = re.compile(r"\b(?:makedirs|mkdir|rmtree|urlopen|snapshot_download)\b|\bopen\(|requests\.")
        offenders = []
        for entry in verbs.discover()["stages"]:
            cls = resolve_stage_class(entry["stage"])
            init = cls.__dict__.get("__init__")
            if init is None:
                continue
            try:
                source = inspect.getsource(init)
            except (OSError, TypeError):
                continue
            if risky.search(source):
                offenders.append(entry["stage"])

        assert offenders == [], (
            f"__init__ performs I/O in {offenders}; describe() constructs stages to resolve "
            f"their contracts, so construction must stay side-effect free"
        )

    def test_resolution_never_calls_setup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called: list[str] = []
        cls = resolve_stage_class("GetAudioDurationStage")
        monkeypatch.setattr(cls, "setup", lambda *_a, **_k: called.append("setup"), raising=False)

        resolved_contract_for("GetAudioDurationStage")

        assert called == []


class TestProducersAnswersWhoMakesAKey:
    def test_it_finds_a_writer_by_key_name(self) -> None:
        out = aa.producers("duration")

        assert "GetAudioDurationStage" in [p["stage"] for p in out["producers"]]

    def test_it_finds_a_writer_by_role_when_the_key_is_named_differently(self) -> None:
        out = aa.producers("segments")
        by_stage = {p["stage"]: p for p in out["producers"]}

        assert by_stage["WhisperXVADStage"]["matched"] == "role"
        assert by_stage["WhisperXVADStage"]["writes_key"] == "vad_segments"

    def test_a_stage_needing_params_is_a_candidate_not_a_producer(self) -> None:
        """A declared role does not say whether the key is read or written, so it cannot be
        presented as an answer -- only as somewhere to look next."""
        out = aa.producers("pred_text")
        candidates = {c["stage"]: c for c in out.get("candidates", [])}

        assert "ASRStage" in candidates
        assert "ASRStage" not in [p["stage"] for p in out["producers"]]
        assert "describe" in candidates["ASRStage"]["confirm_with"]

    def test_an_incomplete_search_never_presents_as_a_complete_one(self) -> None:
        """ "Nothing produces this" and "some stages could not be asked" must not look alike."""
        out = aa.producers("nothing_writes_this_key")

        assert out["producers"] == []
        assert out["not_searched"], "silence about the unsearched would read as a definitive no"

    def test_a_real_role_is_not_drowned_by_the_unsearched(self) -> None:
        out = aa.producers("duration")

        assert len(out["producers"]) > 1


class TestTheCliCarriesParamsThrough:
    def test_params_reach_the_verb(self, capsys: pytest.CaptureFixture[str]) -> None:
        import json

        assert cli.main(["describe", "SplitLongAudioStage", "--params", '{"segments_key": "diar_segments"}']) == 0
        out = json.loads(capsys.readouterr().out)

        assert "diar_segments" in _keys(out["contract"], "reads")

    def test_a_json_array_is_refused_rather_than_blamed_on_the_stage(self, capsys: pytest.CaptureFixture[str]) -> None:
        cli.main(["describe", "GetAudioDurationStage", "--params", "[1, 2]"])
        out = capsys.readouterr().out

        assert "JSON object" in out

    def test_producers_is_reachable(self, capsys: pytest.CaptureFixture[str]) -> None:
        import json

        assert cli.main(["producers", "duration"]) == 0
        out = json.loads(capsys.readouterr().out)

        assert "GetAudioDurationStage" in [p["stage"] for p in out["producers"]]


class TestTheDeclaredParamsAreConsistentWithWhatConstructionNeeds:
    def test_every_required_param_is_accepted_by_the_constructor(self) -> None:
        """``retry_with`` is only useful if passing those names actually works."""
        broken = []
        for entry in verbs.discover()["stages"]:
            cls = resolve_stage_class(entry["stage"])
            specs = stage_params(cls)
            accepted = {spec.name for spec in specs}
            required = {spec.name for spec in specs if spec.required}
            if not required <= accepted:
                broken.append(entry["stage"])

        assert broken == []


class TestConstructingAStageTouchesNothing:
    """``resolved_contract_for`` instantiates a stage to ask what it reads and writes, and
    ``describe`` is an MCP tool -- so an LLM's params reach a real constructor. The docstring
    justifies that with "no audio stage does I/O in ``__init__`` (a regression test pins that)".

    This is that test. It was missing: the property was asserted in prose and enforced by
    nothing, on the one boundary where tool-supplied data meets ``cls(**params)``.
    """

    def _no_io(self, monkeypatch, seen: list[str]):  # noqa: ANN001, ANN202
        """Make every filesystem/network entry point record itself instead of running."""
        import builtins
        import io as _io
        import os
        import pathlib
        import socket

        def trap(label: str):  # noqa: ANN202
            def _boom(*_a: object, **_k: object) -> None:
                seen.append(label)
                msg = f"{label} during __init__"
                raise AssertionError(msg)

            return _boom

        monkeypatch.setattr(builtins, "open", trap("builtins.open"))
        monkeypatch.setattr(_io, "open", trap("io.open"), raising=False)
        monkeypatch.setattr(os, "makedirs", trap("os.makedirs"))
        monkeypatch.setattr(os, "remove", trap("os.remove"))
        monkeypatch.setattr(pathlib.Path, "mkdir", trap("Path.mkdir"))
        monkeypatch.setattr(pathlib.Path, "write_text", trap("Path.write_text"))
        monkeypatch.setattr(pathlib.Path, "read_text", trap("Path.read_text"))
        monkeypatch.setattr(socket, "socket", trap("socket.socket"))

    # Stages KNOWN to touch disk while being constructed. Listed by name, never by shape, so a
    # new one fails this test instead of joining them quietly.
    #
    # AudioDataFilterStage.__init__ calls load_config(config_path) -- with config_path=None it
    # reads the packaged default_config.yaml, and with a caller-supplied path it reads that.
    # Reachable from the ``describe`` MCP tool, and not covered by the workspace lock (describe
    # performs no path check, and ``config_path`` is a deliberately unlocked shared-dependency
    # param), so an LLM-supplied path becomes an existence/parseability oracle. No file content
    # reaches the response. The stage is shared code and out of the audio agent's scope to change.
    _KNOWN_IO_IN_INIT = frozenset({"AudioDataFilterStage"})

    def test_no_agent_ready_stage_touches_disk_or_network_when_constructed(self, monkeypatch) -> None:  # noqa: ANN001
        offenders: list[str] = []
        skipped: list[str] = []
        built = 0
        for name in aa.discover().get("stages", []):
            stage = str(name.get("stage") or "")
            try:
                cls = resolve_stage_class(stage)
            except Exception:  # noqa: BLE001 - an unimportable optional dep is not this test's subject
                skipped.append(stage)
                continue
            seen: list[str] = []
            with monkeypatch.context() as patched:
                self._no_io(patched, seen)
                try:
                    cls()
                    built += 1
                except AssertionError:  # our own trap -- the stage really did I/O
                    offenders.append(f"{stage}: {seen[-1] if seen else 'io'}")
                except Exception:  # noqa: BLE001 - missing required args / validation: not I/O
                    if seen:
                        offenders.append(f"{stage}: {seen[-1]} (before failing)")
        assert built, f"nothing was constructible, so this proves nothing (skipped: {skipped})"
        unexpected = [o for o in offenders if o.split(":")[0] not in self._KNOWN_IO_IN_INIT]
        assert not unexpected, (
            "stage(s) newly performing I/O in __init__, reachable from the describe MCP tool: " + "; ".join(unexpected)
        )
        still_offending = {o.split(":")[0] for o in offenders}
        assert still_offending == set(self._KNOWN_IO_IN_INIT), (
            "the known-exception list is stale -- these no longer do I/O and should be removed "
            f"from _KNOWN_IO_IN_INIT: {sorted(set(self._KNOWN_IO_IN_INIT) - still_offending)}"
        )
