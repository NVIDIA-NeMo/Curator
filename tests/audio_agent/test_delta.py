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

"""Delta reuse: which files changed, how deep per-file work is independent, and the merge.

The property under test throughout is that a delta either produces what a full run would
have produced, or refuses and says which file or stage stopped it. See
``nemo_curator/audio_agent/REUSE_ARCHITECTURE.md`` §7.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import pytest

from nemo_curator.audio_agent import artifacts, delta, profiler
from nemo_curator.audio_agent.recipe import Recipe
from nemo_curator.stages.audio._agent._agent_ready import AgentReady, StageContract
from nemo_curator.stages.base import _STAGE_REGISTRY, ProcessingStage
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_KEY = "dataset-key-now"
_PRIOR = "dataset-key-before"


@pytest.fixture(autouse=True)
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated artifact/run store so tests never touch the developer's real history."""
    monkeypatch.setenv("AUDIO_AGENT_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


def _corpus(root: Path, names: tuple[str, ...]) -> dict[str, str]:
    """A folder of files and the inventory the profiler computes for it."""
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).write_bytes(b"\0" * 16)
    return {name: f"16|{i}" for i, name in enumerate(names)}


def _pipeline(tmp_path: Path, source: Path) -> tuple[Recipe, str]:
    """reader -> duration -> writer: every stage declares per-row independence."""
    out = str(tmp_path / "out.jsonl")
    rec = Recipe.from_dict(
        {
            "stages": [
                {"ref": "ManifestReader", "params": {"manifest_path": str(source)}},
                {"ref": "GetAudioDurationStage", "params": {}},
                {"ref": "ManifestWriterStage", "params": {"output_path": out}},
            ]
        }
    ).freeze()
    return rec, out


def _publish(
    rec: Recipe,
    index: int,
    *,
    dataset_key: str,
    rows: list[dict[str, Any]],
    coverage: dict[str, str] | None,
    **kw: Any,  # noqa: ANN401
) -> artifacts.Artifact:
    """Publish one step of ``rec`` with real output rows and (optionally) its coverage.

    A step whose output is a DIRECTORY (resample, split) gets the directory instead of rows --
    those artifacts are real and a delta has to reason about them, but there is no manifest to
    write into.
    """
    plan = artifacts.plan_steps(rec, dataset_key)[index]
    if os.path.isdir(plan.uri):
        for row in rows:
            name = os.path.basename(str(row.get("audio_filepath") or "row")) or "row"
            open(os.path.join(plan.uri, name), "w", encoding="utf-8").close()
    else:
        with open(plan.uri, "w", encoding="utf-8") as handle:
            handle.writelines(json.dumps(row) + "\n" for row in rows)
    art = artifacts.publish(
        artifacts.Artifact(
            step_key=plan.step_key,
            input_key=plan.input_key,
            stage_ref=plan.stage_ref,
            stage_index=plan.index,
            semantic_params=plan.semantic_params,
            uri=plan.uri,
            kind=plan.kind,
            dataset_key=dataset_key,
            fingerprint_tier="stat",
            impl_version=plan.impl_version,
            code_version=artifacts.code_version(),
            deterministic=plan.deterministic,
            cumulative_sec=kw.pop("cumulative_sec", 600.0),
            covers_files=len(coverage or {}),
            **kw,
        )
    )
    if coverage is not None:
        artifacts.save_coverage(art.step_key, coverage)
    return art


class TestNarrowingIsNotInvisible:
    """``include_files`` must reach the step key, or a partial manifest passes as a whole one.

    Nothing else stops it: a narrowed run reads the same folder, so it carries the same dataset
    key, and it writes the user's manifest path. Were the param filtered out of the semantic
    identity -- it reads like an execution knob, and the two frozensets that do that filtering
    are a plausible place to put it -- a one-file run would publish under the key the full
    pipeline probes, and the next scan would answer ``already_done`` from a manifest holding one
    row of a thousand. The delta relies on this, which is why it is asserted here rather than
    left to whoever next tidies those lists.
    """

    def test_a_narrowed_source_does_not_share_the_full_runs_step_keys(self, tmp_path: Path) -> None:
        source = tmp_path / "m.jsonl"
        source.write_text(json.dumps({"audio_filepath": str(tmp_path / "a.wav")}) + "\n")
        whole, _ = _pipeline(tmp_path, source)
        narrowed = Recipe.from_dict(
            {
                "stages": [
                    {
                        "ref": "ManifestReader",
                        "params": {"manifest_path": str(source), "include_files": [str(tmp_path / "a.wav")]},
                    },
                    *[s.to_dict() for s in whole.stages[1:]],
                ]
            }
        ).freeze()

        mine = artifacts.step_keys(narrowed, _KEY)
        theirs = artifacts.step_keys(whole, _KEY)
        assert mine[0] != theirs[0], "include_files must be part of the source stage's semantic identity"
        assert not set(mine) & set(theirs), "every key below the narrowed source must differ too"

    def test_the_source_is_told_which_column_the_inventory_paths_came_from(self, tmp_path: Path) -> None:
        """Narrowing hands a source paths; it only selects the right rows if both agree on a column.

        They agree by default, which is what made this invisible. Point a manifest's audio column
        somewhere else and the reader matches the inventory's paths against values that are not
        paths, selects nothing, and the delta reports a successful run over zero rows.
        """
        audio = tmp_path / "audio"
        _corpus(audio, ("a.wav", "b.wav"))
        manifest = tmp_path / "m.jsonl"
        manifest.write_text("".join(json.dumps({"wav_path": str(audio / n)}) + "\n" for n in ("a.wav", "b.wav")))

        prof = profiler.profile_data(str(manifest), audio_filepath_key="wav_path")
        assert prof.inventory_key == "wav_path", "the profiler must record the column it indexed"

        rec = Recipe.from_dict(
            {
                "stages": [
                    {"ref": "ManifestReader", "params": {"manifest_path": str(manifest)}},
                    {"ref": "GetAudioDurationStage", "params": {}},
                ]
            }
        ).freeze()
        built, _, err = delta.prefix_recipe(
            rec,
            prefix=2,
            files=(str(audio / "a.wav"),),
            sandbox=str(tmp_path / "sandbox"),
            sinks_=[],
            inventory_key=prof.inventory_key,
        )
        assert built is not None, err
        assert built.stages[0].params["include_files_key"] == "wav_path"

    def test_a_folder_source_is_not_given_a_column_it_does_not_have(self, tmp_path: Path) -> None:
        """A folder scan compares the files themselves, so there is no column to agree on."""
        rec = Recipe.from_dict(
            {
                "stages": [
                    {"ref": "CreateInitialManifestAudioFolderStage", "params": {"data_dir": str(tmp_path)}},
                    {"ref": "GetAudioDurationStage", "params": {}},
                ]
            }
        ).freeze()
        built, _, err = delta.prefix_recipe(
            rec, prefix=2, files=(str(tmp_path / "a.wav"),), sandbox=str(tmp_path / "s"), sinks_=[], inventory_key=""
        )
        assert built is not None, err
        assert "include_files_key" not in built.stages[0].params

    def test_a_narrowable_source_answers_whether_narrowing_it_is_safe(self) -> None:
        """The conformance rule is "must not be silent", not "must be True"."""
        from nemo_curator.stages.audio._agent._conformance import assert_contract_wellformed
        from nemo_curator.stages.audio.common import CreateInitialManifestAudioFolderStage, ManifestReaderStage

        for cls in (CreateInitialManifestAudioFolderStage, ManifestReaderStage):
            contract = assert_contract_wellformed(cls)
            assert contract.gates.per_row_independent is not None, cls.__name__

    def test_a_bounded_source_is_still_declared_narrowable_by_an_accepted_decision(self) -> None:
        """A bounded ``max_samples`` does not stop the region, by decision rather than by proof.

        ``max_samples`` truncates the SORTED listing, so which files are selected is a fact about
        the whole folder: a delta enumerating only the changed files takes the first N of its own
        listing and can admit files a full run would never have selected, silently. This stage
        declared ``False`` while bounded for exactly that reason, and the region stopped at it --
        which for a SOURCE means a full run every time anyone sets the parameter. That cost was
        weighed against the unsoundness and reuse was chosen, so the declaration is now a flat
        ``True``. The assertion is inverted deliberately; it is not that the hazard went away.
        """
        from nemo_curator.stages.audio._agent._conformance import assert_contract_wellformed
        from nemo_curator.stages.audio.common import CreateInitialManifestAudioFolderStage as Folder

        bounded = assert_contract_wellformed(Folder(data_dir="/tmp/x", max_samples=10))  # noqa: S108
        assert bounded.gates.per_row_independent is True
        assert assert_contract_wellformed(Folder(data_dir="/tmp/x")).gates.per_row_independent is True  # noqa: S108

        rec = Recipe.from_dict(
            {
                "stages": [
                    {
                        "ref": "CreateInitialManifestAudioFolderStage",
                        "params": {"data_dir": "/tmp/x", "max_samples": 10},  # noqa: S108
                    },
                    {"ref": "GetAudioDurationStage", "params": {}},
                ]
            }
        ).freeze()
        assert delta.region(rec, upto=2) == (2, "")


class TestInventory:
    """The profiler records WHICH files back the dataset key, not just their digest."""

    def test_a_folder_scan_remembers_every_file(self, tmp_path: Path) -> None:
        _corpus(tmp_path / "audio", ("a.wav", "b.wav"))
        prof = profiler.profile_data(str(tmp_path / "audio"), folder_extensions=[".wav"])
        assert prof.fingerprint_tier == "stat"
        assert sorted(prof.inventory) == ["a.wav", "b.wav"]
        assert prof.inventory_root == str(tmp_path / "audio")

    def test_the_inventory_stays_out_of_the_serialized_profile(self, tmp_path: Path) -> None:
        """It rides on every record and report that carries a profile; a corpus would bury them."""
        _corpus(tmp_path / "audio", ("a.wav",))
        prof = profiler.profile_data(str(tmp_path / "audio"), folder_extensions=[".wav"])
        assert prof.inventory
        assert "inventory" not in prof.to_dict()

    def test_an_edited_manifest_row_moves_that_files_token(self, tmp_path: Path) -> None:
        """A corrected transcript with the wav untouched is a change to that file's work."""
        _corpus(tmp_path / "audio", ("a.wav", "b.wav"))
        manifest = tmp_path / "m.jsonl"
        rows = [{"audio_filepath": str(tmp_path / "audio" / n), "text": "hi"} for n in ("a.wav", "b.wav")]
        manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
        before = profiler.profile_data(str(manifest)).inventory

        rows[0]["text"] = "bye"
        manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
        after = profiler.profile_data(str(manifest)).inventory

        # Keys are relative to the manifest's directory, which is what holds the audio folder.
        assert before["audio/a.wav"] != after["audio/a.wav"]
        assert before["audio/b.wav"] == after["audio/b.wav"]

    def test_an_incomplete_scan_records_nothing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A partial inventory would report unstattable files as removed and delete their rows."""
        _corpus(tmp_path / "audio", ("a.wav", "b.wav"))
        real = profiler.os.stat

        def flaky(path: str, *a: Any, **kw: Any) -> Any:  # noqa: ANN401
            if str(path).endswith("b.wav"):
                msg = "no"
                raise OSError(msg)
            return real(path, *a, **kw)

        monkeypatch.setattr(profiler.os, "stat", flaky)
        prof = profiler.profile_data(str(tmp_path / "audio"), folder_extensions=[".wav"])
        assert prof.fingerprint_tier == "shape"
        assert prof.inventory == {}

    def test_a_corpus_past_the_cap_keeps_the_key_and_declines_the_inventory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(profiler, "_MAX_INVENTORY", 1)
        _corpus(tmp_path / "audio", ("a.wav", "b.wav"))
        prof = profiler.profile_data(str(tmp_path / "audio"), folder_extensions=[".wav"])
        assert prof.fingerprint_tier == "stat"  # reuse is unaffected
        assert prof.inventory == {}
        assert any("delta" in note for note in prof.notes)


class TestClassify:
    """Comparing two inventories names the files, and refuses to guess when one is missing."""

    def test_a_new_file_is_added_only(self) -> None:
        change = delta.classify({"a": "1", "b": "1"}, {"a": "1", "b": "1", "c": "1"})
        assert change is not None
        assert change.kind == "added_only"
        assert change.added == ("c",)
        assert change.touched == ("c",)
        assert change.stale == ()

    def test_an_edited_file_is_stale_and_must_be_rerun(self) -> None:
        change = delta.classify({"a": "1", "b": "1"}, {"a": "2", "b": "1"})
        assert change is not None
        assert change.kind == "changed"
        assert change.modified == ("a",)
        assert change.touched == ("a",)  # rerun it
        assert change.stale == ("a",)  # and drop what it produced before

    def test_a_deleted_file_is_stale_but_not_rerun(self) -> None:
        change = delta.classify({"a": "1", "b": "1"}, {"a": "1"})
        assert change is not None
        assert change.kind == "removed"
        assert change.touched == ()
        assert change.stale == ("b",)

    def test_the_same_files_are_identical(self) -> None:
        change = delta.classify({"a": "1"}, {"a": "1"})
        assert change is not None
        assert change.kind == "identical"

    def test_two_corpora_sharing_no_file_are_not_one_that_changed(self) -> None:
        """Subtracting one dataset from another would drop every row and call it incremental."""
        change = delta.classify({"a": "1"}, {"z": "1"})
        assert change is not None
        assert change.kind == "unrelated"

    def test_an_unrecorded_inventory_is_not_an_empty_one(self) -> None:
        assert delta.classify(None, {"a": "1"}) is None
        assert delta.classify({"a": "1"}, None) is None


class ForeignBatchStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """A stage from outside this repo that never declared the gate, and is handed whole batches.

    Every stage Curator ships now declares ``per_row_independent`` -- pinned by
    ``tests/stages/audio/test_agent_foundation.py`` so a delta can no longer be refused over an
    omission. The derivation for an UNDECLARED stage still has to be exercised, and the case it
    exists for is precisely a stage the audit does not reach: a user's own, or one written before
    the gate was.
    """

    name: str = "ForeignBatchStage"

    def describe(self) -> StageContract:
        return StageContract(cardinality="1:1")

    def process(self, task: AudioTask) -> AudioTask:
        raise NotImplementedError

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        return tasks


class ForeignPerRowStage(AgentReady, ProcessingStage[AudioTask, AudioTask]):
    """The same, one task per call: no batch, nothing kept between calls, no file written."""

    name: str = "ForeignPerRowStage"

    def describe(self) -> StageContract:
        return StageContract(cardinality="1:1")

    def process(self, task: AudioTask) -> AudioTask:
        return task


# ``StageMeta`` registers every ``ProcessingStage`` subclass at class-definition time, so simply
# importing this module would publish these two into the catalog every other test reads -- and
# they are deliberately under-declared, so a sweep like
# ``test_contract_resolution.py::test_no_agent_ready_stage_answers_with_silence`` then reports
# them as stages that describe nothing. They cannot be dropped outright either: the tests below
# name them in a recipe ``ref``, which resolves through that same registry. So: out by default,
# back in for the length of the test that needs them.
_STAGE_REGISTRY.pop("ForeignBatchStage", None)
_STAGE_REGISTRY.pop("ForeignPerRowStage", None)


@pytest.fixture
def foreign_stages() -> Iterator[None]:
    """Publish the two out-of-repo doubles for one test, then withdraw them."""
    _STAGE_REGISTRY["ForeignBatchStage"] = ForeignBatchStage
    _STAGE_REGISTRY["ForeignPerRowStage"] = ForeignPerRowStage
    try:
        yield
    finally:
        _STAGE_REGISTRY.pop("ForeignBatchStage", None)
        _STAGE_REGISTRY.pop("ForeignPerRowStage", None)


class TestRegion:
    """How deep per-file work stays independent, read from what stages declare."""

    def test_a_per_row_pipeline_is_traceable_end_to_end(self, tmp_path: Path) -> None:
        rec, _ = _pipeline(tmp_path, tmp_path / "m.jsonl")
        depth, reason = delta.region(rec, upto=3)
        assert (depth, reason) == (3, "")

    def test_a_corpus_statistic_stage_ends_the_region_and_is_named(self, tmp_path: Path) -> None:
        rec = Recipe.from_dict(
            {
                "stages": [
                    {"ref": "ManifestReader", "params": {"manifest_path": str(tmp_path / "m.jsonl")}},
                    {"ref": "PretrainMetricsAggregatorStage", "params": {"output_path": str(tmp_path / "s.json")}},
                ]
            }
        ).freeze()
        depth, reason = delta.region(rec, upto=2)
        assert depth == 1
        assert "PretrainMetricsAggregator" in reason
        assert "depends on the other rows" in reason

    def test_an_undeclared_batch_stage_ends_the_region_and_says_which_channel_is_open(
        self, tmp_path: Path, foreign_stages: None
    ) -> None:
        """A stage nobody annotated still gets a definite answer, and one that names its reason.

        ``ForeignBatchStage`` declares nothing, but it is handed several rows per call, so
        whether a row's result depends on the batch it landed in cannot be established from the
        outside -- and the refusal says exactly that rather than "undeclared".
        """
        rec = Recipe.from_dict(
            {
                "stages": [
                    {"ref": "ManifestReader", "params": {"manifest_path": str(tmp_path / "m.jsonl")}},
                    {"ref": "ForeignBatchStage", "params": {}},
                ]
            }
        ).freeze()
        depth, reason = delta.region(rec, upto=2)
        assert depth == 1
        assert "ForeignBatchStage" in reason
        assert "several rows at once" in reason

    def test_squim_refuses_by_its_own_declaration_not_by_the_derivation(self, tmp_path: Path) -> None:
        """The counterpart to the ASR test: same shape, opposite answer, and stated outright.

        SQUIM zero-pads each batch to its longest member and calls the model with no lengths, so
        a clip's score moves with the clips beside it. The derivation would refuse it anyway for
        taking a batch, but that is incidental -- a refactor of the batching would flip it -- so
        the stage says so itself.
        """
        rec = Recipe.from_dict(
            {
                "stages": [
                    {"ref": "ManifestReader", "params": {"manifest_path": str(tmp_path / "m.jsonl")}},
                    {"ref": "TorchSquimQualityMetricsStage", "params": {}},
                ]
            }
        ).freeze()
        depth, reason = delta.region(rec, upto=2)
        assert depth == 1
        assert "depends on the other rows" in reason

    def test_the_asr_pipeline_the_feature_exists_for_is_traceable_end_to_end(self, tmp_path: Path) -> None:
        """The headline shape. If this regresses, the delta is decorative.

        ASR is the expensive stage a delta exists to skip, and it batches -- so it is refused by
        the derivation and has to declare instead. ``ASRStage`` prepares each clip independently,
        and ``NeMoASRAdapter`` passes the resulting waveforms through NeMo's length-aware
        transcription path, so a clip's transcript does not move with the clips beside it.
        """
        rec = Recipe.from_dict(
            {
                "stages": [
                    {"ref": "ManifestReader", "params": {"manifest_path": str(tmp_path / "m.jsonl")}},
                    {
                        "ref": "ASRStage",
                        "params": {
                            "adapter_target": "nemo_curator.models.asr.nemo_asr.NeMoASRAdapter",
                            "model_id": "stt_en_conformer_ctc_small",
                            "audio_filepath_key": "audio_filepath",
                        },
                    },
                    {"ref": "GetPairwiseWerStage", "params": {}},
                    {
                        "ref": "PreserveByValueStage",
                        "params": {"input_value_key": "wer", "target_value": 20.0, "operator": "le"},
                    },
                    {"ref": "ManifestWriterStage", "params": {"output_path": str(tmp_path / "out.jsonl")}},
                ]
            }
        ).freeze()
        assert delta.region(rec, upto=5) == (5, "")

    def test_an_undeclared_per_row_stage_is_derived_safe_rather_than_refused(
        self, tmp_path: Path, foreign_stages: None
    ) -> None:
        """The point of deriving: a stage nobody annotated still gets a correct answer.

        ``ForeignPerRowStage`` is handed one task per call, keeps nothing between calls and
        writes no file, so no other file's data can reach its output. Requiring its author to say
        so would be a hand-written claim that can only be wrong -- and would leave the delta
        inert on every pipeline containing a stage written before this feature existed.
        """
        rec = Recipe.from_dict(
            {
                "stages": [
                    {"ref": "ManifestReader", "params": {"manifest_path": str(tmp_path / "m.jsonl")}},
                    {"ref": "ForeignPerRowStage", "params": {}},
                ]
            }
        ).freeze()
        assert delta.region(rec, upto=2) == (2, "")

    def test_a_split_stage_is_independent_only_while_its_outputs_cannot_collide(self, tmp_path: Path) -> None:
        """The surviving example of a gate answered per instance rather than per class.

        Split names come from the source basename alone, so an ``output_dir`` puts every file in
        one flat namespace and ``spk1/utt1.wav`` and ``spk2/utt1.wav`` fight over the same output
        path. Without one they land beside their source and cannot collide. A flat ``False`` would
        cost the delta on the default configuration, which is the one that is actually safe.
        """

        def _region(params: dict[str, object]) -> tuple[int, str]:
            rec = Recipe.from_dict(
                {
                    "stages": [
                        {"ref": "ManifestReader", "params": {"manifest_path": str(tmp_path / "m.jsonl")}},
                        {"ref": "SplitLongAudioStage", "params": params},
                    ]
                }
            ).freeze()
            return delta.region(rec, upto=2)

        assert _region({}) == (2, "")
        depth, reason = _region({"output_dir": str(tmp_path / "splits")})
        assert depth == 1
        assert "SplitLongAudio" in reason

    def test_a_declared_false_still_wins_over_the_derivation(self, tmp_path: Path) -> None:
        """Deriving is the default, not an override: an explicit claim is still authoritative."""
        rec = Recipe.from_dict(
            {
                "stages": [
                    {"ref": "ManifestReader", "params": {"manifest_path": str(tmp_path / "m.jsonl")}},
                    {"ref": "PretrainMetricsAggregatorStage", "params": {"output_path": str(tmp_path / "s.json")}},
                ]
            }
        ).freeze()
        depth, reason = delta.region(rec, upto=2)
        assert depth == 1
        assert "depends on the other rows" in reason


class TestProvenance:
    """Which prior row came from which input file is derived from the rows, never assumed."""

    def test_the_column_naming_the_input_file_is_found(self, tmp_path: Path) -> None:
        inventory = _corpus(tmp_path / "audio", ("a.wav", "b.wav"))
        out = tmp_path / "prior.jsonl"
        out.write_text(
            "".join(
                json.dumps({"audio_filepath": str(tmp_path / "audio" / n), "duration": 1.0}) + "\n"
                for n in ("a.wav", "b.wav")
            )
        )
        key, why = delta.provenance(str(out), inventory=inventory, root=str(tmp_path / "audio"))
        assert (key, why) == ("audio_filepath", "")

    def test_rows_that_no_longer_name_their_origin_get_no_delta(self, tmp_path: Path) -> None:
        """A pipeline that rewrote paths to derived chunks cannot be subtracted per file."""
        inventory = _corpus(tmp_path / "audio", ("a.wav",))
        out = tmp_path / "prior.jsonl"
        out.write_text(json.dumps({"audio_filepath": "/somewhere/else/chunk-000.wav", "duration": 1.0}) + "\n")
        key, why = delta.provenance(str(out), inventory=inventory, root=str(tmp_path / "audio"))
        assert key == ""
        assert "cannot be traced" in why

    def test_an_unreadable_prior_output_is_reported_rather_than_guessed(self, tmp_path: Path) -> None:
        key, why = delta.provenance(str(tmp_path / "missing.jsonl"), inventory={"a.wav": "1"}, root=str(tmp_path))
        assert key == ""
        assert "no rows" in why


class TestPlan:
    """The whole decision, on a real store: ready with names, or refused with a reason."""

    def _prior_run(self, tmp_path: Path, names: tuple[str, ...]) -> tuple[Recipe, dict[str, str]]:
        """Publish a prior run of the pipeline over ``names``, with coverage and real rows."""
        audio = tmp_path / "audio"
        inventory = _corpus(audio, names)
        manifest = tmp_path / "m.jsonl"
        manifest.write_text("".join(json.dumps({"audio_filepath": str(audio / n)}) + "\n" for n in names))
        rec, _ = _pipeline(tmp_path, manifest)
        rows = [{"audio_filepath": str(audio / n), "duration": 1.0} for n in names]
        _publish(rec, 2, dataset_key=_PRIOR, rows=rows, coverage=inventory)
        return rec, inventory

    def test_one_added_file_is_all_that_needs_running(self, tmp_path: Path) -> None:
        rec, inventory = self._prior_run(tmp_path, ("a.wav", "b.wav"))
        now = {**inventory, "c.wav": "16|99"}

        decision = delta.plan(rec, dataset_key=_KEY, inventory=now, inventory_root=str(tmp_path / "audio"))

        assert decision.status == "ready"
        assert decision.change is not None
        assert decision.change.kind == "added_only"
        assert decision.files == (str(tmp_path / "audio" / "c.wav"),)
        assert decision.keeps == 2  # both prior rows survive
        assert decision.drops == 0
        assert decision.provenance_key == "audio_filepath"
        assert decision.estimated_saving_sec > 0

    def test_the_overlapping_corpus_wins_over_the_merely_newest_one(self, tmp_path: Path) -> None:
        """One recipe, several corpora -- the ordinary way to use this agent.

        Picking the prior run by recency meant that the moment a user curated a second corpus,
        the first one's delta died: its inventory was compared against a stranger, answered
        "shares no file with this input", and the refusal blamed a run that had nothing to do
        with it. Overlap is what decides whether prior rows can be kept, so overlap decides.
        """
        audio = tmp_path / "audio"
        inventory = _corpus(audio, ("a.wav", "b.wav"))
        manifest = tmp_path / "m.jsonl"
        manifest.write_text("".join(json.dumps({"audio_filepath": str(audio / n)}) + "\n" for n in ("a.wav", "b.wav")))
        rec, _ = _pipeline(tmp_path, manifest)

        # Each corpus writes to its own output, which is what a user curating two corpora does --
        # a shared path would mean the second run destroyed the first one's manifest. Output
        # locations are outside the reuse identity, so both still share this recipe's step keys.
        def _variant(out: Path) -> Recipe:
            stages = [s.to_dict() for s in rec.stages]
            stages[2]["params"]["output_path"] = str(out)
            return Recipe.from_dict({"stages": stages}).freeze()

        # The corpus we care about, published FIRST so it is the older of the two.
        english = _variant(tmp_path / "en.jsonl")
        _publish(
            english,
            2,
            dataset_key=_PRIOR,
            rows=[{"audio_filepath": str(audio / n), "duration": 1.0} for n in ("a.wav", "b.wav")],
            coverage=inventory,
            created_at="2026-01-01T00:00:00Z",
        )
        # A different corpus curated afterwards, sharing no file with it.
        other = tmp_path / "other"
        other_inventory = _corpus(other, ("x.wav", "y.wav"))
        _publish(
            _variant(tmp_path / "de.jsonl"),
            2,
            dataset_key="dataset-key-unrelated",
            rows=[{"audio_filepath": str(other / n), "duration": 1.0} for n in ("x.wav", "y.wav")],
            coverage=other_inventory,
            created_at="2026-06-01T00:00:00Z",
        )

        # Planned with the recipe that curated the English corpus, which is what its owner reruns.
        decision = delta.plan(
            english, dataset_key=_KEY, inventory={**inventory, "c.wav": "16|99"}, inventory_root=str(audio)
        )

        assert decision.status == "ready", decision.reason
        assert decision.prior_dataset_key == _PRIOR
        assert decision.files == (str(audio / "c.wav"),)

    def test_a_pipeline_that_never_ran_says_exactly_that(self, tmp_path: Path) -> None:
        rec, _ = _pipeline(tmp_path, tmp_path / "m.jsonl")

        decision = delta.plan(rec, dataset_key=_KEY, inventory={"a.wav": "16|1"}, inventory_root=str(tmp_path))

        assert decision.status != "ready"
        assert "no prior run" in decision.reason

    def test_a_pipeline_whose_artifacts_are_unreachable_is_not_told_it_never_ran(self, tmp_path: Path) -> None:
        """The message every existing user meets first, and it used to be false.

        A delta resumes from a published artifact, and artifacts go unreachable for reasons that
        say nothing about whether the work happened -- the step-key version moved, the record was
        pruned, the output changed. Telling someone whose curated manifest is sitting in front of
        them "no prior run" sends them hunting for a run they already have. The run record
        survives all three, so it is the evidence.
        """
        from nemo_curator.audio_agent import run_store

        rec, _ = _pipeline(tmp_path, tmp_path / "m.jsonl")
        run_store.save(
            run_store.RunRecord(
                run_id="run-earlier",
                config_hash=rec.config_hash or "",
                semantic_hash=rec.semantic_hash or "",
                dataset_key="some-older-corpus",
                status="completed",
                data_source=str(tmp_path / "corpus"),
                created_at="2026-01-01T00:00:00Z",
            )
        )

        decision = delta.plan(rec, dataset_key=_KEY, inventory={"a.wav": "16|1"}, inventory_root=str(tmp_path))

        assert decision.status != "ready"
        assert "completed before" in decision.reason
        assert "no prior run" not in decision.reason
        assert "One full run republishes them" in decision.reason

    def test_an_unreadable_stage_source_is_named_rather_than_blamed_on_the_data(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The failure that looks exactly like every other delta miss, and is not one.

        When this process cannot read a stage's source, every step key it computes comes from a
        fallback stamp, so no published artifact can line up no matter what the store holds. The
        prior results are intact and nothing needs rerunning -- but the generic message sends
        someone auditing artifacts and datasets over what is really a broken import path.
        """
        from nemo_curator.audio_agent import code_identity

        rec, _ = _pipeline(tmp_path, tmp_path / "m.jsonl")
        monkeypatch.setattr(code_identity, "unreadable_stages", lambda _refs: ["GetAudioDurationStage"])

        decision = delta.plan(rec, dataset_key=_KEY, inventory={"a.wav": "16|1"}, inventory_root=str(tmp_path))

        assert decision.status != "ready"
        assert "GetAudioDurationStage" in decision.reason
        assert "cannot read the source" in decision.reason
        assert "Prior results are intact" in decision.reason

    def test_readable_sources_leave_the_diagnosis_alone(self, tmp_path: Path) -> None:
        """The new branch must not fire on the ordinary miss it sits in front of."""
        rec, _ = _pipeline(tmp_path, tmp_path / "m.jsonl")

        decision = delta.plan(rec, dataset_key=_KEY, inventory={"a.wav": "16|1"}, inventory_root=str(tmp_path))

        assert "cannot read the source" not in decision.reason
        assert "no prior run" in decision.reason

    def test_a_broken_diagnostic_never_replaces_the_refusal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A diagnostic that raises would turn a useful refusal into a traceback."""
        from nemo_curator.audio_agent import code_identity

        rec, _ = _pipeline(tmp_path, tmp_path / "m.jsonl")
        monkeypatch.setattr(
            code_identity,
            "unreadable_stages",
            lambda _refs: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        decision = delta.plan(rec, dataset_key=_KEY, inventory={"a.wav": "16|1"}, inventory_root=str(tmp_path))

        assert decision.status != "ready"
        assert "no prior run" in decision.reason

    def test_a_directory_resume_point_names_what_shortened_the_region(self, tmp_path: Path) -> None:
        """The realistic GPU shape: resample writes a directory, then a corpus-dependent stage.

        Measured against a real SQUIM pipeline, this refusal said only "ResampleAudioStage does
        not own a manifest the merge can rewrite" -- blaming the stage that happens to hold the
        deepest output, never mentioning the stage that actually shortened the region, and
        offering nothing to do about it. Both halves belong in the sentence.
        """
        audio = tmp_path / "audio"
        inventory = _corpus(audio, ("a.wav", "b.wav"))
        rec = Recipe.from_dict(
            {
                "stages": [
                    {"ref": "CreateInitialManifestAudioFolderStage", "params": {"data_dir": str(audio)}},
                    {"ref": "GetAudioDurationStage", "params": {}},
                    {
                        "ref": "ResampleAudioStage",
                        "params": {"target_sample_rate": 16000, "resampled_audio_dir": str(tmp_path / "rs")},
                    },
                    {"ref": "TorchSquimQualityMetricsStage", "params": {}},
                    {"ref": "ManifestWriterStage", "params": {"output_path": str(tmp_path / "out.jsonl")}},
                ]
            }
        ).freeze()
        (tmp_path / "rs").mkdir(parents=True, exist_ok=True)
        _publish(rec, 2, dataset_key=_PRIOR, rows=[], coverage=inventory)

        decision = delta.plan(
            rec, dataset_key=_KEY, inventory={**inventory, "c.wav": "16|9"}, inventory_root=str(audio)
        )

        assert decision.status != "ready"
        assert "TorchSquimQualityMetrics" in decision.reason, decision.reason
        assert "ResampleAudioStage" in decision.reason
        assert "add-checkpoint" in decision.reason

    def test_a_prior_result_published_elsewhere_is_refused_by_name(self, tmp_path: Path) -> None:
        """Output paths are outside the reuse identity, so one step key can span two files.

        Change where the recipe writes and its step keys do not move -- that is deliberate, so a
        rerun into a new directory can still reuse. But a merge keeps rows from the ARTIFACT and
        rewrites the file the RECIPE names, and when those are different files it is merging two
        unrelated manifests. Refuse saying so, rather than failing later with the far less useful
        "no rows could be read".
        """
        audio = tmp_path / "audio"
        inventory = _corpus(audio, ("a.wav", "b.wav"))
        manifest = tmp_path / "m.jsonl"
        manifest.write_text("".join(json.dumps({"audio_filepath": str(audio / n)}) + "\n" for n in ("a.wav", "b.wav")))
        rec, _ = _pipeline(tmp_path, manifest)

        # Published at one path...
        stages = [s.to_dict() for s in rec.stages]
        stages[2]["params"]["output_path"] = str(tmp_path / "somewhere_else.jsonl")
        elsewhere = Recipe.from_dict({"stages": stages}).freeze()
        _publish(
            elsewhere,
            2,
            dataset_key=_PRIOR,
            rows=[{"audio_filepath": str(audio / n), "duration": 1.0} for n in ("a.wav", "b.wav")],
            coverage=inventory,
        )

        # ...and planned with the recipe that writes somewhere different.
        decision = delta.plan(
            rec, dataset_key=_KEY, inventory={**inventory, "c.wav": "16|99"}, inventory_root=str(audio)
        )

        assert decision.status != "ready"
        assert "somewhere_else.jsonl" in decision.reason
        assert "not the rows at the path it would rewrite" in decision.reason

    def test_an_edited_file_drops_its_prior_row_and_reruns_it(self, tmp_path: Path) -> None:
        rec, inventory = self._prior_run(tmp_path, ("a.wav", "b.wav"))
        now = {**inventory, "a.wav": "16|changed"}

        decision = delta.plan(rec, dataset_key=_KEY, inventory=now, inventory_root=str(tmp_path / "audio"))

        assert decision.status == "ready"
        assert decision.files == (str(tmp_path / "audio" / "a.wav"),)
        assert (decision.drops, decision.keeps) == (1, 1)

    def test_a_removed_file_drops_its_rows_and_runs_nothing(self, tmp_path: Path) -> None:
        rec, inventory = self._prior_run(tmp_path, ("a.wav", "b.wav"))
        now = {k: v for k, v in inventory.items() if k != "b.wav"}

        decision = delta.plan(rec, dataset_key=_KEY, inventory=now, inventory_root=str(tmp_path / "audio"))

        assert decision.status == "ready"
        assert decision.files == ()
        assert (decision.drops, decision.keeps) == (1, 1)
        assert any("gone" in note for note in decision.notes)

    def test_without_an_inventory_there_is_no_delta(self, tmp_path: Path) -> None:
        rec, _ = self._prior_run(tmp_path, ("a.wav",))
        decision = delta.plan(rec, dataset_key=_KEY, inventory=None, inventory_root="")
        assert decision.status == "none"
        assert "no per-file inventory" in decision.reason

    def test_a_prior_run_without_coverage_cannot_be_compared(self, tmp_path: Path) -> None:
        """Runs published before coverage existed are not wrong, they are unusable for a delta."""
        audio = tmp_path / "audio"
        inventory = _corpus(audio, ("a.wav",))
        manifest = tmp_path / "m.jsonl"
        manifest.write_text(json.dumps({"audio_filepath": str(audio / "a.wav")}) + "\n")
        rec, _ = _pipeline(tmp_path, manifest)
        _publish(rec, 2, dataset_key=_PRIOR, rows=[{"audio_filepath": str(audio / "a.wav")}], coverage=None)

        decision = delta.plan(
            rec,
            dataset_key=_KEY,
            inventory={**inventory, "b.wav": "16|1"},
            inventory_root=str(audio),
        )
        assert decision.status == "none"
        assert "no per-file inventory" in decision.reason

    def test_nothing_to_compare_against_is_said_plainly(self, tmp_path: Path) -> None:
        rec, _ = _pipeline(tmp_path, tmp_path / "m.jsonl")
        decision = delta.plan(rec, dataset_key=_KEY, inventory={"a.wav": "1"}, inventory_root=str(tmp_path))
        assert decision.status == "none"
        assert "no prior run" in decision.reason

    def test_an_unfrozen_recipe_is_not_told_a_strangers_run_was_its_own(self, tmp_path: Path) -> None:
        """A recipe with no ``semantic_hash`` has no pipeline identity, and the run query reads a
        missing one as "do not filter" -- so asking with it would return every run on the box and
        let the first completed stranger be described back as this pipeline's own prior run.
        """
        from nemo_curator.audio_agent import run_store
        from nemo_curator.audio_agent.contracts import RunRecord

        run_store.save(
            RunRecord(
                run_id="someone-elses-run",
                semantic_hash="a-completely-different-pipeline",
                status="completed",
                data_source="/data/not-mine.jsonl",
                created_at="2026-08-01T00:00:00Z",
            )
        )
        unfrozen, _ = _pipeline(tmp_path, tmp_path / "m.jsonl")
        unfrozen.semantic_hash = None

        decision = delta.plan(unfrozen, dataset_key=_KEY, inventory={"a.wav": "1"}, inventory_root=str(tmp_path))
        assert decision.status == "none"
        assert "no prior run" in decision.reason
        assert "not-mine.jsonl" not in decision.reason

    def test_a_different_dataset_is_refused_rather_than_subtracted(self, tmp_path: Path) -> None:
        rec, _ = self._prior_run(tmp_path, ("a.wav", "b.wav"))
        decision = delta.plan(
            rec,
            dataset_key=_KEY,
            inventory={"x.wav": "16|1", "y.wav": "16|2"},
            inventory_root=str(tmp_path / "audio"),
        )
        assert decision.status == "none"
        assert "different dataset" in decision.reason

    def test_identical_files_under_a_new_key_say_what_else_moved(self, tmp_path: Path) -> None:
        rec, inventory = self._prior_run(tmp_path, ("a.wav",))
        decision = delta.plan(rec, dataset_key=_KEY, inventory=inventory, inventory_root=str(tmp_path / "audio"))
        assert decision.status == "none"
        assert "identical" in decision.reason

    def test_nothing_persisted_inside_the_region_means_the_corpus_must_be_seen_together(self, tmp_path: Path) -> None:
        """The only saved output sits behind a stage that totals the corpus, so it cannot be split."""
        audio = tmp_path / "audio"
        inventory = _corpus(audio, ("a.wav",))
        rec = Recipe.from_dict(
            {
                "stages": [
                    {"ref": "ManifestReader", "params": {"manifest_path": str(tmp_path / "m.jsonl")}},
                    {"ref": "PretrainMetricsAggregatorStage", "params": {"output_path": str(tmp_path / "s.json")}},
                ]
            }
        ).freeze()
        _publish(rec, 1, dataset_key=_PRIOR, rows=[{"audio_filepath": str(audio / "a.wav")}], coverage=inventory)

        decision = delta.plan(
            rec,
            dataset_key=_KEY,
            inventory={**inventory, "b.wav": "16|9"},
            inventory_root=str(audio),
        )
        assert decision.status == "none"
        assert "PretrainMetricsAggregator" in decision.reason
        assert "corpus together" in decision.reason
