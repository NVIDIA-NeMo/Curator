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

"""Executing a delta: run the changed files, merge, republish -- and prove the equivalence.

The claim these pin down is the one that makes the feature safe to trust: a full run over N
files and a run over N-1 files followed by a delta over the last one produce the same manifest.
"""

from __future__ import annotations

import json
import math
import os
import struct
import wave
from typing import TYPE_CHECKING, Any

import pytest

from nemo_curator.audio_agent import artifacts, cli, delta, verbs
from nemo_curator.audio_agent.recipe import Recipe

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("AUDIO_AGENT_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


def _wav(path: Path, *, seconds: float = 0.25, rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        frames = int(seconds * rate)
        handle.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(i / 20))) for i in range(frames)))


def _recipe(folder: Path, out: Path) -> Recipe:
    """folder source -> duration -> manifest: CPU only, and every stage declares independence."""
    return Recipe.from_dict(
        {
            "stages": [
                {"ref": "CreateInitialManifestAudioFolderStage", "params": {"data_dir": str(folder)}},
                {"ref": "GetAudioDurationStage", "params": {}},
                {"ref": "ManifestWriterStage", "params": {"output_path": str(out)}},
            ]
        }
    ).freeze()


def _rows(path: Path | str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _run(rec: Recipe, folder: Path) -> dict[str, Any]:
    return verbs.run(rec, confirm=True, data=str(folder))


@pytest.mark.usefixtures("store")
class TestDeltaExecution:
    """The verb, against a pipeline that really executes."""

    def test_adding_a_file_runs_only_that_file_and_completes_the_manifest(self, tmp_path: Path) -> None:
        folder, out = tmp_path / "audio", tmp_path / "out" / "m.jsonl"
        for name in ("a.wav", "b.wav"):
            _wav(folder / name)
        rec = _recipe(folder, out)
        assert _run(rec, folder)["status"] == "completed"
        assert len(_rows(out)) == 2

        _wav(folder / "c.wav")
        # Unconfirmed, this is a card and a refusal -- the same shape run() returns.
        card = verbs.delta_run(rec, data=str(folder))
        assert card["status"] == "refused", card
        assert card["delta"]["change"]["added"] == 1
        assert card["delta"]["file_count"] == 1
        assert card["delta"]["change"]["added_files"] == ["c.wav"]

        done = verbs.delta_run(rec, data=str(folder), confirm=True)
        assert done["status"] == "completed", done
        assert done["ran_files"] == [str(folder / "c.wav")]
        # As a set: parallel writers append in completion order, so no run has a fixed row order.
        assert {os.path.basename(r["audio_filepath"]) for r in _rows(out)} == {"a.wav", "b.wav", "c.wav"}
        assert done["merged"][0]["rows_kept"] == 2
        assert done["merged"][0]["rows_added"] == 1
        assert done["published"], "the merged manifest must be findable by an ordinary reuse probe"

    def test_the_merged_manifest_is_what_a_full_run_would_have_produced(self, tmp_path: Path) -> None:
        """The equivalence the whole design rests on: N == (N-1) + delta."""
        folder, out = tmp_path / "audio", tmp_path / "out" / "m.jsonl"
        for name in ("a.wav", "b.wav"):
            _wav(folder / name)
        rec = _recipe(folder, out)
        assert _run(rec, folder)["status"] == "completed"
        _wav(folder / "c.wav")
        verbs.delta_run(rec, data=str(folder), confirm=True)
        incremental = _rows(out)

        whole = tmp_path / "out" / "whole.jsonl"
        assert _run(_recipe(folder, whole), folder)["status"] == "completed"

        by_file = {os.path.basename(r["audio_filepath"]): r for r in incremental}
        assert set(by_file) == {os.path.basename(r["audio_filepath"]) for r in _rows(whole)}
        for row in _rows(whole):
            mine = by_file[os.path.basename(row["audio_filepath"])]
            assert mine == row, "a delta row differs from the row a full run produced"

    def test_the_next_run_reuses_the_merged_result_instead_of_recomputing(self, tmp_path: Path) -> None:
        folder, out = tmp_path / "audio", tmp_path / "out" / "m.jsonl"
        for name in ("a.wav", "b.wav"):
            _wav(folder / name)
        rec = _recipe(folder, out)
        assert _run(rec, folder)["status"] == "completed"
        _wav(folder / "c.wav")
        verbs.delta_run(rec, data=str(folder), confirm=True)

        scan = verbs.reuse_scan(rec, data=str(folder))
        assert scan["decision"] == "already_done", scan["rationale"]

    def test_a_second_delta_resumes_from_what_the_first_one_merged(self, tmp_path: Path) -> None:
        """A corpus grows more than once, so a delta has to be able to follow a delta.

        The merged manifest is published under the full pipeline's own key, which means the next
        delta reads its record like any other. Its ``rows_in`` is deliberately left unrecorded:
        pairing the merged row count with either run's input would describe an execution that
        never happened.
        """
        folder, out = tmp_path / "audio", tmp_path / "out" / "m.jsonl"
        for name in ("a.wav", "b.wav"):
            _wav(folder / name)
        rec = _recipe(folder, out)
        assert _run(rec, folder)["status"] == "completed"

        _wav(folder / "c.wav")
        assert verbs.delta_run(rec, data=str(folder), confirm=True)["status"] == "completed"

        _wav(folder / "d.wav")
        offered = verbs.reuse_scan(rec, data=str(folder))["delta"]
        assert offered["status"] == "ready", offered["reason"]

        done = verbs.delta_run(rec, data=str(folder), confirm=True)
        assert done["status"] == "completed", done
        assert done["ran_files"] == [str(folder / "d.wav")]
        assert done["merged"][0]["rows_kept"] == 3
        assert done["merged"][0]["rows_added"] == 1
        assert {os.path.basename(r["audio_filepath"]) for r in _rows(out)} == {"a.wav", "b.wav", "c.wav", "d.wav"}

    def test_a_delta_that_owns_only_a_prefix_does_not_call_itself_completed(self, tmp_path: Path) -> None:
        """The realistic shape: per-file work, a checkpoint, then a stage that needs the corpus.

        The merge brings the checkpoint up to date and can do nothing about what the stages after
        it wrote, so those files still describe the corpus as it was. Reporting "completed" there
        hands a host a finished-looking answer over a stale deliverable, which is why the status
        names the tail and lists the outputs that are currently a lie.
        """
        folder = tmp_path / "audio"
        checkpoint, groups = tmp_path / "out" / "ck.jsonl", tmp_path / "out" / "groups"
        for name in ("a.wav", "b.wav"):
            _wav(folder / name)
        rec = Recipe.from_dict(
            {
                "stages": [
                    {"ref": "CreateInitialManifestAudioFolderStage", "params": {"data_dir": str(folder)}},
                    {"ref": "GetAudioDurationStage", "params": {}},
                    {"ref": "ManifestWriterStage", "params": {"output_path": str(checkpoint)}},
                    # per_row_independent=False, so the traceable region ends here.
                    {
                        "ref": "ManifestGroupExportStage",
                        "params": {"output_dir": str(groups), "group_by": "audio_filepath", "format": "json"},
                    },
                ]
            }
        ).freeze()
        assert _run(rec, folder)["status"] == "completed"
        exported = sorted(p.name for p in groups.glob("*.jsonl"))
        assert len(exported) == 2, exported

        _wav(folder / "c.wav")
        done = verbs.delta_run(rec, data=str(folder), confirm=True)

        assert done["status"] == "tail_required", done
        assert done["tail"]["stages"] == 1
        assert str(groups) in " ".join(done["tail"]["stale_outputs"])
        assert len(_rows(checkpoint)) == 3, "the prefix's own manifest is merged and current"
        assert sorted(p.name for p in groups.glob("*.jsonl")) == exported, "the tail has not run yet"

        tail = verbs.plan_continuation(rec, data=str(folder), execute=True, choice="extend", confirm=True)
        assert tail.get("status") not in {"error", "failed", "refused"}, tail
        assert len(sorted(groups.glob("*.jsonl"))) == 3, "the tail reran over every row"

    def test_the_merged_deliverable_is_judged_against_the_confirmed_success_bar(self, tmp_path: Path) -> None:
        """A delta rewrites the user's output, so the bar they approved has to be re-checked.

        Without this the verb answers ``completed`` -- and the CLI exits 0 -- over a manifest the
        merge has just pushed past a ``must`` criterion, so a script ships it. The verdict is
        computed over what is on disk after the merge, not over the delta's own narrow run.
        """
        folder, out = tmp_path / "audio", tmp_path / "out" / "m.jsonl"
        for name in ("a.wav", "b.wav"):
            _wav(folder / name)
        rec = Recipe.from_dict(
            {
                "stages": [
                    {"ref": "CreateInitialManifestAudioFolderStage", "params": {"data_dir": str(folder)}},
                    {"ref": "GetAudioDurationStage", "params": {}},
                    {"ref": "ManifestWriterStage", "params": {"output_path": str(out)}},
                ],
                # Two files clear this; the third pushes the corpus past it.
                "acceptance_criteria": [
                    {"id": "cap", "type": "yield", "check": {"op": "<=", "value": 2}, "severity": "must"}
                ],
            }
        ).freeze()
        assert _run(rec, folder)["status"] == "completed"

        _wav(folder / "c.wav")
        done = verbs.delta_run(rec, data=str(folder), confirm=True)

        assert done["status"] == "completed", done
        assert "acceptance" in done, "a delta that rewrites the deliverable must judge it"
        assert done["acceptance"].get("overall") == "not_met", done["acceptance"]
        # And the shell must not read that as success.
        assert cli._result_exit_code("delta-run", done) == 1

    def test_a_prefix_delta_does_not_manufacture_an_acceptance_verdict(self, tmp_path: Path) -> None:
        """The tail has not run, so there is no final output to judge -- and saying so is the point.

        The prefix recipe is also stripped of the criteria before it executes: they describe the
        whole pipeline, and letting a 1-file run over 2 stages record a verdict about them puts a
        number nobody should trust on the run record the merged artifact points at.
        """
        folder = tmp_path / "audio"
        checkpoint, groups = tmp_path / "out" / "ck.jsonl", tmp_path / "out" / "groups"
        for name in ("a.wav", "b.wav"):
            _wav(folder / name)
        rec = Recipe.from_dict(
            {
                "stages": [
                    {"ref": "CreateInitialManifestAudioFolderStage", "params": {"data_dir": str(folder)}},
                    {"ref": "GetAudioDurationStage", "params": {}},
                    {"ref": "ManifestWriterStage", "params": {"output_path": str(checkpoint)}},
                    {
                        "ref": "ManifestGroupExportStage",
                        "params": {"output_dir": str(groups), "group_by": "audio_filepath", "format": "json"},
                    },
                ],
                "acceptance_criteria": [
                    {"id": "some", "type": "yield", "check": {"op": ">=", "value": 1}, "severity": "must"}
                ],
            }
        ).freeze()
        assert _run(rec, folder)["status"] == "completed"

        _wav(folder / "c.wav")
        done = verbs.delta_run(rec, data=str(folder), confirm=True)

        assert done["status"] == "tail_required", done
        assert "acceptance" not in done, "a verdict here would describe output the tail has not written"
        assert done["tail"]["acceptance"]
        # tail_required stays a shell success: the merge did what it promised.
        assert cli._result_exit_code("delta-run", done) == 0

    def test_running_the_same_delta_twice_does_not_duplicate_the_new_rows(self, tmp_path: Path) -> None:
        """A retry has to be safe, because the failure paths tell the caller to retry.

        The merge drops every file the delta RAN, not just the ones whose prior rows went stale,
        so it replaces rather than appends. On a first delta an added file has no prior rows and
        the wider set is a no-op; on a repeat it is what stops the same rows landing twice. This
        matters because a delta that fails on its second sink leaves the first one merged, and
        the coverage sidecar is only rewritten at the very end -- so the natural retry replans
        the identical delta.
        """
        folder, out = tmp_path / "audio", tmp_path / "out" / "m.jsonl"
        for name in ("a.wav", "b.wav"):
            _wav(folder / name)
        rec = _recipe(folder, out)
        assert _run(rec, folder)["status"] == "completed"

        _wav(folder / "c.wav")
        assert verbs.delta_run(rec, data=str(folder), confirm=True)["status"] == "completed"
        after_first = [r["audio_filepath"] for r in _rows(out)]
        assert len(after_first) == 3, after_first

        # Nothing changed in between, so this is the retry shape. Whatever it decides -- a
        # refusal is a perfectly good answer -- it must not leave a duplicated row behind.
        verbs.delta_run(rec, data=str(folder), confirm=True)
        after_second = [r["audio_filepath"] for r in _rows(out)]
        assert len(after_second) == len(set(after_second)), f"a retry duplicated rows: {after_second}"
        assert sorted(after_second) == sorted(after_first)

    def test_a_removed_file_loses_its_rows_without_running_anything(self, tmp_path: Path) -> None:
        folder, out = tmp_path / "audio", tmp_path / "out" / "m.jsonl"
        for name in ("a.wav", "b.wav"):
            _wav(folder / name)
        rec = _recipe(folder, out)
        assert _run(rec, folder)["status"] == "completed"

        (folder / "b.wav").unlink()
        done = verbs.delta_run(rec, data=str(folder), confirm=True)
        assert done["status"] == "completed", done
        assert done["ran_files"] == []
        assert [os.path.basename(r["audio_filepath"]) for r in _rows(out)] == ["a.wav"]

    def test_an_edited_file_replaces_its_row_rather_than_duplicating_it(self, tmp_path: Path) -> None:
        folder, out = tmp_path / "audio", tmp_path / "out" / "m.jsonl"
        _wav(folder / "a.wav", seconds=0.25)
        _wav(folder / "b.wav", seconds=0.25)
        rec = _recipe(folder, out)
        assert _run(rec, folder)["status"] == "completed"
        assert {round(r["duration"], 2) for r in _rows(out)} == {0.25}

        _wav(folder / "b.wav", seconds=0.5)  # same name, different audio
        done = verbs.delta_run(rec, data=str(folder), confirm=True)
        assert done["status"] == "completed", done
        rows = {os.path.basename(r["audio_filepath"]): round(r["duration"], 2) for r in _rows(out)}
        assert rows == {"a.wav": 0.25, "b.wav": 0.5}

    def test_a_delta_needs_confirmation_like_any_other_run(self, tmp_path: Path) -> None:
        folder, out = tmp_path / "audio", tmp_path / "out" / "m.jsonl"
        _wav(folder / "a.wav")
        rec = _recipe(folder, out)
        assert _run(rec, folder)["status"] == "completed"
        _wav(folder / "b.wav")

        refused = verbs.delta_run(rec, data=str(folder), confirm="not-the-hash")
        assert refused["status"] == "refused"
        assert "confirm" in refused["reason"]
        assert len(_rows(out)) == 1, "a refused delta must not touch the prior manifest"

    def test_an_unchanged_corpus_is_told_there_is_no_delta(self, tmp_path: Path) -> None:
        folder, out = tmp_path / "audio", tmp_path / "out" / "m.jsonl"
        _wav(folder / "a.wav")
        rec = _recipe(folder, out)
        assert _run(rec, folder)["status"] == "completed"

        card = verbs.delta_run(rec, data=str(folder))
        assert card["status"] == "no_delta"
        # Nothing changed, so the ordinary probe already serves it; that is not a delta's job.
        assert card["delta"]["reason"]

    def test_a_narrowed_run_records_only_the_files_it_read(self, tmp_path: Path) -> None:
        """Coverage is a claim about work done; a subset run must not claim the whole corpus."""
        folder, out = tmp_path / "audio", tmp_path / "out" / "m.jsonl"
        for name in ("a.wav", "b.wav"):
            _wav(folder / name)
        rec = Recipe.from_dict(
            {
                "stages": [
                    {
                        "ref": "CreateInitialManifestAudioFolderStage",
                        "params": {"data_dir": str(folder), "include_files": [str(folder / "a.wav")]},
                    },
                    {"ref": "GetAudioDurationStage", "params": {}},
                    {"ref": "ManifestWriterStage", "params": {"output_path": str(out)}},
                ]
            }
        ).freeze()
        assert _run(rec, folder)["status"] == "completed"
        assert [os.path.basename(r["audio_filepath"]) for r in _rows(out)] == ["a.wav"]

        published = [a for a in artifacts.list_artifacts() if a.uri == str(out)]
        assert published
        assert list(artifacts.load_coverage(published[0].step_key) or {}) == ["a.wav"]
        assert published[0].covers_files == 1


@pytest.mark.usefixtures("store")
class TestTheCardOffersIt:
    """A delta nobody is told about saves nothing: the miss card is where it has to appear."""

    def test_the_reuse_card_offers_the_delta_on_the_miss_it_belongs_to(self, tmp_path: Path) -> None:
        folder, out = tmp_path / "audio", tmp_path / "out" / "m.jsonl"
        for name in ("a.wav", "b.wav"):
            _wav(folder / name)
        rec = _recipe(folder, out)
        assert _run(rec, folder)["status"] == "completed"
        _wav(folder / "c.wav")

        scan = verbs.reuse_scan(rec, data=str(folder))

        # `decision` names the cheapest correct action, because that is the field a host branches on.
        assert scan["decision"] == "delta"
        assert scan["key_matched"] is False
        assert scan["delta"]["status"] == "ready"
        assert scan["recommended"] == "delta"
        assert scan["prompt_user"] is True, "an available delta is always worth asking about"
        assert [c["id"] for c in scan["choices"]] == ["delta", "fresh"]
        assert "1 file(s) were added" in scan["rationale"]

    def test_a_first_ever_run_is_not_told_about_deltas(self, tmp_path: Path) -> None:
        """Nothing to compare against is the ordinary case, and it must stay quiet."""
        folder, out = tmp_path / "audio", tmp_path / "out" / "m.jsonl"
        _wav(folder / "a.wav")

        scan = verbs.reuse_scan(_recipe(folder, out), data=str(folder))

        assert scan["decision"] == "fresh"
        assert "delta" not in scan

    def test_an_unusable_delta_says_why_on_the_card(self, tmp_path: Path) -> None:
        """The reason the full run is unavoidable is the useful part of a refusal."""
        folder, out = tmp_path / "audio", tmp_path / "out" / "m.jsonl"
        _wav(folder / "a.wav")
        rec = _recipe(folder, out)
        assert _run(rec, folder)["status"] == "completed"

        # A fresh drop of data sharing no file with the last one: a different dataset, not a
        # changed one, and subtracting the two would delete every prior row.
        (folder / "a.wav").unlink()
        for name in ("x.wav", "y.wav"):
            _wav(folder / name)

        scan = verbs.reuse_scan(rec, data=str(folder))

        assert scan["delta"]["status"] != "ready"
        assert scan["delta"]["change"]["kind"] == "unrelated"
        assert "different dataset rather than a changed one" in scan["rationale"]
        assert scan["recommended"] == "fresh"

    def test_a_pipeline_that_saved_nothing_is_told_so_instead_of_never_ran(self, tmp_path: Path) -> None:
        """Coverage lives on artifacts, so a pipeline that persists nothing has no delta to offer.

        What it must not get is the plain miss wording: this pipeline ran yesterday, and "no
        prior artifact matches" reads as "this is new" to anyone who did not write the scanner.
        """
        folder = tmp_path / "audio"
        _wav(folder / "a.wav")
        rec = Recipe.from_dict(
            {
                "stages": [
                    {"ref": "CreateInitialManifestAudioFolderStage", "params": {"data_dir": str(folder)}},
                    {"ref": "GetAudioDurationStage", "params": {}},
                ]
            }
        ).freeze()
        assert _run(rec, folder)["status"] == "completed"
        _wav(folder / "b.wav")

        scan = verbs.reuse_scan(rec, data=str(folder))

        assert scan["decision"] == "fresh"
        assert scan["prior_on_other_data"]["saved"] is False
        assert "persisted nothing" in scan["rationale"]
        assert "add-checkpoint" in scan["rationale"]


@pytest.mark.usefixtures("store")
class TestMergeSafety:
    """The merge refuses rather than producing a manifest no run could have produced."""

    def test_rows_with_different_columns_are_not_merged(self, tmp_path: Path) -> None:
        prior, produced = tmp_path / "prior.jsonl", tmp_path / "new.jsonl"
        prior.write_text(json.dumps({"audio_filepath": "/x/a.wav", "duration": 1.0}) + "\n")
        produced.write_text(json.dumps({"audio_filepath": "/x/b.wav", "duration": 1.0, "extra": 1}) + "\n")
        sink = delta.Sink(index=0, param="output_path", uri=str(prior), step_key="k", key="audio_filepath")

        kept, added, why = delta.merge(sink, produced=str(produced), stale=set(), key="audio_filepath", root="/x")

        assert (kept, added) == (0, 0)
        assert "same columns" in why
        assert len(_rows(prior)) == 1, "the prior manifest must survive a refused merge intact"

    def test_a_manifest_that_grew_since_publication_stops_being_reusable(self, tmp_path: Path) -> None:
        """What protects a delta that dies between two merges.

        The merged manifest is right and its artifact record is not yet, so the record must not
        be honoured. Reuse is bound to the bytes, which is why the half-finished case degrades to
        recomputing rather than to serving a manifest whose record understates it.
        """
        folder, out = tmp_path / "audio", tmp_path / "out" / "m.jsonl"
        _wav(folder / "a.wav")
        rec = _recipe(folder, out)
        assert _run(rec, folder)["status"] == "completed"
        published = [a for a in artifacts.list_artifacts() if a.uri == str(out)]
        assert published
        assert not artifacts.invalid_reasons(published[0])

        with open(out, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"audio_filepath": str(folder / "b.wav"), "duration": 1.0}) + "\n")

        assert any("changed after" in r for r in artifacts.invalid_reasons(published[0]))

    def test_the_swap_is_atomic(self, tmp_path: Path) -> None:
        """A reader sees the old manifest or the new one, never a half-written file."""
        prior, produced = tmp_path / "prior.jsonl", tmp_path / "new.jsonl"
        prior.write_text(json.dumps({"audio_filepath": "/x/a.wav"}) + "\n")
        produced.write_text(json.dumps({"audio_filepath": "/x/b.wav"}) + "\n")
        sink = delta.Sink(index=0, param="output_path", uri=str(prior), step_key="k", key="audio_filepath")
        before = os.stat(prior).st_ino

        kept, added, why = delta.merge(sink, produced=str(produced), stale=set(), key="audio_filepath", root="/x")

        assert (kept, added, why) == (1, 1, "")
        assert os.stat(prior).st_ino != before  # replaced wholesale, not appended in place
        assert [r["audio_filepath"] for r in _rows(prior)] == ["/x/a.wav", "/x/b.wav"]


def _drifted(folder: Path, out: Path) -> Recipe:
    """The same pipeline as :func:`_recipe`, planned a second time.

    One default written out explicitly, which changes nothing about what runs and moves every
    step key -- the whole reason a folder curated an hour ago can look untouched.
    """
    return Recipe.from_dict(
        {
            "stages": [
                {"ref": "CreateInitialManifestAudioFolderStage", "params": {"data_dir": str(folder)}},
                {"ref": "GetAudioDurationStage", "params": {"duration_key": "duration"}},
                {"ref": "ManifestWriterStage", "params": {"output_path": str(out)}},
            ]
        }
    ).freeze()


@pytest.mark.usefixtures("store")
class TestAdoptingAPriorRunsRecipe:
    """From "you curated this folder before" to "only the new file ran", without retyping."""

    def test_a_drifted_plan_misses_everything_and_adopting_the_prior_run_deltas(self, tmp_path: Path) -> None:
        folder, out = tmp_path / "audio", tmp_path / "out" / "m.jsonl"
        for name in ("a.wav", "b.wav"):
            _wav(folder / name)
        first = _run(_recipe(folder, out), folder)
        assert first["status"] == "completed"

        _wav(folder / "c.wav")
        # Session two: same request, planned again into its own output, and every matcher misses.
        second_out = tmp_path / "out" / "second.jsonl"
        scan = verbs.reuse_scan(_drifted(folder, second_out), data=str(folder))
        assert scan["decision"] == "fresh"
        # A delta cannot help here: it resumes from THIS pipeline's artifacts, and the drift moved
        # every key. The notice is the only thing that can still connect the two sessions.
        assert scan.get("delta", {}).get("status") != "ready"
        notice = scan["prior_on_same_path"]
        assert notice["run_id"] == first["run_id"]
        assert notice["recommendation"] == "align"
        assert notice["data_delta"]["added_files"] == ["c.wav"]

        card = verbs.delta_run(from_run=first["run_id"], data=str(folder))
        assert card["adopted_from"]["run_id"] == first["run_id"]
        assert card["delta"]["change"]["added_files"] == ["c.wav"]
        assert card["delta"]["status"] == "ready", card["delta"].get("reason")

        done = verbs.delta_run(from_run=first["run_id"], data=str(folder), confirm=True)
        assert done["status"] == "completed", done
        assert done["ran_files"] == [str(folder / "c.wav")]
        # The deliverable is the prior run's manifest, now covering the whole folder.
        assert {os.path.basename(r["audio_filepath"]) for r in _rows(out)} == {"a.wav", "b.wav", "c.wav"}
        assert not second_out.exists(), "adopting runs the prior pipeline, not the drifted one"

    def test_the_card_names_the_pipeline_being_adopted_before_it_is_confirmed(self, tmp_path: Path) -> None:
        """The user is approving stages they did not write, so the card has to show them."""
        folder, out = tmp_path / "audio", tmp_path / "out" / "m.jsonl"
        _wav(folder / "a.wav")
        first = _run(_recipe(folder, out), folder)
        _wav(folder / "b.wav")

        card = verbs.delta_run(from_run=first["run_id"], data=str(folder))

        assert card["adopted_from"]["pipeline"] == [s["ref"] for s in card["recipe"]["stages"]]
        assert card["adopted_from"]["identity_reproduced"] is True
        assert card["status"] == "refused", "an adopted delta is confirm-gated like any other"
        assert card["config_hash"] in card["confirm_with"]

    def test_adopting_over_an_unchanged_folder_is_sent_to_reuse_not_to_a_rerun(self, tmp_path: Path) -> None:
        """No file moved, so there is nothing for a delta to process -- and running again would
        pay a second time for the result the user came back to reuse."""
        folder, out = tmp_path / "audio", tmp_path / "out" / "m.jsonl"
        _wav(folder / "a.wav")
        first = _run(_recipe(folder, out), folder)

        card = verbs.delta_run(from_run=first["run_id"], data=str(folder))

        assert card["status"] == "no_delta"
        assert "already covers this input" in card["next"]
        assert "as_is" in card["next"]

    def test_a_recipe_and_a_run_together_are_refused_rather_than_one_being_ignored(self, tmp_path: Path) -> None:
        folder, out = tmp_path / "audio", tmp_path / "out" / "m.jsonl"
        _wav(folder / "a.wav")
        first = _run(_recipe(folder, out), folder)

        refused = verbs.delta_run(_recipe(folder, out), from_run=first["run_id"], data=str(folder))

        assert refused["status"] == "refused"
        assert "not both" in refused["reason"]

    def test_neither_a_recipe_nor_a_run_is_refused(self) -> None:
        refused = verbs.delta_run()

        assert refused["status"] == "refused"
        assert "from_run" in refused["reason"]

    def test_an_unknown_run_is_refused_by_id(self) -> None:
        refused = verbs.delta_run(from_run="run-nope")

        assert refused["status"] == "refused"
        assert "run-nope" in refused["reason"]

    def test_a_run_that_did_not_complete_has_nothing_to_extend(self, tmp_path: Path) -> None:
        from nemo_curator.audio_agent import run_store
        from nemo_curator.audio_agent.contracts import RunRecord

        folder, out = tmp_path / "audio", tmp_path / "out" / "m.jsonl"
        _wav(folder / "a.wav")
        rec = _recipe(folder, out)
        run_id = run_store.new_run_id(rec.config_hash)
        run_store.save(
            RunRecord(run_id=run_id, recipe=rec.to_dict(), semantic_hash=rec.semantic_hash, status="failed")
        )

        refused = verbs.delta_run(from_run=run_id, data=str(folder))

        assert refused["status"] == "refused"
        assert "completed" in refused["reason"]


@pytest.mark.usefixtures("store")
class TestAPipelineHistoryCannotReproduce:
    """A masked credential is part of reuse identity, so adopting has to notice it is gone."""

    def _credentialed(self, folder: Path, out: Path) -> Recipe:
        return Recipe.from_dict(
            {
                "stages": [
                    {"ref": "CreateInitialManifestAudioFolderStage", "params": {"data_dir": str(folder)}},
                    {"ref": "PyAnnoteDiarizationStage", "params": {"hf_token": "SECRET-TOKEN"}},
                    {"ref": "ManifestWriterStage", "params": {"output_path": str(out)}},
                ]
            }
        ).freeze()

    def _record_with_masked_recipe(self, rec: Recipe) -> str:
        from nemo_curator.audio_agent import _safety, run_store
        from nemo_curator.audio_agent.contracts import RunRecord

        run_id = run_store.new_run_id(rec.config_hash)
        run_store.save(
            RunRecord(
                run_id=run_id,
                recipe=_safety.redact(rec.to_dict(), redact_transcripts=False),
                config_hash=rec.config_hash,
                semantic_hash=rec.semantic_hash,
                status="completed",
            )
        )
        return run_id

    def test_adopting_it_refuses_and_names_the_param(self, tmp_path: Path) -> None:
        rec = self._credentialed(tmp_path / "audio", tmp_path / "m.jsonl")
        run_id = self._record_with_masked_recipe(rec)

        refused = verbs.delta_run(from_run=run_id)

        assert refused["status"] == "refused"
        assert "PyAnnoteDiarizationStage.hf_token" in refused["reason"]
        assert "recipe with the credential supplied" in refused["next"]

    def test_the_verbatim_copy_a_real_run_keeps_makes_it_adoptable(self, tmp_path: Path) -> None:
        from nemo_curator.audio_agent import run_store

        rec = self._credentialed(tmp_path / "audio", tmp_path / "m.jsonl")
        run_id = self._record_with_masked_recipe(rec)
        path = run_store.save_exact_recipe(run_id, rec.to_dict())

        assert path, "a recipe the record cannot reproduce must be kept verbatim"
        assert oct(os.stat(path).st_mode)[-3:] == "600", "a stored credential is owner-only"
        assert run_store.load_exact_recipe(run_id) == rec.to_dict()

        adopted, provenance, refusal = verbs._adopt_recipe(run_id)
        assert refusal is None, refusal
        assert adopted.semantic_hash == rec.semantic_hash, "the adopted pipeline is the one that ran"
        assert provenance["identity_reproduced"] is True

    def test_an_ordinary_recipe_is_not_copied_a_second_time(self, tmp_path: Path) -> None:
        """Nothing was masked, so the record already reproduces it and a copy would only be one
        more place the same information lives."""
        from nemo_curator.audio_agent import run_store

        rec = _recipe(tmp_path / "audio", tmp_path / "m.jsonl")

        assert run_store.save_exact_recipe("run-x", rec.to_dict()) is None
        assert run_store.load_exact_recipe("run-x") is None
