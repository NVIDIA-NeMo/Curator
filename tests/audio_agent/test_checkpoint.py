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

"""Placing a mid-pipeline manifest so the expensive stages become reusable.

The recipes here are miniatures of the ALM pipeline, which is where the problem was found:
GPU work that persists nothing, and a waveform still resident for several stages after it.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import Any

import pytest  # noqa: TC002

from nemo_curator.audio_agent import checkpoint, reuse, verbs
from nemo_curator.audio_agent.recipe import Recipe

_READER = {"ref": "ManifestReader", "params": {"manifest_path": "/tmp/m.jsonl"}}  # noqa: S108
_DUR = {"ref": "GetAudioDurationStage", "params": {}}
_WRITER = {"ref": "ManifestWriterStage", "params": {"output_path": "/tmp/out.jsonl"}}  # noqa: S108
# Expensive per its card, and persists nothing of its own -- the work worth not repeating.
_ASR = {
    "ref": "ASRStage",
    "params": {
        "adapter_target": "nemo_curator.models.asr.nemo_asr.NeMoASRAdapter",
        "model_id": "nvidia/parakeet-tdt-0.6b-v2",
        "audio_filepath_key": "audio_filepath",
    },
}
# Holds its waveform in the task, so everything downstream of it carries a tensor no manifest
# can serialize, until a sanitizing stage drops it.
_KEEPS_WAVEFORM = {
    "ref": "ResampleAudioStage",
    "params": {"resampled_audio_dir": "/tmp/rs", "keep_waveform_in_task": True, "write_to_disk": False},  # noqa: S108
}
_SANITIZER = {"ref": "AudioToDocumentStage", "params": {}}


def _recipe(*stages: dict[str, Any]) -> Recipe:
    return Recipe.from_dict({"stages": list(stages)}).freeze()


class TestWhereItGoes:
    def test_it_lands_just_past_the_expensive_stage(self) -> None:
        """Not as deep as it will go: the shallowest position clearing the GPU work leaves the
        most of the tail editable, and a tweak below the checkpoint is what reuses it."""
        spot, why = checkpoint.advise(_recipe(_READER, _ASR, _DUR, _WRITER))
        assert spot is not None, why
        assert (spot.index, spot.after_stage) == (2, "ASRStage")
        assert spot.skips == ["ASRStage"]

    def test_a_resident_waveform_pushes_it_to_the_stage_that_drops_one(self) -> None:
        """The ALM finding in miniature. Writing right after the ASR stage would crash on the
        tensor, which no list of stage names would have predicted -- the validator is asked."""
        spot, why = checkpoint.advise(_recipe(_READER, _KEEPS_WAVEFORM, _ASR, _SANITIZER, _DUR, _WRITER))
        assert spot is not None, why
        assert spot.after_stage == "AudioToDocumentStage"
        assert "carrying audio in memory" in spot.not_earlier
        assert "ASRStage" in spot.not_earlier

    def test_a_pipeline_that_never_drops_the_waveform_gets_no_position(self) -> None:
        """Refusing beats advising a writer that would fail at ``json.dumps``."""
        spot, why = checkpoint.advise(_recipe(_READER, _KEEPS_WAVEFORM, _ASR, _DUR, _WRITER))
        assert spot is None
        assert "nowhere after ASRStage" in why

    def test_it_will_not_split_a_pair_that_passes_state_outside_the_row(self) -> None:
        """``OverlapFilterStage`` parks counters in ``task._metadata`` and the aggregator reads
        them. A manifest carries only ``task.data``, and nothing raises when the counters go
        missing -- the aggregator gets an empty dict and reports wrong numbers successfully."""
        overlap = {"ref": "OverlapFilterStage", "params": {}}
        metrics = {"ref": "PretrainMetricsAggregatorStage", "params": {"output_path": "/tmp/metrics.json"}}  # noqa: S108
        recipe = _recipe(_READER, _ASR, overlap, metrics, _WRITER)
        spot, why = checkpoint.at(recipe, index=3)
        assert spot is None
        assert "pretrain_long_form" in why
        assert "PretrainMetricsAggregatorStage" in why
        # Above the pair is fine: the counters are made and read entirely below the checkpoint.
        allowed, why_not = checkpoint.at(recipe, index=2)
        assert allowed is not None, why_not

    def test_cheap_work_is_not_worth_a_checkpoint(self) -> None:
        spot, why = checkpoint.advise(_recipe(_READER, _DUR, _WRITER))
        assert spot is None
        assert "expensive" in why

    def test_a_writer_already_past_the_expensive_work_is_the_checkpoint(self) -> None:
        """Advising a second writer beside one the recipe already has is noise, and would
        repeat itself every scan."""
        mid = {"ref": "ManifestWriterStage", "params": {"output_path": "/tmp/mid.jsonl"}}  # noqa: S108
        spot, why = checkpoint.advise(_recipe(_READER, _ASR, mid, _DUR, _WRITER))
        assert spot is None
        assert "already writes a manifest" in why

    def test_an_output_nothing_can_re_read_is_not_a_checkpoint(self, tmp_path: Any) -> None:  # noqa: ANN401
        """Sortformer fills an RTTM directory and no source stage can start a pipeline from
        one, so counting it would answer "you are covered" to a user whose diarization is
        recomputed on every request."""
        rttm_dir = tmp_path / "rttm"
        rttm_dir.mkdir()
        (rttm_dir / "clip.rttm").write_text("SPEAKER clip 1 0.0 1.0 <NA> <NA> spk0 <NA> <NA>\n")
        diarize = {"ref": "InferenceSortformerStage", "params": {"rttm_out_dir": str(rttm_dir)}}
        spot, why = checkpoint.advise(_recipe(_READER, diarize, _DUR, _WRITER))
        assert spot is not None, why
        assert spot.after_stage == "InferenceSortformerStage"

    def test_the_pipelines_own_final_writer_does_not_count_as_one(self) -> None:
        """Resuming from the final sink only serves a request that was already finished. The
        case this exists for is a changed tail, where that artifact no longer matches and the
        GPU stages are recomputed to rebuild it."""
        spot, _why = checkpoint.advise(_recipe(_READER, _ASR, _DUR, _WRITER))
        assert spot is not None
        assert spot.index < len(_recipe(_READER, _ASR, _DUR, _WRITER).stages)


class TestTheRecipeItHandsBack:
    def test_the_writer_appears_at_the_advised_position_and_nothing_else_moves(self) -> None:
        original = _recipe(_READER, _ASR, _DUR, _WRITER)
        spot, _ = checkpoint.advise(original)
        assert spot is not None
        out, err = checkpoint.insert(original, index=spot.index, output_path="/tmp/ck.jsonl")  # noqa: S108
        assert err == ""
        assert out is not None
        assert [s.ref for s in out.stages] == [
            "ManifestReader",
            "ASRStage",
            "ManifestWriterStage",
            "GetAudioDurationStage",
            "ManifestWriterStage",
        ]
        assert out.stages[spot.index].params == {"output_path": "/tmp/ck.jsonl"}  # noqa: S108
        assert [s.params for s in out.stages if s.ref != "ManifestWriterStage"] == [
            s.params for s in original.stages if s.ref != "ManifestWriterStage"
        ]

    def test_the_checkpointed_recipe_is_not_advised_again(self) -> None:
        original = _recipe(_READER, _ASR, _DUR, _WRITER)
        spot, _ = checkpoint.advise(original)
        assert spot is not None
        out, _ = checkpoint.insert(original, index=spot.index, output_path="/tmp/ck.jsonl")  # noqa: S108
        assert out is not None
        again, why = checkpoint.advise(out)
        assert again is None
        assert "already writes a manifest" in why

    def test_a_checkpoint_needs_somewhere_to_write(self) -> None:
        out, err = checkpoint.insert(_recipe(_READER, _ASR, _WRITER), index=2, output_path="")
        assert out is None
        assert "output path" in err

    def test_a_position_outside_the_recipe_is_refused(self) -> None:
        out, err = checkpoint.insert(_recipe(_READER, _ASR, _WRITER), index=9, output_path="/tmp/ck.jsonl")  # noqa: S108
        assert out is None
        assert "outside the recipe" in err


class TestACallersOwnPosition:
    def test_a_position_that_would_crash_is_refused(self) -> None:
        recipe = _recipe(_READER, _KEEPS_WAVEFORM, _ASR, _SANITIZER, _DUR, _WRITER)
        spot, why = checkpoint.at(recipe, index=3)
        assert spot is None
        assert "carrying audio in memory" in why

    def test_a_position_above_the_expensive_work_is_allowed_and_says_it_saves_nothing(self) -> None:
        """A preference, not an error: a user tuning a stage may want the checkpoint above it."""
        spot, why = checkpoint.at(_recipe(_READER, _ASR, _DUR, _WRITER), index=1)
        assert spot is not None, why
        assert spot.skips == []
        assert "saves little" in spot.as_dict()["effect"]

    def test_an_edge_of_the_recipe_is_not_a_position(self) -> None:
        spot, why = checkpoint.at(_recipe(_READER, _ASR, _WRITER), index=0)
        assert spot is None
        assert "no stage on one side" in why


class TestTheOfferOnAMiss:
    """``reuse`` recommends the simulated position, not the end of the recomputed prefix."""

    def test_the_offer_names_a_position_a_manifest_can_actually_hold(self) -> None:
        recipe = _recipe(_READER, _KEEPS_WAVEFORM, _ASR, _SANITIZER, _DUR, _WRITER)
        unsaved = {"stages": ["ManifestReader", "ResampleAudioStage", "ASRStage"]}
        offer = reuse._persist_offer(recipe, unsaved)
        assert offer is not None
        # The prefix ends at the ASR stage, and a writer there is exactly what crashes.
        assert offer["after_stage"] == "AudioToDocumentStage"
        assert "ASRStage" in offer["effect"]

    def test_a_prefix_that_already_persisted_is_offered_nothing(self) -> None:
        offer = reuse._persist_offer(
            _recipe(_READER, _ASR, _WRITER),
            {"stages": ["ManifestWriterStage"], "resume_point_persists": True},
        )
        assert offer is None

    def test_a_pipeline_with_no_legal_position_says_so_rather_than_advising_one(self) -> None:
        recipe = _recipe(_READER, _KEEPS_WAVEFORM, _ASR, _DUR, _WRITER)
        offer = reuse._persist_offer(recipe, {"stages": ["ManifestReader", "ASRStage"]})
        assert offer is not None
        assert offer["action"] == "no_checkpoint"
        assert "ASRStage" in offer["why"]


class TestTheVerb:
    def test_without_a_path_it_only_advises(self) -> None:
        out = verbs.add_checkpoint(_recipe(_READER, _ASR, _DUR, _WRITER))
        assert out["status"] == "advice"
        assert out["advice"]["after_stage"] == "ASRStage"
        assert "recipe" not in out

    def test_with_a_path_it_returns_the_recipe_and_runs_nothing(self) -> None:
        out = verbs.add_checkpoint(_recipe(_READER, _ASR, _DUR, _WRITER), output_path="/tmp/ck.jsonl")  # noqa: S108
        assert out["status"] == "ok"
        assert [s["ref"] for s in out["recipe"]["stages"]].count("ManifestWriterStage") == 2
        assert "validate" in out["next"]

    def test_a_named_stage_is_still_checked(self) -> None:
        out = verbs.add_checkpoint(
            _recipe(_READER, _KEEPS_WAVEFORM, _ASR, _SANITIZER, _DUR, _WRITER),
            after="ASRStage",
        )
        assert out["status"] == "no_checkpoint"
        assert "carrying audio in memory" in out["reason"]

    def test_a_stage_that_is_not_in_the_recipe_is_an_error(self) -> None:
        out = verbs.add_checkpoint(_recipe(_READER, _ASR, _WRITER), after="NoSuchStage")
        assert out["status"] == "error"
        assert "NoSuchStage" in out["reason"]

    def test_a_path_outside_a_locked_workspace_is_refused(self, monkeypatch: Any, tmp_path: Any) -> None:  # noqa: ANN401
        """The checkpoint is a file the agent told the user to write, so its path answers to the
        same lock as everything else the agent proposes writing."""
        monkeypatch.setenv("AUDIO_AGENT_WORKSPACE", str(tmp_path))
        out = verbs.add_checkpoint(_recipe(_READER, _ASR, _WRITER), output_path="/etc/ck.jsonl")
        assert out["status"] == "refused"
        assert "workspace" in out["reason"].lower()


_KEY = "stat:0123456789abcdef"


def _managed_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the run store at a scratch tree so tests never touch the real one."""
    monkeypatch.setenv("AUDIO_AGENT_RUNS_DIR", str(tmp_path / "runs"))


def _preserve(target: float) -> dict[str, Any]:
    return {
        "ref": "PreserveByValueStage",
        "params": {"input_value_key": "duration", "target_value": target, "operator": "ge"},
    }


class TestTheLocationIsDerived:
    """The user is asked WHETHER to spend a checkpoint, never WHERE to put it.

    Naming a path is a question nobody can answer well: it decides reuse, so a wrong guess
    silently costs the GPU work the checkpoint existed to save. The step key already knows
    the answer.
    """

    def test_data_addresses_the_checkpoint_by_its_step_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from nemo_curator.audio_agent import artifacts, run_store

        _managed_runs(monkeypatch, tmp_path)
        out = verbs.add_checkpoint(_recipe(_READER, _ASR, _DUR, _WRITER), data=_KEY)

        assert out["status"] == "ok"
        assert out["path_source"] == "derived"
        materialized = Recipe.from_dict(out["recipe"]).freeze()
        index = [s.ref for s in materialized.stages].index("ManifestWriterStage")
        step_key = artifacts.plan_steps(materialized, _KEY)[index].step_key
        assert out["output_path"] == run_store.checkpoint_path(step_key)

    def test_the_address_survives_a_downstream_threshold_change(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The whole point: retuning a threshold below the checkpoint must still find it.

        A path named after the recipe hash cannot do this -- inserting the writer changes that
        hash, and so does every later edit. The step key of everything ABOVE the checkpoint
        does not move, which is why it is addressed by that instead.
        """
        _managed_runs(monkeypatch, tmp_path)
        loose = _recipe(_READER, _ASR, _preserve(4.0), _WRITER)
        strict = _recipe(_READER, _ASR, _preserve(9.0), _WRITER)

        assert loose.config_hash != strict.config_hash
        assert (
            verbs.add_checkpoint(loose, data=_KEY)["output_path"]
            == verbs.add_checkpoint(strict, data=_KEY)["output_path"]
        )

    def test_two_datasets_never_share_an_address(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _managed_runs(monkeypatch, tmp_path)
        recipe = _recipe(_READER, _ASR, _DUR, _WRITER)

        mine = verbs.add_checkpoint(recipe, data=_KEY)["output_path"]
        theirs = verbs.add_checkpoint(recipe, data="stat:ffffffffffffffff")["output_path"]
        assert mine != theirs

    def test_an_explicit_path_still_wins(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """For the user who asked to keep the metadata somewhere of their own."""
        _managed_runs(monkeypatch, tmp_path)
        out = verbs.add_checkpoint(
            _recipe(_READER, _ASR, _DUR, _WRITER),
            data=_KEY,
            output_path=str(tmp_path / "mine.jsonl"),
        )
        assert out["path_source"] == "explicit"
        assert out["output_path"] == str(tmp_path / "mine.jsonl")

    def test_without_data_it_still_only_advises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """No dataset key means no step key, so the caller is asked exactly as before."""
        _managed_runs(monkeypatch, tmp_path)
        out = verbs.add_checkpoint(_recipe(_READER, _ASR, _DUR, _WRITER))
        assert out["status"] == "advice"
        assert "recipe" not in out

    def test_an_unkeyable_source_falls_back_to_asking(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Degrades to the old question rather than to a traceback or a wrong address.

        A source that cannot be keyed at all is the only case that reaches this: an
        unreadable path still yields a weaker key tier, and a distinct one per path.
        """
        _managed_runs(monkeypatch, tmp_path)
        monkeypatch.setattr(verbs, "_recipe_dataset_key", lambda *_a, **_k: "")
        out = verbs.add_checkpoint(_recipe(_READER, _ASR, _DUR, _WRITER), data=str(tmp_path))
        assert out["status"] == "advice"
        assert "recipe" not in out

    def test_an_unreadable_source_is_still_keyed_apart_from_another(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Two sources that cannot be profiled must not collapse onto one address.

        Sharing one would hand a run someone else's checkpoint, which is the failure the
        managed location exists to make impossible.
        """
        _managed_runs(monkeypatch, tmp_path)
        recipe = _recipe(_READER, _ASR, _DUR, _WRITER)

        one = verbs.add_checkpoint(recipe, data=str(tmp_path / "absent-a"))
        two = verbs.add_checkpoint(recipe, data=str(tmp_path / "absent-b"))
        assert one["output_path"] != two["output_path"]
