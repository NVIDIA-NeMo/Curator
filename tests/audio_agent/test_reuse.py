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

"""Content-addressed execution reuse: identity split, dataset key, artifacts, decisions.

See ``nemo_curator/audio_agent/REUSE_ARCHITECTURE.md`` for the design these pin down.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any, ClassVar

import pytest

from nemo_curator.audio_agent import artifacts, continuation, profiler, reuse, run_index, verbs
from nemo_curator.audio_agent.recipe import Recipe

_READER = {"ref": "ManifestReader", "params": {"manifest_path": "/tmp/m.jsonl"}}  # noqa: S108
_DUR = {"ref": "GetAudioDurationStage", "params": {}}
_WRITER = {"ref": "ManifestWriterStage", "params": {"output_path": "/tmp/out.jsonl"}}  # noqa: S108
# A stage the cards call expensive, and which persists nothing of its own -- the shape that made
# an untimed hour look free.
_ASR = {
    "ref": "ASRStage",
    "params": {
        "adapter_target": "nemo_curator.models.asr.nemo_asr.NeMoASRAdapter",
        "model_id": "nvidia/parakeet-tdt-0.6b-v2",
        "audio_filepath_key": "audio_filepath",
    },
}


def _frozen(stages: list[dict], **kw: object) -> Recipe:
    return Recipe.from_dict({"stages": stages, **kw}).freeze()


class TestIdentitySplit:
    """semantic_hash answers 'same bytes?'; config_hash still covers everything."""

    def test_execution_knobs_do_not_change_semantic_hash(self) -> None:
        base = _frozen([_READER, _DUR, _WRITER])
        tuned = _frozen(
            [
                _READER,
                {
                    "ref": "GetAudioDurationStage",
                    "params": {"batch_size": 64, "num_workers": 4, "resources": {"gpus": 1}},
                },
                _WRITER,
            ]
        )
        assert tuned.semantic_hash == base.semantic_hash
        # ...but the confirm gate still sees a different plan, as it must.
        assert tuned.config_hash != base.config_hash

    def test_output_location_does_not_change_semantic_hash(self) -> None:
        base = _frozen([_READER, _DUR, _WRITER])
        elsewhere = _frozen(
            [_READER, _DUR, {"ref": "ManifestWriterStage", "params": {"output_path": "/tmp/other.jsonl"}}]  # noqa: S108
        )
        assert elsewhere.semantic_hash == base.semantic_hash
        assert elsewhere.config_hash != base.config_hash

    def test_acceptance_change_leaves_semantic_hash_alone(self) -> None:
        base = _frozen([_READER, _DUR, _WRITER])
        stricter = _frozen(
            [_READER, _DUR, _WRITER],
            acceptance_criteria=[
                {"id": "keep", "type": "yield", "check": {"op": ">=", "value": 0.9}, "severity": "must"}
            ],
        )
        assert stricter.semantic_hash == base.semantic_hash  # same data -> reuse
        assert stricter.contract_hash != base.contract_hash  # different bar -> re-verify
        assert stricter.config_hash != base.config_hash

    def test_semantic_param_change_does_change_semantic_hash(self) -> None:
        base = _frozen([_READER, _DUR, _WRITER])
        other = _frozen(
            [_READER, {"ref": "GetAudioDurationStage", "params": {"input_residency": "waveform"}}, _WRITER]
        )
        assert other.semantic_hash != base.semantic_hash

    def test_config_hash_is_unchanged_by_the_split(self) -> None:
        # The confirm-gate anchor must keep its exact historical value.
        rec = _frozen([_READER, _WRITER])
        assert rec.config_hash == rec.compute_hash()
        assert len(rec.config_hash) == 16


class TestContractShape:
    """A contract that cannot be verified must be refused, not quietly mangled.

    ``list()`` over a mapping yields its keys, so ``{must: [...]}`` used to load as
    ``["must"]`` -- and a run then reported success having never checked its own bar.
    """

    def test_a_mapping_contract_is_refused_with_the_expected_shape(self) -> None:
        with pytest.raises(ValueError, match="must be a LIST of criterion mappings"):
            Recipe.from_dict({"stages": [_READER], "acceptance_criteria": {"must": ["output exists"]}})

    def test_free_text_criteria_are_refused(self) -> None:
        with pytest.raises(ValueError, match="entries must be mappings"):
            Recipe.from_dict({"stages": [_READER], "acceptance_criteria": ["output manifest exists"]})

    def test_the_real_shape_still_loads(self) -> None:
        crit = [{"id": "keep", "type": "yield", "check": {"op": ">=", "value": 0.9}, "severity": "must"}]
        rec = Recipe.from_dict({"stages": [_READER], "acceptance_criteria": crit}).freeze()
        assert rec.acceptance_criteria == crit
        # Strict parsing validates a copy; it must never inject defaults into the
        # stored contract or invalidate existing confirmation/reuse identities.
        assert rec.config_hash == "703a8bd92bc85509"
        assert rec.contract_hash == "ce95ca83d63107b5"

    def test_an_unverifiable_contract_is_reported_not_swallowed(self) -> None:
        # Constructed past the door (a hand-built Recipe skips from_dict), so the run
        # must still say the bar went unchecked rather than imply it passed.
        rec = Recipe(stages=[], acceptance_criteria=["not a criterion"])
        result = verbs._acceptance_result(rec, None, [], [])
        assert result["overall"] == "unverifiable"
        assert "could not verify" in result["reason"]


def _wav(path: Path, *, payload: bytes = b"RIFFfakewavdata") -> Path:
    path.write_bytes(payload)
    return path


class TestTieredDatasetKey:
    """The dataset key must catch in-place edits, and must not be moved by a stage's
    own intermediates landing back in the source folder."""

    def test_stat_entry_reports_whether_metadata_is_complete(self, tmp_path: Path) -> None:
        audio = _wav(tmp_path / "a.wav")
        entry, ok = profiler._stat_entry(str(audio), root=str(tmp_path))
        assert ok is True
        assert entry.startswith("a.wav|")
        assert not entry.endswith("|?")

        missing_entry, missing_ok = profiler._stat_entry(str(tmp_path / "missing.wav"), root=str(tmp_path))
        assert missing_ok is False
        assert missing_entry == "missing.wav|?"

    def test_folder_uses_the_stat_tier(self, tmp_path: Path) -> None:
        _wav(tmp_path / "a.wav")
        prof = profiler.profile_data(str(tmp_path))
        assert prof.fingerprint_tier == "stat"
        assert prof.dataset_key().startswith("stat:")

    def test_folder_stat_failure_downgrades_to_identity_backed_shape(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        audio = _wav(tmp_path / "a.wav")
        target = os.path.abspath(str(audio))
        real_stat = profiler.os.stat

        def fail_target(path: object, *args: object, **kwargs: object) -> os.stat_result:
            if isinstance(path, (str, bytes, os.PathLike)) and os.path.abspath(os.fsdecode(path)) == target:
                raise OSError("simulated stat failure")  # noqa: EM101
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr(profiler.os, "stat", fail_target)
        prof = profiler.profile_data(str(tmp_path))

        assert prof.fingerprint_tier == "shape"
        assert prof.stat_digest == ""
        assert prof.identity_digest
        assert prof.dataset_key() == f"shape:{prof.identity_digest}"
        assert any("could not stat 1 source file" in note for note in prof.notes)

    def test_in_place_edit_changes_the_key(self, tmp_path: Path) -> None:
        f = _wav(tmp_path / "a.wav")
        before = profiler.profile_data(str(tmp_path)).dataset_key()
        _wav(f, payload=b"RIFFdifferentbytes")
        os.utime(f, (1, 1))  # a rewritten file with a different size + mtime
        assert profiler.profile_data(str(tmp_path)).dataset_key() != before

    def test_split_chunks_do_not_self_invalidate(self, tmp_path: Path) -> None:
        # SplitLongAudioStage writes "<stem>.<k>_of_<N>.wav" beside its input; those are the
        # pipeline's own output, so scanning them would make every split run invalidate itself.
        _wav(tmp_path / "long.wav")
        before = profiler.profile_data(str(tmp_path))
        _wav(tmp_path / "long.1_of_2.wav")
        _wav(tmp_path / "long.2_of_2.wav")
        after = profiler.profile_data(str(tmp_path))
        assert after.dataset_key() == before.dataset_key()
        assert after.num_files == before.num_files == 1
        assert after.excluded_intermediates == 2

    def test_source_adapter_can_include_files_the_source_stage_really_emits(self, tmp_path: Path) -> None:
        _wav(tmp_path / "long.wav")
        _wav(tmp_path / "long.1_of_2.wav")

        prof = profiler.profile_data(
            str(tmp_path),
            exclude_stage_intermediates=False,
        )

        assert prof.num_files == 2
        assert prof.excluded_intermediates == 0

    def test_folder_selector_matches_extensions_recursion_and_limit(self, tmp_path: Path) -> None:
        _wav(tmp_path / "a.wav")
        (tmp_path / "b.flac").write_bytes(b"fLaC")
        nested = tmp_path / "nested"
        nested.mkdir()
        (nested / "c.flac").write_bytes(b"fLaC")

        prof = profiler.profile_data(
            str(tmp_path),
            folder_extensions=["flac"],
            recursive=False,
            max_files=1,
            exclude_stage_intermediates=False,
        )

        assert prof.num_files == 1

    def test_a_persons_own_files_are_not_mistaken_for_split_chunks(self, tmp_path: Path) -> None:
        # "part 1 of 3" is an ordinary way to name your own recordings. Excluding these hid them
        # from the dataset key, so editing one would not invalidate a reused result.
        for name in ("interview.1_of_3.wav", "interview.2_of_3.wav", "interview.3_of_3.wav"):
            _wav(tmp_path / name)
        prof = profiler.profile_data(str(tmp_path))
        assert prof.num_files == 3
        assert prof.excluded_intermediates == 0

    def test_editing_such_a_file_invalidates_reuse(self, tmp_path: Path) -> None:
        target = _wav(tmp_path / "interview.2_of_3.wav")
        _wav(tmp_path / "interview.1_of_3.wav")
        before = profiler.profile_data(str(tmp_path)).dataset_key()
        _wav(target, payload=b"RIFFdifferentbytesentirely")
        os.utime(target, (1, 1))
        assert profiler.profile_data(str(tmp_path)).dataset_key() != before

    def test_a_chunk_is_excluded_only_beside_the_file_it_came_from(self, tmp_path: Path) -> None:
        # The source is the corroboration: a chunk was split FROM something still sitting there.
        assert profiler._is_stage_intermediate("long.1_of_2.wav", {"long.wav", "long.1_of_2.wav"}) is True
        assert profiler._is_stage_intermediate("long.1_of_2.wav", {"long.1_of_2.wav"}) is False
        assert profiler._is_stage_intermediate("long.1_of_2.wav", {"long.flac", "long.1_of_2.wav"}) is True
        assert profiler._is_stage_intermediate("ordinary.wav", {"ordinary.wav"}) is False

    def test_the_source_is_matched_whatever_its_case(self, tmp_path: Path) -> None:
        _wav(tmp_path / "Lecture.WAV")
        _wav(tmp_path / "Lecture.1_of_2.wav")
        assert profiler.profile_data(str(tmp_path)).excluded_intermediates == 1

    def test_chunks_in_one_folder_do_not_hide_files_in_another(self, tmp_path: Path) -> None:
        # Corroboration is per-directory: a "long.wav" elsewhere says nothing about this folder.
        (tmp_path / "src").mkdir()
        (tmp_path / "mine").mkdir()
        _wav(tmp_path / "src" / "long.wav")
        _wav(tmp_path / "src" / "long.1_of_2.wav")
        _wav(tmp_path / "mine" / "long.1_of_2.wav")
        prof = profiler.profile_data(str(tmp_path))
        assert prof.excluded_intermediates == 1
        assert prof.num_files == 2  # src/long.wav and mine/long.1_of_2.wav

    def test_manifest_content_change_changes_the_key(self, tmp_path: Path) -> None:
        m = tmp_path / "m.jsonl"
        m.write_text(json.dumps({"audio_filepath": str(_wav(tmp_path / "a.wav"))}) + "\n")
        before = profiler.profile_data(str(m)).dataset_key()
        with m.open("a") as fh:
            fh.write(json.dumps({"audio_filepath": str(_wav(tmp_path / "b.wav"))}) + "\n")
        assert profiler.profile_data(str(m)).dataset_key() != before

    def test_malformed_utf8_manifest_is_structured_and_never_high_trust(
        self,
        tmp_path: Path,
    ) -> None:
        manifest = tmp_path / "bad.jsonl"
        manifest.write_bytes(b"\xff\xfe")

        prof = profiler.profile_data(str(manifest))

        assert prof.kind == "manifest"
        assert prof.source_errors
        assert prof.fingerprint_tier == "shape"
        assert prof.stat_digest == ""

    def test_manifest_with_valid_absolute_reference_stays_stat(self, tmp_path: Path) -> None:
        audio = _wav(tmp_path / "a.wav")
        manifest = tmp_path / "m.jsonl"
        manifest.write_text(json.dumps({"audio_filepath": str(audio)}) + "\n")

        prof = profiler.profile_data(str(manifest))

        assert prof.fingerprint_tier == "stat"
        assert prof.stat_digest
        assert prof.identity_digest == ""

    def test_manifest_stat_failure_downgrades_to_identity_backed_shape(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.wav"
        manifest = tmp_path / "m.jsonl"
        manifest.write_text(json.dumps({"audio_filepath": str(missing)}) + "\n")

        prof = profiler.profile_data(str(manifest))

        assert prof.fingerprint_tier == "shape"
        assert prof.stat_digest == ""
        assert prof.identity_digest
        assert prof.dataset_key() == f"shape:{prof.identity_digest}"
        assert str(missing) in prof.unreadable
        assert any("1 local file stat failure" in note for note in prof.notes)

    def test_remote_manifest_reference_never_gets_stat_tier(self, tmp_path: Path) -> None:
        remote = "memory://bucket/a.wav"
        manifest = tmp_path / "m.jsonl"
        manifest.write_text(json.dumps({"audio_filepath": remote}) + "\n")

        prof = profiler.profile_data(str(manifest))

        assert prof.fingerprint_tier == "shape"
        assert prof.dataset_key() == f"shape:{prof.identity_digest}"
        assert remote not in prof.unreadable
        assert any("1 remote reference" in note for note in prof.notes)

    def test_relative_manifest_reference_never_gets_stat_tier(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _wav(tmp_path / "a.wav")
        manifest = tmp_path / "m.jsonl"
        manifest.write_text(json.dumps({"audio_filepath": "a.wav"}) + "\n")
        monkeypatch.chdir(tmp_path)  # prove even a currently resolvable relative ref stays low-trust

        prof = profiler.profile_data(str(manifest))

        assert prof.fingerprint_tier == "shape"
        assert prof.dataset_key() == f"shape:{prof.identity_digest}"
        assert any("1 relative reference" in note for note in prof.notes)

    def test_explicit_long_form_resolution_can_identify_relative_audio(self, tmp_path: Path) -> None:
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        _wav(audio_dir / "a.wav")
        manifest = tmp_path / "m.jsonl"
        manifest.write_text(json.dumps({"audio_filepath": "a.wav"}) + "\n")

        prof = profiler.profile_data(
            str(manifest),
            audio_dir=str(audio_dir),
            audio_path_resolution="relative",
        )

        assert prof.fingerprint_tier == "stat"
        assert not any("relative reference" in note for note in prof.notes)

    def test_supplemental_definition_file_changes_dataset_key(self, tmp_path: Path) -> None:
        _wav(tmp_path / "a.wav")
        transcript = tmp_path / "dev.tsv"
        transcript.write_text("a.wav\tone\n", encoding="utf-8")
        before = profiler.profile_data(
            str(tmp_path),
            identity_files=[str(transcript)],
        ).dataset_key()

        transcript.write_text("a.wav\ttwo\n", encoding="utf-8")
        after = profiler.profile_data(
            str(tmp_path),
            identity_files=[str(transcript)],
        ).dataset_key()

        assert before != after

    def test_shape_tier_is_the_honest_fallback(self) -> None:
        # An uninterpretable source can't be statted -> low-trust shape tier, never a fake
        # "stat" claim that would make reuse look safer than it is.
        prof = profiler.profile_data("/nonexistent/source/path")
        assert prof.fingerprint_tier == "shape"
        assert prof.dataset_key().startswith("shape:")


# --------------------------------------------------------------------------- artifact fixtures
_KEY = "stat:deadbeefdeadbeef"


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated artifact/run store so tests never touch the developer's real history."""
    monkeypatch.setenv("AUDIO_AGENT_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


def _out(tmp_path: Path, name: str = "out.jsonl") -> str:
    path = tmp_path / name
    path.write_text('{"audio_filepath": "/tmp/a.wav"}\n')
    return str(path)


def _pipeline(tmp_path: Path, *, source: str | None = None) -> tuple[Recipe, str, str]:
    """reader -> writer(mid) -> duration -> writer(final): two persisting reuse points."""
    mid, final = str(tmp_path / "mid.jsonl"), str(tmp_path / "final.jsonl")
    reader = {"ref": "ManifestReader", "params": {"manifest_path": source}} if source is not None else _READER
    rec = _frozen(
        [
            reader,
            {"ref": "ManifestWriterStage", "params": {"output_path": mid}},
            _DUR,
            {"ref": "ManifestWriterStage", "params": {"output_path": final}},
        ]
    )
    return rec, mid, final


def _publish(
    rec: Recipe,
    index: int,
    *,
    dataset_key: str = _KEY,
    duration_sec: float = 120.0,
    **kw: Any,  # noqa: ANN401
) -> artifacts.Artifact:
    plan = artifacts.plan_steps(rec, dataset_key)[index]
    Path(plan.uri).parent.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(plan.uri):
        Path(plan.uri).write_text('{"audio_filepath": "/tmp/a.wav"}\n')
    art = artifacts.Artifact(
        step_key=plan.step_key,
        input_key=plan.input_key,
        stage_ref=plan.stage_ref,
        stage_index=plan.index,
        semantic_params=plan.semantic_params,
        uri=plan.uri,
        kind=plan.kind,
        dataset_key=dataset_key,
        fingerprint_tier=kw.pop("fingerprint_tier", "stat"),
        impl_version=kw.pop("impl_version", plan.impl_version),
        code_version=kw.pop("code_version", artifacts.code_version()),
        deterministic=kw.pop("deterministic", plan.deterministic),
        duration_sec=duration_sec,
        produced_roles=kw.pop("produced_roles", ["audio_filepath"]),
        produced_keys=kw.pop("produced_keys", ["audio_filepath"]),
        **kw,
    )
    return artifacts.publish(art)


class TestReuseDecision:
    def test_an_empty_dataset_key_never_probes_or_claims_reuse(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rec = _frozen([_READER, _DUR, _WRITER])

        def unexpected_probe(*_args: Any, **_kwargs: Any) -> Any:  # noqa: ANN401
            msg = "an unidentified dataset must not enter the artifact namespace"
            raise AssertionError(msg)

        monkeypatch.setattr(artifacts, "plan_steps", unexpected_probe)
        monkeypatch.setattr(artifacts, "lookup", unexpected_probe)

        result = reuse.scan(rec, dataset_key="")

        assert result["decision"] == "fresh"
        assert result["dataset_key"] == ""
        assert result["reuse_point"] is None
        assert result["candidates"] == []
        assert result["prompt_user"] is False
        assert "identity is unavailable" in result["rationale"]

    def test_an_empty_dataset_key_publishes_no_artifact(
        self,
        store: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rec, _mid, _final = _pipeline(store)

        def unexpected_plan(*_args: Any, **_kwargs: Any) -> Any:  # noqa: ANN401
            msg = "publishing must stop before planning an empty-key artifact"
            raise AssertionError(msg)

        monkeypatch.setattr(artifacts, "plan_steps", unexpected_plan)

        published = verbs._publish_artifacts(
            rec,
            [object() for _ in rec.stages],
            dataset_key="",
            fingerprint_tier="",
            per_stage={},
            run_id="unknown-data",
            input_count=1,
            data_profile=None,
            started_at="",
            ended_at="",
        )

        assert published == []
        assert not artifacts.list_artifacts()

    def test_fresh_when_nothing_was_ever_run(self, store: Path) -> None:
        rec, _mid, _final = _pipeline(store)
        result = reuse.scan(rec, dataset_key=_KEY)
        assert result["decision"] == "fresh"
        assert result["prompt_user"] is False  # nothing to offer -> never ask
        assert "no prior artifact" in result["rationale"]

    def test_already_done_when_the_final_step_is_published(self, store: Path) -> None:
        rec, _mid, final = _pipeline(store)
        _publish(rec, 3)
        result = reuse.scan(rec, dataset_key=_KEY)
        assert result["decision"] == "already_done"
        assert result["run_stages"] == []
        assert result["reuse_point"]["uri"] == final
        assert result["prompt_user"] is True  # 120s saved is worth a question
        assert result["candidates"][0]["output"] == final

    def test_incremental_reuses_the_prefix_and_runs_the_rest(self, store: Path) -> None:
        rec, mid, _final = _pipeline(store)
        _publish(rec, 1)
        result = reuse.scan(rec, dataset_key=_KEY)
        assert result["decision"] == "incremental"
        assert result["reuse_point"]["uri"] == mid
        assert result["run_stages"] == ["GetAudioDurationStage", "ManifestWriterStage"]

    def test_deepest_valid_step_wins(self, store: Path) -> None:
        rec, _mid, final = _pipeline(store)
        _publish(rec, 1)
        _publish(rec, 3)
        assert reuse.scan(rec, dataset_key=_KEY)["reuse_point"]["uri"] == final

    def test_trivial_saving_is_taken_without_asking(self, store: Path) -> None:
        rec, _mid, _final = _pipeline(store)
        _publish(rec, 3, duration_sec=2.0)
        result = reuse.scan(rec, dataset_key=_KEY)
        assert result["decision"] == "already_done"
        assert result["prompt_user"] is False  # not worth a question, but still disclosed

    def test_the_saving_counts_the_expensive_steps_that_persisted_nothing(self, store: Path) -> None:
        # A pipeline that only writes at the end: the writer costs milliseconds, the hour of
        # ASR before it costs everything. Charging only the writer would serve yesterday's
        # transcripts without ever asking.
        rec, _mid, _final = _pipeline(store)
        _publish(rec, 3, duration_sec=0.2, cumulative_sec=3600.0)
        result = reuse.scan(rec, dataset_key=_KEY)
        assert result["estimated_saving_sec"] == 3600.0
        assert result["prompt_user"] is True
        assert result["candidates"][0]["estimated_saving_sec"] == 3600.0

    def test_low_trust_candidate_defaults_to_fresh(self, store: Path) -> None:
        rec, _mid, _final = _pipeline(store)
        _publish(rec, 3, fingerprint_tier="shape")
        result = reuse.scan(rec, dataset_key=_KEY)
        assert result["recommended"] == "fresh"
        assert result["candidates"][0]["trust"] == "low"
        assert any("edited in place" in w for w in result["candidates"][0]["weaknesses"])


class TestReuseSurvivesNonSemanticChanges:
    """The headline fix: knobs that cannot change a single output byte must not force a rerun."""

    def test_batch_size_and_resources_change_still_reuses(self, store: Path) -> None:
        rec, _mid, final = _pipeline(store)
        _publish(rec, 3)
        tuned = _frozen(
            [
                _READER,
                {"ref": "ManifestWriterStage", "params": {"output_path": str(store / "mid.jsonl")}},
                {"ref": "GetAudioDurationStage", "params": {"batch_size": 32, "resources": {"gpus": 1}}},
                {"ref": "ManifestWriterStage", "params": {"output_path": final}},
            ]
        )
        assert reuse.scan(tuned, dataset_key=_KEY)["decision"] == "already_done"

    def test_output_path_change_still_reuses(self, store: Path) -> None:
        rec, mid, _final = _pipeline(store)
        _publish(rec, 3)
        elsewhere = _frozen(
            [
                _READER,
                {"ref": "ManifestWriterStage", "params": {"output_path": mid}},
                _DUR,
                {"ref": "ManifestWriterStage", "params": {"output_path": str(store / "somewhere_else.jsonl")}},
            ]
        )
        assert reuse.scan(elsewhere, dataset_key=_KEY)["decision"] == "already_done"

    def test_stricter_acceptance_reuses_data_and_reverifies(self, store: Path) -> None:
        rec, mid, final = _pipeline(store)
        _publish(rec, 3)
        stricter = _frozen(
            [
                _READER,
                {"ref": "ManifestWriterStage", "params": {"output_path": mid}},
                _DUR,
                {"ref": "ManifestWriterStage", "params": {"output_path": final}},
            ],
            acceptance_criteria=[
                {"id": "y", "type": "yield", "check": {"op": ">=", "value": 0.99}, "severity": "must"}
            ],
        )
        assert reuse.scan(stricter, dataset_key=_KEY)["decision"] == "already_done"
        assert stricter.contract_hash != rec.contract_hash  # ...but the bar must be re-checked

    def test_a_real_param_change_forces_a_fresh_run(self, store: Path) -> None:
        rec, mid, final = _pipeline(store)
        _publish(rec, 3)
        changed = _frozen(
            [
                _READER,
                {"ref": "ManifestWriterStage", "params": {"output_path": mid}},
                {"ref": "GetAudioDurationStage", "params": {"input_residency": "waveform"}},
                {"ref": "ManifestWriterStage", "params": {"output_path": final}},
            ]
        )
        assert reuse.scan(changed, dataset_key=_KEY)["decision"] != "already_done"


class TestArtifactValidity:
    """Every rejection here is a way a reused result could silently differ from a fresh one."""

    def test_changed_source_data_is_rejected(self, store: Path) -> None:
        rec, _mid, _final = _pipeline(store)
        art = _publish(rec, 3)
        reasons = artifacts.invalid_reasons(art, dataset_key="stat:something-else")
        assert any("source data changed" in r for r in reasons)
        assert reuse.scan(rec, dataset_key="stat:something-else")["decision"] == "fresh"

    def test_changed_source_data_says_so_instead_of_never_ran_this(self, store: Path) -> None:
        # New data re-roots the whole key chain, so the plain probe finds nothing at all —
        # reporting that as "never ran this before" would be true of the key and useless
        # to the user, who changed one file in a folder they have already processed.
        rec, _mid, _final = _pipeline(store)
        _publish(rec, 3)
        result = reuse.scan(rec, dataset_key="stat:folder-gained-a-file")
        assert result["decision"] == "fresh"
        assert result["prior_on_other_data"]["stage"] == "ManifestWriterStage"
        assert "source identity changed" in result["rationale"]

    def test_a_genuinely_new_pipeline_is_not_blamed_on_the_data(self, store: Path) -> None:
        rec, _mid, _final = _pipeline(store)
        result = reuse.scan(rec, dataset_key=_KEY)
        assert result["prior_on_other_data"] is None
        assert "no prior artifact" in result["rationale"]

    def test_partial_output_without_a_marker_is_rejected(self, store: Path) -> None:
        # A crashed run leaves an appended-but-incomplete JSONL that looks perfectly valid.
        rec, _mid, _final = _pipeline(store)
        art = _publish(rec, 3)
        os.remove(artifacts.marker_path(art.uri))
        _found, reasons = artifacts.lookup(art.step_key, dataset_key=_KEY)
        assert any("_COMPLETE" in r for r in reasons)
        assert reuse.scan(rec, dataset_key=_KEY)["decision"] == "fresh"

    def test_marker_from_a_different_step_is_rejected(self, store: Path) -> None:
        rec, _mid, _final = _pipeline(store)
        art = _publish(rec, 3)
        artifacts.write_marker(art.uri, step_key="some-other-step", rows=1)
        assert any("different step" in r for r in artifacts.invalid_reasons(art, dataset_key=_KEY))

    def test_same_size_same_row_manifest_edit_is_rejected(self, store: Path) -> None:
        rec, _mid, _final = _pipeline(store)
        art = _publish(rec, 3)
        before_size = os.path.getsize(art.uri)
        Path(art.uri).write_text('{"audio_filepath": "/tmp/b.wav"}\n', encoding="utf-8")
        assert os.path.getsize(art.uri) == before_size

        reasons = artifacts.invalid_reasons(art, dataset_key=_KEY)

        assert any("changed after" in reason for reason in reasons)
        assert reuse.scan(rec, dataset_key=_KEY)["decision"] == "fresh"

    def test_same_size_audio_edit_inside_directory_is_rejected(
        self,
        store: Path,
    ) -> None:
        output = store / "audio"
        output.mkdir()
        audio = output / "clip.wav"
        audio.write_bytes(b"RIFFaaaa")
        artifact = artifacts.publish(
            artifacts.Artifact(
                step_key="directory-step",
                uri=str(output),
                kind="audio_dir",
                dataset_key=_KEY,
                fingerprint_tier="stat",
                code_version=artifacts.code_version(),
            )
        )
        audio.write_bytes(b"RIFFbbbb")

        reasons = artifacts.invalid_reasons(artifact, dataset_key=_KEY)

        assert any("changed after" in reason for reason in reasons)

    def test_legacy_marker_without_content_digest_is_rejected(self, store: Path) -> None:
        rec, _mid, _final = _pipeline(store)
        art = _publish(rec, 3)
        marker = artifacts.read_marker(art.uri)
        assert marker is not None
        marker.pop("content_digest")
        Path(artifacts.marker_path(art.uri)).write_text(json.dumps(marker), encoding="utf-8")

        reasons = artifacts.invalid_reasons(art, dataset_key=_KEY)

        assert any("predates serialized-content binding" in reason for reason in reasons)

    def test_artifact_outside_workspace_is_rejected_before_it_is_read(
        self,
        store: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        workspace = store / "workspace"
        workspace.mkdir()
        outside = store / "outside.jsonl"
        outside.write_text('{"audio_filepath":"/tmp/a.wav"}\n', encoding="utf-8")
        artifact = artifacts.publish(
            artifacts.Artifact(
                step_key="outside-step",
                uri=str(outside),
                kind="manifest",
                dataset_key=_KEY,
                fingerprint_tier="stat",
                code_version=artifacts.code_version(),
            )
        )
        monkeypatch.setenv("AUDIO_AGENT_WORKSPACE", str(workspace))
        monkeypatch.setattr(
            artifacts,
            "content_digest",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("outside artifact was read")),
        )

        reasons = artifacts.invalid_reasons(artifact, dataset_key=_KEY)

        assert any("outside the allowed workspace" in reason for reason in reasons)

    def test_deleted_output_is_rejected(self, store: Path) -> None:
        rec, _mid, _final = _pipeline(store)
        art = _publish(rec, 3)
        os.remove(art.uri)
        assert any("no longer exists" in r for r in artifacts.invalid_reasons(art, dataset_key=_KEY))

    def test_a_changed_stage_implementation_is_rejected(self, store: Path) -> None:
        rec, _mid, _final = _pipeline(store)
        art = _publish(rec, 3, impl_version="impl:0000000000000000")
        assert any("implementation changed" in r for r in artifacts.invalid_reasons(art, dataset_key=_KEY))

    def test_an_unreadable_source_is_not_reported_as_a_code_change(
        self, store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The stamp is a fallback when the source cannot be read, and a fallback proves nothing.

        Telling someone their implementation changed when this process simply could not see it
        sends them hunting an edit nobody made -- which is exactly how a broken import path once
        got read as a stale artifact.
        """
        rec, _mid, _final = _pipeline(store)
        art = _publish(rec, 3, impl_version="impl:0000000000000000")
        monkeypatch.setattr(artifacts, "impl_version", lambda _ref: "pkg:1.2.3+abcdef")

        reasons = artifacts.invalid_reasons(art, dataset_key=_KEY)

        assert any("cannot be read here" in r for r in reasons)
        assert any("not a code change" in r for r in reasons)
        assert not any("implementation changed" in r for r in reasons)

    def test_an_artifact_stamped_by_an_unreadable_run_says_so(self, store: Path) -> None:
        """The other direction: the run that PUBLISHED it could not read its own source."""
        rec, _mid, _final = _pipeline(store)
        art = _publish(rec, 3, impl_version="pkg:1.2.3+abcdef")

        reasons = artifacts.invalid_reasons(art, dataset_key=_KEY)

        assert any("could not read its own source" in r for r in reasons)
        assert not any("implementation changed" in r for r in reasons)

    def test_an_unprovable_stamp_still_refuses_reuse(self, store: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only the wording changed. An unproven stamp must still fail toward recompute."""
        rec, _mid, _final = _pipeline(store)
        art = _publish(rec, 3, impl_version="impl:0000000000000000")
        monkeypatch.setattr(artifacts, "impl_version", lambda _ref: "pkg:1.2.3+abcdef")

        assert artifacts.invalid_reasons(art, dataset_key=_KEY) != []

    def test_a_matching_stamp_is_no_reason_at_all(self, store: Path) -> None:
        rec, _mid, _final = _pipeline(store)
        art = _publish(rec, 3)
        assert artifacts.invalid_reasons(art, dataset_key=_KEY) == []

    def test_a_different_curator_build_alone_is_not_a_reason(self, store: Path) -> None:
        # The package version ends in the repository's git SHA, so testing against it meant one
        # commit anywhere -- a README, another modality -- emptied the store. It is provenance now.
        rec, _mid, _final = _pipeline(store)
        art = _publish(rec, 3, code_version="0.0.0-ancient")
        assert artifacts.invalid_reasons(art, dataset_key=_KEY) == []

    def test_a_miss_explains_itself(self, store: Path) -> None:
        # "we found prior work but the data changed" must not be reported as "never ran".
        rec, _mid, _final = _pipeline(store)
        art = _publish(rec, 3)
        os.remove(art.uri)
        assert "not reusable" in reuse.scan(rec, dataset_key=_KEY)["rationale"]

    def test_duplicate_publish_is_idempotent(self, store: Path) -> None:
        rec, _mid, _final = _pipeline(store)
        _publish(rec, 3)
        first = artifacts.load(artifacts.plan_steps(rec, _KEY)[3].step_key)
        _publish(rec, 3)
        second = artifacts.load(artifacts.plan_steps(rec, _KEY)[3].step_key)
        assert len(artifacts.list_artifacts()) == 1
        assert first.rows_out == second.rows_out  # republishing must not double-count rows


class TestMaterializer:
    def test_prefix_is_replaced_by_a_reader(self, store: Path) -> None:
        rec, mid, _final = _pipeline(store)
        out, err = continuation.materialize(rec, uri=mid, kind="manifest", prefix=2)
        assert err == ""
        assert [s.ref for s in out.stages] == ["ManifestReader", "GetAudioDurationStage", "ManifestWriterStage"]
        assert out.stages[0].params == {"manifest_path": mid}

    def test_audio_dir_artifact_uses_the_folder_source(self, store: Path) -> None:
        rec, _mid, _final = _pipeline(store)
        out, err = continuation.materialize(rec, uri=str(store / "wavs"), kind="audio_dir", prefix=2)
        assert err == ""
        assert out.stages[0].ref == "CreateInitialManifestAudioFolderStage"

    def test_audio_dir_resume_does_not_invent_in_memory_metadata(
        self,
        store: Path,
    ) -> None:
        original = store / "original"
        artifact_dir = store / "artifact"
        original.mkdir()
        artifact_dir.mkdir()
        (artifact_dir / "clip.wav").write_bytes(b"RIFF")
        rec = _frozen(
            [
                {
                    "ref": "CreateInitialManifestAudioFolderStage",
                    "params": {"data_dir": str(original)},
                },
                {
                    "ref": "PreserveByValueStage",
                    "params": {
                        "input_value_key": "text",
                        "target_value": 1,
                        "operator": "ge",
                    },
                },
            ]
        )
        plan = {
            "mode": "incremental",
            "source": "artifact_scan",
            "dataset_key": _KEY,
            "reuse_stages": ["CreateInitialManifestAudioFolderStage"],
            "reuse_point": {
                "uri": str(artifact_dir),
                "kind": "audio_dir",
                "stage_index": 0,
                # Cumulative task metadata existed before persistence, but a bare
                # directory serializes neither field.
                "produced_roles": ["audio_filepath", "text"],
                "produced_keys": ["audio_filepath", "text"],
            },
        }

        result = verbs._execute_plan(
            rec,
            plan,
            choice="extend",
            data=str(original),
            confirm=False,
            output_dir=None,
            bootstrap_ray=False,
            goal=None,
            parent=None,
            continuation_mod=continuation,
        )

        assert result["status"] == "refused"
        assert "does not validate" in result["reason"]
        assert any(issue["code"] == "unsatisfied_reads" for issue in result["verdict"]["issues"])

    def test_physical_artifact_replaces_conflicting_source_assertions(
        self,
        store: Path,
    ) -> None:
        artifact = str(store / "mid.jsonl")
        rec = _frozen(
            [_READER, _DUR, _WRITER],
            inputs={
                "manifest_path": "/original/input.jsonl",
                "input_manifest": "/original/longform.jsonl",
                "audio_dir": "/original/audio",
                "raw_data_dir": "/original/raw",
                "data_dir": "/original/folder",
                "output_manifest": "/keep/output.jsonl",
                "dataset_name": "keep-me",
            },
        )

        out, err = continuation.materialize(
            rec,
            uri=artifact,
            kind="manifest",
            prefix=1,
        )

        assert err == ""
        assert out is not None
        assert out.stages[0].params == {"manifest_path": artifact}
        assert out.inputs == {
            "manifest_path": artifact,
            "output_manifest": "/keep/output.jsonl",
            "dataset_name": "keep-me",
        }

    def test_unreadable_artifact_kind_is_refused_not_guessed(self, store: Path) -> None:
        rec, _mid, _final = _pipeline(store)
        out, err = continuation.materialize(rec, uri="/tmp/x", kind="rttm_dir", prefix=2)  # noqa: S108
        assert out is None
        assert "no source stage" in err

    def test_out_of_range_prefix_is_refused(self, store: Path) -> None:
        rec, _mid, _final = _pipeline(store)
        assert continuation.materialize(rec, uri="/tmp/x", kind="manifest", prefix=0)[0] is None  # noqa: S108
        assert continuation.materialize(rec, uri="/tmp/x", kind="manifest", prefix=99)[0] is None  # noqa: S108


class TestIndex:
    def test_probe_finds_a_published_step(self, store: Path) -> None:
        rec, _mid, _final = _pipeline(store)
        art = _publish(rec, 3)
        row = run_index.probe_step(art.step_key)
        assert row is not None
        assert row["stage_ref"] == "ManifestWriterStage"
        assert row["dataset_key"] == _KEY

    def test_index_is_rebuildable_from_the_json_records(self, store: Path) -> None:
        rec, _mid, _final = _pipeline(store)
        art = _publish(rec, 3)
        os.remove(run_index.index_path())
        assert run_index.probe_step(art.step_key) is None
        assert run_index.reindex()["artifacts_indexed"] == 1
        assert run_index.probe_step(art.step_key) is not None

    def test_queries_by_dataset_and_stage(self, store: Path) -> None:
        rec, _mid, _final = _pipeline(store)
        _publish(rec, 1)
        _publish(rec, 3)
        assert len(run_index.find_artifacts(dataset_key=_KEY)) == 2
        assert len(run_index.find_artifacts(dataset_key="stat:other")) == 0

    def test_a_dataset_key_can_be_pasted_back_where_a_path_goes(self, tmp_path: Path) -> None:
        # Every scan prints dataset_key, so it is the obvious thing to paste into
        # `runs --data`. Profiling it as a path would quietly match nothing.
        src, key = _real_source(tmp_path)
        assert verbs._dataset_key_arg(key) == key
        assert verbs._dataset_key_arg(src) == key

    def test_a_failure_on_commit_stays_inside_the_cache(self, store: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # The index is an advisory cache over the JSON records, so a locked database must degrade
        # to "no cache" rather than surface as an error. Opening and using once shared one `try`,
        # which meant a failure at commit ran a SECOND yield -- and a contextmanager that yields
        # twice raises RuntimeError out of whatever the caller was doing.
        import sqlite3

        rec, _mid, _final = _pipeline(store)
        _publish(rec, 3)
        real_connect = sqlite3.connect

        class LockedOnCommit:
            """A working connection whose commit fails, as a contended database does."""

            def __init__(self, conn: Any) -> None:  # noqa: ANN401
                object.__setattr__(self, "_conn", conn)

            def __getattr__(self, name: str) -> Any:  # noqa: ANN401
                return getattr(self._conn, name)

            def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401
                setattr(self._conn, name, value)

            def commit(self) -> None:
                msg = "database is locked"
                raise sqlite3.OperationalError(msg)

        monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: LockedOnCommit(real_connect(*a, **k)))
        assert run_index.reindex()["artifacts_indexed"] >= 0  # no RuntimeError

    def test_an_unopenable_index_degrades_to_no_cache(self, store: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import sqlite3

        rec, _mid, _final = _pipeline(store)
        art = _publish(rec, 3)

        def unopenable(*_a: Any, **_k: Any) -> Any:  # noqa: ANN401
            msg = "unable to open database file"
            raise sqlite3.OperationalError(msg)

        monkeypatch.setattr(sqlite3, "connect", unopenable)
        assert run_index.probe_step(art.step_key) is None  # falls back, does not raise


def _real_source(tmp_path: Path) -> tuple[str, str]:
    """A real on-disk manifest plus the dataset key the verbs will compute for it."""
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFFfakewavdata")
    manifest = tmp_path / "src.jsonl"
    manifest.write_text(json.dumps({"audio_filepath": str(wav)}) + "\n")
    return str(manifest), profiler.profile_data(str(manifest)).dataset_key()


class TestContinueVerb:
    """The verb surface: scan, the three-way choice, and the confirm gate on top of reuse."""

    def test_scan_verb_reports_the_candidate(self, store: Path) -> None:
        from nemo_curator import audio_agent as aa

        src, key = _real_source(store)
        rec, _mid, _final = _pipeline(store, source=src)
        _publish(rec, 3, dataset_key=key)
        result = aa.reuse_scan(rec.to_dict(), data=src)
        assert result["decision"] == "already_done"
        assert result["candidates"][0]["through_stage"] == "ManifestWriterStage"

    def test_artifact_scan_upgrades_a_parentless_plan(self, store: Path) -> None:
        from nemo_curator import audio_agent as aa

        src, key = _real_source(store)
        rec, mid, _final = _pipeline(store, source=src)
        _publish(rec, 1, dataset_key=key)
        plan = aa.plan_continuation(rec.to_dict(), data=src)
        assert plan["mode"] == "incremental"
        assert plan["source"] == "artifact_scan"  # no parent run existed to diff against
        assert plan["reuse_from"] == [mid]

    def test_serving_as_is_does_no_compute_and_rechecks_the_contract(self, store: Path) -> None:
        from nemo_curator import audio_agent as aa

        src, key = _real_source(store)
        rec, _mid, final = _pipeline(store, source=src)
        _publish(rec, 3, dataset_key=key)
        result = aa.plan_continuation(rec.to_dict(), data=src, execute=True, choice="as_is")
        assert result["status"] == "reused"
        assert result["output"] == final
        assert "acceptance" in result  # today's bar is applied to yesterday's bytes

    def test_parentless_as_is_uses_the_source_run_yield_denominator(
        self,
        store: Path,
    ) -> None:
        from nemo_curator.audio_agent import run_store
        from nemo_curator.audio_agent.contracts import RunRecord

        src, key = _real_source(store)
        rec, _mid, final = _pipeline(store, source=src)
        rec.acceptance_criteria = [
            {
                "id": "keep-half",
                "type": "yield",
                "kind": "relative",
                "severity": "must",
                "check": {"op": ">=", "value": 50},
            }
        ]
        rec.freeze()
        Path(final).write_text(
            "".join(json.dumps({"audio_filepath": f"/tmp/{index}.wav"}) + "\n" for index in range(20)),  # noqa: S108
            encoding="utf-8",
        )
        _publish(
            rec,
            3,
            dataset_key=key,
            run_id="source-run-100",
            rows_in=100,
        )
        run_store.save(
            RunRecord(
                run_id="source-run-100",
                recipe=rec.to_dict(),
                dataset_key=key,
                status="completed",
                steps=artifacts.step_keys(rec, key),
                output_paths=[final],
                input_count=100,
                accepted=20,
            )
        )

        result = verbs.plan_continuation(
            rec,
            data=src,
            execute=True,
            choice="as_is",
        )

        assert result["status"] == "reused"
        assert result["source_run_id"] == "source-run-100"
        assert result["acceptance"]["overall"] == "not_met"
        assert "20/100" in result["acceptance"]["criteria"][0]["evidence"]

    def test_artifact_row_proof_can_meet_output_completeness(self, store: Path) -> None:
        src, key = _real_source(store)
        rec, _mid, final = _pipeline(store, source=src)
        rec.acceptance_criteria = [
            {
                "id": "audio_path",
                "type": "output_completeness",
                "check": {"field": "audio_filepath"},
            }
        ]
        rec.freeze()
        _publish(rec, 3, dataset_key=key)

        result = verbs.plan_continuation(
            rec,
            data=src,
            execute=True,
            choice="as_is",
        )

        assert result["status"] == "reused"
        assert result["output"] == final
        assert result["acceptance"]["overall"] == "met"

    def test_artifact_row_proof_detects_manifest_truncation(self, store: Path) -> None:
        src, key = _real_source(store)
        rec, _mid, final = _pipeline(store, source=src)
        rec.acceptance_criteria = [
            {
                "id": "audio_path",
                "type": "output_completeness",
                "check": {"field": "audio_filepath"},
            }
        ]
        rec.freeze()
        Path(final).write_text(
            '{"audio_filepath": "/tmp/a.wav"}\n{"audio_filepath": "/tmp/b.wav"}\n',
            encoding="utf-8",
        )
        _publish(rec, 3, dataset_key=key)
        # Simulate post-publication truncation while the old marker/registry
        # still claims two serialized rows.
        Path(final).write_text(
            '{"audio_filepath": "/tmp/a.wav"}\n',
            encoding="utf-8",
        )

        result = verbs.plan_continuation(
            rec,
            data=src,
            execute=True,
            choice="as_is",
        )

        assert result["status"] == "refused"
        assert "changed after" in result["reason"]

    def test_extending_still_passes_through_the_confirm_gate(self, store: Path) -> None:
        from nemo_curator import audio_agent as aa

        src, key = _real_source(store)
        rec, _mid, _final = _pipeline(store, source=src)
        _publish(rec, 1, dataset_key=key)
        result = aa.plan_continuation(rec.to_dict(), data=src, execute=True, choice="extend")
        # Reuse buys no shortcut past "0 silent full-scale runs".
        assert result["status"] == "refused"
        assert "confirm" in result.get("confirm_with", "")
        assert [s["ref"] for s in result["recipe"]["stages"]] == [
            "ManifestReader",
            "GetAudioDurationStage",
            "ManifestWriterStage",
        ]

    def test_the_documented_confirm_hash_form_reaches_the_run(
        self, store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``continue --execute --choice extend --confirm <hash>`` is what AGENTS.md documents.
        The hash a user holds is their own recipe's; the recipe that executes is derived from it
        and hashes differently, so checking theirs against the derived one refused the only
        invocation the docs give -- pushing hosts onto the weaker bare ``--confirm``.
        """
        from nemo_curator import audio_agent as aa

        src, key = _real_source(store)
        rec, _mid, _final = _pipeline(store, source=src)
        _publish(rec, 1, dataset_key=key)

        seen: dict[str, object] = {}

        def _capture(recipe, **kwargs):  # noqa: ANN001, ANN202
            seen["recipe"] = recipe
            seen["confirm"] = kwargs.get("confirm")
            return {"status": "completed", "run_id": "captured"}

        monkeypatch.setattr(verbs, "run", _capture)
        result = aa.plan_continuation(rec.to_dict(), data=src, execute=True, choice="extend", confirm=rec.config_hash)

        assert result.get("status") == "completed", result
        # Re-anchored: the derived recipe carries its own hash into run()'s integrity check.
        assert seen["confirm"] == seen["recipe"].config_hash
        assert seen["confirm"] != rec.config_hash

    def test_a_wrong_confirm_hash_is_still_refused_on_the_extend_path(
        self, store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The gate must not have been widened into a rubber stamp."""
        from nemo_curator import audio_agent as aa

        src, key = _real_source(store)
        rec, _mid, _final = _pipeline(store, source=src)
        _publish(rec, 1, dataset_key=key)

        def _must_not_run(*_args, **_kwargs):  # noqa: ANN202
            raise AssertionError("a mismatched confirmation reached run()")  # noqa: EM101

        monkeypatch.setattr(verbs, "run", _must_not_run)
        result = aa.plan_continuation(
            rec.to_dict(), data=src, execute=True, choice="extend", confirm="0000000000000000"
        )
        assert result["status"] == "refused"
        assert "integrity check failed" in result["reason"]

    def test_confirmed_extend_executes_the_artifact_under_the_original_dataset_identity(
        self,
        store: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from types import SimpleNamespace

        from nemo_curator.audio_agent import run_store

        src, key = _real_source(store)
        rec, mid, _final = _pipeline(store, source=src)
        reuse_point = _publish(rec, 1, dataset_key=key)
        captured: dict[str, Any] = {}

        env = SimpleNamespace(
            has_gpu=False,
            gpu_count=0,
            to_dict=dict,
        )
        resource_plan = SimpleNamespace(
            feasible=True,
            mode="batch",
            machine_fingerprint="machine",
            escalations=[],
            to_dict=lambda: {"mode": "batch"},
        )

        def fake_build(materialized: Recipe) -> tuple[list[Any], list[Any]]:
            physical_source = materialized.stages[0].params["manifest_path"]
            captured["physical_source"] = physical_source
            return (
                [
                    SimpleNamespace(physical_source=physical_source),
                    *[object() for _ in materialized.stages[1:]],
                ],
                [],
            )

        def fake_execute(stages: list[Any], *_args: Any, **_kwargs: Any) -> tuple[list[Any], str]:  # noqa: ANN401
            captured["executed_source"] = stages[0].physical_source
            return [object()], "batch"

        def fake_report(**kwargs: Any) -> Any:  # noqa: ANN401
            outputs = list(kwargs.get("output_paths") or [])
            return SimpleNamespace(
                accepted=1,
                input_count=1,
                output_paths=outputs,
                per_stage_metrics={},
                to_dict=lambda: {
                    "accepted": 1,
                    "input_count": 1,
                    "output_paths": outputs,
                },
            )

        def fake_publish(_rec: Recipe, _stages: list[Any], **kwargs: Any) -> list[dict[str, Any]]:  # noqa: ANN401
            captured["published_dataset_key"] = kwargs["dataset_key"]
            captured["step_identity"] = kwargs["step_identity"]
            return []

        def fake_record(_rec: Recipe, **kwargs: Any) -> str:  # noqa: ANN401
            captured["recorded_data"] = kwargs["data"]
            captured["recorded_dataset_key"] = kwargs["dataset_key"]
            return "continued-run"

        monkeypatch.delenv("AUDIO_AGENT_REQUIRE_SMOKE", raising=False)
        monkeypatch.setattr(verbs, "validate", lambda *_args, **_kwargs: {"runnable": True})
        monkeypatch.setattr(verbs, "probe_env", lambda: env)
        monkeypatch.setattr(verbs, "build_stages", fake_build)
        monkeypatch.setattr(verbs, "_plan_resources", lambda *_args, **_kwargs: resource_plan)
        monkeypatch.setattr(verbs, "_run_pipeline_autofallback", fake_execute)
        monkeypatch.setattr(verbs, "build_run_report", fake_report)
        monkeypatch.setattr(verbs, "_produced_roles_keys", lambda *_args, **_kw: ([], []))
        monkeypatch.setattr(verbs, "_acceptance_result", lambda *_args, **_kwargs: {})
        monkeypatch.setattr(verbs, "_publish_artifacts", fake_publish)
        monkeypatch.setattr(verbs, "_record_run", fake_record)
        monkeypatch.setattr(run_store, "new_run_id", lambda _config_hash: "continued-run")

        result = verbs.plan_continuation(
            rec,
            data=src,
            execute=True,
            choice="extend",
            confirm=True,
        )

        assert result["status"] == "completed"
        assert captured["physical_source"] == mid
        assert captured["executed_source"] == mid
        assert captured["published_dataset_key"] == key
        assert captured["recorded_dataset_key"] == key
        assert captured["recorded_data"] == src
        assert captured["step_identity"][0][0] == reuse_point.step_key
        assert result["data_binding"]["primary_path"] == mid
        assert result["data_binding"]["logical_dataset_key"] == key
        assert result["data_binding"]["logical_data_source"] == src

    def test_direct_run_cannot_bind_dataset_b_to_dataset_a_artifact(
        self,
        store: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source_a_dir = store / "dataset_a"
        source_b_dir = store / "dataset_b"
        source_a_dir.mkdir()
        source_b_dir.mkdir()
        source_a, key_a = _real_source(source_a_dir)
        source_b, key_b = _real_source(source_b_dir)

        logical_a, mid_a, _final_a = _pipeline(source_a_dir, source=source_a)
        artifact_a = _publish(logical_a, 1, dataset_key=key_a)
        physical_a, err = continuation.materialize(
            logical_a,
            uri=mid_a,
            kind=artifact_a.kind,
            prefix=2,
        )
        assert physical_a is not None, err

        logical_b, _mid_b, _final_b = _pipeline(source_b_dir, source=source_b)
        poisoned = verbs._ContinuationRunContext(
            logical_recipe=logical_b.to_dict(),
            dataset_key=key_b,
            reuse_step_key=artifact_a.step_key,
        )

        def must_not_execute(*_args: Any, **_kwargs: Any) -> Any:  # noqa: ANN401
            msg = "a rejected continuation identity reached execution"
            raise AssertionError(msg)

        monkeypatch.delenv("AUDIO_AGENT_REQUIRE_SMOKE", raising=False)
        monkeypatch.setattr(verbs, "probe_env", must_not_execute)
        monkeypatch.setattr(verbs, "build_stages", must_not_execute)

        result = verbs.run(
            physical_a,
            confirm=True,
            _continuation_context=poisoned,
        )

        assert result["status"] == "refused"
        assert "does not belong to the logical recipe" in result["reason"]
        assert len(artifacts.list_artifacts()) == 1

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("step_identity", [("forged", "forged-parent", 99)]),
            ("logical_steps", ["forged"]),
            (
                "_verified_lineage",
                {
                    "dataset_key": _KEY,
                    "step_key": "forged",
                    "artifact_uri": "/tmp/forged.jsonl",  # noqa: S108
                },
            ),
        ],
    )
    def test_run_no_longer_accepts_independent_identity_overrides(
        self,
        name: str,
        value: Any,  # noqa: ANN401
    ) -> None:
        with pytest.raises(TypeError, match=name):
            verbs.run(
                _frozen([_READER]),
                confirm=True,
                **{name: value},
            )

    def test_an_extended_run_registers_its_tail_under_the_asked_for_identity(self, store: Path) -> None:
        # An extend runs a REWRITTEN recipe (reader-on-artifact + tail). Publishing the tail
        # under that rewritten identity would describe a pipeline nobody asked for, so the same
        # follow-up request would recompute the tail forever.
        from nemo_curator.audio_agent import continuation, verbs
        from nemo_curator.audio_agent.recipe import build_stages

        _src, key = _real_source(store)
        rec, mid, final = _pipeline(store)
        _publish(rec, 1, dataset_key=key)  # the prefix was already done

        materialized, err = continuation.materialize(rec, uri=mid, kind="manifest", prefix=2)
        assert materialized is not None, err
        stages, issues = build_stages(materialized)
        assert stages is not None, issues
        Path(final).write_text('{"audio_filepath": "/tmp/a.wav", "duration": 1.0}\n')

        published = verbs._publish_artifacts(
            materialized,
            stages,
            dataset_key=key,
            fingerprint_tier="stat",
            per_stage={},
            run_id="run-extend",
            input_count=1,
            data_profile=None,
            started_at="",
            ended_at="",
            elapsed_sec=42.0,
            # Exactly what ``run`` derives for a continuation: the logical recipe's step tuples
            # from the reused stage onward (``_verify_continuation_context`` -> step_identity).
            step_identity=[(p.step_key, p.input_key, p.index) for p in artifacts.plan_steps(rec, key)[1:]],
        )
        assert [p["uri"] for p in published] == [final]
        result = reuse.scan(rec, dataset_key=key)
        assert result["decision"] == "already_done"
        assert result["reuse_point"]["uri"] == final
        assert result["saving_is_lower_bound"] is True  # the un-persisted middle steps cost something

    def test_extend_without_a_reuse_point_is_refused(self, store: Path) -> None:
        from nemo_curator import audio_agent as aa

        src, _key = _real_source(store)
        rec, _mid, _final = _pipeline(store, source=src)
        result = aa.plan_continuation(rec.to_dict(), data=src, execute=True, choice="extend")
        assert result["status"] == "refused"
        assert "nothing to extend" in result["reason"]


class TestContractJudgesTheData:
    """A success contract must read the OUTPUT, not the labels ``validate`` declared.

    Declaring a field produced says a column exists; it says nothing about what is in it.
    A black-box test run found `duration_present` satisfied by the string 'duration'
    appearing in a role list, having never read a row -- so every duration could have been
    null and the run would still have reported `met`.
    """

    @staticmethod
    def _manifest(tmp_path: Path, rows: list[dict[str, Any]]) -> str:
        p = tmp_path / "out.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return str(p)

    @staticmethod
    def _verdict(out: str, rec: Recipe) -> dict[str, Any]:
        report = type("R", (), {"accepted": 3, "input_count": 3, "output_paths": [out]})()
        return verbs._acceptance_result(
            TestContractJudgesTheData._with_writer(rec, out),
            report,
            ["duration"],
            ["duration"],
            [out],
        )

    @staticmethod
    def _with_writer(rec: Recipe, output_path: str) -> Recipe:
        payload = rec.to_dict()
        payload["stages"] = [
            *payload["stages"],
            {
                "ref": "ManifestWriterStage",
                "params": {"output_path": output_path},
            },
        ]
        return Recipe.from_dict(payload)

    @staticmethod
    def _recipe() -> Recipe:
        return Recipe.from_dict(
            {
                "stages": [_READER],
                "acceptance_criteria": [
                    {"id": "dur", "type": "output_completeness", "check": {"field": "duration"}, "severity": "must"}
                ],
            }
        )

    def test_a_field_null_on_every_row_is_not_met(self, tmp_path: Path) -> None:
        out = self._manifest(tmp_path, [{"duration": None}, {"duration": None}, {"duration": None}])
        verdict = self._verdict(out, self._recipe())
        assert verdict["overall"] == "not_met"
        assert "EMPTY in every row" in verdict["criteria"][0]["evidence"]

    def test_a_field_on_only_some_rows_is_not_met(self, tmp_path: Path) -> None:
        out = self._manifest(tmp_path, [{"duration": 1.0}, {"duration": 2.0}, {"other": 1}])
        verdict = self._verdict(out, self._recipe())
        assert verdict["overall"] == "not_met"
        assert "only 2/3" in verdict["criteria"][0]["evidence"]

    def test_a_fully_populated_field_is_met_and_says_it_read_rows(self, tmp_path: Path) -> None:
        out = self._manifest(tmp_path, [{"duration": 1.0}, {"duration": 2.0}, {"duration": 3.0}])
        verdict = self._verdict(out, self._recipe())
        assert verdict["overall"] == "met"
        assert "all 3 row(s) read" in verdict["criteria"][0]["evidence"]

    def test_transcript_alias_is_checked_against_the_physical_asr_field(self, tmp_path: Path) -> None:
        # Natural-language "transcript" maps to plain ASR's documented ``pred_text``
        # output. Passing requires the physical values, not merely a producer label.
        out = self._manifest(tmp_path, [{"pred_text": "hello"}, {"pred_text": "world"}])
        rec = Recipe.from_dict(
            {
                "stages": [_READER],
                "acceptance_criteria": [
                    {"id": "t", "type": "output_completeness", "check": {"field": "transcript"}, "severity": "must"}
                ],
            }
        )
        report = type("R", (), {"accepted": 2, "input_count": 2, "output_paths": [out]})()
        verdict = verbs._acceptance_result(
            self._with_writer(rec, out),
            report,
            ["transcript"],
            ["pred_text"],
            [out],
        )
        assert verdict["overall"] == "met"
        assert "'pred_text' present and non-empty in all 2 row(s) read" in verdict["criteria"][0]["evidence"]

    def test_evidence_comes_from_the_terminal_output_not_an_intermediate(self, tmp_path: Path) -> None:
        # An earlier output that never carried the column must not drag coverage down.
        early = tmp_path / "early.jsonl"
        early.write_text('{"audio_filepath": "a.wav"}\n', encoding="utf-8")
        final = self._manifest(tmp_path, [{"duration": 1.0}, {"duration": 2.0}, {"duration": 3.0}])
        report = type("R", (), {"accepted": 3, "input_count": 3, "output_paths": [str(early), final]})()
        verdict = verbs._acceptance_result(
            self._with_writer(self._recipe(), final),
            report,
            ["duration"],
            ["duration"],
            [str(early), final],
        )
        assert verdict["overall"] == "met"

    def test_unreadable_output_is_unverifiable_not_declaration_level_met(self, tmp_path: Path) -> None:
        missing = str(tmp_path / "never_written.jsonl")
        report = type("R", (), {"accepted": 3, "input_count": 3, "output_paths": [missing]})()
        verdict = verbs._acceptance_result(
            self._with_writer(self._recipe(), missing),
            report,
            ["duration"],
            ["duration"],
            [missing],
        )
        assert verdict["overall"] == "not_met"
        assert verdict["criteria"][0]["status"] == "unverifiable"
        assert "terminal output scan was missing" in verdict["criteria"][0]["note"]

    @pytest.mark.parametrize("bad_line", ["not-json\n", "\n"])
    def test_bad_row_after_preview_limit_still_fails(
        self,
        tmp_path: Path,
        bad_line: str,
    ) -> None:
        out = tmp_path / "large.jsonl"
        out.write_text(
            ('{"duration": 1.0}\n' * verbs._EVIDENCE_ROWS) + bad_line,
            encoding="utf-8",
        )
        report = type(
            "R",
            (),
            {
                "accepted": verbs._EVIDENCE_ROWS + 1,
                "input_count": verbs._EVIDENCE_ROWS + 1,
                "output_paths": [str(out)],
            },
        )()

        verdict = verbs._acceptance_result(
            self._with_writer(self._recipe(), str(out)),
            report,
            ["duration"],
            ["duration"],
            [str(out)],
        )

        assert verdict["overall"] == "not_met"
        assert "malformed/blank row" in verdict["criteria"][0]["evidence"]

    @pytest.mark.parametrize("terminal_kind", ["missing", "empty"])
    def test_terminal_output_never_borrows_evidence_from_an_intermediate(
        self,
        tmp_path: Path,
        terminal_kind: str,
    ) -> None:
        early = tmp_path / "early.jsonl"
        early.write_text('{"duration": 1.0}\n', encoding="utf-8")
        terminal = tmp_path / "terminal.jsonl"
        if terminal_kind == "empty":
            terminal.write_text("", encoding="utf-8")
        report = type(
            "R",
            (),
            {"accepted": 1, "input_count": 1, "output_paths": [str(early), str(terminal)]},
        )()

        verdict = verbs._acceptance_result(
            self._with_writer(self._recipe(), str(terminal)),
            report,
            ["duration"],
            ["duration"],
            [str(early), str(terminal)],
        )

        assert verdict["overall"] == "not_met"
        assert verdict["criteria"][0]["status"] == "unverifiable"

    def test_the_row_preview_is_bounded_while_the_scan_still_counts_every_row(self, tmp_path: Path) -> None:
        """The preview is what a caller holds in memory, so it is capped; the summary beside it
        is not, or a bad row past the cap would be invisible to the success contract."""
        rows = verbs._EVIDENCE_ROWS + 50
        out = self._manifest(tmp_path, [{"duration": 1.0}] * rows)
        preview, summary = verbs._scan_terminal_output([out], limit=verbs._EVIDENCE_ROWS)
        assert len(preview) == verbs._EVIDENCE_ROWS
        assert summary["valid_rows"] == rows

    def test_late_per_item_metric_failure_is_checked_beyond_the_preview(self, tmp_path: Path) -> None:
        out = self._manifest(
            tmp_path,
            ([{"mos": 4.5}] * verbs._EVIDENCE_ROWS) + [{"mos": 3.0}],
        )
        rec = Recipe.from_dict(
            {
                "stages": [
                    _READER,
                    {
                        "ref": "ManifestWriterStage",
                        "params": {"output_path": out},
                    },
                ],
                "acceptance_criteria": [
                    {
                        "id": "quality",
                        "type": "quality_standard",
                        "check": {
                            "scope": "per_retained_item",
                            "field": "mos",
                            "op": ">=",
                            "value": 4.0,
                        },
                    }
                ],
            }
        )
        report = type(
            "R",
            (),
            {
                "accepted": verbs._EVIDENCE_ROWS + 1,
                "input_count": verbs._EVIDENCE_ROWS + 1,
                "output_paths": [out],
            },
        )()

        verdict = verbs._acceptance_result(
            rec,
            report,
            ["score"],
            ["mos"],
            [out],
        )

        assert verdict["overall"] == "not_met"
        assert "all 2001 terminal" in verdict["criteria"][0]["evidence"]

    def test_report_only_output_dir_cannot_replace_recipe_terminal(
        self,
        tmp_path: Path,
    ) -> None:
        out = self._manifest(
            tmp_path,
            [{"duration": 1.0}, {"duration": 2.0}, {"duration": 3.0}],
        )
        phantom = tmp_path / "cli-output-dir"
        report = type(
            "R",
            (),
            {
                "accepted": 3,
                "input_count": 3,
                "output_paths": [out, str(phantom)],
            },
        )()

        verdict = verbs._acceptance_result(
            self._with_writer(self._recipe(), out),
            report,
            ["duration"],
            ["duration"],
            [out, str(phantom)],
        )

        assert verdict["overall"] == "met"

    def test_snippet_writer_does_not_assume_one_row_per_returned_task(
        self,
        tmp_path: Path,
    ) -> None:
        out = self._manifest(
            tmp_path,
            [{"pred_text": "one"}, {"pred_text": "two"}],
        )
        rec = Recipe.from_dict(
            {
                "stages": [
                    _READER,
                    {
                        "ref": "SnippetManifestWriterStage",
                        "params": {"output_path": out},
                    },
                ],
                "acceptance_criteria": [
                    {
                        "id": "transcript",
                        "type": "output_completeness",
                        "check": {"field": "pred_text"},
                    }
                ],
            }
        )
        report = type(
            "R",
            (),
            {"accepted": 3, "input_count": 3, "output_paths": [out]},
        )()

        verdict = verbs._acceptance_result(
            rec,
            report,
            ["pred_text"],
            ["pred_text"],
            [out],
        )

        assert verdict["criteria"][0]["status"] == "unverifiable"
        assert "trustworthy serialized-row count" in verdict["criteria"][0]["note"]

    @pytest.mark.parametrize("as_uri", [False, True])
    def test_terminal_scan_supports_local_and_file_uri(
        self,
        tmp_path: Path,
        as_uri: bool,
    ) -> None:
        out = Path(
            self._manifest(
                tmp_path,
                [{"duration": 1.0}, {"duration": 2.0}],
            )
        )
        location = out.as_uri() if as_uri else str(out)

        preview, scan = verbs._scan_terminal_output([location])

        assert preview == [{"duration": 1.0}, {"duration": 2.0}]
        assert scan["status"] == "complete"
        assert scan["valid_rows"] == 2

    def test_terminal_scan_supports_fsspec_memory_manifest(self, tmp_path: Path) -> None:
        from fsspec.core import url_to_fs

        uri = f"memory://audio-agent-{tmp_path.name}/nested/out.jsonl"
        fs, resolved = url_to_fs(uri)
        fs.makedirs(resolved.rsplit("/", 1)[0], exist_ok=True)
        try:
            with fs.open(resolved, "wt", encoding="utf-8") as fh:
                fh.write('{"pred_text": "hello"}\n')
                fh.write('{"pred_text": "world"}\n')

            preview, scan = verbs._scan_terminal_output([uri])

            assert preview == [
                {"pred_text": "hello"},
                {"pred_text": "world"},
            ]
            assert scan["status"] == "complete"
            assert scan["valid_rows"] == 2
        finally:
            fs.rm(f"/audio-agent-{tmp_path.name}", recursive=True)


class TestReuseHonorsTheRequestedPath:
    """Serving a reused result must put it where the recipe asked for it.

    A black-box test asked for the same work written to a NEW file. Reuse fired correctly,
    but the answer pointed at the OLD path -- so the file the user named never appeared, and
    the one they got was never mentioned in their request.
    """

    def test_the_reused_output_is_delivered_to_the_declared_path(self, tmp_path: Path) -> None:
        stored = tmp_path / "old.jsonl"
        stored.write_text('{"duration": 1.0}\n', encoding="utf-8")
        wanted = tmp_path / "new.jsonl"
        rec = _frozen([_READER, {"ref": "ManifestWriterStage", "params": {"output_path": str(wanted)}}])

        out = verbs._serve_as_is(rec, {"reuse_point": {"uri": str(stored), "rows": 1}}, parent=None, lineage={})

        assert out["status"] == "reused"
        assert out["output"] == str(wanted)
        assert wanted.read_text() == '{"duration": 1.0}\n'
        assert "copied to the path the recipe asked for" in out["note"]

    def test_the_same_path_is_served_without_copying(self, tmp_path: Path) -> None:
        stored = tmp_path / "same.jsonl"
        stored.write_text('{"duration": 1.0}\n', encoding="utf-8")
        rec = _frozen([_READER, {"ref": "ManifestWriterStage", "params": {"output_path": str(stored)}}])

        out = verbs._serve_as_is(rec, {"reuse_point": {"uri": str(stored), "rows": 1}}, parent=None, lineage={})

        assert out["output"] == str(stored)
        assert "copied to" not in out["note"]

    def test_an_undeliverable_path_is_refused_not_silently_redirected(self, tmp_path: Path) -> None:
        stored = tmp_path / "old.jsonl"
        stored.write_text('{"duration": 1.0}\n', encoding="utf-8")
        blocker = tmp_path / "blocker.jsonl"  # a FILE where the destination needs a directory
        blocker.write_text("", encoding="utf-8")
        rec = _frozen(
            [_READER, {"ref": "ManifestWriterStage", "params": {"output_path": str(blocker / "want.jsonl")}}]
        )

        out = verbs._serve_as_is(rec, {"reuse_point": {"uri": str(stored), "rows": 1}}, parent=None, lineage={})

        assert out["status"] == "refused"
        assert "could not be copied" in out["reason"]

    def test_a_missing_source_is_refused_before_any_copy_is_attempted(self, tmp_path: Path) -> None:
        stored = tmp_path / "gone.jsonl"  # never created
        rec = _frozen(
            [_READER, {"ref": "ManifestWriterStage", "params": {"output_path": str(tmp_path / "want.jsonl")}}]
        )

        out = verbs._serve_as_is(rec, {"reuse_point": {"uri": str(stored), "rows": 1}}, parent=None, lineage={})

        assert out["status"] == "refused"
        assert "does not exist or is empty" in out["reason"]

    def test_a_recipe_naming_no_output_serves_the_stored_uri(self, tmp_path: Path) -> None:
        stored = tmp_path / "only.jsonl"
        stored.write_text('{"duration": 1.0}\n', encoding="utf-8")
        rec = _frozen([_READER])

        out = verbs._serve_as_is(rec, {"reuse_point": {"uri": str(stored), "rows": 1}}, parent=None, lineage={})

        assert out["output"] == str(stored)


class TestOutputTargetsAreStatedNotGuessed:
    """``validate`` must say what is already at each output path.

    An agent with no way to see this guessed: it read the writer's append-mode open, decided
    reruns would double the rows, and deleted the user's file BEFORE the confirm gate. The
    rerun would in fact have replaced the file cleanly. Facts here remove the motive.
    """

    def test_an_occupied_manifest_reports_its_row_count(self, tmp_path: Path) -> None:
        out = tmp_path / "existing.jsonl"
        out.write_text('{"a": 1}\n{"a": 2}\n{"a": 3}\n', encoding="utf-8")
        rec = _frozen([_READER, {"ref": "ManifestWriterStage", "params": {"output_path": str(out)}}])

        target = next(t for t in verbs._output_targets(rec) if t["path"] == str(out))

        assert target["exists"] is True
        assert target["kind"] == "file"
        assert target["rows"] == 3
        assert "do not clear it yourself" in target["note"]

    def test_a_path_that_does_not_exist_yet_is_plainly_marked(self, tmp_path: Path) -> None:
        out = tmp_path / "not_yet.jsonl"
        rec = _frozen([_READER, {"ref": "ManifestWriterStage", "params": {"output_path": str(out)}}])

        target = next(t for t in verbs._output_targets(rec) if t["path"] == str(out))

        assert target["exists"] is False
        assert "note" not in target  # nothing to warn about

    def test_an_occupied_directory_reports_how_many_files_it_holds(self, tmp_path: Path) -> None:
        groups = tmp_path / "groups"
        groups.mkdir()
        (groups / "a.txt").write_text("x", encoding="utf-8")
        (groups / "b.txt").write_text("y", encoding="utf-8")
        rec = _frozen([_READER, {"ref": "ManifestGroupExportStage", "params": {"output_dir": str(groups)}}])

        target = next(t for t in verbs._output_targets(rec) if t["path"] == str(groups))

        assert target["kind"] == "directory"
        assert target["files"] == 2

    def test_validate_carries_the_targets_through(self, tmp_path: Path) -> None:
        out = tmp_path / "v.jsonl"
        out.write_text('{"a": 1}\n', encoding="utf-8")
        verdict = verbs.validate(
            {"stages": [_READER, {"ref": "ManifestWriterStage", "params": {"output_path": str(out)}}]}
        )

        target = next(t for t in verdict["output_targets"] if t["path"] == str(out))
        assert target["rows"] == 1

    def test_a_recipe_writing_nowhere_reports_no_targets(self) -> None:
        assert verbs._output_targets(_frozen([_READER])) == []


class TestUnsavedPriorPrefixIsDisclosed:
    """The Phase 1 / Phase 3 scenario: the work was done before, nothing was saved.

    Only a stage with an output-location parameter publishes an artifact, so a prefix of
    in-memory stages leaves nothing to resume from even when its step keys match an earlier run
    exactly. Recomputing is correct; reporting "no prior artifact matches this pipeline on this
    data" and leaving it there is not, because the user cannot tell "this is new" from "we are
    paying for this twice".
    """

    @staticmethod
    def _record_prior_run(rec: Recipe, *, dataset_key: str = _KEY, per_stage: dict | None = None) -> None:
        from nemo_curator.audio_agent import run_store
        from nemo_curator.audio_agent.contracts import RunRecord

        run_store.save(
            RunRecord(
                run_id="prior-run",
                recipe=rec.to_dict(),
                dataset_key=dataset_key,
                status="completed",
                steps=artifacts.step_keys(rec, dataset_key),
                per_stage_metrics=per_stage or {},
                created_at="2026-07-01T00:00:00Z",
            )
        )

    # Phase 1: measure durations and write. Phase 3: the same, plus a filter before the writer.
    _PHASE1 = (_READER, _DUR, _WRITER)
    _FILTER: ClassVar[dict] = {
        "ref": "PreserveByValueStage",
        "params": {"input_value_key": "duration", "target_value": 5.0, "operator": "ge"},
    }

    def _phase3(self, source: str | None = None) -> Recipe:
        reader = {"ref": "ManifestReader", "params": {"manifest_path": source}} if source is not None else _READER
        return _frozen(
            [reader, _DUR, self._FILTER, {"ref": "ManifestWriterStage", "params": {"output_path": "kept.jsonl"}}]
        )

    def test_the_shared_prefix_is_named_rather_than_called_new(self, store: Path) -> None:
        self._record_prior_run(_frozen(list(self._PHASE1)))
        scan = reuse.scan(self._phase3(), dataset_key=_KEY)
        assert scan["decision"] == "fresh"  # correct: there is genuinely nothing on disk
        unsaved = scan["prior_unsaved"]
        assert unsaved["count"] == 2
        assert unsaved["stages"] == ["ManifestReader", "GetAudioDurationStage"]
        assert unsaved["run_id"] == "prior-run"

    def test_the_rationale_says_recomputed_not_never_seen(self, store: Path) -> None:
        self._record_prior_run(_frozen(list(self._PHASE1)))
        rationale = reuse.scan(self._phase3(), dataset_key=_KEY)["rationale"]
        assert "already ran for this dataset key" in rationale
        assert "recomputed" in rationale
        assert "nothing was persisted" in rationale

    def test_a_cheap_prefix_is_not_sold_a_writer(self, store: Path) -> None:
        """Reading a manifest and measuring durations is worth disclosing and not worth a file.
        The offer used to name the last stage of the prefix whatever that stage was, which on a
        pipeline this cheap is a chore proposed to save nothing."""
        self._record_prior_run(_frozen(list(self._PHASE1)))
        offer = reuse.scan(self._phase3(), dataset_key=_KEY)["offer"]
        assert offer["action"] == "no_checkpoint"
        assert "expensive" in offer["why"]

    def test_an_expensive_prefix_is_offered_a_position_that_holds(self, store: Path) -> None:
        """And when there IS costly work, the position is one a manifest can be written at --
        simulated rather than read off the end of the prefix. See ``test_checkpoint.py``."""
        prior = _frozen([_READER, _ASR, _WRITER])
        self._record_prior_run(prior)
        later = _frozen(
            [_READER, _ASR, self._FILTER, {"ref": "ManifestWriterStage", "params": {"output_path": "k.jsonl"}}]
        )
        offer = reuse.scan(later, dataset_key=_KEY)["offer"]
        assert offer["action"] == "add_checkpoint"
        assert offer["after_stage"] == "ASRStage"
        assert offer["skips_on_reuse"] == ["ASRStage"]

    # A stage reports metrics under its own ``name`` field, which is neither its class name nor
    # any transformation of it: ManifestReader measures itself as "manifest_reader". Using invented
    # keys here is what let a lookup that could never match in production pass its own test.
    _REAL_METRIC_KEYS: ClassVar[dict] = {
        "manifest_reader": {"process_time": {"sum": 0.5}},
        "GetAudioDurationStage": {"process_time": {"sum": 3.7}},
    }

    def test_recorded_stage_times_are_reported(self, store: Path) -> None:
        self._record_prior_run(_frozen(list(self._PHASE1)), per_stage=dict(self._REAL_METRIC_KEYS))
        scan = reuse.scan(self._phase3(), dataset_key=_KEY)
        assert scan["prior_unsaved"]["recompute_sec"] == 4.2
        assert "about 4.2s last time" in scan["rationale"]

    def test_class_names_are_not_mistaken_for_metric_keys(self, store: Path) -> None:
        # The keys a real run never writes. Matching these would mean the lookup is guessing.
        self._record_prior_run(
            _frozen(list(self._PHASE1)),
            per_stage={
                "ManifestReader": {"process_time": {"sum": 0.5}},
                "GetAudioDuration": {"process_time": {"sum": 3.7}},
            },
        )
        assert reuse.scan(self._phase3(), dataset_key=_KEY)["prior_unsaved"]["recompute_sec"] is None

    def test_any_time_metric_counts_not_only_process_time(self, store: Path) -> None:
        # One reading convention, shared with the publish-time cost, so the two cannot disagree.
        self._record_prior_run(
            _frozen(list(self._PHASE1)),
            per_stage={
                "manifest_reader": {"stage_time": {"sum": 1.0}},
                "GetAudioDurationStage": {"process_time": {"sum": 2.0}},
            },
        )
        assert reuse.scan(self._phase3(), dataset_key=_KEY)["prior_unsaved"]["recompute_sec"] == 3.0

    def test_an_unattributable_stage_reports_no_time_rather_than_a_partial_one(self, store: Path) -> None:
        # A number covering two stages, presented as the cost of three, is worse than no number.
        self._record_prior_run(
            _frozen(list(self._PHASE1)), per_stage={"manifest_reader": {"process_time": {"sum": 0.5}}}
        )
        assert reuse.scan(self._phase3(), dataset_key=_KEY)["prior_unsaved"]["recompute_sec"] is None

    def test_a_run_on_different_data_is_not_claimed_as_prior_work(self, store: Path) -> None:
        self._record_prior_run(_frozen(list(self._PHASE1)), dataset_key="stat:0000000000000000")
        assert reuse.scan(self._phase3(), dataset_key=_KEY)["prior_unsaved"] is None

    def test_a_failed_run_proves_nothing_and_is_ignored(self, store: Path) -> None:
        from nemo_curator.audio_agent import run_store
        from nemo_curator.audio_agent.contracts import RunRecord

        rec = _frozen(list(self._PHASE1))
        run_store.save(
            RunRecord(
                run_id="crashed",
                recipe=rec.to_dict(),
                dataset_key=_KEY,
                status="failed",
                steps=artifacts.step_keys(rec, _KEY),
            )
        )
        assert reuse.scan(self._phase3(), dataset_key=_KEY)["prior_unsaved"] is None

    def test_no_history_means_no_disclosure_and_no_noise(self, store: Path) -> None:
        scan = reuse.scan(self._phase3(), dataset_key=_KEY)
        assert scan["prior_unsaved"] is None
        assert scan["offer"] is None
        assert scan["rationale"] == "no prior artifact matches this pipeline on this data"

    @staticmethod
    def _corpus(tmp_path: Path) -> tuple[str, str]:
        """A real manifest plus the dataset key the verb will derive from it."""
        manifest = tmp_path / "corpus.jsonl"
        manifest.write_text(json.dumps({"audio_filepath": "clip.wav", "duration": 7.0}) + "\n")
        return str(manifest), profiler.profile_data(str(manifest)).dataset_key()

    def test_the_continuation_gate_carries_the_disclosure(self, store: Path, tmp_path: Path) -> None:
        # scan() knowing is not enough: plan_continuation is what the gate shows, and on the
        # fresh path it drops the scan's rationale entirely.
        data, key = self._corpus(tmp_path)
        reader = {"ref": "ManifestReader", "params": {"manifest_path": data}}
        self._record_prior_run(_frozen([reader, _DUR, _WRITER]), dataset_key=key)
        plan = verbs.plan_continuation(self._phase3(data), data=data)
        assert plan["prior_unsaved"]["stages"] == ["ManifestReader", "GetAudioDurationStage"]
        assert plan["offer"]["why"]  # the offer travels with it, whatever it concludes

    def test_the_gate_stays_quiet_when_there_is_nothing_to_disclose(self, store: Path, tmp_path: Path) -> None:
        data, _key = self._corpus(tmp_path)
        plan = verbs.plan_continuation(self._phase3(data), data=data)
        assert "prior_unsaved" not in plan
        assert "offer" not in plan

    def test_unidentified_data_is_never_claimed_as_a_match(self, store: Path) -> None:
        # No data given means no dataset key. Two unknowns are not the same dataset, and saying
        # "you ran this before" on that basis would be a guess presented as a fact.
        self._record_prior_run(_frozen(list(self._PHASE1)), dataset_key="")
        assert reuse.scan(self._phase3(), dataset_key="")["prior_unsaved"] is None

    def test_a_prefix_that_does_write_is_not_said_to_write_nothing(self, store: Path) -> None:
        # The prefix ends in ManifestWriterStage. Claiming "nothing was persisted" about it, and
        # advising a writer after the writer, are both assertions the code cannot support.
        prior = _frozen([_READER, _DUR, _WRITER])
        self._record_prior_run(prior)
        extended = _frozen(
            [
                _READER,
                _DUR,
                _WRITER,
                self._FILTER,
                {"ref": "ManifestWriterStage", "params": {"output_path": "k.jsonl"}},
            ]
        )
        scan = reuse.scan(extended, dataset_key=_KEY)
        unsaved = scan["prior_unsaved"]
        assert unsaved["count"] == 3
        assert unsaved["resume_point_persists"] is True
        assert "writes no file" not in unsaved["note"]
        assert "no valid artifact record remains" in unsaved["note"]
        assert scan["offer"] is None

    def test_a_prefix_that_writes_nothing_still_gets_an_answer(self, store: Path) -> None:
        # The mirror of the test above: that one persisted, so there is nothing to offer. This
        # one did not, so the offer is present and accounts for itself either way -- with a
        # position when there is expensive work to protect, and with a reason when there is not.
        self._record_prior_run(_frozen(list(self._PHASE1)))
        scan = reuse.scan(self._phase3(), dataset_key=_KEY)
        assert scan["prior_unsaved"]["resume_point_persists"] is False
        assert scan["offer"]["action"] == "no_checkpoint"


class TestBothReuseEnginesClearTheSameBar:
    """The parent diff and the artifact scan must be equally proven, not just equally deep.

    Only the scan ever carried an artifact, so a tie in depth handed the plan to the engine that
    had validated nothing -- and the executor, needing a resume URI, then refused to extend. That
    tie is the ordinary case: a parent run whose artifact was published gives both the same depth.
    """

    @staticmethod
    def _corpus(tmp_path: Path) -> tuple[str, str]:
        manifest = tmp_path / "corpus.jsonl"
        manifest.write_text(json.dumps({"audio_filepath": "clip.wav", "duration": 7.0}) + "\n")
        return str(manifest), profiler.profile_data(str(manifest)).dataset_key()

    def _parent(self, tmp_path: Path, key: str, source: str) -> tuple[Recipe, str]:
        """A completed run of reader -> duration -> writer, with its artifact published."""
        from nemo_curator.audio_agent import run_store
        from nemo_curator.audio_agent.contracts import RunRecord

        out = str(tmp_path / "durations.jsonl")
        reader = {"ref": "ManifestReader", "params": {"manifest_path": source}}
        rec = _frozen([reader, _DUR, {"ref": "ManifestWriterStage", "params": {"output_path": out}}])
        _publish(rec, 2, dataset_key=key)
        run_store.save(
            RunRecord(
                run_id="parent",
                recipe=rec.to_dict(),
                dataset_key=key,
                status="completed",
                steps=artifacts.step_keys(rec, key),
                output_paths=[out],
                input_count=1,
                accepted=1,
            )
        )
        return rec, out

    def _extended(self, out: str, tmp_path: Path, source: str) -> Recipe:
        reader = {"ref": "ManifestReader", "params": {"manifest_path": source}}
        return _frozen(
            [
                reader,
                _DUR,
                {"ref": "ManifestWriterStage", "params": {"output_path": out}},
                {
                    "ref": "PreserveByValueStage",
                    "params": {"input_value_key": "duration", "target_value": 5.0, "operator": "ge"},
                },
                {"ref": "ManifestWriterStage", "params": {"output_path": str(tmp_path / "kept.jsonl")}},
            ]
        )

    def test_the_parent_diff_plan_carries_a_validated_resume_point(self, store: Path, tmp_path: Path) -> None:
        data, key = self._corpus(tmp_path)
        _rec, out = self._parent(tmp_path, key, data)
        plan = verbs.plan_continuation(self._extended(out, tmp_path, data), "parent", data=data)
        assert plan["mode"] == "incremental"
        assert plan["reuse_point"]["uri"] == out
        assert plan["reuse_point"]["step_key"]  # resolved from the registry, not from a path list

    def test_extend_is_no_longer_refused_for_want_of_a_uri(self, store: Path, tmp_path: Path) -> None:
        data, key = self._corpus(tmp_path)
        _rec, out = self._parent(tmp_path, key, data)
        result = verbs.plan_continuation(
            self._extended(out, tmp_path, data), "parent", data=data, execute=True, choice="extend", confirm=False
        )
        # It stops at the confirm gate, which is correct; what it must not do is claim there is
        # nothing to extend from while the parent's artifact sits in the registry.
        assert "nothing to extend from" not in str(result.get("reason", ""))

    def test_an_unbacked_parent_claim_says_why_instead_of_failing_late(self, store: Path, tmp_path: Path) -> None:
        from nemo_curator.audio_agent import run_store
        from nemo_curator.audio_agent.contracts import RunRecord

        data, key = self._corpus(tmp_path)
        out = str(tmp_path / "gone.jsonl")
        reader = {"ref": "ManifestReader", "params": {"manifest_path": data}}
        rec = _frozen([reader, _DUR, {"ref": "ManifestWriterStage", "params": {"output_path": out}}])
        run_store.save(  # a completed run, but nothing was ever published for it
            RunRecord(
                run_id="unbacked",
                recipe=rec.to_dict(),
                dataset_key=key,
                status="completed",
                steps=artifacts.step_keys(rec, key),
                output_paths=[out],
            )
        )
        plan = verbs.plan_continuation(self._extended(out, tmp_path, data), "unbacked", data=data)
        assert plan.get("reuse_point_unavailable")
        result = verbs.plan_continuation(
            self._extended(out, tmp_path, data), "unbacked", data=data, execute=True, choice="extend", confirm=False
        )
        assert result["status"] == "refused"
        assert "no prior artifact" in result["reason"]

    def test_a_shallower_but_proven_point_beats_a_deeper_unproven_one(self, store: Path, tmp_path: Path) -> None:
        from nemo_curator.audio_agent import run_store
        from nemo_curator.audio_agent.contracts import RunRecord

        data, key = self._corpus(tmp_path)
        mid = str(tmp_path / "mid.jsonl")
        reader = {"ref": "ManifestReader", "params": {"manifest_path": data}}
        # The scan can prove stage 1 (mid writer). The parent claims all three, backed by nothing.
        deep = _frozen([reader, {"ref": "ManifestWriterStage", "params": {"output_path": mid}}, _DUR])
        _publish(deep, 1, dataset_key=key)
        run_store.save(
            RunRecord(
                run_id="deep",
                recipe=deep.to_dict(),
                dataset_key=key,
                status="completed",
                steps=artifacts.step_keys(deep, key),
                output_paths=[mid],
            )
        )
        extended = _frozen(
            [
                reader,
                {"ref": "ManifestWriterStage", "params": {"output_path": mid}},
                _DUR,
                {"ref": "ManifestWriterStage", "params": {"output_path": str(tmp_path / "end.jsonl")}},
            ]
        )
        plan = verbs.plan_continuation(extended, "deep", data=data)
        assert plan["reuse_point"]["uri"] == mid
        assert plan["source"] == "artifact_scan"
        assert "no reusable artifact" in plan["superseded_parent_diff"]


class TestAContinuedRunRecordsWhatWasAskedFor:
    """A continuation executes a rewritten recipe but DELIVERS the requested one.

    Artifacts are already published under the requested pipeline's step keys. The run record used
    the rewritten recipe's keys instead, so the two identities disagreed and every continued run
    became unmatchable -- it shared no prefix with the request it fulfilled, which silently blinded
    anything keyed on the record, including the disclosure of already-done-but-unsaved work.
    """

    def _recipes(self, tmp_path: Path) -> tuple[Recipe, Recipe]:
        """What the user asked for, and the rewritten recipe that actually runs."""
        mid = str(tmp_path / "mid.jsonl")
        asked = _frozen(
            [
                _READER,
                _DUR,
                {"ref": "ManifestWriterStage", "params": {"output_path": mid}},
                {
                    "ref": "PreserveByValueStage",
                    "params": {"input_value_key": "duration", "target_value": 5.0, "operator": "ge"},
                },
                {"ref": "ManifestWriterStage", "params": {"output_path": str(tmp_path / "final.jsonl")}},
            ]
        )
        materialized, err = continuation.materialize(asked, uri=mid, kind="manifest", prefix=3)
        assert materialized is not None, err
        return asked, materialized

    def test_the_rewritten_recipe_shares_no_prefix_with_the_request(self, store: Path, tmp_path: Path) -> None:
        # The premise of the bug, asserted so the fix cannot be mistaken for a no-op.
        asked, materialized = self._recipes(tmp_path)
        assert artifacts.step_keys(materialized, _KEY)[0] != artifacts.step_keys(asked, _KEY)[0]

    def test_the_record_describes_the_request_not_the_rewrite(self, store: Path, tmp_path: Path) -> None:
        from types import SimpleNamespace

        from nemo_curator.audio_agent import run_store

        asked, materialized = self._recipes(tmp_path)
        report = SimpleNamespace(
            accepted=1,
            input_count=1,
            output_paths=[str(tmp_path / "final.jsonl")],
            per_stage_metrics={},
            stages=[],
            failures=[],
            rows=1,
        )
        run_id = verbs._record_run(
            materialized,
            run_id="continued",
            data=None,
            data_fp=None,
            dataset_key=_KEY,
            fingerprint_tier="stat",
            report=report,
            failed=False,
            # What ``run`` hands over for a continuation: the logical recipe's own chain
            # (``_verify_continuation_context`` -> logical_steps), not the rewrite's.
            logical_steps=artifacts.step_keys(asked, _KEY),
        )
        recorded = list(run_store.load(run_id).steps or [])
        assert recorded == artifacts.step_keys(asked, _KEY)
        assert len(recorded) == 5  # the request, not the 3-stage rewrite

    def test_an_ordinary_run_still_records_its_own_chain(self, store: Path, tmp_path: Path) -> None:
        from types import SimpleNamespace

        from nemo_curator.audio_agent import run_store

        asked, _materialized = self._recipes(tmp_path)
        report = SimpleNamespace(
            accepted=1, input_count=1, output_paths=[], per_stage_metrics={}, stages=[], failures=[], rows=1
        )
        run_id = verbs._record_run(
            asked,
            run_id="plain",
            data=None,
            data_fp=None,
            dataset_key=_KEY,
            fingerprint_tier="stat",
            report=report,
            failed=False,
        )
        assert list(run_store.load(run_id).steps or []) == artifacts.step_keys(asked, _KEY)

    def test_prior_continuation_work_is_credited_in_full(self, store: Path, tmp_path: Path) -> None:
        # The end the fix serves: a later request sharing all five stages is told about all five.
        from nemo_curator.audio_agent import run_store
        from nemo_curator.audio_agent.contracts import RunRecord

        asked, _materialized = self._recipes(tmp_path)
        run_store.save(
            RunRecord(
                run_id="continued",
                recipe=asked.to_dict(),
                dataset_key=_KEY,
                status="completed",
                steps=artifacts.step_keys(asked, _KEY),
            )
        )
        later = _frozen(
            [
                *[{"ref": s.ref, "params": dict(s.params)} for s in asked.stages],
                {"ref": "GetAudioDurationStage", "params": {"input_residency": "waveform"}},
            ]
        )
        unsaved = reuse.scan(later, dataset_key=_KEY)["prior_unsaved"]
        assert unsaved["count"] == 5
        assert unsaved["run_id"] == "continued"


class TestServingAnExistingOutputIsEarned:
    """``as_is`` hands back bytes without running anything, so the bytes must be proven to exist."""

    @staticmethod
    def _corpus(tmp_path: Path) -> tuple[str, str]:
        manifest = tmp_path / "corpus.jsonl"
        manifest.write_text(json.dumps({"audio_filepath": "clip.wav", "duration": 7.0}) + "\n")
        return str(manifest), profiler.profile_data(str(manifest)).dataset_key()

    @staticmethod
    def _record(rec: Recipe, key: str, out: str, *, run_id: str = "p", status: str = "completed") -> None:
        from nemo_curator.audio_agent import run_store
        from nemo_curator.audio_agent.contracts import RunRecord

        run_store.save(
            RunRecord(
                run_id=run_id,
                recipe=rec.to_dict(),
                dataset_key=key,
                status=status,
                steps=artifacts.step_keys(rec, key),
                output_paths=[out],
                input_count=1,
                accepted=1,
            )
        )

    def test_a_path_that_was_never_written_is_refused(self, store: Path, tmp_path: Path) -> None:
        data, key = self._corpus(tmp_path)
        ghost = str(tmp_path / "never_written.jsonl")
        reader = {"ref": "ManifestReader", "params": {"manifest_path": data}}
        rec = _frozen([reader, _DUR, {"ref": "ManifestWriterStage", "params": {"output_path": ghost}}])
        self._record(rec, key, ghost)
        result = verbs.plan_continuation(rec, "p", data=data, execute=True, choice="as_is", confirm=False)
        assert result["status"] == "refused"
        assert "does not exist or is empty" in result["reason"]

    def test_an_empty_file_is_not_a_result(self, store: Path, tmp_path: Path) -> None:
        data, key = self._corpus(tmp_path)
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        reader = {"ref": "ManifestReader", "params": {"manifest_path": data}}
        rec = _frozen([reader, _DUR, {"ref": "ManifestWriterStage", "params": {"output_path": str(empty)}}])
        self._record(rec, key, str(empty))
        assert (
            verbs.plan_continuation(rec, "p", data=data, execute=True, choice="as_is", confirm=False)["status"]
            == "refused"
        )

    def test_output_of_a_run_that_did_not_finish_is_refused(self, store: Path, tmp_path: Path) -> None:
        data, key = self._corpus(tmp_path)
        partial = tmp_path / "partial.jsonl"
        partial.write_text('{"audio_filepath": "clip.wav"}\n')  # looks perfectly valid
        reader = {"ref": "ManifestReader", "params": {"manifest_path": data}}
        rec = _frozen([reader, _DUR, {"ref": "ManifestWriterStage", "params": {"output_path": str(partial)}}])
        self._record(rec, key, str(partial), status="failed")
        result = verbs.plan_continuation(rec, "p", data=data, execute=True, choice="as_is", confirm=False)
        assert result["status"] == "refused"
        assert "not a complete result" in result["reason"]

    def test_legacy_output_outside_workspace_is_refused_before_read_or_copy(
        self,
        store: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        data, key = self._corpus(workspace)
        outside = tmp_path / "outside.jsonl"
        outside.write_text('{"audio_filepath":"clip.wav"}\n', encoding="utf-8")
        desired = workspace / "desired.jsonl"
        reader = {"ref": "ManifestReader", "params": {"manifest_path": data}}
        rec = _frozen(
            [
                reader,
                _DUR,
                {
                    "ref": "ManifestWriterStage",
                    "params": {"output_path": str(desired)},
                },
            ]
        )
        self._record(rec, key, str(outside))
        monkeypatch.setenv("AUDIO_AGENT_WORKSPACE", str(workspace))
        monkeypatch.setattr(
            verbs,
            "_has_content",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("outside legacy output was read")),
        )

        result = verbs.plan_continuation(
            rec,
            "p",
            data=data,
            execute=True,
            choice="as_is",
            confirm=False,
        )

        assert result["status"] == "refused"
        assert "workspace" in result["reason"]
        assert not desired.exists()

    def test_an_artifact_backed_output_is_served_and_says_so(self, store: Path, tmp_path: Path) -> None:
        data, key = self._corpus(tmp_path)
        out = str(tmp_path / "done.jsonl")
        reader = {"ref": "ManifestReader", "params": {"manifest_path": data}}
        rec = _frozen([reader, _DUR, {"ref": "ManifestWriterStage", "params": {"output_path": out}}])
        _publish(rec, 2, dataset_key=key)
        self._record(rec, key, out)
        result = verbs.plan_continuation(rec, "p", data=data, execute=True, choice="as_is", confirm=False)
        assert result["status"] == "reused"
        assert result["evidence"] == "artifact"

    def test_a_record_without_an_artifact_is_served_but_named_as_weaker(self, store: Path, tmp_path: Path) -> None:
        # A run from before artifacts existed, or one whose record was pruned. Refusing it would
        # regress a real case; serving it silently would overstate what was checked.
        data, key = self._corpus(tmp_path)
        out = tmp_path / "legacy.jsonl"
        out.write_text('{"audio_filepath": "clip.wav"}\n')
        reader = {"ref": "ManifestReader", "params": {"manifest_path": data}}
        rec = _frozen([reader, _DUR, {"ref": "ManifestWriterStage", "params": {"output_path": str(out)}}])
        self._record(rec, key, str(out))
        result = verbs.plan_continuation(rec, "p", data=data, execute=True, choice="as_is", confirm=False)
        assert result["status"] == "reused"
        assert result["evidence"] == "run_record"
        assert "No artifact record backs this output" in result["note"]

    def test_a_refusal_leaves_no_copy_behind(self, store: Path, tmp_path: Path) -> None:
        # Delivery copies to the declared path, so validation has to come first.
        data, key = self._corpus(tmp_path)
        ghost, declared = str(tmp_path / "nope.jsonl"), str(tmp_path / "declared.jsonl")
        reader = {"ref": "ManifestReader", "params": {"manifest_path": data}}
        rec = _frozen([reader, _DUR, {"ref": "ManifestWriterStage", "params": {"output_path": declared}}])
        self._record(rec, key, ghost)
        verbs.plan_continuation(rec, "p", data=data, execute=True, choice="as_is", confirm=False)
        assert not os.path.exists(declared)


class TestTheRecommendationIsActedOn:
    """Recommending caution and then not taking it is worse than not warning at all."""

    def test_a_low_trust_candidate_defaults_to_a_fresh_run(self, store: Path) -> None:
        rec, _mid, _final = _pipeline(store)
        _publish(rec, 3, fingerprint_tier="shape")  # cannot see a file edited in place
        scan = reuse.scan(rec, dataset_key=_KEY)
        assert scan["recommended"] == "fresh"
        assert verbs._chosen_by_default({**scan, "mode": "already_done"}) == "fresh"

    def test_a_high_trust_candidate_follows_the_mode(self, store: Path) -> None:
        rec, _mid, _final = _pipeline(store)
        _publish(rec, 3)
        scan = reuse.scan(rec, dataset_key=_KEY)
        assert scan["recommended"] == "as_is"
        assert verbs._chosen_by_default({**scan, "mode": "already_done"}) == "as_is"

    def test_a_plan_with_no_recommendation_falls_back_to_the_mode(self) -> None:
        assert verbs._chosen_by_default({"mode": "incremental"}) == "extend"
        assert verbs._chosen_by_default({"mode": "already_done"}) == "as_is"
        assert verbs._chosen_by_default({"mode": "full_rerun"}) == "fresh"

    def test_an_unoffered_recommendation_is_not_obeyed(self) -> None:
        # Defence against a recommendation the card never presented as a choice.
        plan = {"mode": "already_done", "recommended": "teleport", "choices": [{"id": "as_is"}, {"id": "fresh"}]}
        assert verbs._chosen_by_default(plan) == "as_is"

    def test_low_trust_output_is_not_served_without_being_asked_for(self, store: Path, tmp_path: Path) -> None:
        # End to end: the low-trust path must not come back "reused" when no choice was stated.
        manifest = tmp_path / "corpus.jsonl"
        manifest.write_text(json.dumps({"audio_filepath": "clip.wav", "duration": 7.0}) + "\n")
        key = profiler.profile_data(str(manifest)).dataset_key()
        out = str(tmp_path / "done.jsonl")
        rec = _frozen(
            [
                {"ref": "ManifestReader", "params": {"manifest_path": str(manifest)}},
                _DUR,
                {"ref": "ManifestWriterStage", "params": {"output_path": out}},
            ]
        )
        _publish(rec, 2, dataset_key=key, fingerprint_tier="shape")
        result = verbs.plan_continuation(rec, data=str(manifest), execute=True, confirm=False)
        assert result["status"] != "reused"


class TestFreshnessWindowsApplyWhereDataIsFetched:
    """A re-fetched corpus can differ. A pinned model cannot -- that shows up in model_version."""

    def test_a_corpus_download_has_a_freshness_window(self) -> None:
        for stage in ("CreateInitialManifestFleursStage", "CreateInitialManifestReadSpeechStage"):
            _deterministic, ttl = artifacts.stage_trust(stage)
            assert ttl > 0, f"{stage} fetches its data and should expire"

    def test_a_model_download_does_not(self) -> None:
        # These all carry needs_internet_first_run, but for a checkpoint, not for the corpus.
        for stage in ("ASRStage", "UTMOSFilterStage", "InferenceSortformerStage", "PyAnnoteDiarizationStage"):
            _deterministic, ttl = artifacts.stage_trust(stage)
            assert ttl == 0, f"{stage} downloads a model; expiring its output would recompute for nothing"

    def test_an_ordinary_local_stage_does_not(self) -> None:
        assert artifacts.stage_trust("GetAudioDurationStage")[1] == 0

    def test_the_age_of_an_artifact_is_read_as_utc(self) -> None:
        # created_at is written with gmtime, so reading it as local time shifted every age by the
        # machine's offset -- which, east of UTC, clamped to 0 and made a window never expire.
        made = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 7200))
        assert abs(artifacts._age_sec(artifacts.Artifact(step_key="k", created_at=made)) - 7200) < 10

    def test_an_expired_artifact_is_refused(self, store: Path) -> None:
        rec, _mid, _final = _pipeline(store)
        old = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 60))
        art = _publish(rec, 3, ttl_sec=30, created_at=old)
        assert any("freshness window" in r for r in artifacts.invalid_reasons(art, dataset_key=_KEY))

    def test_a_fresh_artifact_inside_its_window_is_fine(self, store: Path) -> None:
        rec, _mid, _final = _pipeline(store)
        art = _publish(rec, 3, ttl_sec=3600)
        assert not artifacts.invalid_reasons(art, dataset_key=_KEY)


class TestUnmeasuredWorkIsNotAssumedCheap:
    """Zero recorded seconds means nobody looked, not that there was nothing to look at."""

    def test_the_cards_say_which_stages_could_have_been_slow(self) -> None:
        for stage in ("ASRStage", "InferenceSortformerStage", "SpeakerSeparationStage", "UTMOSFilterStage"):
            assert artifacts.stage_is_costly(stage), f"{stage} runs a model; its cost must not be assumed away"
        for stage in ("ManifestReader", "GetAudioDurationStage", "ManifestWriterStage"):
            assert not artifacts.stage_is_costly(stage), f"{stage} is IO-bound; asking about it is a nag"

    def test_a_stage_nobody_wrote_a_card_for_is_not_called_cheap(self) -> None:
        assert artifacts.stage_is_costly("NoCardWasEverWrittenForThisStage")

    def test_a_card_that_does_not_state_its_bound_is_not_called_cheap(self, monkeypatch) -> None:  # noqa: ANN001
        """``bound: null`` is the placeholder a config-dependent stage carries while nobody has
        priced it. It leaves us exactly as uninformed as having no card -- already refused as
        cheap -- yet it used to read as "not gpu" and buy a pass under the auto-take threshold.

        Synthetic cards on purpose: this pins the RULE, so it keeps holding after whichever
        shipped card carried the placeholder gets filled in.
        """
        for resource in ({"bound": None, "cpus": 1.0}, {"cpus": 1.0}, {}):
            monkeypatch.setattr(artifacts, "_card", lambda _ref, r=resource: {"resource": r})
            assert artifacts.stage_is_costly("AnyStage"), f"unstated bound in {resource} rated cheap"

    def test_a_card_that_does_state_a_cheap_bound_is_still_taken_at_its_word(self, monkeypatch) -> None:  # noqa: ANN001
        """The rule must not collapse into "everything is expensive", which would turn the
        reuse gate into a permanent nag -- the failure the costliness check was narrowed to avoid.
        """
        for bound in ("cpu", "io"):
            monkeypatch.setattr(artifacts, "_card", lambda _ref, b=bound: {"resource": {"bound": b}})
            assert not artifacts.stage_is_costly("AnyStage"), f"bound={bound} should stay cheap"

    def test_an_untimed_model_prefix_asks_although_it_scores_as_free(self, store: Path) -> None:
        # The writer took 2 s and the transcription before it was never timed, so the saving
        # reads as 2 s -- comfortably under the threshold meant for milliseconds. Taking that
        # silently is how an hour of ASR came back as yesterday's answer with no question asked.
        rec = _frozen(
            [_READER, _ASR, {"ref": "ManifestWriterStage", "params": {"output_path": str(store / "t.jsonl")}}]
        )
        _publish(rec, 2, duration_sec=2.0)
        result = reuse.scan(rec, dataset_key=_KEY)
        assert result["estimated_saving_sec"] == 2.0
        assert result["unpriced_stages"] == ["ASRStage"]
        assert result["prompt_user"] is True

    def test_an_untimed_io_prefix_still_says_nothing(self, store: Path) -> None:
        # The counterweight. Every pipeline has untimed in-memory stages; treating "unmeasured"
        # alone as a reason to ask turned the gate into a permanent nag.
        rec, _mid, _final = _pipeline(store)
        _publish(rec, 3, duration_sec=2.0)
        result = reuse.scan(rec, dataset_key=_KEY)
        assert result["unpriced_stages"] == []
        assert result["prompt_user"] is False

    def test_a_recorded_cumulative_prices_the_stages_that_persisted_nothing(self, store: Path) -> None:
        # cumulative_sec covers the whole prefix including the model stage, so nothing is unknown
        # and a genuinely quick run is taken at its word.
        rec = _frozen(
            [_READER, _ASR, {"ref": "ManifestWriterStage", "params": {"output_path": str(store / "t.jsonl")}}]
        )
        _publish(rec, 2, duration_sec=2.0, cumulative_sec=5.0)
        result = reuse.scan(rec, dataset_key=_KEY)
        assert result["unpriced_stages"] == []
        assert result["prompt_user"] is False

    def test_the_gate_is_told_why_it_is_asking(self, store: Path) -> None:
        # Without this the approval card reads "saves 2 s" beside a question and looks broken.
        src, key = _real_source(store)
        reader = {"ref": "ManifestReader", "params": {"manifest_path": src}}
        rec = _frozen(
            [reader, _ASR, {"ref": "ManifestWriterStage", "params": {"output_path": str(store / "t.jsonl")}}]
        )
        _publish(rec, 2, dataset_key=key, duration_sec=2.0)
        plan = verbs.plan_continuation(rec, data=src)
        assert plan["unpriced_stages"] == ["ASRStage"]
        assert plan["prompt_user"] is True


class TestNonDeterminismAsksRatherThanHides:
    """A result no rerun is promised to match is still a result. Show it and let them choose."""

    def test_it_no_longer_makes_the_artifact_invalid(self, store: Path) -> None:
        rec, _mid, _final = _pipeline(store)
        art = _publish(rec, 3, deterministic=False)
        assert not artifacts.invalid_reasons(art, dataset_key=_KEY)

    def test_it_is_a_caution_instead(self, store: Path) -> None:
        rec, _mid, _final = _pipeline(store)
        art = _publish(rec, 3, deterministic=False)
        assert any("non-deterministic" in r for r in artifacts.caution_reasons(art))

    def test_a_caller_that_needs_certainty_can_still_refuse(self, store: Path) -> None:
        rec, _mid, _final = _pipeline(store)
        art = _publish(rec, 3, deterministic=False)
        strict = artifacts.invalid_reasons(art, dataset_key=_KEY, require_high_trust=True)
        assert any("non-deterministic" in r for r in strict)

    def test_the_candidate_is_shown_with_its_reason_rather_than_vanishing(self, store: Path) -> None:
        # It used to be dropped as invalid, so the user was never told prior work existed at all
        # and the "pre-select fresh" branch written for exactly this case could never run.
        rec, _mid, _final = _pipeline(store)
        _publish(rec, 3, deterministic=False)
        result = reuse.scan(rec, dataset_key=_KEY)
        assert result["candidates"], "prior work existed and the user was told nothing"
        assert result["candidates"][0]["trust"] == "low"
        assert any("non-deterministic" in w for w in result["candidates"][0]["weaknesses"])

    def test_but_it_is_not_reused_unless_someone_asks_for_it(self, store: Path) -> None:
        rec, _mid, _final = _pipeline(store)
        _publish(rec, 3, deterministic=False)
        result = reuse.scan(rec, dataset_key=_KEY)
        assert result["prompt_user"] is True
        assert result["recommended"] == "fresh"
        assert verbs._chosen_by_default(result) == "fresh"

    def test_the_one_stage_that_documents_its_randomness_declares_it(self) -> None:
        # pyannote's own card note: the internal long-turn VAD re-segments with randomized chunk
        # sizes and no seed is exposed. That is evidence; the field now records it.
        assert artifacts.stage_trust("PyAnnoteDiarizationStage")[0] is False

    def test_silence_is_still_read_as_reproducible(self) -> None:
        # Deliberate: flipping the default would block reuse of every model stage on a suspicion
        # nobody has measured. A card claims non-determinism when someone has evidence for it.
        assert artifacts.stage_trust("InferenceSortformerStage")[0] is True


class TestArtifactKindComesFromTheOutput:
    """The kind picks the source stage that re-reads an artifact, so a guess routes work wrongly."""

    def test_an_archive_is_not_called_a_folder_of_audio(self) -> None:
        # "audio" appears in output_audio_tar_path, so the name-based rule called a .tar an
        # audio_dir -- and a continuation then fed a tarball to a stage that lists a directory.
        from nemo_curator.audio_agent.recipe import StageRef

        stage = StageRef(ref="SnippetExtractionStage", params={"output_audio_tar_path": "/x/snips.tar"})
        assert artifacts.output_uri(stage) == ("/x/snips.tar", "archive")
        assert artifacts._kind_of("/x/snips.tar.gz") == "archive"

    def test_dataset_source_cache_is_not_a_resumable_artifact(self) -> None:
        from nemo_curator.audio_agent.recipe import StageRef

        for ref in (
            "CreateInitialManifestFleursStage",
            "CreateInitialManifestReadSpeechStage",
        ):
            stage = StageRef(
                ref=ref,
                params={"raw_data_dir": "/x/download-cache"},
            )
            assert artifacts.output_uri(stage) == ("", "unknown")

    def test_source_cache_step_never_persists_or_gets_materialized_generically(self) -> None:
        rec = _frozen(
            [
                {
                    "ref": "CreateInitialManifestFleursStage",
                    "params": {
                        "raw_data_dir": "/x/fleurs",
                        "lang": "en_us",
                        "split": "test",
                    },
                },
                _DUR,
            ]
        )

        source_step = artifacts.plan_steps(rec, _KEY)[0]

        assert source_step.persists() is False
        assert source_step.uri == ""

    def test_no_source_stage_claims_to_read_an_archive(self, store: Path, tmp_path: Path) -> None:
        rec = _frozen([_READER, _DUR, _WRITER])
        materialized, err = continuation.materialize(rec, uri="/x/snips.tar", kind="archive", prefix=1)
        assert materialized is None
        assert "archive" in err

    def test_a_directory_is_classified_by_what_is_in_it(self, tmp_path: Path) -> None:
        speakers = tmp_path / "by_speaker"
        speakers.mkdir()
        (speakers / "spk0.txt").write_text("hello\n")
        (speakers / "spk1.txt").write_text("world\n")
        assert artifacts._kind_of(str(speakers)) == "text_dir"

        clips = tmp_path / "clips"
        clips.mkdir()
        (clips / "a.wav").write_bytes(b"RIFF")
        assert artifacts._kind_of(str(clips)) == "audio_dir"

    def test_a_marker_does_not_decide_the_kind(self, tmp_path: Path) -> None:
        clips = tmp_path / "clips"
        clips.mkdir()
        (clips / "a.flac").write_bytes(b"fLaC")
        artifacts.write_marker(str(clips), step_key="k", rows=1)
        assert artifacts._kind_of(str(clips)) == "audio_dir"

    def test_mixed_contents_are_admitted_as_unknown(self, tmp_path: Path) -> None:
        mixed = tmp_path / "mixed"
        mixed.mkdir()
        (mixed / "a.wav").write_bytes(b"RIFF")
        (mixed / "notes.txt").write_text("x")
        assert artifacts._kind_of(str(mixed)) == "unknown"

    def test_a_directory_that_does_not_exist_yet_is_not_guessed(self) -> None:
        assert artifacts._kind_of("/x/not/created/yet") == "unknown"
        assert artifacts.classify_output("/x/not/created/yet") is None

    def test_publishing_classifies_from_the_real_output(self, store: Path, tmp_path: Path) -> None:
        # What plan time could not know, publish time can see.
        groups = tmp_path / "groups"
        groups.mkdir()
        (groups / "spk0.txt").write_text("hi\n")
        assert artifacts.classify_output(str(groups)) == "text_dir"


class TestEveryWritingStageIsReusable:
    """A stage that writes must say where, or reuse cannot see its output at all.

    Output locations are recognised from a fixed set of parameter names. That set is a promise: a
    new writing stage whose parameter is not in it silently drops out of reuse, with no failure
    anywhere to point at it.
    """

    @staticmethod
    def _param_names(stage_cls: type) -> set[str]:
        import dataclasses
        import inspect

        names: set[str] = set()
        if dataclasses.is_dataclass(stage_cls):
            names |= {f.name for f in dataclasses.fields(stage_cls)}
        with contextlib.suppress(TypeError, ValueError):
            names |= set(inspect.signature(stage_cls.__init__).parameters) - {"self"}
        return names

    def test_a_stage_that_writes_to_disk_declares_where(self) -> None:
        from nemo_curator.stages.audio._agent._agent_registry import static_contract
        from nemo_curator.stages.audio._agent._catalog import get_agent_ready_stage_class, list_agent_ready_stages

        undeclared = []
        for name in list_agent_ready_stages():
            stage_cls = get_agent_ready_stage_class(name)
            gates = None
            with contextlib.suppress(Exception):  # contract shape is another test's business
                gates = static_contract(stage_cls).gates
            if (
                gates is not None
                and getattr(gates, "writes_to_disk", False)
                and not (self._param_names(stage_cls) & set(artifacts._URI_PREFERENCE))
            ):
                undeclared.append(name)
        assert not undeclared, (
            "these stages write to disk but name no output location reuse recognises, so their "
            f"output can never be reused: {undeclared}. Add the parameter to "
            "artifacts._URI_PREFERENCE and recipe.OUTPUT_LOCATION_PARAMS."
        )

    def test_the_two_output_param_lists_agree(self) -> None:
        # One list decides what reuse tracks, the other what the step key ignores. If they drift,
        # a stage's output location starts changing its identity and reuse stops matching.
        from nemo_curator.audio_agent.recipe import OUTPUT_LOCATION_PARAMS

        assert set(artifacts._URI_PREFERENCE) == set(OUTPUT_LOCATION_PARAMS)
