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

"""A prior run that read the same folder is disclosed, even when the recipe drifted.

This reproduces a real test that looked like a reuse failure and was not. A folder was curated,
one file was added, and a second run over the same folder built a slightly different pipeline
(a different source stage, a dropped mono stage, a lower quality bar). Every step-key matcher
went silent -- correctly, since neither the corpus nor the recipe matched -- and the agent was
told "fresh", which reads as "never done here". The truth it could not surface was: this exact
folder was curated twenty minutes ago, and only one file has changed since.

``prior_on_path`` adds that missing axis: match by the source PATH, independent of recipe and
dataset key, and report it as advice (a recipe diff + a file-level data delta), never as an
action. These tests pin that it fires on the drifted-recipe/added-file case, stays silent when
the folder differs, never leaks a secret param value through the diff, and leaves the reuse
decision itself untouched.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from nemo_curator.audio_agent import artifacts, profiler, reuse, run_index, run_store, verbs
from nemo_curator.audio_agent.contracts import RunRecord
from nemo_curator.audio_agent.recipe import Recipe

# Output-location params of the prior recipe. Never written -- the recipe is recorded, not run --
# and excluded from reuse identity, so their exact value is irrelevant to what these tests assert.
_PRIOR_RESAMPLE_DIR = "/tmp/rs_prior"  # noqa: S108
_PRIOR_OUT = "/tmp/prior_out.jsonl"  # noqa: S108


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated run/artifact store, so a test never sees the developer's real history."""
    monkeypatch.setenv("AUDIO_AGENT_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


def _folder(tmp_path: Path, names: list[str]) -> str:
    """A folder of distinctly-sized ``.wav`` files (content arbitrary; only stat identity matters)."""
    d = tmp_path / "corpus"
    d.mkdir(exist_ok=True)
    for i, name in enumerate(names):
        (d / name).write_bytes(b"RIFF" + bytes([i % 251]) * (2048 + i * 7))
    return str(d)


def _profile(folder: str):  # noqa: ANN202
    return profiler.profile_data(folder, folder_extensions=[".wav"], recursive=True)


def _prior_recipe(folder: str, *, source: str, mono: bool, mos: float) -> Recipe:
    stages: list[dict] = [
        {
            "ref": source,
            "params": {"data_dir": folder} if source.endswith("AudioFolderStage") else {"raw_data_dir": folder},
        }
    ]
    stages.append(
        {
            "ref": "ResampleAudioStage",
            "params": {"target_sample_rate": 16000, "target_nchannels": 1, "resampled_audio_dir": _PRIOR_RESAMPLE_DIR},
        }
    )
    if mono:
        stages.append({"ref": "MonoConversionStage", "params": {}})
    stages.append({"ref": "VADSegmentationStage", "params": {}})
    stages.append({"ref": "UTMOSFilterStage", "params": {"mos_threshold": mos, "action": "filter"}})
    stages.append({"ref": "ManifestWriterStage", "params": {"output_path": _PRIOR_OUT}})
    return Recipe.from_dict({"stages": stages}).freeze()


def _save_prior_run(
    folder: str,
    recipe: Recipe,
    *,
    prior_profile,  # noqa: ANN001
    created_at: str = "2026-08-17T07:19:19Z",
) -> str:
    """Persist a completed run of ``recipe`` on the folder as ``prior_profile`` saw it."""
    dataset_key = prior_profile.dataset_key()
    steps = [p.step_key for p in artifacts.plan_steps(recipe, dataset_key)]
    # Coverage on the deepest step is what a data delta compares against: the source files the
    # prior run processed. Real runs save it at publish time; here we save it directly.
    artifacts.save_coverage(steps[-1], dict(prior_profile.inventory))
    run_id = run_store.new_run_id(recipe.config_hash)
    run_store.save(
        RunRecord(
            run_id=run_id,
            recipe=recipe.to_dict(),
            config_hash=recipe.config_hash,
            semantic_hash=recipe.semantic_hash,
            goal={"task": "quality_filter"},
            data_source=folder,
            dataset_key=dataset_key,
            status="completed",
            input_count=len(prior_profile.inventory),
            output_paths=[_PRIOR_OUT],
            steps=steps,
            created_at=created_at,
        )
    )
    return run_id


def _current_recipe(folder: str, *, source: str, mono: bool, mos: float) -> dict:
    """The recipe the second session builds -- same shape helper, fresh output paths."""
    return _prior_recipe(folder, source=source, mono=mono, mos=mos).to_dict()


class TestItFiresWhenTheFolderWasCuratedByADifferentPipeline:
    def test_the_drifted_recipe_and_added_file_are_both_reported(self, store: Path) -> None:
        folder = _folder(store, ["a.wav", "b.wav", "c.wav", "d.wav", "e.wav"])
        prior_profile = _profile(folder)
        run_id = _save_prior_run(
            folder,
            _prior_recipe(folder, source="CreateInitialManifestReadSpeechStage", mono=True, mos=3.4),
            prior_profile=prior_profile,
        )
        # A file is added, and the second run builds a materially different pipeline.
        (Path(folder) / "f.wav").write_bytes(b"RIFF" + b"\x05" * 4096)
        current = _current_recipe(folder, source="CreateInitialManifestAudioFolderStage", mono=False, mos=2.5)

        result = verbs.reuse_scan(current, data=folder)
        prior = result.get("prior_on_same_path")

        assert prior is not None, "a prior run on this same folder should be disclosed"
        assert prior["run_id"] == run_id
        assert prior["same_recipe"] is False
        assert "MonoConversionStage" in prior["recipe_diff"]["removed_stages"]
        assert "CreateInitialManifestAudioFolderStage" in prior["recipe_diff"]["added_stages"]
        threshold = [c for c in prior["recipe_diff"]["changed_params"] if c["param"] == "mos_threshold"]
        assert threshold
        assert threshold[0]["from"] == 3.4
        assert threshold[0]["to"] == 2.5

    def test_the_file_level_delta_names_the_one_that_was_added(self, store: Path) -> None:
        folder = _folder(store, ["a.wav", "b.wav", "c.wav", "d.wav", "e.wav"])
        _save_prior_run(
            folder,
            _prior_recipe(folder, source="CreateInitialManifestReadSpeechStage", mono=True, mos=3.4),
            prior_profile=_profile(folder),
        )
        (Path(folder) / "f.wav").write_bytes(b"RIFF" + b"\x05" * 4096)
        current = _current_recipe(folder, source="CreateInitialManifestAudioFolderStage", mono=False, mos=2.5)

        delta = verbs.reuse_scan(current, data=folder)["prior_on_same_path"]["data_delta"]

        assert delta["basis"] == "inventory"
        assert delta["added"] == 1
        assert delta["unchanged"] == 5
        assert "f.wav" in delta["added_files"]

    def test_a_different_pipeline_recommends_aligning(self, store: Path) -> None:
        folder = _folder(store, ["a.wav", "b.wav", "c.wav"])
        _save_prior_run(
            folder,
            _prior_recipe(folder, source="CreateInitialManifestReadSpeechStage", mono=True, mos=3.4),
            prior_profile=_profile(folder),
        )
        (Path(folder) / "d.wav").write_bytes(b"RIFF" + b"\x09" * 5000)
        current = _current_recipe(folder, source="CreateInitialManifestAudioFolderStage", mono=False, mos=2.5)

        assert verbs.reuse_scan(current, data=folder)["prior_on_same_path"]["recommendation"] == "align"

    def test_the_decision_itself_is_left_fresh(self, store: Path) -> None:
        """Advisory only: the notice never turns a correct 'fresh' into a reuse."""
        folder = _folder(store, ["a.wav", "b.wav", "c.wav"])
        _save_prior_run(
            folder,
            _prior_recipe(folder, source="CreateInitialManifestReadSpeechStage", mono=True, mos=3.4),
            prior_profile=_profile(folder),
        )
        (Path(folder) / "d.wav").write_bytes(b"RIFF" + b"\x09" * 5000)
        current = _current_recipe(folder, source="CreateInitialManifestAudioFolderStage", mono=False, mos=2.5)

        result = verbs.reuse_scan(current, data=folder)

        assert result["decision"] == "fresh"
        assert result["reuse_point"] is None

    def test_the_notice_forces_a_prompt_so_the_host_cannot_skip_it(self, store: Path) -> None:
        """The bug this closes: a populated notice beside prompt_user=false was read as
        'nothing to reuse', and a real session reported exactly that over a just-curated folder.
        Disclosure has to flip the one signal the 'never nag' rule cannot ignore."""
        folder = _folder(store, ["a.wav", "b.wav", "c.wav"])
        _save_prior_run(
            folder,
            _prior_recipe(folder, source="CreateInitialManifestReadSpeechStage", mono=True, mos=3.4),
            prior_profile=_profile(folder),
        )
        (Path(folder) / "d.wav").write_bytes(b"RIFF" + b"\x09" * 5000)
        current = _current_recipe(folder, source="CreateInitialManifestAudioFolderStage", mono=False, mos=2.5)

        result = verbs.reuse_scan(current, data=folder)

        assert result["decision"] == "fresh"
        assert result["prompt_user"] is True
        assert result["prior_on_same_path"]["note"] in result["rationale"]


class TestItStaysSilentWhenThereIsNothingToDisclose:
    def test_a_different_folder_is_not_matched(self, store: Path) -> None:
        curated = _folder(store, ["a.wav", "b.wav", "c.wav"])
        _save_prior_run(
            curated,
            _prior_recipe(curated, source="CreateInitialManifestReadSpeechStage", mono=True, mos=3.4),
            prior_profile=_profile(curated),
        )
        other = str(store / "elsewhere")
        Path(other).mkdir()
        (Path(other) / "x.wav").write_bytes(b"RIFF" + b"\x01" * 3000)
        current = _current_recipe(other, source="CreateInitialManifestAudioFolderStage", mono=False, mos=2.5)

        assert "prior_on_same_path" not in verbs.reuse_scan(current, data=other)

    def test_no_prior_run_means_no_notice(self, store: Path) -> None:
        folder = _folder(store, ["a.wav", "b.wav"])
        current = _current_recipe(folder, source="CreateInitialManifestAudioFolderStage", mono=False, mos=2.5)

        assert "prior_on_same_path" not in verbs.reuse_scan(current, data=folder)


class TestASecretParamNeverLeaksThroughTheDiff:
    def test_a_changed_token_is_reported_as_changed_not_shown(self, store: Path) -> None:
        folder = _folder(store, ["a.wav", "b.wav", "c.wav"])
        prior = _prior_recipe(folder, source="PyAnnoteDiarizationStage", mono=False, mos=3.4)
        # Give the prior run a diarizer carrying a token, and a different one this time.
        prior_dict = prior.to_dict()
        prior_dict["stages"].insert(
            1, {"ref": "PyAnnoteDiarizationStage", "params": {"hf_token": "SECRET-PRIOR-TOKEN"}}
        )
        prior = Recipe.from_dict(prior_dict).freeze()
        _save_prior_run(folder, prior, prior_profile=_profile(folder))
        (Path(folder) / "d.wav").write_bytes(b"RIFF" + b"\x09" * 5000)

        cur = prior.to_dict()
        cur["stages"][1]["params"]["hf_token"] = "SECRET-CURRENT-TOKEN"  # noqa: S105
        result = verbs.reuse_scan(cur, data=folder)

        blob = repr(result)
        assert "SECRET-PRIOR-TOKEN" not in blob
        assert "SECRET-CURRENT-TOKEN" not in blob


class TestTheRecipeDiffIsStructural:
    def test_value_changes_and_presence_changes_are_told_apart(self) -> None:
        prior = Recipe.from_dict(
            {"stages": [{"ref": "GetAudioDurationStage", "params": {"audio_filepath_key": "audio_filepath"}}]}
        ).freeze()
        current = Recipe.from_dict(
            {
                "stages": [
                    {"ref": "GetAudioDurationStage", "params": {"audio_filepath_key": "path", "duration_key": "dur"}}
                ]
            }
        ).freeze()

        diff = reuse._recipe_diff(prior, current)

        assert not diff["added_stages"]
        assert not diff["removed_stages"]
        by_param = {c["param"]: c for c in diff["changed_params"]}
        assert by_param["audio_filepath_key"]["from"] == "audio_filepath"
        assert by_param["audio_filepath_key"]["to"] == "path"
        # duration_key went from unset to a value: a presence change, summarised not spelled out.
        assert "default(s) made explicit" in diff["phrase"]

    def test_identical_recipes_diff_to_nothing(self) -> None:
        r = Recipe.from_dict({"stages": [{"ref": "GetAudioDurationStage", "params": {}}]}).freeze()

        diff = reuse._recipe_diff(r, r)

        assert diff["identical"] is True


class TestSeveralPriorRunsAreRankedNotCollapsed:
    """A folder curated more than once has to be reported as a choice, closest pipeline first."""

    def _two_priors(self, store: Path) -> tuple[str, str, str]:
        """A far run (different source stage, no mono, different bar) and a near one (same but
        for one threshold), both on the same folder, the far one written most recently."""
        folder = _folder(store, ["a.wav", "b.wav", "c.wav"])
        profile = _profile(folder)
        near = _save_prior_run(
            folder,
            _prior_recipe(folder, source="CreateInitialManifestAudioFolderStage", mono=False, mos=3.0),
            prior_profile=profile,
            created_at="2026-08-17T07:00:00Z",
        )
        far = _save_prior_run(
            folder,
            _prior_recipe(folder, source="CreateInitialManifestReadSpeechStage", mono=True, mos=3.4),
            prior_profile=profile,
            created_at="2026-08-17T09:00:00Z",
        )
        return folder, near, far

    def test_the_closest_pipeline_leads_even_when_it_is_not_the_newest(self, store: Path) -> None:
        folder, near, far = self._two_priors(store)
        (Path(folder) / "d.wav").write_bytes(b"RIFF" + b"\x09" * 5000)
        current = _current_recipe(folder, source="CreateInitialManifestAudioFolderStage", mono=False, mos=2.5)

        prior = verbs.reuse_scan(current, data=folder)["prior_on_same_path"]

        assert prior["count"] == 2
        assert [m["run_id"] for m in prior["matches"]] == [near, far]
        # The closest match is promoted, so a host reading one level deep still sees a real run.
        assert prior["run_id"] == near
        assert prior["note"] == prior["matches"][0]["note"] + " (1 other run(s) also read this folder; see 'matches'.)"

    def test_the_notice_names_the_commands_that_answer_it(self, store: Path) -> None:
        folder, near, _far = self._two_priors(store)
        (Path(folder) / "d.wav").write_bytes(b"RIFF" + b"\x09" * 5000)
        current = _current_recipe(folder, source="CreateInitialManifestAudioFolderStage", mono=False, mos=2.5)

        nxt = verbs.reuse_scan(current, data=folder)["prior_on_same_path"]["next"]

        assert near in nxt["inspect"]
        assert near in nxt["adopt"]
        assert folder in nxt["adopt"]


class TestTheSameFolderAndSameFilesUnderADifferentPipeline:
    """The corpus never moved, only the plan did -- a case the dataset key cannot distinguish."""

    def test_it_is_still_disclosed_and_does_not_claim_the_data_changed(self, store: Path) -> None:
        folder = _folder(store, ["a.wav", "b.wav", "c.wav"])
        _save_prior_run(
            folder,
            _prior_recipe(folder, source="CreateInitialManifestReadSpeechStage", mono=True, mos=3.4),
            prior_profile=_profile(folder),
        )
        current = _current_recipe(folder, source="CreateInitialManifestAudioFolderStage", mono=False, mos=2.5)

        prior = verbs.reuse_scan(current, data=folder)["prior_on_same_path"]

        assert prior["recommendation"] == "align"
        assert prior["data_delta"]["added"] == 0
        assert prior["data_delta"]["unchanged"] == 3

    def test_the_same_pipeline_on_unchanged_files_is_not_offered_as_a_delta(self, store: Path) -> None:
        """Nothing changed, so there is no changed-file work to do -- saying otherwise sends the
        user to a verb that will refuse."""
        folder = _folder(store, ["a.wav", "b.wav"])
        prior = _prior_recipe(folder, source="CreateInitialManifestAudioFolderStage", mono=False, mos=3.0)
        _save_prior_run(folder, prior, prior_profile=_profile(folder))

        notice = verbs.reuse_scan(prior.to_dict(), data=folder)["prior_on_same_path"]

        assert notice["same_recipe"] is True
        assert notice["recommendation"] == "fresh"
        assert "Nothing reusable remains" in notice["note"]


class TestListingRunsByFolderPath:
    def test_a_run_on_the_same_folder_under_another_corpus_state_is_listed(self, store: Path) -> None:
        folder = _folder(store, ["a.wav", "b.wav"])
        run_id = _save_prior_run(
            folder,
            _prior_recipe(folder, source="CreateInitialManifestAudioFolderStage", mono=False, mos=3.0),
            prior_profile=_profile(folder),
        )
        (Path(folder) / "c.wav").write_bytes(b"RIFF" + b"\x02" * 4096)

        listing = verbs.runs(data=folder)

        assert [r["run_id"] for r in listing["runs"]] == [run_id]
        assert listing["same_folder_only"] == [run_id]
        assert "same_folder_only" in listing["note"]

    def test_a_folder_with_no_history_lists_nothing_extra(self, store: Path) -> None:
        folder = _folder(store, ["a.wav"])

        listing = verbs.runs(data=folder)

        assert listing["runs"] == []
        assert "same_folder_only" not in listing


class TestOneRunIsExplainedWithoutReadingEveryParam:
    def test_the_overview_names_the_pipeline_its_settings_and_its_verdict(self, store: Path) -> None:
        folder = _folder(store, ["a.wav", "b.wav"])
        run_id = _save_prior_run(
            folder,
            _prior_recipe(folder, source="CreateInitialManifestAudioFolderStage", mono=False, mos=3.0),
            prior_profile=_profile(folder),
        )

        overview = verbs.runs(run_id=run_id)["overview"]

        assert "UTMOSFilterStage" in overview["pipeline"]
        assert overview["objective"] == "quality_filter"
        assert overview["prompt"] == "quality_filter"
        assert overview["pipeline_summary"].startswith("CreateInitialManifest")
        assert "UTMOSFilter" in overview["pipeline_summary"]
        assert overview["data"]["source"] == folder
        assert overview["data"]["input_count"] == 2
        assert overview["outputs"] == [_PRIOR_OUT]
        # A run that declared no success contract is not reported as one that passed.
        assert overview["acceptance"]["overall"] == "not_recorded"
        thresholds = [e for e in overview["key_params"] if e["stage"] == "UTMOSFilterStage"]
        assert thresholds[0]["params"]["mos_threshold"] == 3.0

    def test_an_unknown_run_id_is_still_an_error_not_an_empty_overview(self, store: Path) -> None:
        assert "error" in verbs.runs(run_id="run-nope")


class TestFindRunsCanFilterByPath:
    def test_the_data_source_filter_selects_only_that_folder(self, store: Path) -> None:
        for i, src in enumerate(("/data/one", "/data/two", "/data/one")):
            run_store.save(
                RunRecord(
                    run_id=run_store.new_run_id(f"h{i}"),
                    data_source=src,
                    dataset_key=f"stat:{i}",
                    status="completed",
                    created_at=f"2026-08-1{i}T00:00:00Z",
                )
            )
            time.sleep(0.001)

        rows = run_index.find_runs(data_source="/data/one")

        assert rows
        assert all(r["data_source"] == "/data/one" for r in rows)
        assert len(rows) == 2
