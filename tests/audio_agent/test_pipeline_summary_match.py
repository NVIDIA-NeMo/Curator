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

"""Persist a brief pipeline summary and rank same-folder priors by prompt + summary.

Pins the failure mode where inventing a recipe first, then ranking by stage
edit-distance, preferred a convert-only prior over richer work that already
covered most of the folder. Successful runs store ``pipeline_summary``;
``runs --data --goal`` compares the current request to each prior's prompt and
that summary before any recipe exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from nemo_curator.audio_agent import artifacts, profiler, reuse, run_store, verbs
from nemo_curator.audio_agent.contracts import RunRecord
from nemo_curator.audio_agent.recipe import Recipe


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("AUDIO_AGENT_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


def _folder(tmp_path: Path, names: list[str]) -> str:
    d = tmp_path / "corpus"
    d.mkdir(exist_ok=True)
    for i, name in enumerate(names):
        (d / name).write_bytes(b"RIFF" + bytes([i % 251]) * (2048 + i * 7))
    return str(d)


def _profile(folder: str):  # noqa: ANN202
    return profiler.profile_data(folder, folder_extensions=[".wav"], recursive=True)


def _convert_recipe(folder: str, out: str) -> Recipe:
    return Recipe.from_dict(
        {
            "stages": [
                {"ref": "CreateInitialManifestAudioFolderStage", "params": {"data_dir": folder}},
                {
                    "ref": "ResampleAudioStage",
                    "params": {
                        "target_sample_rate": 16000,
                        "target_nchannels": 1,
                        "resampled_audio_dir": "/tmp/rs_convert",  # noqa: S108
                    },
                },
                {"ref": "ManifestWriterStage", "params": {"output_path": out}},
            ]
        }
    ).freeze()


def _quality_recipe(folder: str, out: str, *, mos: float = 3.0) -> Recipe:
    return Recipe.from_dict(
        {
            "stages": [
                {"ref": "CreateInitialManifestAudioFolderStage", "params": {"data_dir": folder}},
                {"ref": "VADSegmentationStage", "params": {"min_duration_sec": 2.0, "max_duration_sec": 30.0}},
                {"ref": "UTMOSFilterStage", "params": {"mos_threshold": mos, "action": "filter"}},
                {
                    "ref": "ResampleAudioStage",
                    "params": {
                        "target_sample_rate": 16000,
                        "target_nchannels": 1,
                        "resampled_audio_dir": "/tmp/rs_quality",  # noqa: S108
                    },
                },
                {"ref": "ManifestWriterStage", "params": {"output_path": out}},
            ]
        }
    ).freeze()


def _long_recipe(folder: str, out: str, *, mos: float = 3.0) -> Recipe:
    """A realistic six-stage pipeline whose names alone nearly fill the summary budget."""
    return Recipe.from_dict(
        {
            "stages": [
                {
                    "ref": "CreateInitialManifestAudioFolderStage",
                    "params": {"data_dir": folder, "max_samples": -1},
                },
                {
                    "ref": "VADSegmentationStage",
                    "params": {"min_duration_sec": 2.0, "max_duration_sec": 30.0, "threshold": 0.5},
                },
                {"ref": "UTMOSFilterStage", "params": {"mos_threshold": mos, "action": "filter"}},
                {
                    "ref": "ResampleAudioStage",
                    "params": {
                        "target_sample_rate": 16000,
                        "target_nchannels": 1,
                        "resampled_audio_dir": "/tmp/rs_long",  # noqa: S108
                    },
                },
                {
                    "ref": "TimestampMapperStage",
                    "params": {
                        "passthrough_keys": [
                            "audio_filepath",
                            "resampled_audio_filepath",
                            "utmos_mos",
                            "segment_num",
                            "start_ms",
                            "end_ms",
                            "segment_clip_id",
                        ]
                    },
                },
                {"ref": "ManifestWriterStage", "params": {"output_path": out}},
            ]
        }
    ).freeze()


def _save_run(  # noqa: PLR0913 - a run record simply has this many fields
    folder: str,
    recipe: Recipe,
    *,
    prior_profile: Any,  # noqa: ANN401
    goal: dict[str, Any] | None,
    created_at: str,
    pipeline_summary: str | None = None,
    input_count: int | None = None,
) -> str:
    dataset_key = prior_profile.dataset_key()
    steps = [p.step_key for p in artifacts.plan_steps(recipe, dataset_key)]
    artifacts.save_coverage(steps[-1], dict(prior_profile.inventory))
    run_id = run_store.new_run_id(recipe.config_hash)
    summary = pipeline_summary if pipeline_summary is not None else reuse.summarize_pipeline(recipe)
    out = next(
        (str(s.params.get("output_path")) for s in recipe.stages if s.ref == "ManifestWriterStage"),
        "",
    )
    run_store.save(
        RunRecord(
            run_id=run_id,
            recipe=recipe.to_dict(),
            config_hash=recipe.config_hash,
            semantic_hash=recipe.semantic_hash,
            goal=goal or {},
            pipeline_summary=summary,
            data_source=folder,
            dataset_key=dataset_key,
            status="completed",
            input_count=len(prior_profile.inventory) if input_count is None else input_count,
            accepted=len(prior_profile.inventory),
            output_paths=[out] if out else [],
            steps=steps,
            created_at=created_at,
            elapsed_sec=12.0,
        )
    )
    return run_id


class TestPipelineSummaryIsDurable:
    def test_summarize_pipeline_names_stages_and_key_params(self) -> None:
        recipe = _quality_recipe("/data", "/out.jsonl", mos=3.0)
        summary = reuse.summarize_pipeline(recipe)

        assert "UTMOSFilterStage" in summary
        assert "mos_threshold=3.0" in summary or "mos_threshold=3" in summary
        assert "VADSegmentationStage" in summary
        assert " -> " in summary
        assert "data_dir=" not in summary

    def test_a_long_pipeline_keeps_every_stage_and_every_behavioural_param(self) -> None:
        """Nothing is clipped: a summary that drops a threshold makes two runs look alike."""
        summary = reuse.summarize_pipeline(_long_recipe("/data", "/out.jsonl", mos=3.0))

        for ref in ("VADSegmentationStage", "UTMOSFilterStage", "ResampleAudioStage", "ManifestWriterStage"):
            assert ref in summary
        assert "mos_threshold=3.0" in summary
        assert "target_sample_rate=16000" in summary
        assert "threshold=0.5" in summary
        assert "…" not in summary
        # Locations and caps that cap nothing say nothing about what the run did.
        assert "/data" not in summary
        assert "max_samples" not in summary

    def test_successful_record_run_persists_the_summary(self, store: Path) -> None:
        folder = _folder(store, ["a.wav"])
        recipe = _convert_recipe(folder, str(store / "out.jsonl"))
        profile = _profile(folder)

        class _Report:
            accepted = 1
            input_count = 1
            output_paths = [str(store / "out.jsonl")]  # noqa: RUF012
            per_stage_metrics: dict = {}  # noqa: RUF012

        run_id = verbs._record_run(
            recipe,
            run_id=run_store.new_run_id(recipe.config_hash),
            data=folder,
            data_fp=None,
            dataset_key=profile.dataset_key(),
            fingerprint_tier="stat",
            report=_Report(),
            failed=False,
            goal={"task": "convert folder to 16 kHz mono"},
            elapsed=1.0,
        )
        record = run_store.load(run_id)

        assert record is not None
        assert record.pipeline_summary
        assert "ResampleAudioStage" in record.pipeline_summary
        assert "16000" in record.pipeline_summary or "16kHz" in record.pipeline_summary
        assert "data_dir=" not in record.pipeline_summary

    def test_failed_run_does_not_store_a_summary(self, store: Path) -> None:
        folder = _folder(store, ["a.wav"])
        recipe = _convert_recipe(folder, str(store / "out.jsonl"))
        profile = _profile(folder)

        class _Report:
            accepted = 0
            input_count = 1
            output_paths: list = []  # noqa: RUF012
            per_stage_metrics: dict = {}  # noqa: RUF012

        run_id = verbs._record_run(
            recipe,
            run_id=run_store.new_run_id(recipe.config_hash),
            data=folder,
            data_fp=None,
            dataset_key=profile.dataset_key(),
            fingerprint_tier="stat",
            report=_Report(),
            failed=True,
            goal={"task": "convert"},
        )
        record = run_store.load(run_id)

        assert record is not None
        assert record.pipeline_summary == ""

    def test_old_records_without_a_summary_are_derived_at_read_time(self, store: Path) -> None:
        folder = _folder(store, ["a.wav"])
        recipe = _quality_recipe(folder, str(store / "q.jsonl"))
        run_id = _save_run(
            folder,
            recipe,
            prior_profile=_profile(folder),
            goal={"task": "quality_filter"},
            created_at="2026-08-17T10:00:00Z",
            pipeline_summary="",
        )

        overview = verbs.runs(run_id=run_id)["overview"]

        assert overview["pipeline_summary"]
        assert "UTMOSFilterStage" in overview["pipeline_summary"]
        assert overview["prompt"] == "quality_filter"


class TestCurrentPromptRanksPriorsByPromptAndSummary:
    def test_quality_request_prefers_quality_prior_over_convert_only(self, store: Path) -> None:
        folder = _folder(store, ["a.wav", "b.wav", "c.wav"])
        profile = _profile(folder)
        convert_id = _save_run(
            folder,
            _convert_recipe(folder, str(store / "convert.jsonl")),
            prior_profile=profile,
            goal={"task": "convert to 16 kHz mono"},
            created_at="2026-08-17T10:00:00Z",
        )
        quality_id = _save_run(
            folder,
            _quality_recipe(folder, str(store / "quality.jsonl"), mos=3.0),
            prior_profile=profile,
            goal={"task": "quality_filter", "note": "keep high-MOS speech"},
            created_at="2026-08-17T11:00:00Z",
        )
        (Path(folder) / "d.wav").write_bytes(b"RIFF" + b"\x09" * 4096)

        listing = verbs.runs(
            data=folder,
            goal="curate into 16 kHz mono clips with a quality filter",
        )

        assert listing["ranked_by"].startswith("current_prompt vs prior_prompt + pipeline_summary")
        assert listing["runs"][0]["run_id"] == quality_id
        assert listing["runs"][0]["match"]["score"] > listing["runs"][1]["match"]["score"]
        matched = listing["runs"][0]["match"]["matched"]
        assert "filter" in matched or "quality" in matched
        by_id = {c["run_id"]: c for c in listing["runs"]}
        assert convert_id in by_id
        assert "UTMOSFilterStage" in by_id[quality_id]["pipeline_summary"]
        assert "UTMOSFilterStage" not in by_id[convert_id]["pipeline_summary"]
        assert "host_directive" in listing

    def test_equal_matches_prefer_the_prior_that_covered_more_of_the_folder(self, store: Path) -> None:
        """Two priors that fit the request equally: adopt the one leaving the smaller delta."""
        folder = _folder(store, ["a.wav", "b.wav"])
        profile = _profile(folder)
        narrow_id = _save_run(
            folder,
            _quality_recipe(folder, str(store / "narrow.jsonl")),
            prior_profile=profile,
            goal={"task": "quality_filter"},
            created_at="2026-08-17T12:00:00Z",
            input_count=3,
        )
        broad_id = _save_run(
            folder,
            _quality_recipe(folder, str(store / "broad.jsonl")),
            prior_profile=profile,
            goal={"task": "quality_filter"},
            created_at="2026-08-17T11:00:00Z",
            input_count=7,
        )

        listing = verbs.runs(data=folder, goal="apply a quality filter")
        order = [c["run_id"] for c in listing["runs"]]
        scores = {c["run_id"]: c["match"]["score"] for c in listing["runs"]}

        assert scores[broad_id] == scores[narrow_id]
        # Ranked ahead despite being the older run -- coverage outranks recency, not score.
        assert order.index(broad_id) < order.index(narrow_id)

    def test_both_payloads_tell_the_host_to_retell_the_summary_rather_than_paste_it(self, store: Path) -> None:
        """The summary is uncapped for comparison; keeping it readable is the host's job."""
        folder = _folder(store, ["a.wav"])
        run_id = _save_run(
            folder,
            _long_recipe(folder, str(store / "long.jsonl")),
            prior_profile=_profile(folder),
            goal={"task": "quality_filter"},
            created_at="2026-08-17T12:00:00Z",
        )

        listing = verbs.runs(data=folder, goal="apply a quality filter")
        single = verbs.runs(run_id=run_id)

        assert "Never paste it verbatim" in listing["host_directive"]
        assert "Never paste it verbatim" in single["host_directive"]

    def test_a_prior_without_a_goal_scores_on_its_pipeline_not_on_placeholder_words(self) -> None:
        match = reuse.prompt_summary_match("run that on the new files", reuse._NO_OBJECTIVE, "")

        assert match["score"] == 0.0
        assert match["matched"] == []

    def test_empty_prior_goal_still_matches_via_pipeline_summary(self, store: Path) -> None:
        folder = _folder(store, ["a.wav"])
        quality_id = _save_run(
            folder,
            _quality_recipe(folder, str(store / "q.jsonl")),
            prior_profile=_profile(folder),
            goal={},
            created_at="2026-08-17T12:00:00Z",
        )

        listing = verbs.runs(data=folder, goal="apply a quality filter")
        card = next(c for c in listing["runs"] if c["run_id"] == quality_id)

        assert card["match"]["score"] > 0
        matched = card["match"]["matched"]
        assert "filter" in matched or "quality" in matched
