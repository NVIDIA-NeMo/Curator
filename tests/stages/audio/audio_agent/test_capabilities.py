# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Onboarding a new source and sink must not require agent-core name tables."""

from __future__ import annotations

import copy
import inspect
import json
import os
import shutil
import subprocess
import sys
import textwrap
import wave
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from nemo_curator.stages.audio._agent._agent_ready import AgentReady, Gates, IOSpec, StageContract
from nemo_curator.stages.audio.audio_agent import _capabilities, _verb_evidence, _verb_smoke_inputs, artifacts, delta
from nemo_curator.stages.audio.audio_agent.card_conformance import check_card
from nemo_curator.stages.audio.audio_agent.contracts import SmokeReport
from nemo_curator.stages.audio.audio_agent.input_identity import resolve_dataset_binding
from nemo_curator.stages.audio.audio_agent.recipe import Recipe, StageRef, build_stages
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask, EmptyTask


@pytest.fixture
def new_modules(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:  # noqa: C901
    from nemo_curator.stages import base
    from nemo_curator.stages.audio._agent import _catalog
    from nemo_curator.stages.audio.audio_agent import index

    _catalog.list_agent_ready_stages()
    monkeypatch.setattr(base, "_STAGE_REGISTRY", dict(base._STAGE_REGISTRY))

    @dataclass
    class FutureRecordsSource(AgentReady, ProcessingStage[EmptyTask, AudioTask]):
        catalog: str
        audio_filepath_key: str = "asset"
        selected_paths: list[str] | None = None
        name: str = "FutureRecordsSource"

        def describe(self) -> StageContract:
            return StageContract(
                writes=IOSpec(data_keys=[self.audio_filepath_key], produces=["disk"]),
                cardinality="1:N fan-out",
                gates=Gates(per_row_independent=True),
            )

        def inputs(self) -> tuple[list[str], list[str]]:
            return [], []

        def outputs(self) -> tuple[list[str], list[str]]:
            return ["data"], [self.audio_filepath_key]

        def process(self, task: EmptyTask) -> list[AudioTask]:
            with open(self.catalog, encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
            return [
                AudioTask(task_id=f"future-{i}", data=row)
                for i, row in enumerate(rows)
                if self.selected_paths is None or row[self.audio_filepath_key] in self.selected_paths
            ]

    @dataclass
    class FutureRecordsWriter(AgentReady, ProcessingStage[AudioTask, AudioTask]):
        destination: str
        review_destination: str = ""
        audio_filepath_key: str = "asset"
        weights_file: str = ""
        name: str = "FutureRecordsWriter"
        AGENT_ATOMIC_OUTPUT_PARAMS: ClassVar[tuple[str, ...]] = ("destination", "review_destination")

        def describe(self) -> StageContract:
            return StageContract(
                reads=IOSpec(data_keys=[self.audio_filepath_key]),
                gates=Gates(
                    writes_to_disk=True,
                    output_path_params=["destination", *(["review_destination"] if self.review_destination else [])],
                    requires_serializable_input=True,
                    per_row_independent=False,
                ),
            )

        def inputs(self) -> tuple[list[str], list[str]]:
            return ["data"], [self.audio_filepath_key]

        def outputs(self) -> tuple[list[str], list[str]]:
            return ["data"], [self.audio_filepath_key]

        def prepare_on_driver(self, *, checkpoint_path: str | None = None) -> None:
            for path in (self.destination, self.review_destination):
                if path:
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                    Path(path).write_text("", encoding="utf-8")

        def process(self, task: AudioTask) -> AudioTask:
            with open(self.destination, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(task.data) + "\n")
            if self.review_destination:
                with open(self.review_destination, "a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"review": "pending"}) + "\n")
            return task

    knowledge = index.KnowledgeIndex()
    common = {"verified": {"params": "mechanical", "io": "mechanical"}, "provenance": {"card_version": 1}}
    source_card = {
        **common,
        "stage_id": "FutureRecordsSource",
        "category": "ingest",
        "summary": "Read local JSONL records.",
        "tags": ["fanout"],
        "io": {
            "source": {
                "kind": "manifest",
                "path_param": "catalog",
                "audio_filepath_key_param": "audio_filepath_key",
                "include_files_param": "selected_paths",
            }
        },
    }
    writer_card = {
        **common,
        "stage_id": "FutureRecordsWriter",
        "category": "export",
        "summary": "Write one JSONL row per task.",
        "tags": ["writes_disk", "sink"],
        "io": {
            "outputs": {
                "destination": {
                    "kind": "manifest",
                    "primary": True,
                    "relocatable": True,
                    "per_item": True,
                },
                "review_destination": {"kind": "manifest", "per_item": True},
            },
            "dependency_params": ["weights_file"],
        },
    }
    knowledge._cards_by_stage.update(FutureRecordsSource=source_card, FutureRecordsWriter=writer_card)
    monkeypatch.setattr(index, "get_index", lambda: knowledge)
    source = tmp_path / "records.jsonl"
    rows = [{"asset": str(tmp_path / f"{i}.wav")} for i in range(3)]
    for row in rows:
        with wave.open(row["asset"], "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(b"\x00\x00" * 320)
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    recipe = Recipe(
        stages=[
            StageRef("FutureRecordsSource", {"catalog": str(source)}),
            StageRef("FutureRecordsWriter", {"destination": str(tmp_path / "curated.jsonl")}),
        ]
    ).freeze()
    return SimpleNamespace(
        source=FutureRecordsSource,
        writer=FutureRecordsWriter,
        recipe=recipe,
        rows=rows,
        source_card=source_card,
        writer_card=writer_card,
        knowledge=knowledge,
    )


@pytest.mark.skipif(
    not (os.environ.get("RAY_ADDRESS") or os.environ.get("CURATOR_TEST_RAY_ADDRESS")),
    reason="requires an explicitly owned Ray cluster",
)
@pytest.mark.parametrize("with_review", [False, True])
def test_new_modules_smoke_run_and_reuse_through_public_verbs(
    new_modules: SimpleNamespace,
    tmp_path: Path,
    with_review: bool,
) -> None:
    import yaml

    import nemo_curator

    # A real package location is required by implementation identity. Do not
    # weaken that boundary to certify classes living only inside a test fixture.
    package = tmp_path / "package"
    source_root = Path(nemo_curator.__file__).parent
    shadow = package / "nemo_curator"
    shutil.copytree(source_root, shadow, ignore=shutil.ignore_patterns("__pycache__"))
    module = shadow / "stages/audio/io/future_fixture.py"
    header = """from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar
from nemo_curator.stages.audio._agent._agent_ready import AgentReady, Gates, IOSpec, StageContract
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask, EmptyTask
"""
    module.write_text(
        header
        + "\n".join(textwrap.dedent(inspect.getsource(stage)) for stage in (new_modules.source, new_modules.writer)),
        encoding="utf-8",
    )
    for card in (new_modules.source_card, new_modules.writer_card):
        path = shadow / "stages/audio/audio_agent/knowledge/cards" / f"{card['stage_id']}.yaml"
        path.write_text(yaml.safe_dump(card), encoding="utf-8")
    recipe = new_modules.recipe
    if with_review:
        recipe.stages[-1].params["review_destination"] = str(tmp_path / "review.jsonl")
    recipe.acceptance_criteria = [
        {"id": "rows", "type": "yield", "check": {"op": ">=", "value": 1}},
        {"id": "paths", "type": "output_completeness", "check": {"field": "asset"}},
    ]
    recipe.freeze()
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe.to_dict()), encoding="utf-8")
    program = """
import json, sys
from pathlib import Path
import nemo_curator
from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.stages.audio.audio_agent import verbs, code_identity
from nemo_curator.stages.audio.audio_agent.recipe import Recipe
assert Path(nemo_curator.__file__).is_relative_to(sys.argv[2])
recipe = Recipe.from_dict(json.loads(Path(sys.argv[1]).read_text())).freeze()
for stage in recipe.stages:
    assert not code_identity.is_fallback(code_identity.impl_version(stage.ref))
assert verbs.validate(recipe)['ok']
final = Path(recipe.stages[-1].params['destination'])
review_path = recipe.stages[-1].params.get('review_destination')
executor = RayActorPoolExecutor(config={'runtime_env': {'env_vars': {'PYTHONPATH': sys.argv[2]}}}, show_progress=False)
assert verbs.run(recipe, executor=executor)['status'] == 'refused'
assert not final.exists()
smoke = verbs.smoke(recipe, sample=1, executor=executor)
assert smoke['status'] == 'completed', smoke
assert smoke['goals_met'] is True, smoke
assert not final.exists()
if review_path:
    assert not Path(review_path).exists()
assert verbs.run(recipe, confirm='wrong-hash', executor=executor)['status'] == 'refused'
done = verbs.run(recipe, confirm=recipe.config_hash, smoke_token=smoke['smoke_token'], executor=executor)
assert done['status'] == 'completed', done
assert done['acceptance']['overall'] == 'met', done
assert done['report']['output_rows_written'] == 3
assert done['report']['cardinality_proven'] is True
assert len(final.read_text().splitlines()) == 3
assert str(final) in done['report']['output_paths']
if review_path:
    review = Path(review_path)
    assert len(review.read_text().splitlines()) == 3
    assert str(review) in done['report']['output_paths']
    assert 'asset' not in json.loads(review.read_text().splitlines()[0])
before = final.read_bytes(), final.stat().st_mtime_ns
reused = verbs.reuse_scan(recipe)
assert reused['decision'] == 'already_done', reused
assert (final.read_bytes(), final.stat().st_mtime_ns) == before
if review_path:
    saved = review.read_bytes()
    review.unlink()
    try:
        assert verbs.reuse_scan(recipe)['decision'] != 'already_done'
    finally:
        review.write_bytes(saved)
    assert verbs.reuse_scan(recipe)['decision'] == 'already_done'
Path(sys.argv[3]).write_text(json.dumps({'smoke': smoke, 'run': done, 'reuse': reused}))
"""
    environment = {
        **os.environ,
        "PYTHONPATH": str(package),
        "RAY_ADDRESS": os.environ.get("CURATOR_TEST_RAY_ADDRESS") or os.environ["RAY_ADDRESS"],
        "AUDIO_AGENT_RUNS_DIR": str(tmp_path / "state"),
        "AUDIO_AGENT_REQUIRE_SMOKE": "1",
    }
    completed = subprocess.run(  # noqa: S603 - fixed interpreter and test-owned package/recipe paths
        [sys.executable, "-c", program, str(recipe_path), str(package), str(tmp_path / "result.json")],
        cwd=package,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, (completed.stdout + completed.stderr)[-12000:]


def test_new_modules_are_discoverable_carded_and_validatable(new_modules: SimpleNamespace) -> None:
    from nemo_curator.stages.audio import agent
    from nemo_curator.stages.audio.audio_agent import verbs

    names = agent.list_agent_ready_stages()
    assert new_modules.source.__name__ in names
    assert new_modules.writer.__name__ in names
    assert check_card(new_modules.source.__name__, new_modules.source_card) == []
    assert check_card(new_modules.writer.__name__, new_modules.writer_card) == []
    result = verbs.validate(new_modules.recipe.to_dict())
    assert result["ok"], result


def test_new_manifest_source_binding_and_smoke_match_actual_selection(new_modules: SimpleNamespace) -> None:
    recipe = new_modules.recipe
    recipe.stages[0].params["selected_paths"] = [new_modules.rows[1]["asset"], new_modules.rows[2]["asset"]]
    original = copy.deepcopy(recipe.to_dict())
    binding = resolve_dataset_binding(recipe, recipe.stages[0].params["catalog"])
    assert binding.status == "resolved"
    assert binding.profile_kwargs == {
        "audio_filepath_key": "asset",
        "include_files": recipe.stages[0].params["selected_paths"],
    }
    bounded = _verb_smoke_inputs._bound_recipe(recipe, 1, SmokeReport(), binding)
    try:
        assert not bounded.error
        stages, errors = build_stages(bounded.recipe)
        assert not errors
        assert [task.data for task in stages[0].process(EmptyTask())] == [new_modules.rows[1]]
        assert bounded.input_count == 1
        assert recipe.to_dict() == original
    finally:
        for path in bounded.tmp_paths:
            Path(path).unlink(missing_ok=True)
    assert resolve_dataset_binding(recipe, "/wrong-input").status == "mismatch"


def test_new_output_is_sandboxed_reported_and_usable_for_evidence(new_modules: SimpleNamespace) -> None:
    recipe = new_modules.recipe
    final = recipe.stages[-1].params["destination"]
    bound = _verb_smoke_inputs._isolate_smoke_outputs(
        _verb_smoke_inputs._SmokeBound(copy.deepcopy(recipe)),
        SmokeReport(),
    )
    try:
        assert not bound.error
        sandbox = bound.recipe.stages[-1].params["destination"]
        assert Path(sandbox).is_relative_to(bound.output_root)
        assert sandbox != final
        stages, errors = build_stages(bound.recipe)
        assert not errors
        assert _verb_smoke_inputs._smoke_write_issues(stages, bound.output_root) == []
        stages[1].prepare_on_driver()
        for task in stages[0].process(EmptyTask()):
            stages[1].process(task)
        assert _verb_evidence._recipe_outputs(bound.recipe, None) == [sandbox]
        assert _verb_evidence._terminal_evidence_outputs(bound.recipe, []) == ([sandbox], True)
        assert _verb_evidence._count_output_rows(sandbox) == 3
        assert not Path(final).exists()
    finally:
        _verb_smoke_inputs._cleanup(list(bound.tmp_paths))


def test_new_output_relocation_and_dependency_identity(new_modules: SimpleNamespace, tmp_path: Path) -> None:
    original = new_modules.recipe
    moved = copy.deepcopy(original)
    moved.stages[-1].params["destination"] = str(tmp_path / "moved.jsonl")
    assert original.compute_hash() != moved.compute_hash()
    assert original.stages[-1].semantic_dict() == moved.stages[-1].semantic_dict()
    assert artifacts.output_uri(original.stages[-1]) == (original.stages[-1].params["destination"], "manifest")
    assert artifacts.step_keys(original, "dataset") == artifacts.step_keys(moved, "dataset")
    weights = tmp_path / "weights"
    weights.write_bytes(b"version1")
    original.stages[-1].params["weights_file"] = str(weights)
    before = artifacts.step_keys(original, "dataset")
    weights.write_bytes(b"version2")
    assert artifacts.step_keys(original, "dataset") != before


def test_unreadable_declared_dependencies_are_not_reusable(new_modules: SimpleNamespace, tmp_path: Path) -> None:
    stage = new_modules.recipe.stages[-1]
    stage.params["weights_file"] = str(tmp_path / "missing.bin")
    assert artifacts.step_keys(new_modules.recipe, "dataset") != artifacts.step_keys(new_modules.recipe, "dataset")
    stage.params["weights_file"] = {"unsupported": "bundle"}
    assert artifacts.step_keys(new_modules.recipe, "dataset") != artifacts.step_keys(new_modules.recipe, "dataset")


def test_nonrelocatable_outputs_remain_generated_dependencies(new_modules: SimpleNamespace) -> None:
    stage = new_modules.recipe.stages[-1]
    new_modules.writer_card["io"]["outputs"]["destination"]["relocatable"] = False
    assert _capabilities.generated_asset_paths(stage.ref, stage.params) == [stage.params["destination"]]
    before = artifacts.step_keys(new_modules.recipe, "dataset")
    new_modules.writer_card["io"]["outputs"]["destination"]["per_item"] = False
    assert artifacts.step_keys(new_modules.recipe, "dataset") != before


def test_relocatable_is_not_optional(new_modules: SimpleNamespace) -> None:
    stage = new_modules.recipe.stages[-1]
    assert _capabilities.generated_asset_paths(stage.ref, stage.params) == [stage.params["destination"]]


@pytest.mark.parametrize(
    "planning",
    [
        {"unknown": {}},
        {"restores_continuity": True},
        {"restores_continuity": {"when": {"imaginary_param": True}}},
        {"restores_continuity": {"when": {"destination": ["expression"]}}},
        {"sample_rate_verification": {"param": "imaginary_param"}},
    ],
)
def test_planning_hints_reject_invented_parameters_or_expressions(
    new_modules: SimpleNamespace, planning: dict
) -> None:
    card = copy.deepcopy(new_modules.writer_card)
    card["planning"] = planning
    card["verified"]["planning"] = "mechanical"
    assert check_card("FutureRecordsWriter", card)


def test_planning_hint_does_not_change_step_keys(new_modules: SimpleNamespace) -> None:
    before = artifacts.step_keys(new_modules.recipe, "dataset")
    card = new_modules.writer_card
    card["planning"] = {"restores_continuity": {}}
    card["verified"]["planning"] = "mechanical"
    assert check_card("FutureRecordsWriter", card) == []
    assert artifacts.step_keys(new_modules.recipe, "dataset") == before


def test_explicit_asset_identity_overrides_legacy_output_parameter_name(
    new_modules: SimpleNamespace,
    tmp_path: Path,
) -> None:
    card = new_modules.knowledge.card("ManifestWriterStage")
    card["io"] = {"outputs": {"output_path": {"kind": "manifest", "relocatable": False}}}
    card["verified"]["io"] = "mechanical"
    stage = StageRef("ManifestWriterStage", {"output_path": str(tmp_path / "embedded-path.jsonl")})
    assert "output_path" in stage.semantic_params()
    assert _capabilities.generated_asset_paths(stage.ref, stage.params) == [stage.params["output_path"]]


def test_undeclared_file_selector_is_not_guessed(new_modules: SimpleNamespace) -> None:
    del new_modules.source_card["io"]["source"]["include_files_param"]
    assert not delta._accepts_include(new_modules.source.__name__)


def test_new_source_delta_uses_declared_selection_parameter(new_modules: SimpleNamespace, tmp_path: Path) -> None:
    selected = (new_modules.rows[1]["asset"],)
    narrowed, _redirect, error = delta.prefix_recipe(
        new_modules.recipe,
        prefix=1,
        files=selected,
        sandbox=str(tmp_path / "delta"),
        sinks_=[],
        inventory_key="asset",
    )
    assert not error
    assert narrowed.stages[0].params["selected_paths"] == list(selected)
    assert "include_files" not in narrowed.stages[0].params
    stages, errors = build_stages(narrowed)
    assert not errors
    assert [task.data for task in stages[0].process(EmptyTask())] == [new_modules.rows[1]]


@pytest.mark.parametrize(
    "mutation",
    [
        {"source": {"kind": "remote_magic", "path_param": "catalog"}},
        {"source": {"kind": [], "path_param": "catalog"}},
        {"source": {"kind": "manifest", "path_param": "typo"}},
        {"source": {"kind": "audio_folder", "path_param": "catalog"}},
        {"unknown": {}},
        {"dependency_params": ["missing"]},
        {"outputs": {"catalog": {"kind": []}}},
    ],
)
def test_invalid_source_declarations_fail_before_binding(
    new_modules: SimpleNamespace, mutation: dict[str, Any]
) -> None:
    new_modules.source_card["io"] = mutation
    assert check_card(new_modules.source.__name__, new_modules.source_card)
    assert resolve_dataset_binding(new_modules.recipe).status == "unsupported"


def test_output_declarations_cannot_invent_writes(new_modules: SimpleNamespace) -> None:
    new_modules.source_card["io"]["outputs"] = {"catalog": {"kind": "manifest", "primary": True}}
    assert _capabilities.output_paths(new_modules.source.__name__, {"catalog": "input.jsonl"}) == {}
    assert artifacts.output_uri(new_modules.recipe.stages[0]) == ("", "unknown")


def test_missing_card_does_not_enable_a_new_source(new_modules: SimpleNamespace) -> None:
    del new_modules.knowledge._cards_by_stage[new_modules.source.__name__]
    assert resolve_dataset_binding(new_modules.recipe).status == "unsupported"


def test_renamed_paths_still_obey_workspace_lock(
    new_modules: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from nemo_curator.stages.audio.audio_agent import _safety, verbs

    monkeypatch.setenv("AUDIO_AGENT_WORKSPACE", str(tmp_path))
    outside = str(tmp_path.parent / "outside.jsonl")
    new_modules.recipe.stages[-1].params["destination"] = outside
    assert _safety.path_violations(_safety.recipe_path_params(new_modules.recipe))
    refused = verbs.run(new_modules.recipe.to_dict(), confirm=True)
    assert refused["status"] == "refused"
    assert not Path(outside).exists()


def test_continuation_rebinds_custom_source_assertions(new_modules: SimpleNamespace) -> None:
    from nemo_curator.stages.audio.audio_agent.continuation import materialize

    recipe = copy.deepcopy(new_modules.recipe)
    recipe.inputs["catalog"] = recipe.stages[0].params["catalog"]
    recipe.stages.append(
        StageRef("PreserveByValueStage", {"input_value_key": "asset", "operator": "ne", "target_value": ""})
    )
    continued, error = materialize(recipe, uri="/new/artifact.jsonl", kind="manifest", prefix=2)
    assert not error
    assert "catalog" not in continued.inputs
    assert continued.inputs["manifest_path"] == "/new/artifact.jsonl"
    assert delta._uri_param(new_modules.recipe, 1) == "destination"


@pytest.mark.parametrize("succeed", [False, True])
def test_new_writer_uses_existing_atomic_publication(new_modules: SimpleNamespace, succeed: bool) -> None:
    from nemo_curator.stages.audio.audio_agent import _verb_outputs

    stages, errors = build_stages(new_modules.recipe)
    assert not errors
    final = Path(stages[-1].destination)
    final.write_text('{"old":true}\n', encoding="utf-8")
    staged = _verb_outputs._prepare_staged_file_outputs(stages)
    try:
        stages[-1].prepare_on_driver()
        for task in stages[0].process(EmptyTask()):
            stages[-1].process(task)
        assert final.read_text(encoding="utf-8") == '{"old":true}\n'
        if succeed:
            _verb_outputs._promote_staged_file_outputs(staged)
            assert _verb_evidence._count_output_rows(str(final)) == 3
    finally:
        _verb_outputs._restore_staged_file_outputs(staged)
        _verb_outputs._discard_staged_file_outputs(staged)
    if not succeed:
        assert final.read_text(encoding="utf-8") == '{"old":true}\n'


def test_folder_source_uses_declared_names_for_inventory_and_bounded_smoke(
    new_modules: SimpleNamespace,
    tmp_path: Path,
) -> None:
    from nemo_curator.stages.audio.audio_agent import _verb_inputs

    @dataclass
    class FutureFolderSource(AgentReady, ProcessingStage[EmptyTask, AudioTask]):
        root: str
        count: int = -1
        descend: bool = True
        extensions: list[str] = field(default_factory=lambda: [".wav"])
        selected: list[str] | None = None
        audio_filepath_key: str = "audio_filepath"
        name: str = "FutureFolderSource"

        def describe(self) -> StageContract:
            return StageContract(
                writes=IOSpec(data_keys=[self.audio_filepath_key], produces=["disk"]),
                cardinality="1:N fan-out",
                gates=Gates(per_row_independent=self.count < 0),
            )

        def inputs(self) -> tuple[list[str], list[str]]:
            return [], []

        def outputs(self) -> tuple[list[str], list[str]]:
            return ["data"], [self.audio_filepath_key]

        def process(self, task: EmptyTask) -> list[AudioTask]:
            listing = Path(self.root).rglob("*") if self.descend else Path(self.root).glob("*")
            paths = sorted(str(path) for path in listing if path.is_file() and path.suffix in self.extensions)
            if self.selected is not None:
                paths = [path for path in paths if path in self.selected]
            if self.count >= 0:
                paths = paths[: self.count]
            return [
                AudioTask(task_id=f"folder-{i}", data={self.audio_filepath_key: path}) for i, path in enumerate(paths)
            ]

    root = tmp_path / "recordings"
    root.mkdir()
    (root / "sub").mkdir()
    for name in ("a.wav", "sub/b.wav", "not-audio.txt"):
        (root / name).write_bytes(b"identity fixture; deliberately not decoded")
    card = copy.deepcopy(new_modules.source_card)
    card["stage_id"] = "FutureFolderSource"
    card["io"]["source"] = {
        "kind": "audio_folder",
        "path_param": "root",
        "limit_param": "count",
        "recursive_param": "descend",
        "extensions_param": "extensions",
        "include_files_param": "selected",
    }
    new_modules.knowledge._cards_by_stage["FutureFolderSource"] = card
    recipe = Recipe(stages=[StageRef("FutureFolderSource", {"root": str(root)})])
    assert check_card("FutureFolderSource", card) == []
    binding = resolve_dataset_binding(recipe)
    profile = _verb_inputs._profile_binding(binding)
    assert profile.num_files == 2
    assert set(profile.inventory) == {"a.wav", "sub/b.wav"}
    bound = _verb_smoke_inputs._bound_recipe(recipe, 1, SmokeReport(), binding)
    assert not bound.error
    stages, errors = build_stages(bound.recipe)
    assert not errors
    assert len(stages[0].process(EmptyTask())) == 1
    assert "count" not in recipe.stages[0].params
    assert delta.region(recipe, upto=1) == (1, "")
    bounded_recipe = copy.deepcopy(recipe)
    bounded_recipe.stages[0].params["count"] = 1
    depth, reason = delta.region(bounded_recipe, upto=1)
    assert depth == 0
    assert "bounded" in reason
    selected = str(root / "sub/b.wav")
    recipe.stages[0].params["selected"] = [selected]
    binding = resolve_dataset_binding(recipe)
    profile = _verb_inputs._profile_binding(binding)
    assert profile.num_files == 1
    before = profile.dataset_key()
    (root / "a.wav").write_bytes(b"unselected change")
    assert _verb_inputs._profile_binding(binding).dataset_key() == before
    (root / "sub/b.wav").write_bytes(b"selected change")
    assert _verb_inputs._profile_binding(binding).dataset_key() != before
