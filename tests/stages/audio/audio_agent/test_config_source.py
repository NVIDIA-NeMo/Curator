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

"""Generic generated-source protocol, independent of any simulation toolkit."""

from pathlib import Path

import pytest

from nemo_curator.stages.audio.audio_agent import _capabilities
from nemo_curator.stages.audio.audio_agent._verb_smoke_inputs import _bound_recipe
from nemo_curator.stages.audio.audio_agent.contracts import SmokeReport
from nemo_curator.stages.audio.audio_agent.input_identity import resolve_dataset_binding
from nemo_curator.stages.audio.audio_agent.recipe import Recipe, StageRef


@pytest.fixture
def source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    config = tmp_path / "generator.cfg"
    config.write_text("records=99")
    spec = {"kind": "config_file", "path_param": "configuration", "limit_param": "row_cap"}
    monkeypatch.setattr(_capabilities, "io_for", lambda _ref: {"source": spec})
    monkeypatch.setattr(_capabilities, "configured_values", lambda _ref, params: {"row_cap": -1, **params})
    return Recipe(stages=[StageRef(ref="FutureGenerator", params={"configuration": str(config)})])


def test_config_identity_is_generated_and_not_profiled(source: Recipe, tmp_path: Path):
    binding = resolve_dataset_binding(source)
    assert binding.status == "resolved"
    assert binding.generated
    assert binding.profile_source is None
    assert resolve_dataset_binding(source, str(tmp_path / "other.cfg")).status == "mismatch"
    assert not binding.selected_manifest_files


@pytest.mark.parametrize("value", [-1, 1, 9])
def test_smoke_caps_output_without_rewriting_configuration(source: Recipe, value: object):
    source.stages[0].params["row_cap"] = value
    bound = _bound_recipe(source, 2, SmokeReport(), resolve_dataset_binding(source))
    assert bound.recipe.stages[0].params["row_cap"] == (1 if value == 1 else 2)
    assert source.stages[0].params["row_cap"] == value


@pytest.mark.parametrize("value", [0, -2, True, None, "9"])
def test_invalid_cap_fails_closed(source: Recipe, value: object):
    source.stages[0].params["row_cap"] = value
    bound = _bound_recipe(source, 2, SmokeReport(), resolve_dataset_binding(source))
    assert bound.recipe is None


def test_config_protocol_rejects_directories_and_selector_lists(source: Recipe, tmp_path: Path):
    for value in (str(tmp_path), [source.stages[0].params["configuration"]]):
        source.stages[0].params["configuration"] = value
        assert resolve_dataset_binding(source).status != "resolved"


def test_config_card_requires_real_bound_parameter():
    card = {"io": {"source": {"kind": "config_file", "path_param": "configuration"}}, "verified": {"io": "mechanical"}}
    assert _capabilities.io_violations("Generator", card, {"configuration"})
    card["io"]["source"]["limit_param"] = "row_cap"
    assert not _capabilities.io_violations("Generator", card, {"configuration", "row_cap"})


@pytest.fixture
def generator(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Recipe:
    from dataclasses import dataclass

    from nemo_curator.stages import base
    from nemo_curator.stages.audio._agent import _catalog
    from nemo_curator.stages.audio._agent._agent_ready import AgentReady, Gates, IOSpec, StageContract
    from nemo_curator.stages.audio.audio_agent import index
    from nemo_curator.stages.base import ProcessingStage
    from nemo_curator.tasks import AudioTask, EmptyTask

    _catalog.list_agent_ready_stages()
    monkeypatch.setattr(base, "_STAGE_REGISTRY", dict(base._STAGE_REGISTRY))

    @dataclass
    class FutureConfigGenerator(AgentReady, ProcessingStage[EmptyTask, AudioTask]):
        configuration: str
        output_dir: str
        row_cap: int = -1
        safe: bool = True
        name: str = "future_config_generator"

        def inputs(self) -> tuple[list[str], list[str]]:
            return [], []

        def outputs(self) -> tuple[list[str], list[str]]:
            return ["data"], ["audio_filepath"]

        def describe(self) -> StageContract:
            return StageContract(
                writes=IOSpec(data_keys=["audio_filepath"]),
                cardinality="1:N fan-out",
                wrappable=self.safe,
                gates=Gates(writes_to_disk=True, output_path_params=["output_dir"], per_row_independent=False),
            )

        def process(self, task: EmptyTask) -> list[AudioTask]:
            count = int(Path(self.configuration).read_text())
            if self.row_cap >= 0:
                count = min(count, self.row_cap)
            output = Path(self.output_dir)
            output.mkdir(parents=True)
            rows = []
            for i in range(count):
                audio = output / f"{i}.wav"
                audio.touch()
                rows.append(AudioTask(data={"audio_filepath": str(audio)}))
            return rows

    knowledge = index.KnowledgeIndex()
    knowledge._cards_by_stage["FutureConfigGenerator"] = {
        "stage_id": "FutureConfigGenerator",
        "category": "ingest",
        "summary": "Generate fixture rows from one config",
        "tags": ["writes_disk", "fanout"],
        "verified": {"params": "mechanical", "io": "mechanical"},
        "provenance": {"card_version": 1},
        "io": {"source": {"kind": "config_file", "path_param": "configuration", "limit_param": "row_cap"}},
    }
    monkeypatch.setattr(index, "get_index", lambda: knowledge)
    config = tmp_path / "generator.cfg"
    config.write_text("1")
    return Recipe(
        stages=[
            StageRef(
                ref="FutureConfigGenerator",
                params={"configuration": str(config), "output_dir": str(tmp_path / "production")},
            )
        ]
    )


@pytest.mark.parametrize(("rows", "cap"), [(8, 1), (1, 5), (0, 5)])
def test_public_generated_smoke_does_not_treat_cap_as_input_count(generator: Recipe, rows: int, cap: int) -> None:
    from nemo_curator.backends.base import BaseExecutor, BaseStageAdapter
    from nemo_curator.stages.audio.audio_agent import smoke, verify
    from nemo_curator.stages.base import ProcessingStage
    from nemo_curator.tasks import EmptyTask, Task

    class LocalExecutor(BaseExecutor):
        def execute(self, stages: list[ProcessingStage], initial_tasks: list[Task] | None = None) -> list[Task]:
            tasks = initial_tasks or [EmptyTask()]
            for stage in stages:
                tasks = BaseStageAdapter(stage).process_batch(tasks)
            return tasks

    params = generator.stages[0].params
    config = Path(params["configuration"])
    config.write_text(str(rows))
    params["row_cap"] = cap
    result = smoke(generator, sample=10, executor=LocalExecutor())
    assert result["status"] == "completed", result
    assert result["retained"] == min(rows, cap), result
    assert result["input_count"] is None, result
    assert result["rejected"] is None, result
    assert any("unmeasured" in note for note in result["notes"])
    assert not Path(params["output_dir"]).exists()
    assert config.read_text() == str(rows)
    criterion = {"id": "yield", "type": "yield", "kind": "relative", "check": {"op": ">=", "value": 50}}
    evidence = {"retained": result["retained"], "input_count": result.get("input_count")}
    verification = verify([criterion], evidence=evidence)
    assert verification["overall"] == "not_met"
    assert verification["criteria"][0]["status"] == "unverifiable"


def test_public_smoke_refuses_unwrappable_source_before_execution(generator: Recipe) -> None:
    from nemo_curator.stages.audio.audio_agent import smoke

    class MustNotExecute:
        def execute(self, *_args: object, **_kwargs: object) -> None:
            pytest.fail("Unwrappable source reached execution")

    generator.stages[0].params["safe"] = False
    result = smoke(generator, executor=MustNotExecute())
    assert result["status"] == "refused", result
    assert any("not wrappable" in issue for issue in result["issues"]), result
    assert not Path(generator.stages[0].params["output_dir"]).exists()
