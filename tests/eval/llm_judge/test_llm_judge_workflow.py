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

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from jinja2 import Environment, StrictUndefined

from eval.llm_judge import llm_judge_workflow as subject

EXAMPLE_DIR = Path(__file__).parents[3] / "tutorials" / "eval" / "llm_judge" / "cc_extract_example"


def _config_with_filters() -> tuple[dict[str, object], list[dict[str, object]]]:
    stages: list[dict[str, object]] = [
        {
            "name": "quality",
            "judges": [{"name": "quality_judge", "scores": [{"name": "quality"}]}],
            "filters": [{"judge": "quality_judge", "score": "quality", "operator": "gte", "value": 4}],
        },
        {
            "name": "safety",
            "judges": [{"name": "safety_judge", "scores": [{"name": "safe"}]}],
        },
    ]
    return ({"filters": [{"judge": "safety_judge", "score": "safe", "operator": "eq", "value": "yes"}]}, stages)


def test_load_yaml_rejects_a_non_mapping_document(tmp_path: Path) -> None:
    config_path = tmp_path / "judge.yaml"
    config_path.write_text("- not\n- a mapping\n", encoding="utf-8")

    with pytest.raises(TypeError, match="must contain a mapping"):
        subject._load_yaml(config_path)


@pytest.mark.parametrize(
    ("filename", "models", "stages", "judges"),
    [
        ("text_extraction_qwen_judge.yaml", 1, 2, 2),
        ("text_extraction_qwen_gemma_judges.yaml", 2, 4, 4),
    ],
)
def test_example_configs_and_templates_are_valid(filename: str, models: int, stages: int, judges: int) -> None:
    config_path = EXAMPLE_DIR / filename
    config = subject._load_yaml(config_path)
    configured_stages = config["execution"]["stages"]

    assert len(config["models"]) == models
    assert len(configured_stages) == stages
    assert sum(len(stage["judges"]) for stage in configured_stages) == judges
    subject._validate_filter_references(config, configured_stages)

    aliases = {model["alias"] for model in config["models"]}
    environment = Environment(undefined=StrictUndefined)  # noqa: S701
    for stage in configured_stages:
        for judge in stage["judges"]:
            assert judge["model_alias"] in aliases
            prompt = subject._read_template(judge["prompt_path"], config_path=config_path)
            assert environment.from_string(prompt).render(
                raw_text=None, justext_text="clean text", trafilatura_text=None
            )
            system_prompt = subject._read_template(judge["system_prompt_path"], config_path=config_path)
            assert environment.from_string(system_prompt).render(
                raw_text=None, justext_text="clean text", trafilatura_text=None
            )


def test_place_filters_after_their_producing_stage() -> None:
    config, stages = _config_with_filters()

    placed = subject._place_filters(config, stages)

    assert [[item["judge"] for item in filters] for filters in placed] == [
        ["quality_judge"],
        ["safety_judge"],
    ]


@pytest.mark.parametrize(
    ("filter_config", "message"),
    [
        ({"judge": "missing", "score": "quality"}, "unknown judge output column"),
        ({"judge": "quality_judge", "score": "missing"}, "unknown score"),
    ],
)
def test_filter_validation_rejects_unknown_references(filter_config: dict[str, object], message: str) -> None:
    _config, stages = _config_with_filters()
    with pytest.raises(ValueError, match=message):
        subject._validate_filter_references({"filters": [filter_config]}, stages)


def test_filter_validation_rejects_stage_local_filter_for_later_judge() -> None:
    config, stages = _config_with_filters()
    stages[0]["filters"] = [{"judge": "safety_judge", "score": "safe", "operator": "eq", "value": "yes"}]

    with pytest.raises(ValueError, match="produced by a later stage"):
        subject._validate_filter_references(config, stages)


@pytest.mark.parametrize(
    ("judge_result", "score_name", "operator", "expected", "keep"),
    [
        ({"quality": {"score": 4}}, "quality", "eq", 4, True),
        ({"quality": {"score": 4}}, "quality", "ne", 4, False),
        ({"quality": {"score": 4}}, "quality", "gt", 3, True),
        ({"quality": {"score": 4}}, "quality", "gte", 4, True),
        ({"quality": {"score": 4}}, "quality", "lt", 5, True),
        ({"quality": {"score": 4}}, "quality", "lte", 4, True),
        ({"quality": {"score": "good"}}, "quality", "in", ["good", "bad"], True),
        ({"quality": {"score": "good"}}, "quality", "not_in", ["bad"], True),
        ({}, "quality", "eq", 4, False),
        ({"quality": {"score": "four"}}, "quality", "gt", 3, False),
    ],
)
def test_keep_judge_score(
    judge_result: object, score_name: str, operator: subject.FilterOperator, expected: object, keep: bool
) -> None:
    assert subject._keep_judge_score(judge_result, score_name=score_name, operator=operator, expected=expected) is keep


def test_start_inference_server_forwards_dynamo_configuration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    class FakeModelConfig:
        def __init__(self, **kwargs: object) -> None:
            captured.setdefault("models", []).append(kwargs)

    class FakeServerConfig:
        def __init__(self, **kwargs: object) -> None:
            captured["backend"] = kwargs

    class FakeServer:
        def __init__(self, **kwargs: object) -> None:
            captured["server"] = kwargs

        def start(self) -> None:
            captured["started"] = True

    monkeypatch.setattr(subject, "DynamoVLLMModelConfig", FakeModelConfig)
    monkeypatch.setattr(subject, "DynamoServerConfig", FakeServerConfig)
    monkeypatch.setattr(subject, "InferenceServer", FakeServer)
    config = {"dynamo_server": {"subprocess_env": {"PYTHONPATH": "patches"}, "port": 9000}}
    models = [{"model": "local-weights", "served_model_name": "served-name", "dynamo_model": {"num_replicas": 2}}]

    server = subject._start_inference_server(config, models, config_path=tmp_path / "judge.yaml")

    assert isinstance(server, FakeServer)
    assert captured["models"] == [
        {"model_identifier": "local-weights", "model_name": "served-name", "num_replicas": 2}
    ]
    assert captured["backend"] == {
        "subprocess_env": {"PYTHONPATH": str((tmp_path / "patches").resolve())},
        "port": 9000,
    }
    assert captured["started"] is True


def test_build_pipeline_orders_reader_judges_filters_and_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subject.DataDesignerStage, "_init_data_designer", lambda self: None)  # noqa: ARG005
    judge_stages = [
        (
            "quality",
            object(),
            [],
            {"env": "one"},
            1,
            [{"judge": "quality", "score": "score", "operator": "eq", "value": 1}],
        ),
        ("safety", object(), [], {"env": "two"}, 2, []),
    ]

    pipeline = subject.build_pipeline(
        input_path="input.jsonl",
        input_format="jsonl",
        output_path="output",
        output_format="jsonl",
        judge_stages=judge_stages,
        language_filter_stage=None,
        files_per_partition=4,
    )

    assert [stage.name for stage in pipeline.stages] == [
        "jsonl_reader",
        "ndd_quality",
        "judge_filter_quality_01",
        "ndd_safety",
        "jsonl_writer",
    ]
    assert isinstance(pipeline.stages[0], subject.JsonlReader)
    assert isinstance(pipeline.stages[2], subject.Filter)
    assert isinstance(pipeline.stages[-1], subject.JsonlWriter)
    ndd_stages = [stage for stage in pipeline.stages if stage.name.startswith("ndd_")]
    assert [stage.runtime_env for stage in ndd_stages] == [{"env": "one"}, {"env": "two"}]
    assert [stage.num_workers() for stage in ndd_stages] == [1, 2]


def test_build_language_filter_stage_returns_none_when_language_not_set() -> None:
    assert (
        subject._build_language_filter_stage(language=None, model_path=None, min_score=0.3, text_field="raw_text")
        is None
    )


@pytest.mark.parametrize(
    ("model_path", "min_score", "message"),
    [
        (None, 0.3, "fasttext_langid_model_path is required"),
        ("/path/to/model", -0.1, "min_langid_score must be between 0 and 1"),
        ("/path/to/model", 1.1, "min_langid_score must be between 0 and 1"),
    ],
)
def test_build_language_filter_stage_rejects_invalid_config(
    model_path: str | None, min_score: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        subject._build_language_filter_stage(
            language="en", model_path=model_path, min_score=min_score, text_field="raw_text"
        )


def test_build_language_filter_stage_builds_score_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_curator.stages.text.filters.doc_filter import DocumentFilter

    class FakeFastTextLangId(DocumentFilter):
        def __init__(self, **kwargs: object) -> None:
            super().__init__()
            self.kwargs = kwargs

        def score_document(self, text: str) -> float:  # noqa: ARG002
            return 1.0

        def keep_document(self, scores: float) -> bool:  # noqa: ARG002
            return True

    monkeypatch.setattr("nemo_curator.stages.text.filters.fasttext.FastTextLangId", FakeFastTextLangId)

    stage = subject._build_language_filter_stage(
        language="en", model_path="/path/to/model", min_score=0.5, text_field="raw_text"
    )

    assert isinstance(stage, subject.ScoreFilter)
    assert stage.name == "fasttext_language_filter"
    assert isinstance(stage.filter_obj, list)
    assert len(stage.filter_obj) == 1
    assert isinstance(stage.filter_obj[0], FakeFastTextLangId)
    assert stage.filter_obj[0].kwargs == {"model_path": "/path/to/model", "min_langid_score": 0.5, "lang": "en"}
    assert stage.text_field == ["raw_text"]


def test_workflow_run_builds_pipeline_and_returns_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured, result = _run_workflow_with_fakes(monkeypatch, tmp_path, output_tasks=["task-a", "task-b"])

    assert captured["builder_judges"] == [["quality_judge"], ["safety_judge"]]
    stage_details = [
        (stage[0], stage[3], stage[4], [item["judge"] for item in stage[5]]) for stage in captured["judge_stages"]
    ]
    assert stage_details == [
        ("quality", {"env": "quality"}, 1, ["quality_judge"]),
        ("safety", {"env": "safety"}, 2, ["safety_judge"]),
    ]
    assert captured["build_pipeline_kwargs"]["language_filter_stage"] is None
    assert captured["server_stopped"] is True
    assert captured["run_kwargs"]["checkpoint_path"] == "checkpoint"
    assert result.workflow_name == "llm_judge"
    assert result.pipeline_tasks == {"llm_judge": ["task-a", "task-b"]}
    assert result.get_metadata("total_time") >= 0


def test_workflow_run_builds_language_filter_stage_when_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sentinel_stage = object()
    monkeypatch.setattr(subject, "_build_language_filter_stage", lambda **kwargs: sentinel_stage)  # noqa: ARG005

    captured, _result = _run_workflow_with_fakes(
        monkeypatch,
        tmp_path,
        workflow_kwargs={
            "language": "en",
            "fasttext_langid_model_path": "/path/to/model",
        },
    )

    assert captured["build_pipeline_kwargs"]["language_filter_stage"] is sentinel_stage


def test_workflow_run_stops_inference_server_even_when_pipeline_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(RuntimeError, match="pipeline exploded"):
        _run_workflow_with_fakes(monkeypatch, tmp_path, pipeline_run_error=RuntimeError("pipeline exploded"))


def test_llm_judge_workflow_post_init_validates_filters_eagerly(tmp_path: Path) -> None:
    config_path = tmp_path / "judge.yaml"
    config_path.write_text(
        """
models:
  - alias: judge
    model: model
execution:
  stages:
    - name: quality
      judges:
        - name: quality_judge
          scores:
            - name: quality
      filters:
        - judge: missing_judge
          score: quality
          operator: gte
          value: 4
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown judge output column"):
        subject.LLMJudgeWorkflow(judge_config=config_path, input_path="input.jsonl", output_path="output")


def _run_workflow_with_fakes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    output_tasks: list[object] | None = None,
    pipeline_run_error: Exception | None = None,
    workflow_kwargs: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], subject.WorkflowRunResult]:
    config, stages = _config_with_filters()
    config.update(
        {
            "models": [{"alias": "judge", "model": "model"}],
            "execution": {"stages": stages},
        }
    )
    stages[0]["runtime_env"] = {"env": "quality"}
    stages[0]["num_workers"] = 1
    stages[1]["runtime_env"] = {"env": "safety"}
    stages[1]["num_workers"] = 2
    captured: dict[str, Any] = {"builder_judges": []}

    class FakeServer:
        endpoint = "http://judge"

        def stop(self) -> None:
            captured["server_stopped"] = True

    class FakePipeline:
        def run(self, **kwargs: object) -> list[object]:
            captured["run_kwargs"] = kwargs
            if pipeline_run_error is not None:
                raise pipeline_run_error
            return output_tasks if output_tasks is not None else []

    def fake_builder(*args: object, **kwargs: object) -> tuple[str, list[str]]:  # noqa: ARG001
        captured["builder_judges"].append([judge["name"] for judge in kwargs["judges"]])
        return "builder", ["provider"]

    def fake_build_pipeline(**kwargs: object) -> FakePipeline:
        captured["judge_stages"] = kwargs["judge_stages"]
        captured["build_pipeline_kwargs"] = kwargs
        return FakePipeline()

    monkeypatch.setattr(subject, "_load_yaml", lambda path: config)  # noqa: ARG005
    monkeypatch.setattr(subject, "_start_inference_server", lambda *args, **kwargs: FakeServer())  # noqa: ARG005
    monkeypatch.setattr(subject, "build_config_builder", fake_builder)
    monkeypatch.setattr(subject, "build_pipeline", fake_build_pipeline)
    monkeypatch.setattr(subject, "RayDataExecutor", lambda: "executor")

    config_path = tmp_path / "judge.yaml"
    config_path.write_text("models: []\n", encoding="utf-8")
    workflow = subject.LLMJudgeWorkflow(
        judge_config=config_path,
        input_path="input.jsonl",
        output_path="output",
        checkpoint_path="checkpoint",
        **(workflow_kwargs or {}),
    )
    try:
        result = workflow.run()
    finally:
        assert captured.get("server_stopped") is True
    return captured, result
