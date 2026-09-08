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

"""
Config-driven text LLM judge workflow, run through a NeMo Curator pipeline.

The input records may have any text schema. The Jinja templates and score
rubrics in the judge config define which fields are evaluated, what the
judge returns, and how judges are grouped into ``execution.stages``, each of
which runs as its own NDD stage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Literal

import data_designer.config as dd
import yaml
from loguru import logger

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.core.serve import DynamoServerConfig, DynamoVLLMModelConfig, InferenceServer
from nemo_curator.pipeline import Pipeline
from nemo_curator.pipeline.workflow import WorkflowBase, WorkflowRunResult
from nemo_curator.stages.synthetic.nemo_data_designer import DataDesignerStage
from nemo_curator.stages.text.filters import Filter, ScoreFilter
from nemo_curator.stages.text.io.reader import JsonlReader, ParquetReader
from nemo_curator.stages.text.io.writer import JsonlWriter, ParquetWriter

DataFormat = Literal["jsonl", "parquet"]
FilterOperator = Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in"]


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        msg = f"Judge config must contain a mapping: {path}"
        raise TypeError(msg)
    return config


def _read_template(path: str, *, config_path: Path) -> str:
    template_path = Path(path)
    if not template_path.is_absolute():
        template_path = config_path.parent / template_path
    return template_path.read_text(encoding="utf-8")


def _place_filters(config: dict[str, object], stages: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    """Place top-level filters after the NDD stage that produces their judge column."""
    producer_stage_by_judge = {
        str(judge["name"]): index for index, stage in enumerate(stages) for judge in stage["judges"]
    }
    stage_filters = [list(stage.get("filters", [])) for stage in stages]
    for filter_config in config.get("filters", []):
        stage_filters[producer_stage_by_judge[str(filter_config["judge"])]].append(filter_config)
    return stage_filters


def _validate_filter_references(config: dict[str, object], stages: list[dict[str, object]]) -> None:
    """Ensure filters refer to a configured judge output column and rubric score."""
    judge_scores = {
        str(judge["name"]): {str(score["name"]) for score in judge["scores"]}
        for stage in stages
        for judge in stage["judges"]
    }
    producer_stage_by_judge = {
        str(judge["name"]): index for index, stage in enumerate(stages) for judge in stage["judges"]
    }
    filters = [(filter_config, None) for filter_config in config.get("filters", [])]
    filters.extend(
        (filter_config, stage_index)
        for stage_index, stage in enumerate(stages)
        for filter_config in stage.get("filters", [])
    )
    for filter_config, filter_stage_index in filters:
        judge_name = str(filter_config["judge"])
        score_name = str(filter_config["score"])
        if judge_name not in judge_scores:
            msg = f"Filter refers to unknown judge output column {judge_name!r}."
            raise ValueError(msg)
        if score_name not in judge_scores[judge_name]:
            msg = f"Filter refers to unknown score {score_name!r} on judge {judge_name!r}."
            raise ValueError(msg)
        if filter_stage_index is not None and producer_stage_by_judge[judge_name] > filter_stage_index:
            msg = (
                f"Stage {stages[filter_stage_index].get('name', '<unnamed>')!r} filter refers to judge {judge_name!r} "
            )
            msg += "produced by a later stage."
            raise ValueError(msg)


def _keep_judge_score(  # noqa: PLR0911
    judge_result: object,
    *,
    score_name: str,
    operator: FilterOperator,
    expected: object,
) -> bool:
    """Return whether one NDD judge result satisfies a declarative comparison."""
    try:
        actual = judge_result[score_name]["score"]
    except (KeyError, TypeError):
        return False

    try:
        if operator == "eq":
            return actual == expected
        if operator == "ne":
            return actual != expected
        if operator == "gt":
            return actual > expected
        if operator == "gte":
            return actual >= expected
        if operator == "lt":
            return actual < expected
        if operator == "lte":
            return actual <= expected
        if operator == "in":
            return actual in expected
        return actual not in expected  # noqa: TRY300
    except TypeError:
        return False


def _build_filter_stages(filters: list[dict[str, object]], *, name_prefix: str) -> list[Filter]:
    """Build Curator filters that retain rows satisfying every configured condition."""
    return [
        Filter(
            partial(
                _keep_judge_score,
                score_name=str(filter_config["score"]),
                operator=str(filter_config["operator"]),
                expected=filter_config["value"],
            ),
            filter_field=str(filter_config["judge"]),
        ).with_(name=f"{name_prefix}_{index:02d}")
        for index, filter_config in enumerate(filters, start=1)
    ]


def _build_language_filter_stage(
    *,
    language: str | None,
    model_path: str | None,
    min_score: float,
    text_field: str,
) -> ScoreFilter | None:
    """Build an optional FastText language gate without retaining its score column."""
    if not language:
        return None
    if not model_path:
        msg = "fasttext_langid_model_path is required when language is provided."
        raise ValueError(msg)
    if not 0.0 <= min_score <= 1.0:
        msg = "min_langid_score must be between 0 and 1."
        raise ValueError(msg)

    # FastText is optional, so import it only for jobs that enable this stage.
    from nemo_curator.stages.text.filters.fasttext import FastTextLangId

    return ScoreFilter(
        filter_obj=FastTextLangId(
            model_path=model_path,
            min_langid_score=min_score,
            lang=language,
        ),
        text_field=text_field,
        verbose=True,
    ).with_(name="fasttext_language_filter")


def build_config_builder(
    config_path: str | Path,
    *,
    endpoint: str,
    models: list[dict[str, object]],
    judges: list[dict[str, object]],
) -> tuple[dd.DataDesignerConfigBuilder, list[dd.ModelProvider]]:
    """Build one NDD configuration for a selected group of judge columns."""
    config_path = Path(config_path)
    provider_name = "local-judge"
    config_builder = dd.DataDesignerConfigBuilder(
        model_configs=[
            dd.ModelConfig(
                alias=str(model["alias"]),
                model=str(model.get("served_model_name", model["model"])),
                provider=provider_name,
                skip_health_check=bool(model.get("skip_health_check", True)),
                inference_parameters=dd.ChatCompletionInferenceParams(**model.get("inference_parameters", {})),
            )
            for model in models
        ]
    )

    for judge in judges:
        judge_name = str(judge["name"])
        model_alias = str(judge.get("model_alias", models[0]["alias"]))
        scores = [
            dd.Score(
                name=str(score["name"]),
                description=str(score["description"]),
                options=score["options"],
            )
            for score in judge["scores"]
        ]
        judge_kwargs: dict[str, object] = {
            "name": judge_name,
            "model_alias": model_alias,
            "prompt": _read_template(str(judge["prompt_path"]), config_path=config_path),
            "scores": scores,
            "extract_reasoning_content": bool(judge.get("extract_reasoning_content", False)),
        }
        if system_prompt_path := judge.get("system_prompt_path"):
            judge_kwargs["system_prompt"] = _read_template(str(system_prompt_path), config_path=config_path)
        if trace := judge.get("with_trace"):
            judge_kwargs["with_trace"] = dd.TraceType(trace)
        config_builder.add_column(dd.LLMJudgeColumnConfig(**judge_kwargs))

    model_providers = [
        dd.ModelProvider(
            name=provider_name,
            endpoint=endpoint,
            api_key="unused",  # pragma: allowlist secret
        )
    ]
    return config_builder, model_providers


def _start_inference_server(
    config: dict[str, object], models: list[dict[str, object]], *, config_path: Path
) -> InferenceServer:
    """Start all configured Dynamo models behind one OpenAI-compatible endpoint."""
    dynamo_server = dict(config.get("dynamo_server", {}))
    subprocess_env = dynamo_server.get("subprocess_env", {})
    if pythonpath := subprocess_env.get("PYTHONPATH"):
        patch_dir = Path(pythonpath)
        if not patch_dir.is_absolute():
            dynamo_server["subprocess_env"] = {
                **subprocess_env,
                "PYTHONPATH": str((config_path.parent / patch_dir).resolve()),
            }
    inference_server = config.get("inference_server", {})
    model_configs = []
    for model in models:
        dynamo_model = dict(model.get("dynamo_model", {}))
        model_configs.append(
            DynamoVLLMModelConfig(
                model_identifier=str(model["model"]),
                model_name=str(model.get("served_model_name", model["model"])),
                **dynamo_model,
            )
        )
    server = InferenceServer(
        models=model_configs,
        backend=DynamoServerConfig(**dynamo_server),
        **inference_server,
    )
    server.start()
    return server


def build_pipeline(  # noqa: PLR0913
    *,
    input_path: str,
    input_format: DataFormat,
    output_path: str,
    output_format: DataFormat,
    judge_stages: list[
        tuple[
            str,
            dd.DataDesignerConfigBuilder,
            list[dd.ModelProvider],
            dict[str, object] | None,
            int | None,
            list[dict[str, object]],
        ]
    ],
    language_filter_stage: ScoreFilter | None,
    files_per_partition: int | None,
) -> Pipeline:
    """Build a streaming pipeline with an optional language gate, NDD stages, filters, and writer."""
    # TODO: Add an optional TokenLengthFilter stage before NDD stages so prompts
    # can be bounded by model tokens instead of task-specific Jinja character caps.
    reader = (
        JsonlReader(file_paths=input_path, files_per_partition=files_per_partition)
        if input_format == "jsonl"
        else ParquetReader(file_paths=input_path, files_per_partition=files_per_partition)
    )
    writer = JsonlWriter(path=output_path) if output_format == "jsonl" else ParquetWriter(path=output_path)
    processing_stages = []
    for stage_name, config_builder, model_providers, runtime_env, num_workers, stage_filters in judge_stages:
        processing_stages.append(
            DataDesignerStage(config_builder=config_builder, model_providers=model_providers).with_(
                name=f"ndd_{stage_name}", runtime_env=runtime_env, num_workers=num_workers
            )
        )
        processing_stages.extend(_build_filter_stages(stage_filters, name_prefix=f"judge_filter_{stage_name}"))
    return Pipeline(
        name="llm_judge",
        description="Evaluate text records with a config-driven NDD LLM judge.",
        stages=[reader, *([language_filter_stage] if language_filter_stage else []), *processing_stages, writer],
    )


@dataclass
class LLMJudgeWorkflow(WorkflowBase):
    """
    End-to-end config-driven LLM judge workflow.

    Loads a judge config YAML (models, Jinja prompt templates, score rubrics,
    and ``execution.stages``), starts a Dynamo/vLLM inference server hosting
    the configured judge models, then runs one Curator pipeline containing:
    reader -> optional FastText language gate -> one NDD ``DataDesignerStage``
    (+ its filters) per judge stage -> writer.
    """

    # required args
    judge_config: str | Path
    input_path: str
    output_path: str

    # I/O
    input_format: DataFormat = "jsonl"
    output_format: DataFormat = "jsonl"
    files_per_partition: int | None = None

    # optional FastText language gate
    language: str | None = None
    fasttext_langid_model_path: str | None = None
    min_langid_score: float = 0.3
    language_text_field: str = "raw_text"

    # execution
    checkpoint_path: str | None = None

    config_path: Path = field(init=False)
    config: dict[str, object] = field(init=False)

    def __post_init__(self) -> None:
        self.config_path = Path(self.judge_config).resolve()
        self.config = _load_yaml(self.config_path)
        stages = self.config["execution"]["stages"]
        _validate_filter_references(self.config, stages)

    def _build_judge_stages(
        self, *, endpoint: str
    ) -> list[
        tuple[
            str,
            dd.DataDesignerConfigBuilder,
            list[dd.ModelProvider],
            dict[str, object] | None,
            int | None,
            list[dict[str, object]],
        ]
    ]:
        models = self.config["models"]
        stages = self.config["execution"]["stages"]
        stage_filters = _place_filters(self.config, stages)
        judge_stages = []
        for stage, filters_after_stage in zip(stages, stage_filters, strict=True):
            config_builder, model_providers = build_config_builder(
                self.judge_config,
                endpoint=endpoint,
                models=models,
                judges=stage["judges"],
            )
            judge_stages.append(
                (
                    str(stage["name"]),
                    config_builder,
                    model_providers,
                    stage.get("runtime_env"),
                    stage.get("num_workers"),
                    filters_after_stage,
                )
            )
        return judge_stages

    def run(self) -> WorkflowRunResult:
        """
        Run the complete LLM judge pipeline.

        Returns:
            WorkflowRunResult containing the pipeline output tasks and timing metadata.
        """
        executor = RayDataExecutor()
        workflow_result = WorkflowRunResult(workflow_name="llm_judge")

        language_filter_stage = _build_language_filter_stage(
            language=self.language,
            model_path=self.fasttext_langid_model_path,
            min_score=self.min_langid_score,
            text_field=self.language_text_field,
        )

        inference_server: InferenceServer | None = None
        start_time = time.time()
        try:
            inference_server = _start_inference_server(
                self.config, self.config["models"], config_path=self.config_path
            )
            judge_stages = self._build_judge_stages(endpoint=inference_server.endpoint)
            pipeline = build_pipeline(
                input_path=self.input_path,
                input_format=self.input_format,
                output_path=self.output_path,
                output_format=self.output_format,
                judge_stages=judge_stages,
                language_filter_stage=language_filter_stage,
                files_per_partition=self.files_per_partition,
            )
            output_tasks = pipeline.run(executor=executor, checkpoint_path=self.checkpoint_path)
        except Exception as e:
            logger.error(f"LLM judge pipeline failed: {e}")
            raise
        finally:
            if inference_server is not None:
                inference_server.stop()

        execution_time = time.time() - start_time
        workflow_result.add_pipeline_tasks("llm_judge", output_tasks)
        workflow_result.add_metadata("total_time", execution_time)
        return workflow_result
