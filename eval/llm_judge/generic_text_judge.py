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

"""Run a config-driven text LLM judge through a NeMo Curator pipeline.

The input records may have any text schema. The Jinja templates and score
rubrics in ``--judge-config`` define which fields are evaluated, what the
judge returns, and whether groups share or use separate NDD stages.

Example:
    python eval/llm_judge/generic_text_judge.py \
        --judge-config eval/llm_judge/examples/text_extraction_judge.yaml \
        --input-path extracted.jsonl --input-format jsonl \
        --output-path judged --output-format jsonl
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from typing import Any, Literal

import data_designer.config as dd
import yaml
from loguru import logger

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.core.serve import DynamoServerConfig, DynamoVLLMModelConfig, InferenceServer
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.synthetic.nemo_data_designer import DataDesignerStage
from nemo_curator.stages.text.filters import Filter, ScoreFilter
from nemo_curator.stages.text.io.reader import JsonlReader, ParquetReader
from nemo_curator.stages.text.io.writer import JsonlWriter, ParquetWriter
from nemo_curator.tasks import DocumentBatch


DataFormat = Literal["jsonl", "parquet"]
FilterOperator = Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in"]
_FILTER_OPERATORS = {"eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in"}


class _LoggingLanguageFilter(ScoreFilter):
    """Report the number of records retained by the optional FastText gate."""

    def process(self, batch: DocumentBatch) -> DocumentBatch | None:
        rows_before = len(batch.to_pandas())
        filtered_batch = super().process(batch)
        rows_after = len(filtered_batch.to_pandas()) if filtered_batch is not None else 0
        logger.info(
            "FastText language filter batch {}: {} -> {} rows",
            batch.task_id,
            rows_before,
            rows_after,
        )
        return filtered_batch


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Judge config must contain a mapping: {path}")
    return config


def _read_template(path: str, *, config_path: Path) -> str:
    template_path = Path(path)
    if not template_path.is_absolute():
        template_path = config_path.parent / template_path
    return template_path.read_text(encoding="utf-8")


def _load_models(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Load model mappings, accepting the original single-model shorthand."""
    models = config.get("models")
    if models is None:
        model = config.get("model")
        models = [model] if model is not None else None
    if not isinstance(models, list) or not models or not all(isinstance(model, dict) for model in models):
        raise ValueError("Judge config requires a non-empty 'models' list of mappings.")

    aliases: set[str] = set()
    for model in models:
        missing = [key for key in ("alias", "model") if not model.get(key)]
        if missing:
            raise ValueError(f"Model config is missing required fields: {', '.join(missing)}")
        alias = str(model["alias"])
        if alias in aliases:
            raise ValueError(f"Model aliases must be unique; {alias!r} appears more than once.")
        aliases.add(alias)
    return models


def _load_execution_stages(config: dict[str, Any]) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    """Load judge groups and the requested one- or many-NDD-stage execution mode."""
    execution = config.get("execution", {})
    if not isinstance(execution, dict):
        raise ValueError("'execution' must be a mapping when provided.")
    mode = execution.get("mode", "single_stage")
    if mode not in {"single_stage", "multi_stage"}:
        raise ValueError("execution.mode must be 'single_stage' or 'multi_stage'.")
    if execution.get("runtime_env") is not None and not isinstance(execution["runtime_env"], dict):
        raise ValueError("execution.runtime_env must be a mapping when provided.")

    stages = execution.get("stages")
    if stages is None:
        judges = config.get("judges")
        if judges is None:
            single_judge = config.get("judge")
            judges = [single_judge] if single_judge is not None else None
        stages = [{"name": "judges", "judges": judges}]
    if not isinstance(stages, list) or not stages or not all(isinstance(stage, dict) for stage in stages):
        raise ValueError("execution.stages must be a non-empty list of mappings.")

    stage_names: set[str] = set()
    judge_names: set[str] = set()
    for stage in stages:
        if not stage.get("name"):
            raise ValueError("Every execution stage requires a unique 'name'.")
        name = str(stage["name"])
        if name in stage_names:
            raise ValueError(f"Execution stage names must be unique; {name!r} appears more than once.")
        stage_names.add(name)
        judges = stage.get("judges")
        if not isinstance(judges, list) or not judges or not all(isinstance(judge, dict) for judge in judges):
            raise ValueError(f"Execution stage {name!r} requires a non-empty 'judges' list.")
        for judge in judges:
            judge_name = judge.get("name")
            if not judge_name:
                raise ValueError(f"Every judge in execution stage {name!r} requires a unique 'name'.")
            if str(judge_name) in judge_names:
                raise ValueError(f"Judge names must be unique; {judge_name!r} appears more than once.")
            judge_names.add(str(judge_name))
        if stage.get("runtime_env") is not None and not isinstance(stage["runtime_env"], dict):
            raise ValueError(f"execution stage {name!r}.runtime_env must be a mapping.")
    return str(mode), execution, stages


def _validate_filters(
    filters: Any,
    *,
    judge_scores: dict[str, set[str]],
    location: str,
) -> list[dict[str, Any]]:
    """Validate declarative keep/drop filters over LLM-judge rubric scores."""
    if not isinstance(filters, list) or not all(isinstance(item, dict) for item in filters):
        raise ValueError(f"{location} filters must be a list of mappings when provided.")
    for item in filters:
        missing = [key for key in ("judge", "score", "operator", "value") if key not in item]
        if missing:
            raise ValueError(f"Every filter requires: {', '.join(missing)}")
        judge_name = str(item["judge"])
        if judge_name not in judge_scores:
            raise ValueError(f"Filter refers to unknown judge {item['judge']!r}.")
        if str(item["score"]) not in judge_scores[judge_name]:
            raise ValueError(f"Filter refers to unknown score {item['score']!r} on judge {judge_name!r}.")
        if str(item["operator"]) not in _FILTER_OPERATORS:
            allowed = ", ".join(sorted(_FILTER_OPERATORS))
            raise ValueError(f"Unknown filter operator {item['operator']!r}; choose one of: {allowed}.")
        if str(item["operator"]) in {"in", "not_in"} and not isinstance(item["value"], list):
            raise ValueError(f"Filter operator {item['operator']!r} requires 'value' to be a YAML list.")
    return filters


def _load_filters(
    config: dict[str, Any], stages: list[dict[str, Any]]
) -> list[list[dict[str, Any]]]:
    """Place top-level filters after the NDD stage that produces their judge column."""
    all_judge_scores = {
        str(judge["name"]): {str(score["name"]) for score in judge["scores"]}
        for stage in stages
        for judge in stage["judges"]
    }
    filters = _validate_filters(
        config.get("filters", []), judge_scores=all_judge_scores, location="Top-level"
    )
    producer_stage_by_judge = {
        str(judge["name"]): index
        for index, stage in enumerate(stages)
        for judge in stage["judges"]
    }

    available_judge_scores: dict[str, set[str]] = {}
    stage_filters: list[list[dict[str, Any]]] = []
    for stage in stages:
        available_judge_scores.update(
            {
                str(judge["name"]): {str(score["name"]) for score in judge["scores"]}
                for judge in stage["judges"]
            }
        )
        stage_filters.append(
            _validate_filters(
                stage.get("filters", []),
                judge_scores=available_judge_scores,
                location=f"Execution stage {stage['name']!r}",
            )
        )
    for filter_config in filters:
        stage_filters[producer_stage_by_judge[str(filter_config["judge"])]].append(filter_config)
    return stage_filters


def _keep_judge_score(
    judge_result: Any,
    *,
    score_name: str,
    operator: FilterOperator,
    expected: Any,
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
        return actual not in expected
    except TypeError:
        return False


def _build_filter_stages(filters: list[dict[str, Any]], *, name_prefix: str) -> list[Filter]:
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
        raise ValueError("--fasttext-langid-model-path is required when --language is provided.")
    if not 0.0 <= min_score <= 1.0:
        raise ValueError("--min-langid-score must be between 0 and 1.")

    # FastText is optional, so import it only for jobs that enable this stage.
    from nemo_curator.stages.text.filters.fasttext import FastTextLangId

    return _LoggingLanguageFilter(
        filter_obj=FastTextLangId(
            model_path=model_path,
            min_langid_score=min_score,
            lang=language,
        ),
        text_field=text_field,
    ).with_(name="fasttext_language_filter")


def build_config_builder(
    config_path: str | Path,
    *,
    endpoint: str,
    models: list[dict[str, Any]],
    judges: list[dict[str, Any]],
) -> tuple[dd.DataDesignerConfigBuilder, list[dd.ModelProvider]]:
    """Build one NDD configuration for a selected group of judge columns."""
    config_path = Path(config_path)
    provider_name = "local-judge"
    aliases = {str(model["alias"]) for model in models}
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

    judge_names: set[str] = set()
    for judge in judges:
        if not judge.get("name") or not judge.get("prompt_path") or not judge.get("scores"):
            raise ValueError("Every judge requires 'name', 'prompt_path', and non-empty 'scores'.")
        judge_name = str(judge["name"])
        if judge_name in judge_names:
            raise ValueError(f"Judge names must be unique; {judge_name!r} appears more than once.")
        judge_names.add(judge_name)
        model_alias = str(judge.get("model_alias", models[0]["alias"]))
        if model_alias not in aliases:
            raise ValueError(f"Judge {judge_name!r} refers to unknown model_alias {model_alias!r}.")

        scores = [
            dd.Score(
                name=str(score["name"]),
                description=str(score["description"]),
                options=score["options"],
            )
            for score in judge["scores"]
        ]
        judge_kwargs: dict[str, Any] = {
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


def _start_inference_server(config: dict[str, Any], models: list[dict[str, Any]]) -> InferenceServer:
    """Start all configured Dynamo models behind one OpenAI-compatible endpoint."""
    dynamo_server = config.get("dynamo_server", {})
    if not isinstance(dynamo_server, dict):
        raise ValueError("dynamo_server must be a mapping when provided.")

    model_configs = []
    for model in models:
        dynamo_model = model.get("dynamo_model", {})
        if not isinstance(dynamo_model, dict):
            raise ValueError(f"models.{model['alias']}.dynamo_model must be a mapping when provided.")
        model_configs.append(
            DynamoVLLMModelConfig(
                model_identifier=str(model["model"]),
                model_name=str(model.get("served_model_name", model["model"])),
                **dynamo_model,
            )
        )
    server = InferenceServer(models=model_configs, backend=DynamoServerConfig(**dynamo_server))
    server.start()
    return server


def build_pipeline(
    *,
    input_path: str,
    input_format: DataFormat,
    output_path: str,
    output_format: DataFormat,
    judge_stages: list[
        tuple[str, dd.DataDesignerConfigBuilder, list[dd.ModelProvider], dict[str, Any] | None, list[dict[str, Any]]]
    ],
    language_filter_stage: ScoreFilter | None,
    files_per_partition: int | None,
) -> Pipeline:
    """Build one streaming Curator reader → optional language gate → NDD stages → filters → writer pipeline."""
    # TODO: Add an optional TokenLengthFilter stage before NDD stages so prompts
    # can be bounded by model tokens instead of task-specific Jinja character caps.
    reader = (
        JsonlReader(file_paths=input_path, files_per_partition=files_per_partition)
        if input_format == "jsonl"
        else ParquetReader(file_paths=input_path, files_per_partition=files_per_partition)
    )
    writer = JsonlWriter(path=output_path) if output_format == "jsonl" else ParquetWriter(path=output_path)
    processing_stages = []
    for stage_name, config_builder, model_providers, runtime_env, stage_filters in judge_stages:
        processing_stages.append(
            DataDesignerStage(config_builder=config_builder, model_providers=model_providers).with_(
                name=f"ndd_{stage_name}", runtime_env=runtime_env
            )
        )
        processing_stages.extend(_build_filter_stages(stage_filters, name_prefix=f"judge_filter_{stage_name}"))
    return Pipeline(
        name="generic_text_llm_judge",
        description="Evaluate text records with a config-driven NDD LLM judge.",
        stages=[reader, *([language_filter_stage] if language_filter_stage else []), *processing_stages, writer],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--judge-config",
        required=True,
        help="YAML file defining the model, Jinja templates, and rubrics.",
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="JSONL/Parquet path or glob accepted by the Curator reader.",
    )
    parser.add_argument("--input-format", required=True, choices=("jsonl", "parquet"))
    parser.add_argument("--output-path", required=True, help="Directory for Curator output partitions.")
    parser.add_argument("--output-format", default="jsonl", choices=("jsonl", "parquet"))
    parser.add_argument("--files-per-partition", type=int, default=None)
    parser.add_argument(
        "--language",
        default=None,
        help=(
            "FastText language code to retain, such as 'en'. "
            "Omit this option to disable language filtering."
        ),
    )
    parser.add_argument(
        "--fasttext-langid-model-path",
        default=None,
        help="Path to the FastText language-ID model; required only with --language.",
    )
    parser.add_argument(
        "--min-langid-score",
        type=float,
        default=0.3,
        help="Minimum FastText language-ID confidence when --language is used (default: 0.3).",
    )
    parser.add_argument(
        "--language-text-field",
        default="raw_text",
        help="Input column used for FastText language ID (default: raw_text).",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional durable Curator checkpoint directory for this pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = _load_yaml(Path(args.judge_config))
    models = _load_models(config)
    execution_mode, execution, configured_stages = _load_execution_stages(config)
    stage_filters = _load_filters(config, configured_stages)
    language_filter_stage = _build_language_filter_stage(
        language=args.language,
        model_path=args.fasttext_langid_model_path,
        min_score=args.min_langid_score,
        text_field=args.language_text_field,
    )

    client = RayClient(include_dashboard=False)
    client.start()
    inference_server: InferenceServer | None = None
    try:
        inference_server = _start_inference_server(config, models)
        if execution_mode == "single_stage":
            judges = [judge for stage in configured_stages for judge in stage["judges"]]
            config_builder, model_providers = build_config_builder(
                args.judge_config,
                endpoint=inference_server.endpoint,
                models=models,
                judges=judges,
            )
            pipeline = build_pipeline(
                input_path=args.input_path,
                input_format=args.input_format,
                output_path=args.output_path,
                output_format=args.output_format,
                judge_stages=[
                    (
                        "all_judges",
                        config_builder,
                        model_providers,
                        execution.get("runtime_env"),
                        [filter_config for filters in stage_filters for filter_config in filters],
                    )
                ],
                language_filter_stage=language_filter_stage,
                files_per_partition=args.files_per_partition,
            )
            pipeline.run(executor=RayDataExecutor(), checkpoint_path=args.checkpoint_path)
        else:
            judge_stages = []
            for stage, filters_after_stage in zip(configured_stages, stage_filters, strict=True):
                config_builder, model_providers = build_config_builder(
                    args.judge_config,
                    endpoint=inference_server.endpoint,
                    models=models,
                    judges=stage["judges"],
                )
                judge_stages.append(
                    (str(stage["name"]), config_builder, model_providers, stage.get("runtime_env"), filters_after_stage)
                )
            pipeline = build_pipeline(
                input_path=args.input_path,
                input_format=args.input_format,
                output_path=args.output_path,
                output_format=args.output_format,
                judge_stages=judge_stages,
                language_filter_stage=language_filter_stage,
                files_per_partition=args.files_per_partition,
            )
            pipeline.run(executor=RayDataExecutor(), checkpoint_path=args.checkpoint_path)
    finally:
        if inference_server is not None:
            inference_server.stop()
        client.stop()


if __name__ == "__main__":
    main()
