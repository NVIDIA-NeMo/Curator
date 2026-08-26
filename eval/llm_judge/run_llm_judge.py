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
Run a config-driven text LLM judge through a NeMo Curator pipeline.

The input records may have any text schema. The Jinja templates and score
rubrics in ``--judge-config`` define which fields are evaluated, what the
judge returns, and whether groups share or use separate NDD stages.

Example:
    python eval/llm_judge/run_llm_judge.py \
        --judge-config eval/llm_judge/cc_extract_example/text_extraction_qwen_judge.yaml \
        --input-path extracted.jsonl --input-format jsonl \
        --output-path judged --output-format jsonl
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from typing import Literal

import data_designer.config as dd
import yaml

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.core.serve import DynamoServerConfig, DynamoVLLMModelConfig, InferenceServer
from nemo_curator.pipeline import Pipeline
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


def _validate_filter_references(
    config: dict[str, object], stages: list[dict[str, object]], *, enforce_stage_order: bool = True
) -> None:
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
        if (
            enforce_stage_order
            and filter_stage_index is not None
            and producer_stage_by_judge[judge_name] > filter_stage_index
        ):
            msg = (
                f"Stage {stages[filter_stage_index].get('name', '<unnamed>')!r} "
                f"filter refers to judge {judge_name!r} "
            )
            msg += "produced by a later stage."
            raise ValueError(msg)


def _get_num_workers(config: dict[str, object], *, owner: str) -> int | None:
    """Return an optional fixed Ray worker count for one NDD stage."""
    num_workers = config.get("num_workers")
    if num_workers is None:
        return None
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers <= 0:
        msg = f"{owner} must be a positive integer."
        raise ValueError(msg)
    return num_workers


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
        msg = "--fasttext-langid-model-path is required when --language is provided."
        raise ValueError(msg)
    if not 0.0 <= min_score <= 1.0:
        msg = "--min-langid-score must be between 0 and 1."
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--judge-config",
        required=True,
        help="YAML file defining the model, Jinja templates, and rubrics.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=("single_stage", "multi_stage"),
        default="single_stage",
        help="Run all judges in one NDD stage or use one NDD stage per configured group (default: single_stage).",
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
        help=("FastText language code to retain, such as 'en'. Omit this option to disable language filtering."),
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
    parser.add_argument(
        "--ray-temp-dir",
        default="/tmp/ray",  # noqa: S108
        help="Ray runtime directory (default: /tmp/ray).",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        help="Optional CPU count for the local Ray client (default: all available CPUs).",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Optional GPU count for the local Ray client (default: all available GPUs).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = Path(args.judge_config).resolve()
    config = _load_yaml(config_path)
    models = config["models"]
    execution = config["execution"]
    configured_stages = execution["stages"]
    _validate_filter_references(
        config,
        configured_stages,
        enforce_stage_order=args.execution_mode == "multi_stage",
    )
    stage_filters = _place_filters(config, configured_stages)
    language_filter_stage = _build_language_filter_stage(
        language=args.language,
        model_path=args.fasttext_langid_model_path,
        min_score=args.min_langid_score,
        text_field=args.language_text_field,
    )

    client = RayClient(
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        include_dashboard=False,
        ray_temp_dir=args.ray_temp_dir,
    )
    client.start()
    inference_server: InferenceServer | None = None
    try:
        inference_server = _start_inference_server(config, models, config_path=config_path)
        if args.execution_mode == "single_stage":
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
                        _get_num_workers(execution, owner="execution.num_workers"),
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
                    (
                        str(stage["name"]),
                        config_builder,
                        model_providers,
                        stage.get("runtime_env"),
                        _get_num_workers(stage, owner=f"Stage {stage.get('name', '<unnamed>')!r} num_workers"),
                        filters_after_stage,
                    )
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
