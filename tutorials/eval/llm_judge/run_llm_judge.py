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
CLI entry point for LLMJudgeWorkflow.

The input records may have any text schema. The Jinja templates and score
rubrics in ``--judge-config`` define which fields are evaluated, what the
judge returns, and how judges are grouped into ``execution.stages``, each of
which runs as its own NDD stage.

Example:
    python tutorials/eval/llm_judge/run_llm_judge.py \
        --judge-config tutorials/eval/llm_judge/cc_extract_example/text_extraction_qwen_judge.yaml \
        --input-path data/cc_extractions --input-format jsonl \
        --output-path data/qwen_judgements --output-format jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from nemo_curator.core.client import RayClient
from nemo_curator.eval.llm_judge import LLMJudgeWorkflow
from nemo_curator.utils.file_utils import FILETYPE_TO_DEFAULT_EXTENSIONS, get_all_file_paths_and_size_under


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


def _preflight_errors(args: argparse.Namespace) -> list[str]:
    """Collect config mistakes that would otherwise surface only after the model server has loaded."""
    errors: list[str] = []
    config_path = Path(args.judge_config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    aliases = {str(model["alias"]) for model in config.get("models", [])}
    for stage in config.get("execution", {}).get("stages", []):
        for judge in stage.get("judges", []):
            name = judge.get("name")
            alias = judge.get("model_alias")
            if alias is not None and str(alias) not in aliases:
                errors.append(f"Judge {name!r}: unknown model_alias {str(alias)!r} (configured: {sorted(aliases)}).")
            for key in ("prompt_path", "system_prompt_path"):
                if template := judge.get(key):
                    template_path = config_path.parent / template  # absolute paths replace the parent
                    if not template_path.is_file():
                        errors.append(f"Judge {name!r}: {key} not found: {template_path}")

    try:
        input_files = get_all_file_paths_and_size_under(
            args.input_path,
            recurse_subdirectories=True,
            keep_extensions=FILETYPE_TO_DEFAULT_EXTENSIONS[args.input_format],
        )
    except FileNotFoundError:
        input_files = []
    if not input_files:
        errors.append(f"No {args.input_format} files found at --input-path: {args.input_path}")
    return errors


def main() -> None:
    args = _parse_args()
    if errors := _preflight_errors(args):
        sys.exit("Config errors (found before starting the model server):\n  " + "\n  ".join(errors))
    workflow = LLMJudgeWorkflow(
        judge_config=args.judge_config,
        input_path=args.input_path,
        output_path=args.output_path,
        input_format=args.input_format,
        output_format=args.output_format,
        files_per_partition=args.files_per_partition,
        language=args.language,
        fasttext_langid_model_path=args.fasttext_langid_model_path,
        min_langid_score=args.min_langid_score,
        language_text_field=args.language_text_field,
        checkpoint_path=args.checkpoint_path,
    )
    ray_client = RayClient(
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        include_dashboard=False,
        ray_temp_dir=args.ray_temp_dir,
    )
    ray_client.start()
    try:
        workflow.run()
    finally:
        ray_client.stop()


if __name__ == "__main__":
    main()
