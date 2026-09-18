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
Step 4 of the fuzzy-dedup-eval example: judge the labeled pairs produced by
`3_build_pair_dataset.py` with `LLMJudgeWorkflow`.

This is a thin wrapper around `LLMJudgeWorkflow`.
Edit `judge_config/fuzzy_pair_judge.yaml` first -- set `models[0].model` to a
local model path or HF repo id, and size `num_replicas`/`tensor_parallel_size`
to your GPUs (bundled default: 1 GPU).

Example:
    python tutorials/eval/dedup/4_run_llm_judge.py \
        --input-path output/dedup_eval/keeper_removed_pairs \
        --output-path output/dedup_eval/judged_pairs
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nemo_curator.core.client import RayClient
from nemo_curator.eval.llm_judge import LLMJudgeWorkflow

_DEFAULT_JUDGE_CONFIG = Path(__file__).resolve().parent / "judge_config" / "fuzzy_pair_judge.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--judge-config",
        default=str(_DEFAULT_JUDGE_CONFIG),
        help="YAML file defining the model, Jinja templates, and rubric (default: judge_config/fuzzy_pair_judge.yaml).",
    )
    parser.add_argument(
        "--input-path", required=True, help="Directory of pairs part_*.jsonl files written by 3_build_pair_dataset.py."
    )
    parser.add_argument("--input-format", default="jsonl", choices=("jsonl", "parquet"))
    parser.add_argument("--output-path", required=True, help="Directory for judged-pair output partitions.")
    parser.add_argument("--output-format", default="jsonl", choices=("jsonl", "parquet"))
    parser.add_argument("--files-per-partition", type=int, default=None)
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
    workflow = LLMJudgeWorkflow(
        judge_config=args.judge_config,
        input_path=args.input_path,
        output_path=args.output_path,
        input_format=args.input_format,
        output_format=args.output_format,
        files_per_partition=args.files_per_partition,
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
