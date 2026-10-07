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
Step 6: review saved main results with coverage and optional subject verification.

Run from the repository root:
    python tutorials/eval/dedup/6_run_critics.py \
        --input-path output/dedup_eval/judged_pairs \
        --output-path output/dedup_eval/reviewed_pairs

Reuses the main judge YAML's serving/model settings, but does not run its judges
or score filters. Main results must already be present in the input records.
Add --subject to check fixed subject-conflict proposals after coverage, sharing
the same service and Pipeline.

The critics only ever remove unsafe "can replace" directions from the saved judgment. The
result is written to `final_decision` alongside `coverage_*` columns, and `subject_*` columns
with --subject.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import data_designer.config as dd

import critics
import ray
from critics.coverage import CoverageApplyStage, CoveragePrepareStage, coverage_column
from critics.spans import DEFAULT_MAX_EVIDENCE_CHARS
from critics.subject import (
    SubjectApplyStage,
    SubjectPrepareStage,
    SubjectVerifierApplyStage,
    subject_column,
    verifier_column,
)

from nemo_curator.core.client import RayClient
from nemo_curator.eval.llm_judge.workflow import LLMJudgeWorkflow, build_config_builder
from nemo_curator.stages.synthetic.nemo_data_designer import DataDesignerStage

_DEFAULT_JUDGE_CONFIG = Path(__file__).resolve().parent / "judge_config" / "fuzzy_pair_judge.yaml"


@dataclass
class CriticWorkflow(LLMJudgeWorkflow):
    """
    `LLMJudgeWorkflow` that runs the coverage critic (and optionally the subject critic and verifier)
    instead of the YAML's judges.

    Pipeline: reader -> coverage prepare -> coverage LLM -> coverage apply
    [-> subject prepare -> subject LLM -> subject apply -> verifier LLM -> verifier apply] -> writer.
    Serving settings, `num_workers` and `runtime_env` come from the first stage in the judge YAML's
    `execution.stages`.
    """

    model_alias: str | None = None
    source_judge: str = "pair_semantic_judgment"
    subject: bool = False
    max_evidence_chars: int = DEFAULT_MAX_EVIDENCE_CHARS

    def __post_init__(self) -> None:
        super().__post_init__()
        aliases = [str(model["alias"]) for model in self.config["models"]]
        self.model_alias = self.model_alias or aliases[0]
        if self.model_alias not in aliases:
            msg = f"Unknown model alias {self.model_alias!r}; the judge config defines {aliases}."
            raise ValueError(msg)

        self._user_postprocessing = list(self.postprocessing_stages)
        self.preprocessing_stages = [
            *self.preprocessing_stages,
            CoveragePrepareStage(self.source_judge, self.max_evidence_chars),
        ]
        self.postprocessing_stages = [
            CoverageApplyStage(self.source_judge, self.max_evidence_chars),
            *self._user_postprocessing,
        ]

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
        stage = self.config["execution"]["stages"][0]

        def llm_config(
            column: dd.LLMStructuredColumnConfig,
        ) -> tuple[dd.DataDesignerConfigBuilder, list[dd.ModelProvider]]:
            builder, providers = build_config_builder(
                self.config_path, endpoint=endpoint, models=self.config["models"], judges=[]
            )
            builder.add_column(column)
            return builder, providers

        def llm_stage(column: dd.LLMStructuredColumnConfig) -> DataDesignerStage:
            builder, providers = llm_config(column)
            return DataDesignerStage(config_builder=builder, model_providers=providers).with_(
                name=f"ndd_{column.name}", runtime_env=stage.get("runtime_env"), num_workers=stage.get("num_workers")
            )

        if self.subject:
            # These LLM stages need the live inference endpoint, so they can only be built here.
            self.postprocessing_stages = [
                CoverageApplyStage(self.source_judge, self.max_evidence_chars),
                SubjectPrepareStage(self.max_evidence_chars),
                llm_stage(subject_column(self.model_alias)),
                SubjectApplyStage(self.max_evidence_chars),
                llm_stage(verifier_column(self.model_alias)),
                SubjectVerifierApplyStage(),
                *self._user_postprocessing,
            ]

        # Dynamo startup detaches from Ray, so ship this package to the workers of the pipeline job that follows.
        ray.init(ignore_reinit_error=True, runtime_env={"py_modules": [critics]})
        builder, providers = llm_config(coverage_column(self.model_alias))
        return [("coverage", builder, providers, stage.get("runtime_env"), stage.get("num_workers"), [])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge-config", default=str(_DEFAULT_JUDGE_CONFIG))
    parser.add_argument("--input-path", required=True, help="Saved main-judge output from step 5.")
    parser.add_argument("--output-path", required=True, help="A separate directory for reviewed pairs.")
    parser.add_argument("--input-format", default="jsonl", choices=("jsonl", "parquet"))
    parser.add_argument("--output-format", default="jsonl", choices=("jsonl", "parquet"))
    parser.add_argument("--source-judge", default="pair_semantic_judgment", help="Judge column to review.")
    parser.add_argument("--model-alias", default=None, help="Model alias from the YAML; defaults to the first model.")
    parser.add_argument("--subject", action="store_true", help="Verify subject-conflict proposals after coverage.")
    parser.add_argument(
        "--max-evidence-chars",
        type=int,
        default=DEFAULT_MAX_EVIDENCE_CHARS,
        help="Longest span the critics accept. Must be at least 4_span_alignment.py's --max-span-chunk-chars.",
    )
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--ray-temp-dir", default="/tmp/ray")  # noqa: S108
    args = parser.parse_args()

    workflow = CriticWorkflow(
        judge_config=args.judge_config,
        input_path=args.input_path,
        output_path=args.output_path,
        input_format=args.input_format,
        output_format=args.output_format,
        checkpoint_path=args.checkpoint_path,
        model_alias=args.model_alias,
        source_judge=args.source_judge,
        subject=args.subject,
        max_evidence_chars=args.max_evidence_chars,
    )
    with RayClient(ray_temp_dir=args.ray_temp_dir):
        try:
            workflow.run()
        finally:
            ray.shutdown()


if __name__ == "__main__":
    main()
