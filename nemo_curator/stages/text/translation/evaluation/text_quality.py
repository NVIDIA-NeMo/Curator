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

"""Text quality metrics for translated and backtranslated text."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch


def compute_text_quality_metric(
    hypothesis: str,
    reference: str,
    metric_type: str,
    threshold: float,
) -> tuple[float, bool]:
    """Compute one text quality metric and its pass/fail flag."""
    try:
        import sacrebleu
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "sacrebleu is required for round-trip quality metrics. "
            "Install with: pip install sacrebleu"
        ) from exc

    references = [reference]
    if metric_type == "sacrebleu":
        score = sacrebleu.sentence_bleu(hypothesis, references).score
        return score, score >= threshold
    if metric_type == "chrf":
        score = sacrebleu.sentence_chrf(hypothesis, references).score
        return score, score >= threshold
    if metric_type == "ter":
        score = sacrebleu.sentence_ter(hypothesis, references).score
        return score, score <= threshold
    raise ValueError(f"Unsupported round-trip quality metric: {metric_type}")


@dataclass
class TextQualityMetricStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Compute quality metrics for two text columns."""

    name: str = "TextQualityMetricStage"
    reference_text_field: str = "text"
    hypothesis_text_field: str = "backtranslated_text"
    metrics: list[dict[str, Any]] = field(default_factory=list)
    filter_enabled: bool = False
    pass_column: str = "is_quality_metric_passed"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.reference_text_field, self.hypothesis_text_field]

    def outputs(self) -> tuple[list[str], list[str]]:
        columns = [self.pass_column]
        for metric in self.metrics:
            metric_type = str(metric["type"])
            columns.extend([f"score_{metric_type}", f"score_{metric_type}_passed"])
        return ["data"], columns

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas().copy()

        if not self.metrics:
            df[self.pass_column] = True
            return DocumentBatch(
                task_id=batch.task_id,
                dataset_name=batch.dataset_name,
                data=df,
                _metadata=batch._metadata,
                _stage_perf=batch._stage_perf,
            )

        passed_columns: list[str] = []
        for metric in self.metrics:
            metric_type = str(metric["type"])
            threshold = float(metric["threshold"])
            score_column = f"score_{metric_type}"
            passed_column = f"score_{metric_type}_passed"

            scores: list[float] = []
            passed: list[bool] = []
            for _, row in df.iterrows():
                score, did_pass = compute_text_quality_metric(
                    hypothesis=str(row[self.hypothesis_text_field]),
                    reference=str(row[self.reference_text_field]),
                    metric_type=metric_type,
                    threshold=threshold,
                )
                scores.append(score)
                passed.append(did_pass)

            df[score_column] = scores
            df[passed_column] = passed
            passed_columns.append(passed_column)

        df[self.pass_column] = df[passed_columns].all(axis=1) if passed_columns else True
        if self.filter_enabled:
            df = df[df[self.pass_column]].reset_index(drop=True)

        return DocumentBatch(
            task_id=batch.task_id,
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )
