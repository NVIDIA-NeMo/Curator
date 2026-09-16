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

"""LLM task relevance filter stage."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._utils import is_missing_value
from .analysis import DEFAULT_ANALYSIS_USER_TEMPLATE, LLMAnalysisFilterStage

DEFAULT_TASK_RELEVANCE_DIMENSIONS = [
    "topical_relevance",
    "linguistic_style_match",
    "task_match",
    "knowledge_alignment",
    "potential_utility",
]

DEFAULT_TASK_RELEVANCE_SYSTEM_PROMPT = """You are evaluating whether a training sample is useful for a downstream task.
Return only a JSON object with these keys:
- dimension_scores: object with numeric 1-5 scores for topical_relevance, linguistic_style_match, task_match, knowledge_alignment, and potential_utility.
- tags: object or list of short labels.
- flags: list of short issue labels.
- rationale: short explanation grounded only in the provided data and validation context.
Focus on alignment with the task and validation examples, not general writing quality."""


@dataclass
class LLMTaskRelevanceFilterStage(LLMAnalysisFilterStage):
    """Use an LLM to score sample relevance to a downstream validation task."""

    task_desc: str | None = None
    validation_examples: list[dict[str, Any]] | None = None
    validation_examples_path: str | None = None
    n_shot: int | None = None
    allow_empty_validation_context: bool = False
    min_score: float = 0.5
    max_score: float = 1.0
    dimension_keys: list[str] = field(default_factory=lambda: list(DEFAULT_TASK_RELEVANCE_DIMENSIONS))
    system_prompt: str = DEFAULT_TASK_RELEVANCE_SYSTEM_PROMPT
    input_template: str = DEFAULT_ANALYSIS_USER_TEMPLATE
    keep_field: str = "llm_task_relevance_keep"
    score_field: str | None = "llm_task_relevance_score"
    record_field: str | None = "llm_task_relevance_record"
    tags_field: str | None = "llm_task_relevance_tags"
    parse_error_field: str | None = "llm_task_relevance_parse_error"
    provenance_field: str | None = "llm_task_relevance_provenance"
    name: str = "llm_task_relevance_filter"
    _validation_context: str = field(init=False, repr=False, default="")

    def __post_init__(self) -> None:
        """Load validation examples and validate configuration."""
        if self.task_desc is not None:
            self.task_desc = self.task_desc.strip()

        if self.n_shot is not None and self.n_shot <= 0:
            msg = "n_shot must be positive when provided"
            raise ValueError(msg)

        if self.validation_examples_path is not None:
            loaded_examples = self._load_validation_examples(self.validation_examples_path)
            self.validation_examples = (self.validation_examples or []) + loaded_examples

        if self.validation_examples is not None:
            self._validate_validation_examples(self.validation_examples)

        if not self.allow_empty_validation_context and not self.task_desc and not self.validation_examples:
            msg = "Provide task_desc, validation_examples, or set allow_empty_validation_context=True"
            raise ValueError(msg)

        super().__post_init__()
        self._validation_context = self._build_validation_context()

    def build_messages(self, row: dict[str, Any]) -> list[dict[str, str]] | None:
        """Build the task relevance prompt for one row."""
        data = self.format_input(row)
        if not data:
            return None
        prompt = self.validation_context
        prompt += self.input_template.format(data=data)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

    @property
    def validation_context(self) -> str:
        """Return cached task description and validation examples as prompt context."""
        return self._validation_context

    def _build_validation_context(self) -> str:
        """Build task description and validation examples as prompt context."""
        blocks = []
        if self.task_desc:
            blocks.append(f"# Task Description\n{self.task_desc}")

        examples = self.validation_examples or []
        if examples:
            limit = self.n_shot if self.n_shot is not None else len(examples)
            formatted_examples = []
            for example in examples[:limit]:
                formatted = self._format_validation_example(example)
                if formatted:
                    formatted_examples.append(formatted)
            if formatted_examples:
                blocks.append("# Validation Examples\n" + "\n\n".join(formatted_examples))

        return ("\n\n".join(blocks) + "\n\n") if blocks else ""

    def _format_validation_example(self, example: dict[str, Any]) -> str:
        labels = self.field_names or self.input_fields
        parts = []
        for field_name, label in zip(self.input_fields, labels, strict=True):
            if field_name in example and not is_missing_value(example[field_name]):
                parts.append(f"**{label}**\n{example[field_name]}")
        return "'''\n" + "\n\n".join(parts) + "\n'''" if parts else ""

    @staticmethod
    def _load_validation_examples(path: str) -> list[dict[str, Any]]:
        source = Path(path)
        if not source.exists():
            msg = f"validation_examples_path does not exist: {path}"
            raise FileNotFoundError(msg)

        text = source.read_text(encoding="utf-8").strip()
        if not text:
            return []

        if source.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in text.splitlines() if line.strip()]

        loaded = json.loads(text)
        if isinstance(loaded, list):
            return loaded
        if isinstance(loaded, dict):
            return [loaded]
        msg = "validation examples must be a JSON object, JSON list, or JSONL file"
        raise ValueError(msg)

    @staticmethod
    def _validate_validation_examples(examples: list[dict[str, Any]]) -> None:
        for idx, example in enumerate(examples):
            if not isinstance(example, dict):
                msg = f"validation example {idx} must be a JSON object"
                raise TypeError(msg)
