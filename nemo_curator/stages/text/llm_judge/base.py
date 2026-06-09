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

"""Shared implementation for LLM-backed judge stages."""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from nemo_curator.models.client.llm_client import AsyncLLMClient, GenerationConfig, LLMClient
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import DocumentBatch

from ._utils import is_missing_value, stable_json_dumps, to_plain_dict

if TYPE_CHECKING:
    import pandas as pd

    from nemo_curator.backends.base import WorkerMetadata

FailurePolicy = Literal["keep", "drop", "mark_only"]


@dataclass
class LLMJudgeResult:
    """Structured result from a single LLM judge call."""

    keep: bool
    score: float | None = None
    record_json: str = ""
    tags_json: str = ""
    raw_response: str = ""
    parse_error: str = ""
    provenance_json: str = ""


@dataclass
class LLMJudgeStage(ProcessingStage[DocumentBatch, DocumentBatch], ABC):
    """Base stage for text filters that call an existing Curator LLM client."""

    client: AsyncLLMClient | LLMClient
    model_name: str
    input_fields: list[str] = field(default_factory=lambda: ["text"])
    field_names: list[str] | None = None
    generation_config: GenerationConfig | dict | None = None
    max_chars_per_field: int | None = None
    filter: bool = True
    keep_field: str = "llm_judge_keep"
    score_field: str | None = None
    record_field: str | None = None
    tags_field: str | None = None
    raw_response_field: str | None = None
    parse_error_field: str | None = None
    provenance_field: str | None = None
    prompt_version: str = "v1"
    run_id: str | None = None
    on_failure: FailurePolicy = "keep"
    name: str = "llm_judge"

    def __post_init__(self) -> None:
        """Validate the stage configuration."""
        if self.client is None:
            msg = "client must be provided"
            raise ValueError(msg)
        self.model_name = self.model_name.strip() if self.model_name else self.model_name
        if not self.model_name:
            msg = "model_name must be provided"
            raise ValueError(msg)
        if not self.input_fields:
            msg = "input_fields must contain at least one field"
            raise ValueError(msg)
        if self.field_names is not None and len(self.field_names) != len(self.input_fields):
            msg = "field_names must match input_fields length"
            raise ValueError(msg)
        if self.max_chars_per_field is not None and self.max_chars_per_field <= 0:
            msg = "max_chars_per_field must be positive when provided"
            raise ValueError(msg)
        if self.on_failure not in {"keep", "drop", "mark_only"}:
            msg = "on_failure must be one of: keep, drop, mark_only"
            raise ValueError(msg)
        self.is_async_client = isinstance(self.client, AsyncLLMClient)

    def inputs(self) -> tuple[list[str], list[str]]:
        """Return required task attributes and dataframe columns."""
        return ["data"], list(dict.fromkeys(self.input_fields + self.extra_input_fields()))

    def outputs(self) -> tuple[list[str], list[str]]:
        """Return output task attributes and dataframe columns."""
        columns = [self.keep_field]
        for field_name in (
            self.score_field,
            self.record_field,
            self.tags_field,
            self.raw_response_field,
            self.parse_error_field,
            self.provenance_field,
        ):
            if field_name is not None:
                columns.append(field_name)
        return ["data"], list(dict.fromkeys(columns))

    def setup(self, _: WorkerMetadata | None = None) -> None:
        """Initialize the configured LLM client."""
        self.client.setup()

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        """Judge each row in a document batch and optionally filter rows."""
        df = batch.to_pandas()
        if df.empty:
            logger.info(f"Empty dataset for batch {batch.task_id}")
            return batch

        rows = df.to_dict(orient="records")
        results = self._process_async(rows) if self.is_async_client else self._process_sync(rows)
        self._write_results(df, results)

        if self.filter:
            df = df[df[self.keep_field].astype(bool)]
            if len(df) == 0:
                logger.info(f"All documents filtered out for batch {batch.task_id}")

        return DocumentBatch(
            dataset_name=batch.dataset_name,
            data=df,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )

    def extra_input_fields(self) -> list[str]:
        """Return subclass-specific input fields."""
        return []

    def format_input(self, row: dict[str, Any]) -> str:
        """Format configured input fields for a judge prompt."""
        labels = self.field_names or self.input_fields
        parts = []
        for field_name, label in zip(self.input_fields, labels, strict=True):
            value = row.get(field_name)
            if is_missing_value(value):
                continue
            text = str(value)
            if self.max_chars_per_field is not None:
                text = text[: self.max_chars_per_field]
            if text.strip():
                parts.append(f"**{label}**\n{text}")
        return "\n\n".join(parts)

    @abstractmethod
    def build_messages(self, row: dict[str, Any]) -> list[dict[str, str]] | None:
        """Build chat messages for a row.

        Return ``None`` when no model call is needed and subclass-specific
        no-call behavior should be used.
        """

    @abstractmethod
    def parse_response(
        self,
        raw_response: str,
        row: dict[str, Any],
        messages: list[dict[str, str]],
    ) -> LLMJudgeResult:
        """Parse a model response into a judge result."""

    @abstractmethod
    def no_call_result(self, row: dict[str, Any]) -> LLMJudgeResult:
        """Return a result when a row does not require an LLM call."""

    def _process_sync(self, rows: list[dict[str, Any]]) -> list[LLMJudgeResult]:
        return [self._judge_row(row) for row in rows]

    def _process_async(self, rows: list[dict[str, Any]]) -> list[LLMJudgeResult]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._judge_rows_async(rows))

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, self._judge_rows_async(rows))
            return future.result()

    async def _judge_rows_async(self, rows: list[dict[str, Any]]) -> list[LLMJudgeResult]:
        return await asyncio.gather(*(self._judge_row_async(row) for row in rows))

    def _judge_row(self, row: dict[str, Any]) -> LLMJudgeResult:
        messages = self.build_messages(row)
        if messages is None:
            result = self.no_call_result(row)
            result.provenance_json = self._provenance_json([], result.parse_error)
            return result

        raw_response = ""
        try:
            response = self.client.query_model(
                messages=messages,
                model=self.model_name,
                generation_config=self.generation_config,
            )
            raw_response = response[0] if response else ""
            result = self.parse_response(raw_response, row, messages)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"{self.name} failed for one row: {exc}")
            result = self.failure_result(str(exc))
            result.raw_response = raw_response

        result.provenance_json = self._provenance_json(messages, result.parse_error)
        if raw_response and not result.raw_response:
            result.raw_response = raw_response
        return result

    async def _judge_row_async(self, row: dict[str, Any]) -> LLMJudgeResult:
        messages = self.build_messages(row)
        if messages is None:
            result = self.no_call_result(row)
            result.provenance_json = self._provenance_json([], result.parse_error)
            return result

        raw_response = ""
        try:
            response = await self.client.query_model(
                messages=messages,
                model=self.model_name,
                generation_config=self.generation_config,
            )
            raw_response = response[0] if response else ""
            result = self.parse_response(raw_response, row, messages)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"{self.name} failed for one row: {exc}")
            result = self.failure_result(str(exc))
            result.raw_response = raw_response

        result.provenance_json = self._provenance_json(messages, result.parse_error)
        if raw_response and not result.raw_response:
            result.raw_response = raw_response
        return result

    def failure_result(self, reason: str) -> LLMJudgeResult:
        """Build a result for parse/client failures according to the stage policy."""
        keep = self.on_failure in {"keep", "mark_only"}
        return LLMJudgeResult(keep=keep, parse_error=reason)

    def _write_results(self, df: pd.DataFrame, results: list[LLMJudgeResult]) -> None:
        df[self.keep_field] = [result.keep for result in results]
        if self.score_field is not None:
            df[self.score_field] = [result.score for result in results]
        if self.record_field is not None:
            df[self.record_field] = [result.record_json for result in results]
        if self.tags_field is not None:
            df[self.tags_field] = [result.tags_json for result in results]
        if self.raw_response_field is not None:
            df[self.raw_response_field] = [result.raw_response for result in results]
        if self.parse_error_field is not None:
            df[self.parse_error_field] = [result.parse_error for result in results]
        if self.provenance_field is not None:
            df[self.provenance_field] = [result.provenance_json for result in results]

    def _provenance_json(self, messages: list[dict[str, str]], failure_reason: str = "") -> str:
        prompt_text = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        generation_config = to_plain_dict(self.generation_config)
        provenance = {
            "model_name": self.model_name,
            "client_type": type(self.client).__name__,
            "prompt_version": self.prompt_version,
            "prompt_hash": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
            "generation_config": generation_config,
            "run_id": self.run_id,
            "failure_reason": failure_reason,
        }
        return stable_json_dumps(provenance)
