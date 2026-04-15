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

"""Text-only LLM request stage for punctuation, capitalization, and text refinement.

Sends text-only messages (no audio/image) to an LLM via the OpenAI API and
stores the response under a configurable output column. Supports both inline
prompts and YAML prompt configs (same format as the transcription cascade).

This is the stage used for Pass 3 (PnC) of the cascade and for standalone
punctuation/capitalization.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd
from loguru import logger

from nemo_curator.stages.audio.request.prompt_template import (
    build_prompt_conversation,
    load_prompt_config,
)
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch

if TYPE_CHECKING:
    from nemo_curator.models.client import AsyncLLMClient, LLMClient


@dataclass
class TextOnlyLLMRequestStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Send text-only prompts to an LLM and store the response.

    Two prompt modes:

    1. **Inline**: Set ``system_prompt`` and ``user_prompt_template`` directly.
       Placeholders ``{field_name}`` in the template are interpolated from row data.
    2. **YAML config**: Set ``prompt_config_path`` to a YAML file with the cascade
       prompt format (``input_fields``, ``conversation``, ``output_field``).

    Args:
        client: ``LLMClient`` or ``AsyncLLMClient`` instance.
        model_name: Model name for the API.
        generation_config: Optional generation parameters (temperature, etc.).
        prompt_config_path: Path to YAML prompt config (mode 2).
        system_prompt: System message (mode 1, ignored if YAML provided).
        user_prompt_template: User message template with ``{field}`` placeholders.
        input_text_key: Key in row data for the input text (mode 1 fallback).
        output_text_key: Key for the LLM response in output data.
    """

    client: Any = None  # LLMClient | AsyncLLMClient
    model_name: str = ""
    generation_config: dict[str, Any] = field(default_factory=dict)
    prompt_config_path: str = ""
    system_prompt: str = ""
    user_prompt_template: str = "{input_text}"
    input_text_key: str = "input_text"
    output_text_key: str = "predicted_text"

    name: str = "TextOnlyLLMRequest"
    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self) -> None:
        from nemo_curator.models.client import AsyncLLMClient

        self._prompt_cfg: dict[str, Any] | None = None
        if self.prompt_config_path:
            self._prompt_cfg = load_prompt_config(self.prompt_config_path)
            if "output_field" in self._prompt_cfg:
                self.output_text_key = self._prompt_cfg["output_field"]
        self._is_async = isinstance(self.client, AsyncLLMClient)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.output_text_key]

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        df = batch.to_pandas() if hasattr(batch, "to_pandas") else batch.data
        results: list[str] = []

        for _, row in df.iterrows():
            messages = self._build_messages(row.to_dict())
            try:
                text = self._query(messages)
            except Exception as e:
                logger.warning(f"LLM request failed: {e}")
                text = ""
            results.append(text)

        df = df.copy()
        df[self.output_text_key] = results
        return DocumentBatch(data=df, dataset_name=batch.dataset_name, task_id=batch.task_id)

    def _build_messages(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        """Build OpenAI-style messages from row data."""
        if self._prompt_cfg is not None:
            return build_prompt_conversation(row, self._prompt_cfg)

        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        import re
        placeholders = re.findall(r"\{([^{}]+)\}", self.user_prompt_template)
        values = {k: str(row.get(k, "")) for k in placeholders}
        user_text = self.user_prompt_template.format(**values)
        messages.append({"role": "user", "content": user_text})
        return messages

    def _query(self, messages: list[dict[str, Any]]) -> str:
        """Query the LLM client (sync or async)."""
        if self._is_async:
            return self._query_async(messages)
        response = self.client.query_model(
            model=self.model_name,
            messages=messages,
            **self.generation_config,
        )
        return self._clean_response(response)

    def _query_async(self, messages: list[dict[str, Any]]) -> str:
        async def _run() -> str:
            response = await self.client.query_model(
                model=self.model_name,
                messages=messages,
                **self.generation_config,
            )
            return self._clean_response(response)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_run())
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as pool:
            return loop.run_in_executor(pool, asyncio.run, _run())

    @staticmethod
    def _clean_response(response: Any) -> str:
        if isinstance(response, list):
            response = response[0] if response else ""
        return str(response).strip().replace("**", "")
