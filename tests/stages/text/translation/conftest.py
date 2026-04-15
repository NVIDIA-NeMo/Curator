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

"""Shared test fixtures for the translation pipeline test suite."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
import pytest

from nemo_curator.models.client.llm_client import AsyncLLMClient, GenerationConfig
from nemo_curator.tasks import DocumentBatch


class MockAsyncLLMClient(AsyncLLMClient):
    """Mock LLM client that returns bracketed translations or FAITH JSON scores.

    Behaviour:
    - If the system message mentions "FAITH" or "evaluating the quality",
      returns a valid FAITH JSON score string.
    - Otherwise, returns the last 50 characters of the user message wrapped
      in〘…〙 brackets (simulating a translation response).
    """

    def __init__(self) -> None:
        super().__init__(max_concurrent_requests=5, max_retries=0, base_delay=0.0)

    def setup(self) -> None:
        pass

    async def _query_model_impl(
        self,
        *,
        messages: Iterable,
        model: str,  # noqa: ARG002
        conversation_formatter=None,  # noqa: ARG002
        generation_config: GenerationConfig | dict | None = None,  # noqa: ARG002
    ) -> list[str]:
        messages_list = list(messages)
        system_msg = ""
        user_msg = ""
        for msg in messages_list:
            if msg.get("role") == "system":
                system_msg = msg.get("content", "")
            elif msg.get("role") == "user":
                user_msg = msg.get("content", "")

        # Detect FAITH evaluation requests
        if "evaluating the quality" in system_msg.lower() or "faith" in system_msg.lower():
            return [
                '{"Fluency": 4.0, "Accuracy": 4.5, "Idiomaticity": 3.5, '
                '"Terminology": 4.0, "Handling_of_Format": 5.0}'
            ]

        # Default: return a bracketed mock translation
        snippet = user_msg[-50:] if len(user_msg) > 50 else user_msg
        return [f"\u3018mock translation of: {snippet}\u3019"]


@pytest.fixture()
def mock_client() -> MockAsyncLLMClient:
    """Return a fresh ``MockAsyncLLMClient`` instance."""
    return MockAsyncLLMClient()


@pytest.fixture()
def sample_batch() -> DocumentBatch:
    """A small ``DocumentBatch`` suitable for pipeline integration tests.

    Contains two documents:
    1. A multi-line document with a code block.
    2. A simple single-sentence document.
    """
    df = pd.DataFrame(
        {
            "text": [
                "Hello world. This is a test.\n```python\nprint('hi')\n```\nGoodbye.",
                "Simple sentence.",
            ],
            "id": [1, 2],
        }
    )
    return DocumentBatch(data=df, dataset_name="test", task_id="1")


@pytest.fixture()
def messages_batch() -> DocumentBatch:
    """A ``DocumentBatch`` with an OpenAI-style ``messages`` column.

    Each row's ``messages`` value is a JSON-serialized list of dicts with
    ``role`` and ``content`` keys, representing the canonical chat format
    used by Speaker and other tooling.
    """
    import json

    messages_doc1 = json.dumps([
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I am fine, thank you."},
    ])
    messages_doc2 = json.dumps([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about NLP."},
        {"role": "assistant", "content": "NLP stands for Natural Language Processing."},
    ])
    df = pd.DataFrame(
        {
            "messages": [messages_doc1, messages_doc2],
            "id": [10, 20],
        }
    )
    return DocumentBatch(data=df, dataset_name="messages-test", task_id="1")


@pytest.fixture()
def batch_with_existing_translations() -> DocumentBatch:
    """A ``DocumentBatch`` simulating partially-translated data for skip/resume tests.

    Row 0 already has a ``translated_text`` value (simulating a prior run).
    Row 1 has an empty ``translated_text`` (needs translation).
    Row 2 has no ``translated_text`` at all (NaN -- needs translation).
    """
    import numpy as np

    df = pd.DataFrame(
        {
            "text": [
                "Already translated document.",
                "Needs translation.",
                "Also needs translation.",
            ],
            "translated_text": [
                "Bereits uebersetztes Dokument.",
                "",
                np.nan,
            ],
            "id": [100, 200, 300],
        }
    )
    return DocumentBatch(data=df, dataset_name="resume-test", task_id="1")
