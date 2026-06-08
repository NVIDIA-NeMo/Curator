"""
Unit tests for nemo_curator.stages.synthetic.nemotron_cc.base module.
"""

import asyncio
from collections.abc import Iterable

import pandas as pd

from nemo_curator.models.client.llm_client import AsyncLLMClient, GenerationConfig, LLMClient
from nemo_curator.stages.synthetic.nemotron_cc.base import BaseSyntheticStage
from nemo_curator.tasks import DocumentBatch


class MockSyncLLMClient(LLMClient):
    """Mock synchronous LLM client for testing BaseSyntheticStage."""

    def __init__(self, responses: list[list[str]] | None = None):
        self.responses = responses or [["ok"]]
        self.call_count = 0
        self.setup_called = False
        self.received_messages: list[list[dict[str, str]]] = []

    def setup(self) -> None:
        self.setup_called = True

    def query_model(
        self,
        *,
        messages: Iterable,
        model: str,
        generation_config: GenerationConfig | None = None,
        **kwargs: object,
    ) -> list[str]:
        del model, generation_config, kwargs
        msgs = list(messages)
        self.received_messages.append(msgs)
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response


class MockAsyncLLMClient(AsyncLLMClient):
    """Mock asynchronous LLM client for testing BaseSyntheticStage."""

    def __init__(self, responses: list[list[str]] | None = None, delay: float = 0.0):
        super().__init__()
        self.responses = responses or [["ok"]]
        self.call_count = 0
        self.setup_called = False
        self.delay = delay
        self.received_messages: list[list[dict[str, str]]] = []

    def setup(self) -> None:
        self.setup_called = True

    async def _query_model_impl(
        self,
        *,
        messages: Iterable,
        model: str,
        generation_config: GenerationConfig | None = None,
        **kwargs: object,
    ) -> list[str]:
        del model, generation_config, kwargs
        msgs = list(messages)
        self.received_messages.append(msgs)
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response


def test_setup_calls_client_setup() -> None:
    client = MockSyncLLMClient()
    stage = BaseSyntheticStage(
        system_prompt=None,
        prompt="Echo: {document}",
        input_field="text",
        output_field="out",
        client=client,
        model_name="test-model",
    )
    assert client.setup_called is False
    stage.setup()
    assert client.setup_called is True


def test_process_llm_prompt_and_response_defaults() -> None:
    client = MockSyncLLMClient()
    stage = BaseSyntheticStage(
        system_prompt=None,
        prompt="Echo: {document}",
        input_field="text",
        output_field="out",
        client=client,
        model_name="test-model",
    )
    # Prompt formatting
    prompt = stage._process_llm_prompt({"text": "hello"})
    assert prompt == "Echo: hello"
    # Response default behavior (first element)
    result = stage._process_llm_response(["first", "second"])
    assert result == "first"
    assert stage._process_llm_response([]) == ""


def test_process_sync_single_row_no_system_prompt() -> None:
    client = MockSyncLLMClient(responses=[["resp-1"]])
    stage = BaseSyntheticStage(
        system_prompt=None,
        prompt="Echo: {document}",
        input_field="text",
        output_field="out",
        client=client,
        model_name="test-model",
    )
    df = pd.DataFrame([{"text": "hello"}])
    batch = DocumentBatch(data=df, dataset_name="ds", task_id="t1")

    out_batch = stage.process(batch)
    assert isinstance(out_batch, DocumentBatch)
    assert out_batch.dataset_name == "ds"
    assert out_batch.task_id.endswith(stage.name)
    assert "out" in out_batch.data.columns
    assert out_batch.data["out"].iloc[0] == "resp-1"
    # Ensure user-only message when no system prompt
    assert client.call_count == 1
    msgs = client.received_messages[0]
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "Echo: hello"


def test_process_sync_with_system_prompt() -> None:
    client = MockSyncLLMClient(responses=[["ok"]])
    stage = BaseSyntheticStage(
        system_prompt="SYS",
        prompt="Doc: {document}",
        input_field="text",
        output_field="out",
        client=client,
        model_name="test-model",
    )
    df = pd.DataFrame([{"text": "abc"}])
    batch = DocumentBatch(data=df, dataset_name="ds", task_id="t2")

    out_batch = stage.process(batch)
    assert out_batch.data["out"].iloc[0] == "ok"
    # Ensure system + user messages
    msgs = client.received_messages[0]
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "SYS"
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "Doc: abc"


def test_process_async_multiple_rows() -> None:
    responses = [["a"], ["b"], ["c"]]
    client = MockAsyncLLMClient(responses=responses, delay=0.0)
    stage = BaseSyntheticStage(
        system_prompt=None,
        prompt="Q: {document}",
        input_field="text",
        output_field="out",
        client=client,
        model_name="test-model",
    )
    df = pd.DataFrame([{"text": "x"}, {"text": "y"}, {"text": "z"}])
    batch = DocumentBatch(data=df, dataset_name="ds", task_id="t3")

    out_batch = stage.process(batch)
    assert len(out_batch.data) == 3
    assert set(out_batch.data["out"].tolist()) == {"a", "b", "c"}
    assert client.call_count == 3
