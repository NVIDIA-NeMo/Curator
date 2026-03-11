"""
Unit tests for nemo_curator.stages.synthetic.nemotron_cc.nemotron_cc module.
"""

import asyncio
from collections.abc import Iterable
from unittest.mock import patch

import pandas as pd

from nemo_curator.models.client.llm_client import AsyncLLMClient, GenerationConfig, LLMClient
from nemo_curator.stages.synthetic.nemotron_cc.nemotron_cc import (
    DistillStage,
    DiverseQAPostProcessingStage,
    DiverseQAStage,
    ExtractKnowledgeStage,
    KnowledgeListPostProcessingStage,
    KnowledgeListStage,
    WikipediaParaphrasingStage,
)
from nemo_curator.tasks import DocumentBatch


class MockSyncLLMClient(LLMClient):
    """Mock synchronous LLM client."""

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
    """Mock asynchronous LLM client."""

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


def _build_diverseqa_response(prefix: str) -> str:
    # Includes prefix and "- " bullet markers to be stripped
    lines = [
        prefix,
        "- Question: Q1?",
        "- Answer: A1.",
        "- Question: Q2?",
        "- Answer: A2.",
        "- Question: Q3?",
        "- Answer: A3.",
    ]
    return "\n".join(lines)


def test_diverseqa_post_processing_basic() -> None:
    # Create batch with raw QA output and run post-processing
    pp = DiverseQAPostProcessingStage()
    generated_text = _build_diverseqa_response(pp.prefix)
    df = pd.DataFrame([{"text": "DOC", "diverse_qa": generated_text}])
    batch = DocumentBatch(data=df, dataset_name="ds", task_id="t0")
    # Deterministic behavior: no shuffle and pick 2 pairs
    with patch("nemo_curator.stages.synthetic.nemotron_cc.nemotron_cc.random.shuffle", lambda _: None), patch(
        "nemo_curator.stages.synthetic.nemotron_cc.nemotron_cc.random.randint", return_value=2
    ):
        out_batch = pp.process(batch)
    out = out_batch.data["diverse_qa"].iloc[0]
    expected = "DOC\n\nQuestion: Q1?\nAnswer: A1.\n\nQuestion: Q2?\nAnswer: A2."
    assert out == expected


def test_diverseqa_sync_end_to_end() -> None:
    raw = _build_diverseqa_response(DiverseQAPostProcessingStage.prefix)
    stage = DiverseQAStage(
        client=MockSyncLLMClient(responses=[[raw]]),
        model_name="m",
        input_field="text",
        output_field="diverse_qa",
    )
    pp = DiverseQAPostProcessingStage()
    df = pd.DataFrame([{"text": "DOC"}])
    batch = DocumentBatch(data=df, dataset_name="ds", task_id="t1")
    with patch("nemo_curator.stages.synthetic.nemotron_cc.nemotron_cc.random.shuffle", lambda _: None), patch(
        "nemo_curator.stages.synthetic.nemotron_cc.nemotron_cc.random.randint", return_value=1
    ):
        raw_batch = stage.process(batch)
        out_batch = pp.process(raw_batch)
    val = out_batch.data["diverse_qa"].iloc[0]
    assert val.startswith("DOC\n\nQuestion: Q1?")
    assert "Answer: A1." in val


def test_diverseqa_async_multiple_rows() -> None:
    resp = _build_diverseqa_response(DiverseQAPostProcessingStage.prefix)
    client = MockAsyncLLMClient(responses=[[resp], [resp], [resp]], delay=0.01)
    stage = DiverseQAStage(
        client=client,
        model_name="m",
        input_field="text",
        output_field="diverse_qa",
    )
    pp = DiverseQAPostProcessingStage()
    df = pd.DataFrame([{"text": "D1"}, {"text": "D2"}, {"text": "D3"}])
    batch = DocumentBatch(data=df, dataset_name="ds", task_id="t2")
    with patch("nemo_curator.stages.synthetic.nemotron_cc.nemotron_cc.random.shuffle", lambda _: None), patch(
        "nemo_curator.stages.synthetic.nemotron_cc.nemotron_cc.random.randint", return_value=1
    ):
        raw_batch = stage.process(batch)
        out_batch = pp.process(raw_batch)
    assert len(out_batch.data) == 3
    texts = out_batch.data["diverse_qa"].tolist()
    assert all("Question:" in t for t in texts)
    assert client.call_count == 3


def test_knowledge_list_process_llm_response() -> None:
    pp = KnowledgeListPostProcessingStage()
    # First line not starting with "-" should be skipped
    generated = (
        "Header line\n"
        "- item one\n"
        "  continuation\n"
        "- item two"
    )
    df = pd.DataFrame([{"knowledge_list": generated}])
    batch = DocumentBatch(data=df, dataset_name="ds", task_id="tkl")
    out_batch = pp.process(batch)
    assert out_batch.data["knowledge_list"].iloc[0] == "item one\ncontinuation\nitem two"


def test_wikipedia_paraphrasing_smoke() -> None:
    client = MockSyncLLMClient(responses=[["rephrased"]])
    stage = WikipediaParaphrasingStage(client=client, model_name="m")
    df = pd.DataFrame([{"text": "original"}])
    batch = DocumentBatch(data=df, dataset_name="ds", task_id="t3")
    out_batch = stage.process(batch)
    assert out_batch.data["rephrased"].iloc[0] == "rephrased"


def test_distill_stage_smoke() -> None:
    client = MockSyncLLMClient(responses=[["distilled"]])
    stage = DistillStage(client=client, model_name="m")
    df = pd.DataFrame([{"text": "doc"}])
    batch = DocumentBatch(data=df, dataset_name="ds", task_id="t4")
    out_batch = stage.process(batch)
    assert out_batch.data["distill"].iloc[0] == "distilled"
    # Ensure system prompt is present in messages
    msgs = client.received_messages[0]
    assert msgs[0]["role"] == "system"


def test_extract_knowledge_stage_smoke() -> None:
    client = MockSyncLLMClient(responses=[["facts"]])
    stage = ExtractKnowledgeStage(client=client, model_name="m")
    df = pd.DataFrame([{"text": "doc"}])
    batch = DocumentBatch(data=df, dataset_name="ds", task_id="t5")
    out_batch = stage.process(batch)
    assert out_batch.data["extract_knowledge"].iloc[0] == "facts"


def test_knowledge_list_stage_smoke() -> None:
    # Stage should pass through raw LLM output; post-processing is covered separately
    generated = "- item one\n  continuation\n- item two"
    client = MockSyncLLMClient(responses=[[generated]])
    stage = KnowledgeListStage(client=client, model_name="m")
    df = pd.DataFrame([{"text": "doc"}])
    batch = DocumentBatch(data=df, dataset_name="ds", task_id="t6")
    out_batch = stage.process(batch)
    assert out_batch.data["knowledge_list"].iloc[0] == generated
