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

import asyncio
import json
from collections.abc import Iterable

import pandas as pd
import pytest

from nemo_curator.models.client.llm_client import AsyncLLMClient, GenerationConfig, LLMClient
from nemo_curator.stages.text.llm_judge import (
    LLMAnalysisFilterStage,
    LLMConditionFilterStage,
    LLMTaskRelevanceFilterStage,
)
from nemo_curator.tasks import DocumentBatch


class MockSyncLLMClient(LLMClient):
    """Mock synchronous LLM client for judge stage tests."""

    def __init__(self, responses: list[list[str]] | None = None):
        self.responses = responses or [["ok"]]
        self.call_count = 0
        self.setup_called = False
        self.received_messages: list[list[dict[str, str]]] = []
        self.received_generation_configs: list[GenerationConfig | dict | None] = []

    def setup(self) -> None:
        self.setup_called = True

    def query_model(
        self,
        *,
        messages: Iterable,
        model: str,
        generation_config: GenerationConfig | dict | None = None,
        **kwargs: object,
    ) -> list[str]:
        del model, kwargs
        self.received_messages.append(list(messages))
        self.received_generation_configs.append(generation_config)
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response


class MockAsyncLLMClient(AsyncLLMClient):
    """Mock asynchronous LLM client for judge stage tests."""

    def __init__(self, responses: list[list[str]] | None = None):
        super().__init__()
        self.responses = responses or [["ok"]]
        self.call_count = 0
        self.setup_called = False
        self.received_messages: list[list[dict[str, str]]] = []

    def setup(self) -> None:
        self.setup_called = True

    async def _query_model_impl(
        self,
        *,
        messages: Iterable,
        model: str,
        generation_config: GenerationConfig | dict | None = None,
        **kwargs: object,
    ) -> list[str]:
        del model, generation_config, kwargs
        self.received_messages.append(list(messages))
        await asyncio.sleep(0)
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response


def _analysis_response(scores: dict[str, int], tags: dict[str, str] | None = None) -> str:
    return json.dumps(
        {
            "dimension_scores": scores,
            "tags": tags or {"topic": "test"},
            "flags": [],
            "rationale": "short rationale",
            "recommendation": "keep",
        }
    )


def test_analysis_stage_scores_filters_and_preserves_batch_metadata() -> None:
    client = MockSyncLLMClient(
        responses=[
            [_analysis_response({"clarity": 5, "relevance": 5, "usefulness": 5, "fluency": 5})],
            [_analysis_response({"clarity": 1, "relevance": 1, "usefulness": 1, "fluency": 1})],
        ]
    )
    metadata = {"source": "unit"}
    stage_perf = []
    batch = DocumentBatch(
        data=pd.DataFrame({"text": ["excellent", "poor"]}),
        dataset_name="ds",
        _metadata=metadata,
        _stage_perf=stage_perf,
    )
    stage = LLMAnalysisFilterStage(client=client, model_name="judge", min_score=0.8)
    stage.setup()

    out = stage.process(batch)
    df = out.to_pandas()

    assert client.setup_called is True
    assert out.dataset_name == "ds"
    assert out._metadata is metadata
    assert out._stage_perf is stage_perf
    assert df["text"].tolist() == ["excellent"]
    assert df["llm_analysis_score"].tolist() == [1.0]
    assert df["llm_analysis_keep"].tolist() == [True]
    assert json.loads(df["llm_analysis_record"].iloc[0])["recommendation"] == ["keep"]
    assert json.loads(df["llm_analysis_tags"].iloc[0]) == {"topic": "test"}
    assert json.loads(df["llm_analysis_provenance"].iloc[0])["model_name"] == "judge"
    out.to_pyarrow()


def test_analysis_stage_parse_failure_keeps_raw_response_when_policy_keeps() -> None:
    client = MockSyncLLMClient(responses=[["not json"]])
    batch = DocumentBatch(data=pd.DataFrame({"text": ["bad response"]}), dataset_name="ds")
    stage = LLMAnalysisFilterStage(
        client=client,
        model_name="judge",
        raw_response_field="llm_analysis_raw",
        on_failure="keep",
    )

    out = stage.process(batch)
    df = out.to_pandas()

    assert len(df) == 1
    assert bool(df["llm_analysis_keep"].iloc[0]) is True
    assert df["llm_analysis_raw"].iloc[0] == "not json"
    assert "JSON" in df["llm_analysis_parse_error"].iloc[0] or "json" in df["llm_analysis_parse_error"].iloc[0]


def test_analysis_stage_min_max_normalizes_scores() -> None:
    client = MockSyncLLMClient(
        responses=[
            [_analysis_response({"clarity": 1, "relevance": 1, "usefulness": 1, "fluency": 1})],
            [_analysis_response({"clarity": 5, "relevance": 5, "usefulness": 5, "fluency": 5})],
        ]
    )
    batch = DocumentBatch(data=pd.DataFrame({"text": ["low", "high"]}), dataset_name="ds")
    stage = LLMAnalysisFilterStage(client=client, model_name="judge", min_score=0.0, filter=False)

    out = stage.process(batch)

    assert out.to_pandas()["llm_analysis_score"].tolist() == [0.0, 1.0]


def test_analysis_stage_treats_nan_as_empty_input() -> None:
    client = MockSyncLLMClient()
    batch = DocumentBatch(data=pd.DataFrame({"text": [pd.NA]}), dataset_name="ds")
    stage = LLMAnalysisFilterStage(client=client, model_name="judge", filter=False)

    out = stage.process(batch)
    df = out.to_pandas()

    assert client.call_count == 0
    assert df["llm_analysis_keep"].tolist() == [False]
    assert df["llm_analysis_score"].tolist() == [0.0]
    assert df["llm_analysis_parse_error"].tolist() == ["empty input"]


def test_analysis_stage_extracts_json_after_quoted_brace_text() -> None:
    response = 'log "{not json}" ' + _analysis_response(
        {"clarity": 5, "relevance": 5, "usefulness": 5, "fluency": 5}
    )
    client = MockSyncLLMClient(responses=[[response]])
    batch = DocumentBatch(data=pd.DataFrame({"text": ["sample"]}), dataset_name="ds")
    stage = LLMAnalysisFilterStage(client=client, model_name="judge", min_score=0.0)

    out = stage.process(batch)
    df = out.to_pandas()

    assert df["llm_analysis_score"].tolist() == [1.0]
    assert df["llm_analysis_parse_error"].tolist() == [""]


def test_analysis_stage_rejects_out_of_range_dimension_scores() -> None:
    response = _analysis_response({"clarity": 6, "relevance": 5, "usefulness": 5, "fluency": 5})
    client = MockSyncLLMClient(responses=[[response]])
    batch = DocumentBatch(data=pd.DataFrame({"text": ["sample"]}), dataset_name="ds")
    stage = LLMAnalysisFilterStage(client=client, model_name="judge", on_failure="drop", filter=False)

    out = stage.process(batch)
    df = out.to_pandas()

    assert df["llm_analysis_keep"].tolist() == [False]
    assert "between 1 and 5" in df["llm_analysis_parse_error"].iloc[0]


def test_task_relevance_stage_includes_validation_context() -> None:
    client = MockSyncLLMClient(
        responses=[
            [
                _analysis_response(
                    {
                        "topical_relevance": 5,
                        "linguistic_style_match": 5,
                        "task_match": 5,
                        "knowledge_alignment": 5,
                        "potential_utility": 5,
                    }
                )
            ]
        ]
    )
    batch = DocumentBatch(data=pd.DataFrame({"text": ["Q: 1+1? A: 2"]}), dataset_name="ds")
    stage = LLMTaskRelevanceFilterStage(
        client=client,
        model_name="judge",
        task_desc="Solve arithmetic word problems.",
        validation_examples=[{"text": "Q: 2+2? A: 4"}, {"text": "Q: 3+3? A: 6"}],
        n_shot=1,
        filter=False,
    )

    out = stage.process(batch)
    user_message = client.received_messages[0][1]["content"]

    assert "Solve arithmetic word problems." in user_message
    assert "Q: 2+2? A: 4" in user_message
    assert "Q: 3+3? A: 6" not in user_message
    assert out.to_pandas()["llm_task_relevance_score"].iloc[0] == 1.0


def test_task_relevance_stage_caches_validation_context(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MockSyncLLMClient(
        responses=[
            [
                _analysis_response(
                    {
                        "topical_relevance": 5,
                        "linguistic_style_match": 5,
                        "task_match": 5,
                        "knowledge_alignment": 5,
                        "potential_utility": 5,
                    }
                )
            ]
        ]
    )
    batch = DocumentBatch(data=pd.DataFrame({"text": ["Q: 1+1? A: 2"]}), dataset_name="ds")
    stage = LLMTaskRelevanceFilterStage(
        client=client,
        model_name="judge",
        task_desc="Solve arithmetic word problems.",
        validation_examples=[{"text": "Q: 2+2? A: 4"}],
        filter=False,
    )

    def fail_format(_: dict[str, object]) -> str:
        msg = "validation context should have been cached during initialization"
        raise AssertionError(msg)

    monkeypatch.setattr(stage, "_format_validation_example", fail_format)

    out = stage.process(batch)

    assert "Q: 2+2? A: 4" in client.received_messages[0][1]["content"]
    assert out.to_pandas()["llm_task_relevance_score"].iloc[0] == 1.0


def test_task_relevance_stage_rejects_nonpositive_n_shot() -> None:
    client = MockSyncLLMClient()

    with pytest.raises(ValueError, match="n_shot"):
        LLMTaskRelevanceFilterStage(
            client=client,
            model_name="judge",
            validation_examples=[{"text": "example"}],
            n_shot=0,
        )


def test_condition_stage_handles_empty_text_and_empty_condition_without_model_call() -> None:
    client = MockSyncLLMClient(responses=[["yes"]])
    batch = DocumentBatch(data=pd.DataFrame({"text": ["", "content"]}), dataset_name="ds")

    empty_text_stage = LLMConditionFilterStage(
        client=client,
        model_name="judge",
        condition="Text is non-empty.",
        filter=False,
    )
    out = empty_text_stage.process(batch)
    df = out.to_pandas()

    assert client.call_count == 1
    assert df["llm_condition_result"].tolist() == [False, True]
    assert df["llm_condition_keep"].tolist() == [False, True]

    no_condition_client = MockSyncLLMClient()
    no_condition_stage = LLMConditionFilterStage(
        client=no_condition_client,
        model_name="judge",
        condition="",
        filter=False,
    )
    out = no_condition_stage.process(DocumentBatch(data=pd.DataFrame({"text": ["content"]}), dataset_name="ds"))

    assert no_condition_client.call_count == 0
    assert out.to_pandas()["llm_condition_result"].tolist() == [True]


def test_condition_parse_failure_result_differs_from_keep_policy() -> None:
    client = MockSyncLLMClient(responses=[["maybe"]])
    batch = DocumentBatch(data=pd.DataFrame({"text": ["unclear"]}), dataset_name="ds")
    stage = LLMConditionFilterStage(
        client=client,
        model_name="judge",
        condition="Contains a question.",
        on_failure="keep",
        filter=False,
    )

    out = stage.process(batch)
    df = out.to_pandas()

    assert df["llm_condition_keep"].tolist() == [True]
    assert df["llm_condition_result"].tolist() == [False]
    assert "yes or no" in df["llm_condition_parse_error"].iloc[0]


def test_condition_stage_rejects_ambiguous_no_prefix() -> None:
    client = MockSyncLLMClient(responses=[["not sure"]])
    batch = DocumentBatch(data=pd.DataFrame({"text": ["unclear"]}), dataset_name="ds")
    stage = LLMConditionFilterStage(
        client=client,
        model_name="judge",
        condition="Contains a question.",
        on_failure="drop",
        filter=False,
    )

    out = stage.process(batch)
    df = out.to_pandas()

    assert df["llm_condition_keep"].tolist() == [False]
    assert df["llm_condition_result"].tolist() == [False]
    assert "yes or no" in df["llm_condition_parse_error"].iloc[0]


def test_async_analysis_stage_uses_async_client() -> None:
    client = MockAsyncLLMClient(
        responses=[
            [_analysis_response({"clarity": 5, "relevance": 5, "usefulness": 5, "fluency": 5})],
            [_analysis_response({"clarity": 4, "relevance": 4, "usefulness": 4, "fluency": 4})],
        ]
    )
    batch = DocumentBatch(data=pd.DataFrame({"text": ["one", "two"]}), dataset_name="ds")
    stage = LLMAnalysisFilterStage(client=client, model_name="judge", min_score=0.0)
    stage.setup()

    out = stage.process(batch)

    assert client.setup_called is True
    assert client.call_count == 2
    assert out.to_pandas()["llm_analysis_score"].tolist() == [1.0, 0.75]
