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

# ruff: noqa: INP001

import json
from collections.abc import Callable

import httpx
import pytest
from openai import AsyncOpenAI

from nemo_curator.models.client import openai_client
from nemo_curator.stages.audio.text_filtering.audio_text_llm import AudioTextLLMStage
from nemo_curator.stages.audio.text_filtering.text_windows import split_text_windows
from nemo_curator.tasks import AudioTask


def _sentences(n: int, words: int = 5) -> str:
    return " ".join(" ".join(f"s{i}w{j}" for j in range(words - 1)) + f" s{i}end." for i in range(n))


# ---------------------------------------------------------------- split_text_windows


@pytest.mark.parametrize("text", [_sentences(40), _sentences(3), "  a  b\n c.  d ", "no punctuation " * 100])
def test_windows_are_lossless(text: str) -> None:
    windows = split_text_windows(text, 12, 20)
    assert " ".join(windows) == " ".join(text.split())
    assert all(1 <= len(w.split()) <= 20 for w in windows)


def test_windows_break_at_sentence_ends() -> None:
    windows = split_text_windows(_sentences(10), 12, 20)
    assert all(w.endswith(".") for w in windows)
    # greedy packing: 5-word sentences fill to >= 12 words (3 sentences), never past 20
    assert [len(w.split()) for w in windows] == [15, 15, 15, 5]


def test_unpunctuated_text_is_split_at_clauses_then_hard_cap() -> None:
    clauses = " ".join(["one two three four five six seven eight,"] * 5)
    assert all(w.endswith(",") for w in split_text_windows(clauses, 10, 12))
    flat = " ".join(f"w{i}" for i in range(50))
    assert [len(w.split()) for w in split_text_windows(flat, 10, 14)] == [14, 14, 14, 8]


def test_short_and_empty_text() -> None:
    assert split_text_windows("Hello there.", 250) == ["Hello there."]
    assert split_text_windows("   ", 250) == []


def test_default_cap_and_validation() -> None:
    assert max(len(w.split()) for w in split_text_windows("x " * 100, 10)) == 14
    with pytest.raises(ValueError, match="target_words"):
        split_text_windows("a", 0)
    with pytest.raises(ValueError, match="max_words"):
        split_text_windows("a", 10, 5)


# ---------------------------------------------------------------- AudioTextLLMStage windows


def _completion(content: str | None, finish_reason: str = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "c",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [
                {"index": 0, "finish_reason": finish_reason, "message": {"role": "assistant", "content": content}}
            ],
        },
    )


@pytest.fixture
def endpoint(monkeypatch: pytest.MonkeyPatch) -> Callable:
    def install(handler: Callable) -> None:
        def make_client(**kwargs) -> AsyncOpenAI:
            return AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

        monkeypatch.setattr(openai_client, "AsyncOpenAI", make_client)

    return install


def _stage(**kwargs) -> AudioTextLLMStage:
    return AudioTextLLMStage(
        **{
            "endpoint": "http://dynamo.test/v1",
            "model_name": "test-model",
            "prompt": "Rewrite.",
            "text_key": "text",
            "output_text_key": "out",
            "window_target_words": 12,
            "window_max_words": 20,
            **kwargs,
        }
    )


def _user_text(request: httpx.Request) -> str:
    return json.loads(request.content)["messages"][-1]["content"]


def test_each_window_is_one_request_and_outputs_join_in_order(endpoint: Callable) -> None:
    seen = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(_user_text(request))
        return _completion(_user_text(request).upper())

    endpoint(handle)
    text = _sentences(10)
    task = AudioTask(data={"text": text})
    stage = _stage()
    stage.process(task)
    stage.teardown()

    assert sorted(seen) == sorted(split_text_windows(text, 12, 20))
    assert task.data["out"] == text.upper()
    assert task.data["additional_notes"][stage.name] == "applied (modified); 4 windows"


@pytest.mark.parametrize(
    ("content", "finish_reason", "reason"), [("CUT", "length", "truncated"), ("", "stop", "empty_output")]
)
def test_bad_window_keeps_its_input_and_others_apply(
    endpoint: Callable, content: str, finish_reason: str, reason: str
) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        window = _user_text(request)
        if window.startswith("s3w0"):
            return _completion(content, finish_reason)
        return _completion(window.upper())

    endpoint(handle)
    text = _sentences(10)
    windows = split_text_windows(text, 12, 20)
    task = AudioTask(data={"text": text})
    stage = _stage()
    stage.process(task)
    stage.teardown()

    expected = " ".join(w if w.startswith("s3w0") else w.upper() for w in windows)
    assert task.data["out"] == expected
    assert task.data["additional_notes"][stage.name] == f"applied (modified); fallback 1/4 windows ({reason})"


def test_short_text_is_a_single_request(endpoint: Callable) -> None:
    seen = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(_user_text(request))
        return _completion("Hi.")

    endpoint(handle)
    task = AudioTask(data={"text": "hi"})
    stage = _stage()
    stage.process(task)
    stage.teardown()
    assert seen == ["hi"]
    assert task.data["out"] == "Hi."


def test_language_id_sends_only_the_middle_window(endpoint: Callable) -> None:
    seen = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(_user_text(request))
        return _completion("English")

    endpoint(handle)
    text = _sentences(10)
    task = AudioTask(data={"text": text, "source_lang": "en"})
    stage = _stage(mode="language_id")
    stage.process(task)
    stage.teardown()
    windows = split_text_windows(text, 12, 20)
    assert seen == [windows[len(windows) // 2]]
    assert task.data["out"] == "English"


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"window_target_words": 0}, ValueError),
        ({"window_target_words": True}, TypeError),
        ({"window_target_words": 10, "window_max_words": 5}, ValueError),
        ({"window_target_words": None, "window_max_words": 5}, ValueError),
        ({"mode": "recover_entities", "ground_truth_key": "gt"}, ValueError),
    ],
)
def test_window_config_is_validated(options: dict, error: type[Exception]) -> None:
    with pytest.raises(error):
        _stage(**options)


def test_default_max_is_forty_percent_over_target() -> None:
    assert _stage(window_max_words=None, window_target_words=250).window_max_words == 350
