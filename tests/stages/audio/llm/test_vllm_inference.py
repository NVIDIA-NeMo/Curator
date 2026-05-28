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

# modality: audio

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from nemo_curator.stages.audio.llm.vllm_inference import (
    DocumentToAudioStage,
    TopicExpander,
    vLLMInference,
)
from nemo_curator.tasks import AudioTask


VALID_CONVERSATION = {
    "turns": [
        {"speaker": "speaker_0", "overlap": 0.0, "utterance": "Hello, how are you today?"},
        {"speaker": "speaker_1", "overlap": -0.5, "utterance": "I am doing great, thanks for asking!"},
        {"speaker": "speaker_0", "overlap": 0.3, "utterance": "That is wonderful to hear."},
    ]
}

VALID_CONVERSATION_JSON = json.dumps(VALID_CONVERSATION)

GERMAN_CONVERSATION = {
    "turns": [
        {"speaker": "speaker_0", "overlap": 0.0, "utterance": "Hallo, wie geht es dir heute?"},
        {"speaker": "speaker_1", "overlap": -0.3, "utterance": "Mir geht es gut, danke der Nachfrage!"},
    ]
}


def _make_vllm_output(text: str):
    """Create a fake vLLM output object."""
    return SimpleNamespace(outputs=[SimpleNamespace(text=text)])


def _make_prompt_yaml(tmp_path: Path, content: str | None = None) -> Path:
    prompt_file = tmp_path / "prompt.yaml"
    if content is None:
        content = (
            'system: |\n'
            '  You are a dialogue simulator.\n'
            'user: |\n'
            '  Topic: "$topic"\n'
        )
    prompt_file.write_text(content, encoding="utf-8")
    return prompt_file


def _make_stage(tmp_path: Path, **overrides) -> vLLMInference:
    pf = _make_prompt_yaml(tmp_path)

    defaults = {
        "prompt_file": str(pf),
        "model": {"model": "test-model"},
        "inference": {"temperature": 0.7, "max_tokens": 512},
        "apply_chat_template": {"tokenize": False, "add_generation_prompt": True},
    }
    defaults.update(overrides)
    return vLLMInference(**defaults)


def _setup_stage_with_mocks(stage: vLLMInference) -> mock.MagicMock:
    """Inject a fake LLM and tokenizer, return the mock LLM."""
    fake_tokenizer = mock.MagicMock()
    fake_tokenizer.apply_chat_template.return_value = "formatted prompt"
    fake_tokenizer.decode.return_value = "formatted prompt"

    fake_llm = mock.MagicMock()

    stage.llm = fake_llm
    stage.tokenizer = fake_tokenizer

    return fake_llm


def _make_task(topic: str = "morning routine") -> AudioTask:
    return AudioTask(
        data={"topic": topic},
        task_id="test_task_1",
        dataset_name="test_ds",
    )


def _make_tasks(topics: list[str]) -> list[AudioTask]:
    return [
        AudioTask(
            data={"topic": t},
            task_id=f"test_task_{i}",
            dataset_name="test_ds",
        )
        for i, t in enumerate(topics)
    ]


class TestCleanText:
    @pytest.fixture(autouse=True)
    def _stage(self, tmp_path):
        self.stage = _make_stage(tmp_path)

    def test_empty(self):
        assert self.stage._clean_text("") == ""

    def test_normal_text(self):
        assert self.stage._clean_text("Hello world") == "Hello world"

    def test_strips_control_chars(self):
        assert self.stage._clean_text("Hello\x00world") == "Helloworld"

    def test_normalises_whitespace(self):
        assert self.stage._clean_text("Hello   world  ") == "Hello world"

    def test_unicode_zero_width(self):
        result = self.stage._clean_text("Guten\u200bTag")
        assert "\u200b" not in result

    def test_preserves_tab_space(self):
        result = self.stage._clean_text("Hello world")
        assert "Hello" in result and "world" in result


class TestIsValidText:
    @pytest.fixture(autouse=True)
    def _stage(self, tmp_path):
        self.stage = _make_stage(tmp_path)

    def test_valid(self):
        assert self.stage._is_valid_text("Hello there") is True

    def test_empty(self):
        assert self.stage._is_valid_text("") is False

    def test_single_char(self):
        assert self.stage._is_valid_text("a") is False

    def test_only_numbers(self):
        assert self.stage._is_valid_text("12345") is False

    def test_only_punctuation(self):
        assert self.stage._is_valid_text("...") is False

    def test_german_text(self):
        assert self.stage._is_valid_text("Wie geht es dir?") is True

    def test_cyrillic(self):
        assert self.stage._is_valid_text("Привет мир") is True

    def test_cjk(self):
        assert self.stage._is_valid_text("你好世界") is True


class TestGenerateConversationId:
    @pytest.fixture(autouse=True)
    def _stage(self, tmp_path):
        self.stage = _make_stage(tmp_path)

    def test_deterministic(self):
        turns = [
            {"speaker": "s0", "utterance": "hi"},
            {"speaker": "s1", "utterance": "hello"},
        ]
        id1 = self.stage.generate_conversation_id(turns)
        id2 = self.stage.generate_conversation_id(turns)
        assert id1 == id2
        assert len(id1) == 16

    def test_different_content_different_id(self):
        turns_a = [{"speaker": "s0", "utterance": "hi"}]
        turns_b = [{"speaker": "s0", "utterance": "bye"}]
        assert self.stage.generate_conversation_id(
            turns_a
        ) != self.stage.generate_conversation_id(turns_b)


class TestValidateJsonOutput:
    @pytest.fixture(autouse=True)
    def _stage(self, tmp_path):
        self.stage = _make_stage(tmp_path)

    def test_valid_json(self):
        result = self.stage.validate_json_output(VALID_CONVERSATION_JSON)
        assert result is not None
        assert len(result["turns"]) == 3

    def test_wrapped_in_markdown(self):
        text = f"```json\n{VALID_CONVERSATION_JSON}\n```"
        assert self.stage.validate_json_output(text) is not None

    def test_extra_text_around_json(self):
        text = f"Here is the conversation:\n{VALID_CONVERSATION_JSON}\nDone."
        assert self.stage.validate_json_output(text) is not None

    def test_invalid_json(self):
        assert self.stage.validate_json_output("not json at all") is None

    def test_missing_turns_key(self):
        assert self.stage.validate_json_output('{"speakers": []}') is None

    def test_too_few_turns(self):
        data = {"turns": [{"speaker": "s0", "overlap": 0, "utterance": "Hi there friend"}]}
        assert self.stage.validate_json_output(json.dumps(data)) is None

    def test_missing_speaker(self):
        data = {"turns": [
            {"overlap": 0, "utterance": "Hi there friend"},
            {"speaker": "s1", "overlap": 0, "utterance": "Hello there"},
        ]}
        assert self.stage.validate_json_output(json.dumps(data)) is None

    def test_empty_speaker(self):
        data = {"turns": [
            {"speaker": "", "overlap": 0, "utterance": "Hi there friend"},
            {"speaker": "s1", "overlap": 0, "utterance": "Hello there"},
        ]}
        assert self.stage.validate_json_output(json.dumps(data)) is None

    def test_invalid_utterance(self):
        data = {"turns": [
            {"speaker": "s0", "overlap": 0, "utterance": "..."},
            {"speaker": "s1", "overlap": 0, "utterance": "Hello there"},
        ]}
        assert self.stage.validate_json_output(json.dumps(data)) is None

    def test_invalid_overlap_type(self):
        data = {"turns": [
            {"speaker": "s0", "overlap": "fast", "utterance": "Hello there friend"},
            {"speaker": "s1", "overlap": 0, "utterance": "Hi back to you"},
        ]}
        assert self.stage.validate_json_output(json.dumps(data)) is None

    def test_cleans_utterances(self):
        data = {"turns": [
            {"speaker": "s0", "overlap": 0, "utterance": "Hello\u200b   world"},
            {"speaker": "s1", "overlap": 0, "utterance": "Hi  there   friend"},
        ]}
        result = self.stage.validate_json_output(json.dumps(data))
        assert result is not None
        assert result["turns"][0]["utterance"] == "Hello world"
        assert result["turns"][1]["utterance"] == "Hi there friend"

    def test_german_conversation(self):
        result = self.stage.validate_json_output(json.dumps(GERMAN_CONVERSATION))
        assert result is not None
        assert len(result["turns"]) == 2

    def test_nested_braces_in_utterance(self):
        data = {"turns": [
            {"speaker": "s0", "overlap": 0, "utterance": "He said {wow} amazing"},
            {"speaker": "s1", "overlap": 0, "utterance": "Really interesting stuff"},
        ]}
        assert self.stage.validate_json_output(json.dumps(data)) is not None

    def test_cyrillic_conversation(self):
        data = {"turns": [
            {"speaker": "s0", "overlap": 0, "utterance": "Привет, как дела?"},
            {"speaker": "s1", "overlap": 0, "utterance": "Все хорошо, спасибо!"},
        ]}
        assert self.stage.validate_json_output(json.dumps(data)) is not None

    def test_cjk_conversation(self):
        data = {"turns": [
            {"speaker": "s0", "overlap": 0, "utterance": "你好，最近怎么样？"},
            {"speaker": "s1", "overlap": 0, "utterance": "很好，谢谢你的关心！"},
        ]}
        assert self.stage.validate_json_output(json.dumps(data)) is not None


class TestStageConstruction:
    def test_prompt_file_mode(self, tmp_path):
        stage = _make_stage(tmp_path)
        assert stage.prompt_file is not None
        assert stage.prompt is None
        assert stage.prompt_field is None

    def test_prompt_string_mode(self):
        stage = vLLMInference(
            prompt="system:You are helpful",
            model={"model": "test"},
        )
        assert stage.prompt == "system:You are helpful"

    def test_no_prompt_raises(self):
        with pytest.raises(ValueError, match="Exactly one"):
            vLLMInference(model={"model": "test"})

    def test_multiple_prompts_raises(self, tmp_path):
        pf = _make_prompt_yaml(tmp_path)
        with pytest.raises(ValueError, match="Exactly one"):
            vLLMInference(
                prompt="system:hello",
                prompt_file=str(pf),
                model={"model": "test"},
            )

    def test_default_gpu_resources(self, tmp_path):
        stage = _make_stage(tmp_path)
        assert stage.resources.gpus == 1


class TestPromptBuilding:
    def test_prompt_file_topic_substitution(self, tmp_path):
        stage = _make_stage(tmp_path)
        _setup_stage_with_mocks(stage)

        stage.get_entry_prompt({"topic": "morning routine"})
        call_args = stage.tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]
        assert any("morning routine" in m["content"] for m in messages)

    def test_prompt_string_mode(self):
        stage = vLLMInference(
            prompt="system:Generate a conversation",
            model={"model": "test"},
        )
        _setup_stage_with_mocks(stage)

        stage.get_entry_prompt({})
        call_args = stage.tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Generate a conversation"

    def test_prompt_field_mode(self):
        stage = vLLMInference(
            prompt_field="my_prompt",
            model={"model": "test"},
        )
        _setup_stage_with_mocks(stage)

        entry = {"my_prompt": "user:Tell me about dogs"}
        stage.get_entry_prompt(entry)
        call_args = stage.tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]
        assert messages[0]["role"] == "user"

    def test_prompt_field_missing_raises(self):
        stage = vLLMInference(
            prompt_field="missing_field",
            model={"model": "test"},
        )
        _setup_stage_with_mocks(stage)
        with pytest.raises(ValueError, match="not found"):
            stage.get_entry_prompt({"other_key": "value"})

    def test_tokenizer_returns_ids_decoded(self, tmp_path):
        stage = _make_stage(tmp_path)
        _setup_stage_with_mocks(stage)
        stage.tokenizer.apply_chat_template.return_value = [1, 2, 3, 4]
        stage.tokenizer.decode.return_value = "decoded prompt string"

        prompt = stage.get_entry_prompt({"topic": "test"})
        stage.tokenizer.decode.assert_called_once()
        assert prompt == "decoded prompt string"

    def test_empty_topic_uses_empty_string(self, tmp_path):
        stage = _make_stage(tmp_path)
        _setup_stage_with_mocks(stage)

        stage.get_entry_prompt({})
        call_args = stage.tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "$topic" not in user_msg["content"]

    def test_multilingual_topic_substitution(self, tmp_path):
        content = (
            'system: |\n'
            '  Sie sind ein Dialogsimulator.\n'
            'user: |\n'
            '  Thema: "$topic"\n'
        )
        pf = _make_prompt_yaml(tmp_path, content)
        stage = vLLMInference(
            prompt_file=str(pf),
            model={"model": "test"},
        )
        _setup_stage_with_mocks(stage)

        stage.get_entry_prompt({"topic": "Wanderung planen"})
        call_args = stage.tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "Wanderung planen" in user_msg["content"]
        sys_msg = next(m for m in messages if m["role"] == "system")
        assert "Dialogsimulator" in sys_msg["content"]


class TestBatchGenerationWithRetry:
    def test_all_succeed_first_round(self, tmp_path):
        stage = _make_stage(tmp_path)
        fake_llm = _setup_stage_with_mocks(stage)
        fake_llm.generate.return_value = [
            _make_vllm_output(VALID_CONVERSATION_JSON),
            _make_vllm_output(json.dumps(GERMAN_CONVERSATION)),
        ]

        results = stage.generate_batch_with_retry(fake_llm, ["p1", "p2"])
        assert len(results) == 2
        assert all(r is not None for r in results)
        assert fake_llm.generate.call_count == 1

    def test_retry_on_failure(self, tmp_path):
        stage = _make_stage(tmp_path)
        fake_llm = _setup_stage_with_mocks(stage)
        fake_llm.generate.side_effect = [
            [_make_vllm_output(VALID_CONVERSATION_JSON), _make_vllm_output("invalid")],
            [_make_vllm_output(json.dumps(GERMAN_CONVERSATION))],
        ]

        results = stage.generate_batch_with_retry(fake_llm, ["p1", "p2"])
        assert results[0] is not None
        assert results[1] is not None
        assert fake_llm.generate.call_count == 2

    def test_all_fail(self, tmp_path):
        stage = _make_stage(tmp_path)
        fake_llm = _setup_stage_with_mocks(stage)
        fake_llm.generate.return_value = [_make_vllm_output("bad output")]

        results = stage.generate_batch_with_retry(fake_llm, ["p1"], max_retry_rounds=2)
        assert results == [None]
        assert fake_llm.generate.call_count == 2

    def test_empty_prompts(self, tmp_path):
        stage = _make_stage(tmp_path)
        fake_llm = _setup_stage_with_mocks(stage)

        results = stage.generate_batch_with_retry(fake_llm, [])
        assert results == []
        fake_llm.generate.assert_not_called()


class TestProcessBatch:
    def test_successful_generation(self, tmp_path):
        stage = _make_stage(tmp_path)
        fake_llm = _setup_stage_with_mocks(stage)
        fake_llm.generate.return_value = [
            _make_vllm_output(VALID_CONVERSATION_JSON),
        ]

        tasks = _make_tasks(["morning routine"])
        results = stage.process_batch(tasks)

        assert len(results) == 3  # 3 turns
        assert all(isinstance(r, AudioTask) for r in results)
        assert results[0].data["turn_index"] == 0
        assert results[2].data["turn_index"] == 2
        assert results[0].data["topic"] == "morning routine"
        conv_id = results[0].data["conversation_id"]
        assert all(r.data["conversation_id"] == conv_id for r in results)

    def test_empty_batch(self, tmp_path):
        stage = _make_stage(tmp_path)
        _setup_stage_with_mocks(stage)

        results = stage.process_batch([])
        assert results == []

    def test_partial_failure(self, tmp_path):
        stage = _make_stage(tmp_path)
        fake_llm = _setup_stage_with_mocks(stage)
        call_count = {"n": 0}

        def _generate_side_effect(prompts, _sp):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return [
                    _make_vllm_output(VALID_CONVERSATION_JSON),
                    _make_vllm_output("invalid json garbage"),
                ]
            return [_make_vllm_output("still invalid") for _ in prompts]

        fake_llm.generate.side_effect = _generate_side_effect

        tasks = _make_tasks(["topic A", "topic B"])
        results = stage.process_batch(tasks)

        # Only first conversation succeeded → 3 turns
        assert len(results) == 3
        assert all(r.data["topic"] == "topic A" for r in results)

    def test_all_failures(self, tmp_path):
        stage = _make_stage(tmp_path)
        fake_llm = _setup_stage_with_mocks(stage)
        fake_llm.generate.return_value = [
            _make_vllm_output("bad"),
            _make_vllm_output("also bad"),
        ]

        tasks = _make_tasks(["A", "B"])
        results = stage.process_batch(tasks)
        assert results == []

    def test_multiple_conversations(self, tmp_path):
        stage = _make_stage(tmp_path)
        fake_llm = _setup_stage_with_mocks(stage)
        fake_llm.generate.return_value = [
            _make_vllm_output(VALID_CONVERSATION_JSON),
            _make_vllm_output(json.dumps(GERMAN_CONVERSATION)),
        ]

        tasks = _make_tasks(["english topic", "german topic"])
        results = stage.process_batch(tasks)

        # 3 turns from english + 2 turns from german = 5 AudioTasks
        assert len(results) == 5
        english_turns = [r for r in results if r.data["topic"] == "english topic"]
        german_turns = [r for r in results if r.data["topic"] == "german topic"]
        assert len(english_turns) == 3
        assert len(german_turns) == 2

    def test_output_has_conversation_id(self, tmp_path):
        stage = _make_stage(tmp_path)
        fake_llm = _setup_stage_with_mocks(stage)
        fake_llm.generate.return_value = [
            _make_vllm_output(VALID_CONVERSATION_JSON),
        ]

        tasks = _make_tasks(["test"])
        results = stage.process_batch(tasks)
        conv_id = results[0].data["conversation_id"]
        assert conv_id and len(conv_id) == 16
        assert all(r.data["conversation_id"] == conv_id for r in results)
        assert "conv_" in results[0].task_id

    def test_each_turn_has_unique_task_id(self, tmp_path):
        stage = _make_stage(tmp_path)
        fake_llm = _setup_stage_with_mocks(stage)
        fake_llm.generate.return_value = [
            _make_vllm_output(VALID_CONVERSATION_JSON),
        ]

        tasks = _make_tasks(["test"])
        results = stage.process_batch(tasks)
        task_ids = [r.task_id for r in results]
        assert len(set(task_ids)) == len(task_ids)


class TestProcessSingle:
    def test_process_delegates_to_process_batch(self, tmp_path):
        stage = _make_stage(tmp_path)
        fake_llm = _setup_stage_with_mocks(stage)
        fake_llm.generate.return_value = [
            _make_vllm_output(VALID_CONVERSATION_JSON),
        ]

        task = _make_task("morning routine")
        results = stage.process(task)

        assert len(results) == 3
        assert all(isinstance(r, AudioTask) for r in results)
        assert results[0].data["topic"] == "morning routine"


class TestSetupLifecycle:
    def test_setup_initialises_llm(self, tmp_path):
        stage = _make_stage(tmp_path)

        with (
            mock.patch("vllm.LLM") as mock_llm_cls,
            mock.patch("transformers.AutoTokenizer") as mock_tok_cls,
        ):
            mock_llm_cls.return_value = mock.MagicMock()
            mock_tok_cls.from_pretrained.return_value = mock.MagicMock()

            stage.setup()

            mock_llm_cls.assert_called_once_with(model="test-model")
            mock_tok_cls.from_pretrained.assert_called_once_with(
                "test-model", trust_remote_code=True
            )
            assert stage.llm is not None
            assert stage.tokenizer is not None

    def test_lazy_init_on_process_batch(self, tmp_path):
        stage = _make_stage(tmp_path)
        assert stage.llm is None

        with (
            mock.patch("vllm.LLM") as mock_llm_cls,
            mock.patch("transformers.AutoTokenizer") as mock_tok_cls,
        ):
            mock_llm = mock.MagicMock()
            mock_llm.generate.return_value = [
                _make_vllm_output(VALID_CONVERSATION_JSON)
            ]
            mock_llm_cls.return_value = mock_llm
            mock_tok = mock.MagicMock()
            mock_tok.apply_chat_template.return_value = "prompt"
            mock_tok_cls.from_pretrained.return_value = mock_tok

            tasks = _make_tasks(["test"])
            stage.process_batch(tasks)
            assert stage.llm is not None


class TestDocumentToAudioStage:
    def test_converts_document_batch(self):
        import pandas as pd
        from nemo_curator.tasks import DocumentBatch

        df = pd.DataFrame([{"topic": "hello"}, {"topic": "world"}])
        doc_batch = DocumentBatch(
            data=df, task_id="doc_1", dataset_name="test"
        )

        stage = DocumentToAudioStage()
        results = stage.process(doc_batch)

        assert len(results) == 2
        assert all(isinstance(r, AudioTask) for r in results)
        assert results[0].data["topic"] == "hello"
        assert results[1].data["topic"] == "world"
        assert results[0].task_id == "doc_1_0"
        assert results[1].task_id == "doc_1_1"

    def test_converts_audio_task_passthrough(self):
        task = AudioTask(
            data={"topic": "test"}, task_id="at_1", dataset_name="ds"
        )
        stage = DocumentToAudioStage()
        results = stage.process(task)

        assert len(results) == 1
        assert isinstance(results[0], AudioTask)
        assert results[0].data["topic"] == "test"


class TestTopicExpander:
    def test_expands_single_topic(self):
        stage = TopicExpander(num_conversations=5, seed=42)
        task = AudioTask(
            data={"topic": "morning routine"},
            task_id="t",
            dataset_name="ds",
        )

        results = stage.process(task)

        assert len(results) == 5
        assert all(isinstance(r, AudioTask) for r in results)
        assert all(r.data["topic"] == "morning routine" for r in results)
        assert all("conversation_index" in r.data for r in results)

    def test_batch_expands_multiple_topics(self):
        stage = TopicExpander(num_conversations=10, seed=42)
        tasks = _make_tasks(["A", "B", "C"])

        results = stage.process_batch(tasks)

        assert len(results) == 10
        topics_used = {r.data["topic"] for r in results}
        assert topics_used.issubset({"A", "B", "C"})

    def test_no_topic_returns_empty(self):
        stage = TopicExpander(num_conversations=5)
        task = AudioTask(
            data={"no_topic_key": "value"},
            task_id="t",
            dataset_name="ds",
        )

        results = stage.process(task)
        assert results == []

    def test_seed_reproducibility(self):
        tasks = _make_tasks(["A", "B", "C"])

        stage1 = TopicExpander(num_conversations=20, seed=123)
        results1 = stage1.process_batch(tasks)
        topics1 = [r.data["topic"] for r in results1]

        stage2 = TopicExpander(num_conversations=20, seed=123)
        results2 = stage2.process_batch(tasks)
        topics2 = [r.data["topic"] for r in results2]

        assert topics1 == topics2

    def test_xenna_stage_spec(self):
        stage = TopicExpander(num_conversations=10)
        assert stage.xenna_stage_spec() == {"num_workers": 1}

    def test_conversation_indices_sequential(self):
        stage = TopicExpander(num_conversations=5)
        task = AudioTask(
            data={"topic": "test"},
            task_id="t",
            dataset_name="ds",
        )
        results = stage.process(task)
        indices = [r.data["conversation_index"] for r in results]
        assert indices == [0, 1, 2, 3, 4]
