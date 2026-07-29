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

"""Tests for the concrete ``QwenOmniASRAdapter`` internals (no GPU / no real vLLM required)."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nemo_curator.models.asr.qwen_omni import QwenOmniASRAdapter

if TYPE_CHECKING:
    from collections.abc import Iterator

_SR = 16000


class _FakeProcessor:
    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        return repr(messages)


def _fake_process_mm_info(
    messages: list[dict[str, object]],
    *,
    use_audio_in_video: bool,
) -> tuple[list[np.ndarray] | None, None, None]:
    assert use_audio_in_video is False
    audios = [
        part["audio"]
        for message in messages
        for part in message["content"]  # type: ignore[union-attr]
        if part["type"] == "audio"
    ]
    return audios or None, None, None


def _vllm_output(text: str) -> SimpleNamespace:
    return SimpleNamespace(outputs=[SimpleNamespace(text=text, token_ids=[0], finish_reason="stop")])


@contextmanager
def _mock_external_qwen_runtime(
    adapter: QwenOmniASRAdapter,
    *,
    generated_batches: list[list[SimpleNamespace]] | None = None,
) -> Iterator[tuple[MagicMock, MagicMock]]:
    llm = MagicMock()
    if generated_batches is not None:
        llm.generate.side_effect = generated_batches
    adapter._llm = llm
    adapter._sampling_params = object()
    adapter._processor = _FakeProcessor()
    with patch(
        "nemo_curator.models.asr.qwen_omni.process_mm_info",
        side_effect=_fake_process_mm_info,
    ) as mm_info:
        yield llm, mm_info


@contextmanager
def _mock_qwen_model_load(
    *, processor_side_effect: Exception | None = None
) -> Iterator[tuple[MagicMock, MagicMock, MagicMock]]:
    processor_cls = MagicMock()
    processor_cls.from_pretrained.side_effect = processor_side_effect
    with (
        patch("nemo_curator.models.asr.qwen_omni.process_mm_info", MagicMock()),
        patch(
            "nemo_curator.models.asr.qwen_omni.create_vllm_llm",
            return_value=MagicMock(),
        ) as llm_ctor,
        patch(
            "nemo_curator.models.asr.qwen_omni.Qwen3OmniMoeProcessor",
            processor_cls,
        ),
        patch("nemo_curator.models.asr.qwen_omni.SamplingParams") as sampling_ctor,
    ):
        yield llm_ctor, processor_cls.from_pretrained, sampling_ctor


@pytest.mark.parametrize("num_gpus", [0, -1, 1.5, True])
def test_qwen_adapter_load_model_requires_positive_integer_stage_gpu_count(num_gpus: object) -> None:
    adapter = QwenOmniASRAdapter(model_id="mock/qwen-omni")

    with pytest.raises(ValueError, match="requires a positive integer num_gpus"):
        adapter.load_model(num_gpus=num_gpus)  # type: ignore[arg-type]


def test_qwen_adapter_rejects_invalid_prompt_content_order() -> None:
    with pytest.raises(ValueError, match="prompt_content_order must be one of"):
        QwenOmniASRAdapter(model_id="mock/qwen-omni", prompt_content_order="invalid")


def test_qwen_adapter_infer_batch_returns_length_stopped_output() -> None:
    adapter = QwenOmniASRAdapter(model_id="mock/qwen-omni", max_output_tokens=2)
    with _mock_external_qwen_runtime(
        adapter,
        generated_batches=[
            [SimpleNamespace(outputs=[SimpleNamespace(text="partial", token_ids=[0, 1], finish_reason="length")])]
        ],
    ):
        texts = adapter._infer_batch(inputs=[{"prompt": "a"}], indices=[0], n=1)

    assert texts == ["partial"]
    assert adapter._last_output_metrics == [{"output_tokens": 2.0, "model_finish_reason_length_count": 1.0}]


def test_qwen_adapter_infer_batch_accepts_explicit_stop_at_token_cap() -> None:
    adapter = QwenOmniASRAdapter(model_id="mock/qwen-omni", max_output_tokens=2)
    with _mock_external_qwen_runtime(
        adapter,
        generated_batches=[
            [SimpleNamespace(outputs=[SimpleNamespace(text="complete", token_ids=[0, 1], finish_reason="stop")])]
        ],
    ):
        texts = adapter._infer_batch(inputs=[{"prompt": "a"}], indices=[0], n=1)

    assert texts == ["complete"]


def test_qwen_adapter_infer_batch_scatters_outputs_by_index() -> None:
    """``_infer_batch`` scatters vLLM outputs back to original positions."""
    adapter = QwenOmniASRAdapter(model_id="mock/qwen-omni")

    # Length-4 batch where only positions 1 and 3 produced valid inputs.
    with _mock_external_qwen_runtime(
        adapter,
        generated_batches=[[_vllm_output("t0"), _vllm_output("t1")]],
    ):
        texts = adapter._infer_batch(
            inputs=[{"prompt": "a"}, {"prompt": "b"}],
            indices=[1, 3],
            n=4,
        )

    assert texts == ["", "t0", "", "t1"]


def test_qwen_adapter_infer_batch_raises_on_vllm_count_mismatch() -> None:
    """A short vLLM result list must fail loud (strict=True), not silently drop utterances."""
    adapter = QwenOmniASRAdapter(model_id="mock/qwen-omni")

    with (
        _mock_external_qwen_runtime(
            adapter,
            generated_batches=[[_vllm_output("only-one")]],
        ),
        pytest.raises(ValueError, match="zip"),
    ):
        adapter._infer_batch(inputs=[{"prompt": "a"}, {"prompt": "b"}], indices=[0, 1], n=2)


def test_qwen_adapter_audio_text_prompt_order_matches_official_asr_recipe() -> None:
    adapter = QwenOmniASRAdapter(
        model_id="mock/qwen-omni",
        prompt_text="Transcribe the English audio into text.",
        prompt_content_order="audio_text",
    )
    waveform = np.zeros(_SR, dtype=np.float32)

    messages = adapter._build_messages(waveform, "English")

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0]["type"] == "audio"
    assert messages[0]["content"][0]["audio"] is waveform
    assert messages[0]["content"][1] == {
        "type": "text",
        "text": "Transcribe the English audio into text.",
    }


def test_qwen_adapter_prompt_replaces_language() -> None:
    adapter = QwenOmniASRAdapter(
        model_id="mock/qwen-omni",
        prompt_text="Transcribe {language}",
        en_prompt_text="English prompt",
    )
    waveform = np.zeros(_SR, dtype=np.float32)

    messages = adapter._build_messages(waveform, "English")

    assert messages[-1]["content"][0]["text"] == "English prompt"
    assert messages[-1]["content"][1]["audio"] is waveform


def test_qwen_adapter_transcribe_batch_packages_results() -> None:
    adapter = QwenOmniASRAdapter(model_id="mock/qwen-omni")
    items = [
        {
            "waveform": np.zeros(_SR, dtype=np.float32),
            "sample_rate": _SR,
            "language": "English",
        },
        {
            "waveform": np.zeros(_SR, dtype=np.float32),
            "sample_rate": _SR,
            "language": "English",
        },
        {"waveform": np.zeros(0, dtype=np.float32), "sample_rate": _SR, "language": None},
    ]
    with _mock_external_qwen_runtime(
        adapter,
        generated_batches=[
            [_vllm_output("text-a"), _vllm_output("text-b")],
        ],
    ) as (llm, _):
        results = adapter.transcribe_batch(items)

    assert [r.text for r in results] == ["text-a", "text-b", ""]
    assert [r.skipped for r in results] == [False, False, True]
    assert [r.extras for r in results] == [
        {"output_tokens": 1.0, "model_finish_reason_stop_count": 1.0},
        {"output_tokens": 1.0, "model_finish_reason_stop_count": 1.0},
        {},
    ]
    assert llm.generate.call_count == 1
    assert len(llm.generate.call_args_list[0].args[0]) == 2


def test_qwen_adapter_rejects_non_16khz_audio_before_inference() -> None:
    adapter = QwenOmniASRAdapter(model_id="mock/qwen-omni")
    with (
        _mock_external_qwen_runtime(adapter) as (llm, mm_info),
        pytest.raises(ValueError, match=r"requires 16000 Hz audio.*decoded at 8000 Hz"),
    ):
        adapter.transcribe_batch(
            [
                {
                    "waveform": np.zeros(8000, dtype=np.float32),
                    "sample_rate": 8000,
                }
            ]
        )

    mm_info.assert_not_called()
    llm.generate.assert_not_called()


def test_qwen_adapter_prepare_single_uses_stage_normalized_waveform_unchanged() -> None:
    adapter = QwenOmniASRAdapter(model_id="mock/qwen-omni")
    waveform = np.ones(_SR, dtype=np.float32)

    with _mock_external_qwen_runtime(adapter):
        prepared = adapter._prepare_single(waveform, "English")

    assert prepared is not None
    assert len(prepared["multi_modal_data"]["audio"]) == 1
    assert prepared["multi_modal_data"]["audio"][0] is waveform


def test_qwen_adapter_prepare_single_skips_too_short_waveform_before_preprocess() -> None:
    adapter = QwenOmniASRAdapter(model_id="mock/qwen-omni")

    with _mock_external_qwen_runtime(adapter) as (llm, mm_info):
        assert adapter._prepare_single(np.zeros(100, dtype=np.float32), "English") is None

    mm_info.assert_not_called()
    llm.generate.assert_not_called()


def test_qwen_adapter_accepts_nested_vllm_kwargs() -> None:
    """Engine knobs remain directly configurable through YAML ``adapter_kwargs``."""
    vllm_kwargs = {
        "enable_prefix_caching": False,
        "prefix_caching_hash_algo": "sha256",
        "limit_mm_per_prompt": {"image": 1, "video": 1, "audio": 1},
        "max_num_batched_tokens": 49152,
        "seed": 99,
    }
    adapter = QwenOmniASRAdapter(
        model_id="mock/qwen-omni",
        vllm_kwargs=vllm_kwargs,
    )
    assert adapter.vllm_kwargs == vllm_kwargs
    assert adapter.vllm_kwargs is not vllm_kwargs
    vllm_kwargs["limit_mm_per_prompt"]["audio"] = 9
    assert adapter.vllm_kwargs["limit_mm_per_prompt"]["audio"] == 1


def test_qwen_adapter_load_model_threads_vllm_kwargs_into_shared_llm_ctor() -> None:
    """load_model() forwards engine kwargs through Curator's shared helper."""
    adapter = QwenOmniASRAdapter(
        model_id="mock/qwen-omni",
        vllm_kwargs={
            "enable_prefix_caching": False,
            "prefix_caching_hash_algo": "sha256",
            "limit_mm_per_prompt": {"image": 1, "video": 1, "audio": 3},
            "max_num_batched_tokens": 49152,
            "seed": 42,
        },
    )
    with _mock_qwen_model_load() as (llm_ctor, _, sampling_ctor):
        adapter.load_model(num_gpus=2)

    llm_ctor.assert_called_once_with(
        "mock/qwen-omni",
        enable_prefix_caching=False,
        prefix_caching_hash_algo="sha256",
        limit_mm_per_prompt={"image": 1, "video": 1, "audio": 3},
        max_num_batched_tokens=49152,
        seed=42,
        tensor_parallel_size=2,
    )
    sampling_ctor.assert_called_once_with(
        temperature=0.0,
        top_k=1,
        repetition_penalty=1.0,
        max_tokens=256,
    )


@pytest.mark.parametrize(
    ("adapter_kwargs", "expected_sampling_kwargs"),
    [
        (
            {
                "sampling_kwargs": {"temperature": 0.01, "top_p": 0.1, "top_k": 1},
                "max_output_tokens": 8192,
            },
            {"temperature": 0.01, "top_k": 1, "max_tokens": 8192, "top_p": 0.1},
        ),
        (
            {"sampling_kwargs": {"repetition_penalty": 1.15}},
            {"max_tokens": 256, "repetition_penalty": 1.15},
        ),
    ],
)
def test_qwen_adapter_load_model_threads_sampling_kwargs(
    adapter_kwargs: dict[str, object],
    expected_sampling_kwargs: dict[str, float | int],
) -> None:
    adapter = QwenOmniASRAdapter(model_id="mock/qwen-omni", **adapter_kwargs)
    with _mock_qwen_model_load() as (_, _, sampling_ctor):
        adapter.load_model(num_gpus=1)

    sampling_ctor.assert_called_once_with(**expected_sampling_kwargs)


def test_qwen_adapter_load_model_forwards_revision_to_llm_and_processor() -> None:
    """Tier-1 revision must reach inference loaders, not only the weight download."""
    adapter = QwenOmniASRAdapter(
        model_id="mock/qwen-omni",
        revision="abc123",
    )
    with _mock_qwen_model_load() as (llm_ctor, proc_ctor, _):
        adapter.load_model(num_gpus=1)

    assert llm_ctor.call_args.kwargs["revision"] == "abc123"
    proc_ctor.assert_called_once_with("mock/qwen-omni", revision="abc123")


def test_qwen_adapter_load_model_cleans_up_partial_engine_when_processor_fails() -> None:
    adapter = QwenOmniASRAdapter(model_id="mock/qwen-omni")
    with (
        _mock_qwen_model_load(processor_side_effect=RuntimeError("processor failed")),
        pytest.raises(RuntimeError, match="processor failed"),
    ):
        adapter.load_model(num_gpus=1)

    assert adapter._llm is None
    assert adapter._sampling_params is None
    assert adapter._processor is None


def test_qwen_adapter_marks_empty_outputs_skipped() -> None:
    adapter = QwenOmniASRAdapter(model_id="mock/qwen-omni")
    waveform_a = np.ones(_SR, dtype=np.float32)
    waveform_b = np.ones(_SR, dtype=np.float32)
    with _mock_external_qwen_runtime(
        adapter,
        generated_batches=[
            [_vllm_output(""), _vllm_output("text-b")],
        ],
    ) as (llm, _):
        pred_texts, skipped_indices = adapter._run_inference(
            [waveform_a, waveform_b],
            ["English", "English"],
        )

    assert pred_texts == ["", "text-b"]
    assert skipped_indices == {0}
    assert llm.generate.call_count == 1
