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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from nemo_curator.models.asr.base import ASRAdapter
from nemo_curator.models.asr.qwen_omni import QwenOmniASRAdapter

_SR = 16000


# ----------------------------------------------------------------------
# Protocol conformance (requires @runtime_checkable)
# ----------------------------------------------------------------------


def test_qwen_adapter_conforms_to_asr_protocol() -> None:
    """QwenOmniASRAdapter satisfies the ASRAdapter contract (requires @runtime_checkable)."""
    adapter = QwenOmniASRAdapter(model_id="mock/qwen-omni")
    assert isinstance(adapter, ASRAdapter)


# ----------------------------------------------------------------------
# QwenOmniASRAdapter helpers (no GPU, no vLLM required)
# ----------------------------------------------------------------------


def test_qwen_adapter_first_output_text_handles_empty_vllm_output() -> None:
    assert QwenOmniASRAdapter._first_output_text(SimpleNamespace(outputs=[])) == ""


def test_qwen_adapter_count_output_tokens_handles_empty_vllm_output() -> None:
    assert QwenOmniASRAdapter._count_output_tokens([SimpleNamespace(outputs=[])]) == 0.0


def test_qwen_adapter_infer_turn_scatters_outputs_by_index() -> None:
    """``_infer_turn`` scatters vLLM outputs back to original positions and reports time + tokens."""
    adapter = QwenOmniASRAdapter(model_id="mock/qwen-omni")

    def _fake_generate(inputs: list[dict[str, object]]) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(outputs=[SimpleNamespace(text=f"t{i}", token_ids=[0, 1])]) for i, _ in enumerate(inputs)
        ]

    adapter._generate = _fake_generate  # type: ignore[method-assign]

    # Length-4 batch where only positions 1 and 3 produced valid inputs.
    texts, generation_s, tokens = adapter._infer_turn(
        inputs=[{"prompt": "a"}, {"prompt": "b"}],
        indices=[1, 3],
        n=4,
    )

    assert texts == ["", "t0", "", "t1"]
    assert tokens == 4.0  # 2 outputs x 2 token_ids each
    assert generation_s >= 0.0


def test_qwen_adapter_infer_turn_raises_on_vllm_count_mismatch() -> None:
    """A short vLLM result list must fail loud (strict=True), not silently drop utterances."""
    adapter = QwenOmniASRAdapter(model_id="mock/qwen-omni")

    def _short_generate(_inputs: list[dict[str, object]]) -> list[SimpleNamespace]:
        # vLLM returns fewer outputs than inputs (e.g. a scheduler drop).
        return [SimpleNamespace(outputs=[SimpleNamespace(text="only-one", token_ids=[0])])]

    adapter._generate = _short_generate  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="zip"):
        adapter._infer_turn(inputs=[{"prompt": "a"}, {"prompt": "b"}], indices=[0, 1], n=2)


def test_qwen_adapter_turn2_extends_shared_audio_prompt_messages() -> None:
    adapter = QwenOmniASRAdapter(
        model_id="mock/qwen-omni",
        prompt_text="Transcribe {language}.",
        followup_prompt="Refine {language}.",
        system_prompt="System {language}.",
    )
    waveform = np.zeros(_SR, dtype=np.float32)

    turn1_messages = adapter._build_messages(waveform, "English")
    turn2_messages = adapter._build_turn2_messages(waveform, "draft text", "English")

    assert [message["role"] for message in turn2_messages[:2]] == [message["role"] for message in turn1_messages]
    assert turn2_messages[0]["content"][0]["text"] == turn1_messages[0]["content"][0]["text"]
    assert turn2_messages[1]["content"][0]["text"] == turn1_messages[1]["content"][0]["text"]
    assert turn2_messages[1]["content"][1]["audio"] is waveform
    assert turn2_messages[2] == {"role": "assistant", "content": [{"type": "text", "text": "draft text"}]}
    assert turn2_messages[3] == {"role": "user", "content": [{"type": "text", "text": "Refine English."}]}


def test_qwen_adapter_prompt_replaces_language_and_reference_transcript() -> None:
    adapter = QwenOmniASRAdapter(
        model_id="mock/qwen-omni",
        prompt_text="Transcribe {language}: {transcript}",
        en_prompt_text="English prompt {transcript}",
        followup_prompt="Refine {language}: {transcript}",
    )
    waveform = np.zeros(_SR, dtype=np.float32)

    turn1_messages = adapter._build_messages(waveform, "English", "hello reference")
    turn2_messages = adapter._build_turn2_messages(waveform, "draft", "Spanish", "hola ref")

    assert turn1_messages[-1]["content"][0]["text"] == "English prompt hello reference"
    assert turn2_messages[0]["content"][0]["text"] == "Transcribe Spanish: hola ref"
    assert turn2_messages[2]["content"][0]["text"] == "Refine Spanish: hola ref"


def test_qwen_adapter_transcribe_batch_packages_results() -> None:
    adapter = QwenOmniASRAdapter(model_id="mock/qwen-omni", followup_prompt="refine")
    adapter._run_two_turn = MagicMock(  # type: ignore[method-assign]
        return_value=(
            ["text-a", "text-b", ""],
            ["refined-a", "", ""],
            {2},
        ),
    )
    items = [
        {
            "waveform": np.zeros(_SR, dtype=np.float32),
            "sample_rate": _SR,
            "language": "English",
            "reference_text": "ref-a",
        },
        {
            "waveform": np.zeros(_SR, dtype=np.float32),
            "sample_rate": _SR,
            "language": "English",
            "reference_text": "ref-b",
        },
        {"waveform": np.zeros(0, dtype=np.float32), "sample_rate": _SR, "language": None},
    ]
    results = adapter.transcribe_batch(items)

    assert [r.text for r in results] == ["text-a", "text-b", ""]
    assert [r.secondary_text for r in results] == ["refined-a", "", ""]
    assert [r.skipped for r in results] == [False, False, True]
    assert all(r.model_id == "mock/qwen-omni" for r in results)

    adapter._run_two_turn.assert_called_once()
    _waveforms, _srs, langs, refs = adapter._run_two_turn.call_args[0]
    assert langs == ["English", "English", None]
    assert refs == ["ref-a", "ref-b", None]


def test_qwen_adapter_single_turn_drops_secondary_text() -> None:
    adapter = QwenOmniASRAdapter(model_id="mock/qwen-omni", followup_prompt=None)
    adapter._run_two_turn = MagicMock(  # type: ignore[method-assign]
        return_value=(["text-a"], [""], set()),
    )
    results = adapter.transcribe_batch(
        [
            {"waveform": np.zeros(_SR, dtype=np.float32), "sample_rate": _SR},
        ]
    )
    assert results[0].secondary_text is None


def test_qwen_adapter_prepare_single_accepts_canonical_torch_2d_waveform() -> None:
    adapter = QwenOmniASRAdapter(model_id="mock/qwen-omni")
    adapter._build_messages = MagicMock(return_value=[{"role": "user", "content": []}])  # type: ignore[method-assign]
    adapter._pack_vllm_inputs = MagicMock(return_value={"prompt": "p"})  # type: ignore[method-assign]
    waveform = torch.stack([torch.ones(_SR), torch.zeros(_SR)])

    prepared = adapter._prepare_single(waveform, _SR, "English")

    assert prepared is not None
    inputs, waveform_16k = prepared
    assert inputs == {"prompt": "p"}
    assert waveform_16k.shape == (_SR,)
    assert waveform_16k.dtype == np.float32
    np.testing.assert_allclose(waveform_16k, np.full(_SR, 0.5, dtype=np.float32))


def test_qwen_adapter_prepare_single_skips_too_short_waveform_before_preprocess() -> None:
    adapter = QwenOmniASRAdapter(model_id="mock/qwen-omni")
    adapter._build_messages = MagicMock(return_value=[{"role": "user", "content": []}])  # type: ignore[method-assign]
    adapter._pack_vllm_inputs = MagicMock(return_value={"prompt": "p"})  # type: ignore[method-assign]

    assert adapter._prepare_single(np.zeros(100, dtype=np.float32), _SR, "English") is None
    adapter._build_messages.assert_not_called()
    adapter._pack_vllm_inputs.assert_not_called()


# ----------------------------------------------------------------------
# Elevated vLLM knobs
# ----------------------------------------------------------------------


def test_qwen_adapter_has_elevated_vllm_knobs_as_dataclass_fields() -> None:
    """vLLM knobs are dataclass fields settable from YAML ``adapter_kwargs``."""
    adapter = QwenOmniASRAdapter(
        model_id="mock/qwen-omni",
        enable_prefix_caching=False,
        prefix_caching_hash_algo="sha256",
        limit_mm_per_prompt_audio=1,
        max_num_batched_tokens=49152,
        seed=99,
    )
    assert adapter.enable_prefix_caching is False
    assert adapter.prefix_caching_hash_algo == "sha256"
    assert adapter.limit_mm_per_prompt_audio == 1
    assert adapter.max_num_batched_tokens == 49152
    assert adapter.seed == 99


def test_qwen_adapter_vllm_knob_defaults_match_doc() -> None:
    """Default vLLM knob values match the tutorial when YAML omits overrides."""
    adapter = QwenOmniASRAdapter(model_id="mock/qwen-omni")
    assert adapter.enable_prefix_caching is True
    assert adapter.prefix_caching_hash_algo == "xxhash"
    assert adapter.limit_mm_per_prompt_audio == 2
    assert adapter.max_num_batched_tokens is None
    assert adapter.seed == 1234


def test_qwen_adapter_rejects_invalid_max_num_batched_tokens() -> None:
    with pytest.raises(ValueError, match="max_num_batched_tokens must be positive"):
        QwenOmniASRAdapter(model_id="mock/qwen-omni", max_num_batched_tokens=0)


def test_qwen_adapter_setup_threads_vllm_knobs_into_llm_ctor() -> None:
    """setup() forwards the elevated knobs to the vLLM LLM ctor."""
    adapter = QwenOmniASRAdapter(
        model_id="mock/qwen-omni",
        enable_prefix_caching=False,
        prefix_caching_hash_algo="sha256",
        limit_mm_per_prompt_audio=3,
        max_num_batched_tokens=49152,
        seed=42,
        tensor_parallel_size=1,
    )
    fake_llm = MagicMock()
    fake_processor = MagicMock()
    with (
        patch("nemo_curator.models.asr.qwen_omni.VLLM_AVAILABLE", new=True),
        patch("nemo_curator.models.asr.qwen_omni.process_mm_info", MagicMock()),
        patch("nemo_curator.models.vllm_model.LLM", return_value=fake_llm) as llm_ctor,
        patch(
            "nemo_curator.models.asr.qwen_omni.Qwen3OmniMoeProcessor.from_pretrained",
            return_value=fake_processor,
        ),
        patch("nemo_curator.models.vllm_model.SamplingParams"),
    ):
        adapter.setup()

    llm_ctor.assert_called_once()
    kwargs = llm_ctor.call_args.kwargs
    assert kwargs["enable_prefix_caching"] is False
    assert kwargs["prefix_caching_hash_algo"] == "sha256"
    assert kwargs["limit_mm_per_prompt"] == {"image": 1, "video": 1, "audio": 3}
    assert kwargs["max_num_batched_tokens"] == 49152
    assert kwargs["seed"] == 42
    assert "revision" not in kwargs


def test_qwen_adapter_setup_forwards_revision_to_llm_and_processor() -> None:
    """Tier-1 revision must reach inference loaders, not only prefetch_weights."""
    adapter = QwenOmniASRAdapter(
        model_id="mock/qwen-omni",
        revision="abc123",
        tensor_parallel_size=1,
    )
    fake_llm = MagicMock()
    fake_processor = MagicMock()
    with (
        patch("nemo_curator.models.asr.qwen_omni.VLLM_AVAILABLE", new=True),
        patch("nemo_curator.models.asr.qwen_omni.process_mm_info", MagicMock()),
        patch("nemo_curator.models.vllm_model.LLM", return_value=fake_llm) as llm_ctor,
        patch(
            "nemo_curator.models.asr.qwen_omni.Qwen3OmniMoeProcessor.from_pretrained",
            return_value=fake_processor,
        ) as proc_ctor,
        patch("nemo_curator.models.vllm_model.SamplingParams"),
    ):
        adapter.setup()

    assert llm_ctor.call_args.kwargs["revision"] == "abc123"
    proc_ctor.assert_called_once_with("mock/qwen-omni", revision="abc123")


def test_qwen_adapter_setup_cleans_up_partial_engine_when_processor_fails() -> None:
    adapter = QwenOmniASRAdapter(model_id="mock/qwen-omni", tensor_parallel_size=1)
    fake_llm = MagicMock()
    with (
        patch("nemo_curator.models.asr.qwen_omni.VLLM_AVAILABLE", new=True),
        patch("nemo_curator.models.asr.qwen_omni.process_mm_info", MagicMock()),
        patch("nemo_curator.models.vllm_model.LLM", return_value=fake_llm),
        patch(
            "nemo_curator.models.asr.qwen_omni.Qwen3OmniMoeProcessor.from_pretrained",
            side_effect=RuntimeError("processor failed"),
        ),
        patch("nemo_curator.models.vllm_model.SamplingParams"),
        pytest.raises(RuntimeError, match="processor failed"),
    ):
        adapter.setup()

    assert adapter._llm is None
    assert adapter._sampling_params is None
    assert adapter._processor is None
    assert adapter._prep_pool is None


def test_qwen_adapter_marks_empty_turn1_outputs_skipped_and_excludes_turn2() -> None:
    adapter = QwenOmniASRAdapter(model_id="mock/qwen-omni", followup_prompt="refine")
    waveform_a = np.ones(_SR, dtype=np.float32)
    waveform_b = np.ones(_SR, dtype=np.float32)
    adapter._prepare_batch = MagicMock(  # type: ignore[method-assign]
        return_value=[
            ({"prompt": "a"}, waveform_a),
            ({"prompt": "b"}, waveform_b),
        ],
    )
    adapter._prepare_turn2_batch = MagicMock(return_value=[{"prompt": "turn2-b"}])  # type: ignore[method-assign]
    adapter._infer_turn = MagicMock(  # type: ignore[method-assign]
        side_effect=[
            (["", "text-b"], 0.1, 2.0),
            (["", "refined-b"], 0.2, 3.0),
        ],
    )

    pred_texts, disfluency_texts, skipped_indices = adapter._run_two_turn(
        [waveform_a, waveform_b],
        [_SR, _SR],
        ["English", "English"],
    )

    assert pred_texts == ["", "text-b"]
    assert disfluency_texts == ["", "refined-b"]
    assert skipped_indices == {0}
    adapter._prepare_turn2_batch.assert_called_once_with([waveform_b], ["text-b"], ["English"], [None])
    assert adapter.last_metrics["utterances_skipped_empty_output"] == 1.0
