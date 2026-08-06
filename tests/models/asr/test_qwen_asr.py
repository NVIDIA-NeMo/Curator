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

"""Tests for the Qwen3-ASR implementation of the shared ASR adapter."""

from __future__ import annotations

import inspect
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from nemo_curator.models.asr.base import ASRAdapter
from nemo_curator.models.asr.qwen_asr import _MIN_SAMPLES, QwenASRAdapter, _patch_transformers_compat
from nemo_curator.stages.audio.inference.asr.stage import ASRStage
from nemo_curator.tasks import AudioTask

_MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
_SAMPLE_RATE = 16_000
_FIXTURE_PATH = Path(__file__).parents[2] / "fixtures/audio/qwen_omni/audio_1_5s_16khz_mono.wav"


def _item(
    samples: int = _SAMPLE_RATE,
    *,
    sample_rate: int = _SAMPLE_RATE,
    language: str | None = None,
) -> dict[str, object]:
    return {
        "waveform": np.zeros(samples, dtype=np.float32),
        "sample_rate": sample_rate,
        "audio_seconds": (float(samples) / float(sample_rate)) if sample_rate else 0.0,
        "language": language,
    }


def _mock_model(texts: list[str], languages: list[str] | None = None) -> MagicMock:
    langs = languages if languages is not None else [""] * len(texts)
    model = MagicMock()
    model.transcribe.return_value = [
        SimpleNamespace(text=text, language=lang) for text, lang in zip(texts, langs, strict=True)
    ]
    return model


def _adapter_with_model(model: MagicMock) -> QwenASRAdapter:
    adapter = QwenASRAdapter()
    adapter._model = model
    return adapter


def test_qwen_adapter_conforms_to_asr_protocol() -> None:
    assert isinstance(QwenASRAdapter(), ASRAdapter)


def test_qwen_adapter_default_checkpoint() -> None:
    assert QwenASRAdapter().model_id == "Qwen/Qwen3-ASR-0.6B"


def test_qwen_adapter_rejects_empty_model_id() -> None:
    with pytest.raises(ValueError, match="model_id must be non-empty"):
        QwenASRAdapter(model_id="")


@pytest.mark.parametrize("utilization", [0.0, -0.1, 1.5])
def test_qwen_adapter_rejects_out_of_range_gpu_utilization(utilization: float) -> None:
    with pytest.raises(ValueError, match="gpu_memory_utilization must be in"):
        QwenASRAdapter(gpu_memory_utilization=utilization)


@pytest.mark.parametrize("max_new_tokens", [0, -1, 1.5, True])
def test_qwen_adapter_rejects_invalid_max_new_tokens(max_new_tokens: object) -> None:
    with pytest.raises(ValueError, match="max_new_tokens must be a positive integer"):
        QwenASRAdapter(max_new_tokens=max_new_tokens)  # type: ignore[arg-type]


@pytest.mark.parametrize("max_inference_batch_size", [0, -1, 1.5, True])
def test_qwen_adapter_rejects_invalid_inference_batch_size(max_inference_batch_size: object) -> None:
    with pytest.raises(ValueError, match="max_inference_batch_size must be a positive integer"):
        QwenASRAdapter(max_inference_batch_size=max_inference_batch_size)  # type: ignore[arg-type]


def test_qwen_adapter_copies_nested_vllm_kwargs() -> None:
    vllm_kwargs = {"compilation_config": {"cudagraph_mode": "NONE"}}

    adapter = QwenASRAdapter(vllm_kwargs=vllm_kwargs)
    vllm_kwargs["compilation_config"]["cudagraph_mode"] = "FULL"

    assert adapter.vllm_kwargs == {"compilation_config": {"cudagraph_mode": "NONE"}}


@pytest.mark.parametrize(
    "reserved_key",
    [
        "model",
        "revision",
        "gpu_memory_utilization",
        "max_inference_batch_size",
        "max_new_tokens",
        "trust_remote_code",
        "enforce_eager",
        "enable_prefix_caching",
        "prefix_caching_hash_algo",
    ],
)
def test_qwen_adapter_rejects_adapter_owned_vllm_kwargs(reserved_key: str) -> None:
    adapter = QwenASRAdapter(vllm_kwargs={reserved_key: object()})

    with pytest.raises(ValueError, match="cannot override adapter-owned arguments"):
        adapter.load_model(num_gpus=1)


def test_download_weights_on_node_downloads_snapshot_without_constructing_model() -> None:
    with patch("nemo_curator.models.asr.qwen_asr.snapshot_download") as snapshot_download:
        QwenASRAdapter.download_weights_on_node("Qwen/Qwen3-ASR-0.6B", "abc123")
    snapshot_download.assert_called_once_with("Qwen/Qwen3-ASR-0.6B", revision="abc123")


@pytest.mark.parametrize("num_gpus", [0, -1, 2, 1.5, True])
def test_load_model_requires_exactly_one_integer_gpu(num_gpus: object) -> None:
    adapter = QwenASRAdapter()

    with pytest.raises(ValueError, match="requires exactly one integer GPU"):
        adapter.load_model(num_gpus=num_gpus)  # type: ignore[arg-type]


def test_load_model_builds_one_worker_local_model_and_is_idempotent() -> None:
    model_cls = MagicMock()
    adapter = QwenASRAdapter(max_inference_batch_size=64, max_new_tokens=256)

    with patch("nemo_curator.models.asr.qwen_asr._qwen_asr_model_cls", return_value=model_cls):
        adapter.load_model(num_gpus=1)
        adapter.load_model(num_gpus=1)

    model_cls.LLM.assert_called_once()
    kwargs = model_cls.LLM.call_args.kwargs
    assert kwargs["model"] == "Qwen/Qwen3-ASR-0.6B"
    assert kwargs["gpu_memory_utilization"] == 0.7
    assert kwargs["max_inference_batch_size"] == 64
    assert kwargs["max_new_tokens"] == 256
    assert kwargs["trust_remote_code"] is True
    assert kwargs["enforce_eager"] is True
    assert kwargs["enable_prefix_caching"] is True
    assert kwargs["prefix_caching_hash_algo"] == "xxhash"


def test_load_model_forwards_gpu_utilization_to_vllm() -> None:
    model_cls = MagicMock()
    adapter = QwenASRAdapter(gpu_memory_utilization=0.75)

    with patch("nemo_curator.models.asr.qwen_asr._qwen_asr_model_cls", return_value=model_cls):
        adapter.load_model(num_gpus=1)

    assert model_cls.LLM.call_args.kwargs["gpu_memory_utilization"] == 0.75


def test_load_model_forwards_revision() -> None:
    model_cls = MagicMock()
    adapter = QwenASRAdapter(revision="abc123")

    with patch("nemo_curator.models.asr.qwen_asr._qwen_asr_model_cls", return_value=model_cls):
        adapter.load_model(num_gpus=1)

    assert model_cls.LLM.call_args.kwargs["revision"] == "abc123"


def test_load_model_forwards_additional_vllm_kwargs() -> None:
    model_cls = MagicMock()
    adapter = QwenASRAdapter(vllm_kwargs={"max_model_len": 8192})

    with patch("nemo_curator.models.asr.qwen_asr._qwen_asr_model_cls", return_value=model_cls):
        adapter.load_model(num_gpus=1)

    assert model_cls.LLM.call_args.kwargs["max_model_len"] == 8192


def test_load_model_applies_nkoluguri_transformers_compat_patch() -> None:
    model_cls = MagicMock()
    adapter = QwenASRAdapter()

    with (
        patch("nemo_curator.models.asr.qwen_asr._patch_transformers_compat") as compat,
        patch("nemo_curator.models.asr.qwen_asr._qwen_asr_model_cls", return_value=model_cls),
    ):
        adapter.load_model(num_gpus=1)

    compat.assert_called_once_with()


def test_transformers_compat_wraps_plain_decorator_for_factory_syntax() -> None:
    def plain_decorator(func):  # noqa: ANN001, ANN202
        return func

    transformers_stub = SimpleNamespace(check_model_inputs=plain_decorator)
    with patch("nemo_curator.models.asr.qwen_asr.transformers", transformers_stub):
        _patch_transformers_compat()

    def decorated() -> str:
        return "ok"

    assert transformers_stub.check_model_inputs()(decorated)() == "ok"
    assert transformers_stub.check_model_inputs(decorated)() == "ok"


def test_load_model_without_qwen_asr_names_required_extras() -> None:
    adapter = QwenASRAdapter()
    with (
        patch.dict("sys.modules", {"qwen_asr": None}),
        pytest.raises(ImportError, match="audio_cuda12 and vllm"),
    ):
        adapter.load_model(num_gpus=1)


def test_load_model_cleans_up_after_model_construction_failure() -> None:
    model_cls = MagicMock()
    model_cls.LLM.side_effect = RuntimeError("construction failed")
    adapter = QwenASRAdapter()

    with (
        patch("nemo_curator.models.asr.qwen_asr._qwen_asr_model_cls", return_value=model_cls),
        patch.object(adapter, "unload_model", wraps=adapter.unload_model) as unload_model,
        pytest.raises(RuntimeError, match="construction failed"),
    ):
        adapter.load_model(num_gpus=1)

    unload_model.assert_called_once_with()
    assert adapter._model is None


def test_unload_model_releases_the_model() -> None:
    adapter = _adapter_with_model(_mock_model(["x"]))
    adapter.unload_model()
    assert adapter._model is None


def test_transcribe_batch_uses_one_exact_transcribe_call() -> None:
    model = _mock_model(["one", "two"])
    adapter = _adapter_with_model(model)

    results = adapter.transcribe_batch([_item(), _item()])

    assert model.transcribe.call_count == 1
    assert [r.text for r in results] == ["one", "two"]
    assert all(r.skipped is False for r in results)
    assert len(model.transcribe.call_args.kwargs["audio"]) == 2


def test_transcribe_batch_returns_empty_for_empty_input() -> None:
    adapter = _adapter_with_model(_mock_model([]))
    assert adapter.transcribe_batch([]) == []


def test_transcribe_batch_preserves_skipped_positions() -> None:
    """A too-short row keeps its slot so the caller's 1:1 mapping survives."""
    model = _mock_model(["only valid"])
    adapter = _adapter_with_model(model)

    results = adapter.transcribe_batch([_item(_MIN_SAMPLES - 1), _item()])

    assert results[0].skipped is True
    assert results[0].text == ""
    assert results[1].skipped is False
    assert results[1].text == "only valid"
    assert len(model.transcribe.call_args.kwargs["audio"]) == 1


def test_transcribe_batch_skips_rows_without_a_sample_rate() -> None:
    model = _mock_model(["kept"])
    adapter = _adapter_with_model(model)

    results = adapter.transcribe_batch([_item(sample_rate=0), _item()])

    assert [r.skipped for r in results] == [True, False]


def test_transcribe_batch_returns_all_skipped_when_nothing_is_usable() -> None:
    model = _mock_model([])
    adapter = _adapter_with_model(model)

    results = adapter.transcribe_batch([_item(_MIN_SAMPLES - 1), _item(10)])

    assert [r.skipped for r in results] == [True, True]
    model.transcribe.assert_not_called()


def test_transcribe_batch_forwards_language_per_item() -> None:
    model = _mock_model(["a", "b"])
    adapter = _adapter_with_model(model)

    adapter.transcribe_batch([_item(language="English"), _item(language="Spanish")])

    assert model.transcribe.call_args.kwargs["language"] == ["English", "Spanish"]


def test_transcribe_batch_reports_detected_language_in_extras() -> None:
    model = _mock_model(["hola"], languages=["Spanish"])
    adapter = _adapter_with_model(model)

    results = adapter.transcribe_batch([_item()])

    assert results[0].extras["detected_language"] == "Spanish"


def test_transcribe_batch_omits_detected_language_when_absent() -> None:
    model = _mock_model(["hi"], languages=[""])
    adapter = _adapter_with_model(model)

    results = adapter.transcribe_batch([_item()])

    assert results[0].extras == {}


@pytest.mark.parametrize("empty_text", ["", "   ", None])
def test_transcribe_batch_marks_empty_outputs_skipped(empty_text: str | None) -> None:
    model = MagicMock()
    model.transcribe.return_value = [SimpleNamespace(text=empty_text, language="English")]
    adapter = _adapter_with_model(model)

    result = adapter.transcribe_batch([_item()])[0]

    assert result.text == ("" if empty_text is None else empty_text)
    assert result.skipped is True


def test_transcribe_batch_rejects_output_count_mismatch() -> None:
    model = _mock_model(["only one"])
    adapter = _adapter_with_model(model)

    with pytest.raises(RuntimeError, match="1 transcriptions for 2 valid inputs"):
        adapter.transcribe_batch([_item(), _item()])


def test_transcribe_batch_accepts_plain_string_outputs() -> None:
    model = MagicMock()
    model.transcribe.return_value = ["plain text"]
    adapter = _adapter_with_model(model)

    assert adapter.transcribe_batch([_item()])[0].text == "plain text"


def test_asr_stage_drives_qwen_adapter_end_to_end() -> None:
    model_cls = MagicMock()
    model_cls.LLM.return_value = _mock_model(
        ["one", "two"],
        languages=["English", "Spanish"],
    )
    stage = ASRStage(
        adapter_target="nemo_curator.models.asr.qwen_asr.QwenASRAdapter",
        model_id="Qwen/Qwen3-ASR-0.6B",
        batch_size=2,
        waveform_key="waveform",
        sample_rate_key="sample_rate",
        extras_key="asr_extras",
    )

    with (
        patch("hydra.utils.get_class", return_value=QwenASRAdapter),
        patch("nemo_curator.models.asr.qwen_asr._qwen_asr_model_cls", return_value=model_cls),
    ):
        stage.setup()

    tasks = [
        AudioTask(data={"waveform": np.zeros(_SAMPLE_RATE), "sample_rate": _SAMPLE_RATE, "source_lang": "en"}),
        AudioTask(data={"waveform": np.zeros(2 * _SAMPLE_RATE), "sample_rate": _SAMPLE_RATE, "source_lang": "es"}),
    ]
    results = stage.process_batch(tasks)

    assert [task.data["pred_text"] for task in results] == ["one", "two"]
    assert [task.data["asr_extras"] for task in results] == [
        {"detected_language": "English"},
        {"detected_language": "Spanish"},
    ]
    # The stage resolved ISO codes to the names the adapter forwards to Qwen.
    assert model_cls.LLM.return_value.transcribe.call_args.kwargs["language"] == ["English", "Spanish"]


def _load_short_fixture() -> np.ndarray:
    """Decode Curator's bundled five-second, 16 kHz mono WAV."""
    with wave.open(str(_FIXTURE_PATH), "rb") as wav_file:
        assert wav_file.getframerate() == _SAMPLE_RATE
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getnframes() == 5 * _SAMPLE_RATE
        pcm = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype="<i2")
    return np.ascontiguousarray(pcm.astype(np.float32) / 32768.0)


def _require_real_qwen_stack() -> type:
    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        pytest.fail(f"The Qwen3-ASR GPU test environment is missing qwen-asr: {exc}")
    return Qwen3ASRModel


@pytest.mark.gpu
def test_qwen_asr_real_package_api_contract() -> None:
    """Exercise the external API surface that mocked CPU tests cannot validate."""
    model_cls = _require_real_qwen_stack()

    load_parameters = inspect.signature(model_cls.LLM).parameters
    transcribe_parameters = inspect.signature(model_cls.transcribe).parameters

    assert "model" in load_parameters
    assert "max_inference_batch_size" in load_parameters
    assert "max_new_tokens" in load_parameters
    assert list(transcribe_parameters) == ["self", "audio", "context", "language", "return_time_stamps"]
    assert callable(model_cls.LLM)


@pytest.mark.gpu
def test_qwen_asr_real_single_gpu_smoke() -> None:
    """Load the real model through the adapter and transcribe one sample."""
    _require_real_qwen_stack()
    if torch.cuda.device_count() < 1:
        pytest.fail("Qwen3-ASR smoke test requires one visible GPU")

    adapter = QwenASRAdapter(
        model_id=_MODEL_ID,
        max_new_tokens=64,
        max_inference_batch_size=1,
        # The 12 GB local smoke-test GPU cannot allocate Qwen3-ASR's full
        # 65,536-token default KV cache. Production/reference defaults remain
        # unchanged because this is an explicit test-only engine override.
        vllm_kwargs={"max_model_len": 8192},
    )
    adapter.load_model(num_gpus=1)
    try:
        assert adapter._model.max_inference_batch_size == 1
        assert adapter._model.backend == "vllm"
        results = adapter.transcribe_batch(
            [
                {
                    "waveform": _load_short_fixture(),
                    "sample_rate": _SAMPLE_RATE,
                    "language": "English",
                    "language_code": "en",
                    "task_id": "qwen-asr-gpu-smoke",
                }
            ]
        )
    finally:
        adapter.unload_model()

    assert len(results) == 1
    assert results[0].text.strip()
    assert results[0].skipped is False
