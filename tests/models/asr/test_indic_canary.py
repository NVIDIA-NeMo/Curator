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

"""CPU-only lifecycle and inference-contract tests for Indic Canary."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from nemo_curator.models.asr.base import ASRAdapter
from nemo_curator.models.asr.indic_canary import IndicCanaryTRTLLMASR
from nemo_curator.stages.audio.inference import indic_canary_trtllm_runtime

_SAMPLE_RATE = 16_000
_REQUIRED_ENGINE_FILES = (
    "encoder/encoder.plan",
    "encoder/config.json",
    "decoder/config.json",
    "decoder/rank0.engine",
    "decoder/vocab.json",
    "preprocessor/config.json",
    "preprocessor/mel_basis.pt",
)


def _write_engine_files(engine_dir: Path) -> None:
    for relative_path in _REQUIRED_ENGINE_FILES:
        path = engine_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")


def _item(
    language: str,
    *,
    samples: int = _SAMPLE_RATE,
    sample_rate: int = _SAMPLE_RATE,
) -> dict[str, object]:
    return {
        "waveform": np.zeros(samples, dtype=np.float32),
        "sample_rate": sample_rate,
        "language_code": language,
    }


class _RecordingRuntime:
    def __init__(self, *, max_batch_size: int = 2, languages: tuple[str, ...] = ("hi", "ta")) -> None:
        language_set = set(languages)
        self.max_batch_size = max_batch_size
        self.tokenizer = SimpleNamespace(
            langs=list(languages),
            supports_prompt_language=lambda language: language in language_set,
        )
        self.calls: list[dict[str, object]] = []

    def process_batch(
        self,
        padded: list[object],
        durations: list[int],
        prompts: list[dict[str, object]],
        *,
        num_beams: int,
        max_new_tokens: int,
    ) -> list[str]:
        self.calls.append(
            {
                "padded": padded,
                "durations": durations,
                "prompts": prompts,
                "num_beams": num_beams,
                "max_new_tokens": max_new_tokens,
            }
        )
        return [f"text-{prompt['source_language']}" for prompt in prompts]


def test_adapter_conforms_to_asr_protocol() -> None:
    assert isinstance(IndicCanaryTRTLLMASR("/models/indic-canary"), ASRAdapter)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"engine_dir": ""}, "engine_dir must point"),
        ({"engine_dir": "/engine", "num_beams": 0}, "at least 1"),
        ({"engine_dir": "/engine", "max_new_tokens": 0}, "at least 1"),
        ({"engine_dir": "/engine", "max_duration_sec": 0}, "must both be positive"),
        ({"engine_dir": "/engine", "min_duration_sec": 0}, "must both be positive"),
        ({"engine_dir": "/engine", "kv_cache_free_gpu_memory_fraction": 1.0}, "between 0 and 1"),
        ({"engine_dir": "/engine", "cross_kv_cache_fraction": 0.0}, "between 0 and 1"),
    ],
)
def test_adapter_rejects_invalid_configuration(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        IndicCanaryTRTLLMASR(**kwargs)  # type: ignore[arg-type]


def test_download_weights_validates_required_engine_files(tmp_path: Path) -> None:
    adapter = IndicCanaryTRTLLMASR(str(tmp_path))

    with pytest.raises(FileNotFoundError, match="missing required file"):
        adapter.download_weights_on_node()

    _write_engine_files(tmp_path)
    adapter.download_weights_on_node()


def test_load_model_validates_then_constructs_runtime_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_engine_files(tmp_path)
    runtime = MagicMock()
    runtime_cls = MagicMock(return_value=runtime)
    monkeypatch.setattr(indic_canary_trtllm_runtime, "CanaryTRTLLM", runtime_cls)
    adapter = IndicCanaryTRTLLMASR(
        str(tmp_path),
        kv_cache_free_gpu_memory_fraction=0.1,
        cross_kv_cache_fraction=0.3,
    )

    adapter.load_model(num_gpus=1)
    adapter.load_model(num_gpus=1)

    assert adapter._model is runtime
    runtime_cls.assert_called_once_with(
        str(tmp_path),
        device="cuda:0",
        kv_cache_free_gpu_memory_fraction=0.1,
        cross_kv_cache_fraction=0.3,
    )


@pytest.mark.parametrize("num_gpus", [0, 2, -1, 0.5, True])
def test_load_model_requires_exactly_one_gpu(num_gpus: object) -> None:
    adapter = IndicCanaryTRTLLMASR("/models/indic-canary")

    with pytest.raises(ValueError, match="requires exactly one GPU"):
        adapter.load_model(num_gpus=num_gpus)  # type: ignore[arg-type]


def test_unload_model_releases_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = IndicCanaryTRTLLMASR("/models/indic-canary")
    adapter._model = object()
    collect = MagicMock()
    empty_cache = MagicMock()
    monkeypatch.setattr("nemo_curator.models.asr.indic_canary.gc.collect", collect)
    monkeypatch.setattr("torch.cuda.empty_cache", empty_cache)

    adapter.unload_model()

    assert adapter._model is None
    collect.assert_called_once_with()
    empty_cache.assert_called_once_with()


def test_language_normalization_and_prompt_contract() -> None:
    adapter = IndicCanaryTRTLLMASR("/models/indic-canary", pnc=True)
    adapter._model = _RecordingRuntime(languages=("hi",))

    assert adapter._normalize_language("hi-IN") == "hi"
    assert adapter._normalize_language("zz") is None
    assert adapter._prompt_config("hi") == {
        "task": "transcribe",
        "pnc": True,
        "source_language": "hi",
        "target_language": "hi",
        "itn": False,
        "romanized": False,
        "timestamp": False,
        "diarize": False,
    }


def test_transcribe_batch_preserves_order_and_chunks_to_engine_limit() -> None:
    runtime = _RecordingRuntime(max_batch_size=2)
    adapter = IndicCanaryTRTLLMASR("/models/indic-canary", num_beams=3, max_new_tokens=77)
    adapter._model = runtime

    results = adapter.transcribe_batch(
        [
            _item("hi", samples=100),
            _item("zz"),
            _item("ta", samples=2 * _SAMPLE_RATE),
            _item("hi-IN", samples=40 * _SAMPLE_RATE + 1),
        ]
    )

    assert [result.text for result in results] == ["text-hi", "", "text-ta", "text-hi"]
    assert [result.skipped for result in results] == [False, True, False, False]
    assert results[1].skip_reason == "language_not_supported"
    assert results[1].unsupported_language == "zz"
    assert results[1].extras == {"language_code": "zz", "language_unsupported": True}
    assert results[3].extras == {
        "language_code": "hi",
        "truncated": True,
        "audio_duration_sec": pytest.approx(40.0 + 1 / _SAMPLE_RATE),
    }
    assert len(runtime.calls) == 2
    assert runtime.calls[0]["durations"] == [400, 2 * _SAMPLE_RATE]
    assert runtime.calls[1]["durations"] == [40 * _SAMPLE_RATE]
    assert all(call["num_beams"] == 3 for call in runtime.calls)
    assert all(call["max_new_tokens"] == 77 for call in runtime.calls)


def test_transcribe_batch_requires_loaded_runtime() -> None:
    adapter = IndicCanaryTRTLLMASR("/models/indic-canary")

    with pytest.raises(RuntimeError, match="not initialized"):
        adapter.transcribe_batch([_item("hi")])

    assert adapter.transcribe_batch([]) == []


@pytest.mark.parametrize(
    ("item", "message"),
    [
        (
            {"waveform": np.zeros((1, 8), dtype=np.float32), "sample_rate": _SAMPLE_RATE, "language_code": "hi"},
            "mono 1-D",
        ),
        ({"waveform": np.zeros(8, dtype=np.float32), "sample_rate": 8_000, "language_code": "hi"}, "16000 Hz"),
    ],
)
def test_transcribe_batch_rejects_invalid_audio(item: dict[str, object], message: str) -> None:
    adapter = IndicCanaryTRTLLMASR("/models/indic-canary")
    adapter._model = _RecordingRuntime()

    with pytest.raises(ValueError, match=message):
        adapter.transcribe_batch([item])


def test_transcribe_batch_rejects_runtime_cardinality_mismatch() -> None:
    adapter = IndicCanaryTRTLLMASR("/models/indic-canary")
    runtime = _RecordingRuntime()
    runtime.process_batch = MagicMock(return_value=[])
    adapter._model = runtime

    with pytest.raises(RuntimeError, match="returned 0 transcriptions for 1 inputs"):
        adapter.transcribe_batch([_item("hi")])
