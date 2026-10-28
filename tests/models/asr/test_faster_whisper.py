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

from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nemo_curator.models.asr import faster_whisper as faster_whisper_module
from nemo_curator.models.asr.base import ASRAdapter
from nemo_curator.models.asr.faster_whisper import _LANGUAGE_ALIASES, FasterWhisperASR


def _item(*, sample_rate: int = 16_000, language_code: str = "en") -> dict[str, object]:
    return {
        "waveform": np.zeros(160, dtype=np.float32),
        "sample_rate": sample_rate,
        "language_code": language_code,
    }


def test_adapter_conforms_to_shared_protocol() -> None:
    assert isinstance(FasterWhisperASR(), ASRAdapter)


def test_faster_whisper_stack_resolves_lazy_dependency() -> None:
    model_class = type("StubWhisperModel", (), {})
    download_model = MagicMock()
    package = ModuleType("faster_whisper")
    package.__path__ = []
    package.WhisperModel = model_class
    utils = ModuleType("faster_whisper.utils")
    utils.download_model = download_model

    with patch.dict(
        "sys.modules",
        {"faster_whisper": package, "faster_whisper.utils": utils},
    ):
        resolved_model_class, resolved_download_model = faster_whisper_module._faster_whisper_stack()

    assert resolved_model_class is model_class
    assert resolved_download_model is download_model


def test_faster_whisper_stack_explains_missing_dependency() -> None:
    with (
        patch.dict("sys.modules", {"faster_whisper": None}),
        pytest.raises(ImportError, match="requires the 'faster-whisper' package"),
    ):
        faster_whisper_module._faster_whisper_stack()


def test_stack_helpers_delegate_model_resolution_and_download() -> None:
    model_class = type("StubWhisperModel", (), {})
    download_model = MagicMock()
    with patch.object(
        faster_whisper_module,
        "_faster_whisper_stack",
        return_value=(model_class, download_model),
    ):
        assert faster_whisper_module._whisper_model_class() is model_class
        faster_whisper_module._download_whisper_model("tiny", "abc123")

    download_model.assert_called_once_with("tiny", revision="abc123")


def test_adapter_rejects_empty_model_id() -> None:
    with pytest.raises(ValueError, match="model_id must be non-empty"):
        FasterWhisperASR(model_id="")


def test_download_weights_uses_download_only() -> None:
    adapter = FasterWhisperASR(revision="abc123")
    with patch("nemo_curator.models.asr.faster_whisper._download_whisper_model") as download_model:
        adapter.download_weights_on_node()

    download_model.assert_called_once_with("large-v3", "abc123")


def test_load_model_uses_stage_owned_gpu_count() -> None:
    model_class = MagicMock()
    adapter = FasterWhisperASR(compute_type="float16", revision="abc123")

    with patch("nemo_curator.models.asr.faster_whisper._whisper_model_class", return_value=model_class):
        adapter.load_model(num_gpus=1)

    model_class.assert_called_once_with("large-v3", device="cuda", compute_type="float16", revision="abc123")


def test_load_model_uses_cpu_compute_type() -> None:
    model_class = MagicMock()
    adapter = FasterWhisperASR(compute_type="float16", cpu_compute_type="int8_float32")

    with patch("nemo_curator.models.asr.faster_whisper._whisper_model_class", return_value=model_class):
        adapter.load_model(num_gpus=0)

    model_class.assert_called_once_with(
        "large-v3",
        device="cpu",
        compute_type="int8_float32",
        revision=None,
    )


def test_load_model_is_idempotent() -> None:
    adapter = FasterWhisperASR()
    adapter._model = object()

    with patch("nemo_curator.models.asr.faster_whisper._whisper_model_class") as model_class:
        adapter.load_model(num_gpus=1)

    model_class.assert_not_called()


def test_load_model_rejects_negative_gpu_count() -> None:
    with pytest.raises(ValueError, match="num_gpus must be non-negative"):
        FasterWhisperASR().load_model(num_gpus=-1)


def test_unload_model_releases_model_and_cuda_cache() -> None:
    adapter = FasterWhisperASR()
    adapter._model = object()

    with (
        patch("nemo_curator.models.asr.faster_whisper.gc.collect") as collect,
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.empty_cache") as empty_cache,
    ):
        adapter.unload_model()

    assert adapter._model is None
    collect.assert_called_once_with()
    empty_cache.assert_called_once_with()


def test_unload_model_suppresses_cuda_cleanup_errors() -> None:
    adapter = FasterWhisperASR()
    adapter._model = object()

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.empty_cache", side_effect=RuntimeError("driver unavailable")),
    ):
        adapter.unload_model()

    assert adapter._model is None


def test_adapter_requires_loaded_model() -> None:
    with pytest.raises(RuntimeError, match=r"call load_model\(\) first"):
        FasterWhisperASR().transcribe_batch([_item()])


@pytest.mark.parametrize(("source_code", "whisper_code"), sorted(_LANGUAGE_ALIASES.items()))
def test_adapter_forces_every_reference_language_alias(source_code: str, whisper_code: str) -> None:
    adapter = FasterWhisperASR()
    adapter._model = MagicMock()
    adapter._model.transcribe.return_value = (iter([SimpleNamespace(text="hello")]), object())

    result = adapter.transcribe_batch([_item(language_code=source_code)])[0]

    assert result.text == "hello"
    assert result.extras == {"language_code": whisper_code}
    assert adapter._model.transcribe.call_args.kwargs["language"] == whisper_code


def test_adapter_calls_reference_transcribe_contract_sequentially_per_audio() -> None:
    adapter = FasterWhisperASR()
    adapter._model = MagicMock()
    adapter._model.transcribe.side_effect = [
        (iter([SimpleNamespace(text="first")]), object()),
        (iter([SimpleNamespace(text="second"), SimpleNamespace(text="utterance")]), object()),
    ]
    first = np.arange(160, dtype=np.float64)[::2]
    second = np.arange(120, dtype=np.float32)[::2]
    items = [
        {"waveform": first, "sample_rate": 16_000, "language_code": "en"},
        {"waveform": second, "sample_rate": 16_000, "language_code": "fil"},
    ]

    results = adapter.transcribe_batch(items)

    assert [result.text for result in results] == ["first", "second utterance"]
    assert [result.extras for result in results] == [{"language_code": "en"}, {"language_code": "tl"}]
    assert adapter._model.transcribe.call_count == 2
    expected = [(first, "en"), (second, "tl")]
    for model_call, (expected_waveform, expected_language) in zip(
        adapter._model.transcribe.call_args_list,
        expected,
        strict=True,
    ):
        args, kwargs = model_call
        assert len(args) == 1
        np.testing.assert_array_equal(args[0], np.asarray(expected_waveform, dtype=np.float32))
        assert args[0].dtype == np.float32
        assert args[0].flags.c_contiguous
        assert kwargs == {
            "language": expected_language,
            "beam_size": 5,
            "vad_filter": True,
            "without_timestamps": True,
        }


def test_adapter_discards_transcription_info_and_timestamp_metadata() -> None:
    adapter = FasterWhisperASR()
    adapter._model = MagicMock()
    word = SimpleNamespace(start=0.1, end=0.2, word="hello", probability=0.99)
    segment = SimpleNamespace(text="hello", start=0.0, end=0.4, words=[word])
    transcription_info = SimpleNamespace(
        language="fr",
        language_probability=0.98,
        duration=0.4,
        duration_after_vad=0.3,
        all_language_probs=[("fr", 0.98)],
    )
    adapter._model.transcribe.return_value = (iter([segment]), transcription_info)

    result = adapter.transcribe_batch([_item(language_code="en")])[0]

    assert result.text == "hello"
    assert result.extras == {"language_code": "en"}
    assert set(result.extras) == {"language_code"}


def test_adapter_preserves_empty_position_and_forced_language() -> None:
    adapter = FasterWhisperASR()
    adapter._model = MagicMock()
    item = _item(sample_rate=8_000, language_code="fil")
    item["waveform"] = np.zeros(0, dtype=np.float32)

    result = adapter.transcribe_batch([item])[0]

    assert result.text == ""
    assert result.skipped is False
    assert result.skip_reason is None
    assert result.extras == {"language_code": "tl"}
    adapter._model.transcribe.assert_not_called()


def test_adapter_requires_upstream_resampling() -> None:
    adapter = FasterWhisperASR()
    adapter._model = MagicMock()

    with pytest.raises(ValueError, match="ASRStage must provide 16000 Hz"):
        adapter.transcribe_batch([_item(sample_rate=8_000)])


def test_adapter_rejects_non_mono_waveform() -> None:
    adapter = FasterWhisperASR()
    adapter._model = MagicMock()
    item = _item()
    item["waveform"] = np.zeros((1, 160), dtype=np.float32)

    with pytest.raises(ValueError, match="mono 1-D waveform"):
        adapter.transcribe_batch([item])
