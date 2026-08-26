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

from nemo_curator.models import faster_whisper_asr as faster_whisper_module
from nemo_curator.models.asr.base import ASRAdapter
from nemo_curator.models.faster_whisper_asr import FasterWhisperASR


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
    with patch("nemo_curator.models.faster_whisper_asr._download_whisper_model") as download_model:
        adapter.download_weights_on_node()

    download_model.assert_called_once_with("large-v3", "abc123")


def test_load_model_uses_stage_owned_gpu_count() -> None:
    model_class = MagicMock()
    adapter = FasterWhisperASR(compute_type="float16", revision="abc123")

    with patch("nemo_curator.models.faster_whisper_asr._whisper_model_class", return_value=model_class):
        adapter.load_model(num_gpus=1)

    model_class.assert_called_once_with("large-v3", device="cuda", compute_type="float16", revision="abc123")


def test_load_model_uses_cpu_compute_type() -> None:
    model_class = MagicMock()
    adapter = FasterWhisperASR(compute_type="float16", cpu_compute_type="int8_float32")

    with patch("nemo_curator.models.faster_whisper_asr._whisper_model_class", return_value=model_class):
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

    with patch("nemo_curator.models.faster_whisper_asr._whisper_model_class") as model_class:
        adapter.load_model(num_gpus=1)

    model_class.assert_not_called()


def test_load_model_rejects_negative_gpu_count() -> None:
    with pytest.raises(ValueError, match="num_gpus must be non-negative"):
        FasterWhisperASR().load_model(num_gpus=-1)


def test_unload_model_releases_model_and_cuda_cache() -> None:
    adapter = FasterWhisperASR()
    adapter._model = object()

    with (
        patch("nemo_curator.models.faster_whisper_asr.gc.collect") as collect,
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.empty_cache") as empty_cache,
    ):
        adapter.unload_model()

    assert adapter._model is None
    collect.assert_called_once_with()
    empty_cache.assert_called_once_with()


def test_adapter_requires_loaded_model() -> None:
    with pytest.raises(RuntimeError, match=r"call load_model\(\) first"):
        FasterWhisperASR().transcribe_batch([_item()])


def test_adapter_transcribes_waveform_and_maps_language_alias() -> None:
    adapter = FasterWhisperASR()
    adapter._model = MagicMock()
    adapter._model.transcribe.return_value = (
        iter([SimpleNamespace(text="hello"), SimpleNamespace(text="world")]),
        object(),
    )

    results = adapter.transcribe_batch([_item(language_code="fil")])

    assert [result.text for result in results] == ["hello world"]
    assert results[0].extras["language_code"] == "tl"
    assert adapter._model.transcribe.call_args.kwargs["language"] == "tl"


def test_adapter_preserves_empty_positions() -> None:
    adapter = FasterWhisperASR()
    adapter._model = MagicMock()
    item = _item()
    item["waveform"] = np.zeros(0, dtype=np.float32)

    results = adapter.transcribe_batch([item])

    assert results[0].skipped is True
    assert results[0].skip_reason == "empty_audio"
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
