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

"""Tests for the PANNs implementation of the SED adapter contract."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nemo_curator.models.sed import panns  # noqa: E402
from nemo_curator.models.sed.panns import PANNsSEDAdapter  # noqa: E402

_SR = 16000
_HOP = 320
_CLASSES = 527
_CHECKPOINT = "/weights/Cnn14.pth"


class _FakeCNN14:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def __call__(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, samples = tensor.shape
        self.calls.append((batch, samples))
        frames = samples // _HOP
        return {
            "framewise_output": torch.full((batch, frames, _CLASSES), 0.5),
            "clipwise_output": torch.full((batch, _CLASSES), 0.5),
        }


def _adapter(**kwargs: object) -> tuple[PANNsSEDAdapter, _FakeCNN14]:
    adapter = PANNsSEDAdapter(
        checkpoint_path=_CHECKPOINT,
        sample_rate=_SR,
        hop_size=_HOP,
        classes_num=_CLASSES,
        **kwargs,
    )
    model = _FakeCNN14()
    adapter._model = model
    adapter._device = torch.device("cpu")
    return adapter, model


def _item(seconds: float) -> dict[str, object]:
    return {"waveform": np.zeros(int(seconds * _SR), dtype=np.float32)}


def test_ragged_waveforms_are_padded_into_one_model_call() -> None:
    adapter, model = _adapter()
    adapter.infer_batch([_item(1.0), _item(3.0)])
    assert model.calls == [(2, 3 * _SR)]


def test_short_audio_is_padded_to_the_cnn14_minimum() -> None:
    adapter, model = _adapter(window_size=1024)
    adapter.infer_batch([_item(0.001)])
    assert model.calls[0][1] == max(1024, _HOP * 32)


def test_short_audio_padding_can_be_disabled() -> None:
    adapter, model = _adapter(window_size=1024, pad_short_segments=False)
    adapter.infer_batch([_item(0.001)])
    assert model.calls[0][1] == 16


def test_results_preserve_padded_matrix_and_real_valid_frames() -> None:
    adapter, _ = _adapter()
    short, long = adapter.infer_batch([_item(1.0), _item(3.0)])
    assert short.framewise_output.shape == long.framewise_output.shape
    assert short.valid_frames == _SR / _HOP
    assert long.valid_frames == 3 * _SR / _HOP
    assert short.original_num_samples == _SR
    assert short.fps == _SR / _HOP


def test_default_adapter_matches_the_registered_checkpoint_frontend() -> None:
    adapter = PANNsSEDAdapter()

    assert adapter.model_type == panns._DEFAULT_MODEL_TYPE
    assert {
        name: getattr(adapter, name) for name in panns._DEFAULT_CHECKPOINT_CONFIG
    } == panns._DEFAULT_CHECKPOINT_CONFIG


def test_explicit_checkpoint_bypasses_default_prefetch() -> None:
    adapter = PANNsSEDAdapter(checkpoint_path=_CHECKPOINT)
    with (
        patch("torch.hub.load_state_dict_from_url") as download,
        patch("torch.load") as load,
    ):
        adapter.download_weights_on_node()

    download.assert_not_called()
    load.assert_not_called()


def test_missing_explicit_checkpoint_never_falls_back_to_network(tmp_path: Path) -> None:
    adapter = PANNsSEDAdapter(checkpoint_path=str(tmp_path / "missing.pth"))
    with (
        patch("nemo_curator.models.sed.panns.get_model_class", return_value=MagicMock()),
        patch("torch.hub.load_state_dict_from_url") as download,
        pytest.raises(FileNotFoundError),
    ):
        adapter.load_model(num_gpus=0)
    download.assert_not_called()


def test_download_weights_on_node_prefetches_default_without_constructing_model() -> None:
    adapter = PANNsSEDAdapter()
    with (
        patch("torch.hub.load_state_dict_from_url", return_value={"model": {}}) as download,
        patch("nemo_curator.models.sed.panns.get_model_class") as model_resolver,
    ):
        adapter.download_weights_on_node()

    download.assert_called_once_with(
        panns._DEFAULT_CHECKPOINT_URL,
        map_location="cpu",
        progress=False,
        file_name=panns._DEFAULT_CHECKPOINT_FILENAME,
        weights_only=True,
    )
    model_resolver.assert_not_called()


def test_default_prefetch_propagates_provider_failure() -> None:
    adapter = PANNsSEDAdapter()
    with (
        patch("torch.hub.load_state_dict_from_url", side_effect=RuntimeError("offline")),
        pytest.raises(RuntimeError, match="offline"),
    ):
        adapter.download_weights_on_node()


def test_unsupported_model_requires_an_explicit_checkpoint() -> None:
    adapter = PANNsSEDAdapter(model_type="Cnn14_DecisionLevelAvg")
    with (
        patch("torch.hub.load_state_dict_from_url") as download,
        pytest.raises(ValueError, match="provide checkpoint_path"),
    ):
        adapter.download_weights_on_node()
    download.assert_not_called()


def test_automatic_checkpoint_rejects_frontend_mismatch() -> None:
    adapter = PANNsSEDAdapter(sample_rate=16000)
    with (
        patch("torch.hub.load_state_dict_from_url") as download,
        pytest.raises(ValueError, match="published frontend"),
    ):
        adapter.download_weights_on_node()
    download.assert_not_called()


def test_load_model_uses_cpu_and_restricted_checkpoint_loading(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "custom.pth"
    checkpoint_path.touch()
    adapter = PANNsSEDAdapter(checkpoint_path=str(checkpoint_path))
    model = MagicMock()
    model_cls = MagicMock(return_value=model)
    with (
        patch("nemo_curator.models.sed.panns.get_model_class", return_value=model_cls),
        patch("torch.load", return_value={"model": {"weight": "value"}}) as torch_load,
        patch("torch.cuda.is_available", return_value=False),
    ):
        adapter.load_model(num_gpus=0)

    assert torch_load.call_args.args == (checkpoint_path.resolve(),)
    assert torch_load.call_args.kwargs == {"map_location": "cpu", "weights_only": True}
    model.load_state_dict.assert_called_once_with({"weight": "value"})
    model.to.assert_called_once_with(torch.device("cpu"))
    model.eval.assert_called_once_with()


def test_load_model_resolves_the_default_through_the_provider_cache() -> None:
    adapter = PANNsSEDAdapter()
    model = MagicMock()
    with (
        patch("nemo_curator.models.sed.panns.get_model_class", return_value=MagicMock(return_value=model)),
        patch(
            "torch.hub.load_state_dict_from_url",
            return_value={"model": {"weight": "value"}},
        ) as download,
        patch("torch.cuda.is_available", return_value=False),
    ):
        adapter.load_model(num_gpus=0)

    download.assert_called_once_with(
        panns._DEFAULT_CHECKPOINT_URL,
        map_location="cpu",
        progress=False,
        file_name=panns._DEFAULT_CHECKPOINT_FILENAME,
        weights_only=True,
    )
    model.load_state_dict.assert_called_once_with({"weight": "value"})


def test_load_model_forwards_checkpoint_frontend_configuration(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "custom.pth"
    checkpoint_path.touch()
    adapter = PANNsSEDAdapter(
        checkpoint_path=str(checkpoint_path),
        sample_rate=22050,
        model_type="Cnn14_DecisionLevelAvg",
        window_size=2048,
        hop_size=512,
        mel_bins=80,
        fmin=20,
        fmax=10000,
        classes_num=100,
    )
    model_cls = MagicMock(return_value=MagicMock())
    with (
        patch("nemo_curator.models.sed.panns.get_model_class", return_value=model_cls) as resolver,
        patch("torch.load", return_value={"model": {}}),
    ):
        adapter.load_model(num_gpus=0)

    resolver.assert_called_once_with("Cnn14_DecisionLevelAvg")
    model_cls.assert_called_once_with(
        sample_rate=22050,
        window_size=2048,
        hop_size=512,
        mel_bins=80,
        fmin=20,
        fmax=10000,
        classes_num=100,
    )


def test_panns_adapter_rejects_multi_gpu_loading() -> None:
    adapter = PANNsSEDAdapter(checkpoint_path=_CHECKPOINT)
    with pytest.raises(ValueError, match="zero or one"):
        adapter.load_model(num_gpus=2)


def test_unload_releases_model_and_device() -> None:
    adapter, _ = _adapter()
    adapter.unload_model()
    assert adapter._model is None
    assert adapter._device is None


def test_model_specific_loading_is_absent_from_the_stage_source() -> None:
    stage_source = Path(__file__).parents[3] / "nemo_curator" / "stages" / "audio" / "inference" / "sed" / "stage.py"
    source = stage_source.read_text()
    assert "torch.load" not in source
    assert "get_model_class" not in source
    assert "adapter_target" in source
