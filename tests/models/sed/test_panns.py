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
import torch

from nemo_curator.models.sed import panns
from nemo_curator.models.sed.panns import PANNsSEDAdapter

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


def test_existing_checkpoint_path_skips_prefetch_download(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "weights" / "custom.pth"
    checkpoint_path.parent.mkdir()
    checkpoint_path.touch()
    adapter = PANNsSEDAdapter(checkpoint_path=str(checkpoint_path))
    with (
        patch("torch.hub.load_state_dict_from_url") as download,
        patch("torch.load") as load,
    ):
        adapter.download_weights_on_node()

    download.assert_not_called()
    load.assert_not_called()


def test_missing_checkpoint_path_is_used_as_download_destination(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model-cache" / "custom-name.pth"
    adapter = PANNsSEDAdapter(checkpoint_path=str(checkpoint_path))
    with (
        patch("torch.hub.load_state_dict_from_url", return_value={"model": {}}) as download,
        patch("nemo_curator.models.sed.panns.get_model_class") as model_resolver,
    ):
        adapter.download_weights_on_node()

    download.assert_called_once_with(
        panns._DEFAULT_CHECKPOINT_URL,
        model_dir=str(checkpoint_path.parent),
        map_location="cpu",
        progress=False,
        file_name=checkpoint_path.name,
        weights_only=True,
    )
    model_resolver.assert_not_called()


def test_download_weights_on_node_prefetches_default_without_constructing_model(tmp_path: Path) -> None:
    hub_dir = tmp_path / "hub"
    checkpoint_path = hub_dir / "checkpoints" / panns._DEFAULT_CHECKPOINT_FILENAME
    adapter = PANNsSEDAdapter()
    with (
        patch("torch.hub.get_dir", return_value=str(hub_dir)),
        patch("torch.hub.load_state_dict_from_url", return_value={"model": {}}) as download,
        patch("nemo_curator.models.sed.panns.get_model_class") as model_resolver,
    ):
        adapter.download_weights_on_node()

    download.assert_called_once_with(
        panns._DEFAULT_CHECKPOINT_URL,
        model_dir=str(checkpoint_path.parent),
        map_location="cpu",
        progress=False,
        file_name=checkpoint_path.name,
        weights_only=True,
    )
    model_resolver.assert_not_called()


def test_prefetched_checkpoint_path_is_loaded_from_the_same_file(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model-cache" / "custom-name.pth"
    adapter = PANNsSEDAdapter(checkpoint_path=str(checkpoint_path))
    model = MagicMock()

    def materialize_checkpoint(*_args: object, **_kwargs: object) -> dict[str, object]:
        checkpoint_path.parent.mkdir()
        checkpoint_path.touch()
        return {"model": {"prefetched": "state"}}

    with (
        patch("torch.hub.load_state_dict_from_url", side_effect=materialize_checkpoint) as download,
        patch("torch.load", return_value={"model": {"loaded": "state"}}) as torch_load,
        patch("nemo_curator.models.sed.panns.get_model_class", return_value=MagicMock(return_value=model)),
    ):
        adapter.download_weights_on_node()
        adapter.load_model(num_gpus=0)

    download.assert_called_once_with(
        panns._DEFAULT_CHECKPOINT_URL,
        model_dir=str(checkpoint_path.parent),
        map_location="cpu",
        progress=False,
        file_name=checkpoint_path.name,
        weights_only=True,
    )
    torch_load.assert_called_once_with(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict.assert_called_once_with({"loaded": "state"})


def test_checkpoint_path_must_include_a_filename(tmp_path: Path) -> None:
    adapter = PANNsSEDAdapter(checkpoint_path=str(tmp_path))
    with (
        patch("torch.hub.load_state_dict_from_url") as download,
        pytest.raises(IsADirectoryError, match="must include a checkpoint filename"),
    ):
        adapter.download_weights_on_node()
    download.assert_not_called()


def test_default_prefetch_propagates_provider_failure(tmp_path: Path) -> None:
    adapter = PANNsSEDAdapter()
    with (
        patch("torch.hub.get_dir", return_value=str(tmp_path / "hub")),
        patch("torch.hub.load_state_dict_from_url", side_effect=RuntimeError("offline")),
        pytest.raises(RuntimeError, match="offline"),
    ):
        adapter.download_weights_on_node()


@pytest.mark.parametrize("use_default_location", [False, True])
def test_missing_checkpoint_destination_rejects_unsupported_model(
    tmp_path: Path,
    *,
    use_default_location: bool,
) -> None:
    adapter = PANNsSEDAdapter(
        checkpoint_path=None if use_default_location else str(tmp_path / "avg.pth"),
        model_type="Cnn14_DecisionLevelAvg",
    )
    with (
        patch("torch.hub.get_dir", return_value=str(tmp_path / "hub")),
        patch("torch.hub.load_state_dict_from_url") as download,
        pytest.raises(ValueError, match="provide checkpoint_path"),
    ):
        adapter.download_weights_on_node()
    download.assert_not_called()


@pytest.mark.parametrize("use_default_location", [False, True])
def test_missing_checkpoint_destination_rejects_frontend_mismatch(
    tmp_path: Path,
    *,
    use_default_location: bool,
) -> None:
    adapter = PANNsSEDAdapter(
        checkpoint_path=None if use_default_location else str(tmp_path / "custom.pth"),
        sample_rate=16000,
    )
    with (
        patch("torch.hub.get_dir", return_value=str(tmp_path / "hub")),
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
        patch("torch.cuda.is_available", return_value=True),
    ):
        adapter.load_model(num_gpus=0)

    assert torch_load.call_args.args == (checkpoint_path.resolve(),)
    assert torch_load.call_args.kwargs == {"map_location": "cpu", "weights_only": True}
    model.load_state_dict.assert_called_once_with({"weight": "value"})
    model.to.assert_called_once_with(torch.device("cpu"))
    model.eval.assert_called_once_with()


def test_load_model_resolves_the_default_through_the_provider_cache(tmp_path: Path) -> None:
    hub_dir = tmp_path / "hub"
    checkpoint_path = hub_dir / "checkpoints" / panns._DEFAULT_CHECKPOINT_FILENAME
    adapter = PANNsSEDAdapter()
    model = MagicMock()
    with (
        patch("torch.hub.get_dir", return_value=str(hub_dir)),
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
        model_dir=str(checkpoint_path.parent),
        map_location="cpu",
        progress=False,
        file_name=checkpoint_path.name,
        weights_only=True,
    )
    model.load_state_dict.assert_called_once_with({"weight": "value"})


def test_cached_default_checkpoint_matches_existing_explicit_path_behavior(tmp_path: Path) -> None:
    hub_dir = tmp_path / "hub"
    checkpoint_path = hub_dir / "checkpoints" / panns._DEFAULT_CHECKPOINT_FILENAME
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.touch()
    adapter = PANNsSEDAdapter(sample_rate=16000)
    model = MagicMock()

    with (
        patch("torch.hub.get_dir", return_value=str(hub_dir)),
        patch("torch.hub.load_state_dict_from_url") as download,
        patch("torch.load", return_value={"model": {"cached": "state"}}) as torch_load,
        patch("nemo_curator.models.sed.panns.get_model_class", return_value=MagicMock(return_value=model)),
    ):
        adapter.download_weights_on_node()
        adapter.load_model(num_gpus=0)

    download.assert_not_called()
    torch_load.assert_called_once_with(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict.assert_called_once_with({"cached": "state"})


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


@pytest.mark.parametrize("num_gpus", [1, 2])
def test_load_model_uses_cuda_for_any_positive_gpu_count(tmp_path: Path, num_gpus: int) -> None:
    checkpoint_path = tmp_path / "custom.pth"
    checkpoint_path.touch()
    adapter = PANNsSEDAdapter(checkpoint_path=str(checkpoint_path))
    model = MagicMock()
    with (
        patch("nemo_curator.models.sed.panns.get_model_class", return_value=MagicMock(return_value=model)),
        patch("torch.load", return_value={"model": {}}),
        patch("torch.cuda.is_available", return_value=True),
    ):
        adapter.load_model(num_gpus=num_gpus)

    assert adapter._device == torch.device("cuda")
    model.to.assert_called_once_with(torch.device("cuda"))


@pytest.mark.parametrize("num_gpus", [-1, 1.5, True])
def test_panns_adapter_rejects_invalid_worker_gpu_counts(num_gpus: object) -> None:
    adapter = PANNsSEDAdapter(checkpoint_path=_CHECKPOINT)
    with (
        patch("nemo_curator.models.sed.panns.get_model_class") as model_resolver,
        pytest.raises(ValueError, match="requires a non-negative integer num_gpus"),
    ):
        adapter.load_model(num_gpus=num_gpus)  # type: ignore[arg-type]
    model_resolver.assert_not_called()


@pytest.mark.parametrize("num_gpus", [1, 2])
def test_panns_adapter_requires_cuda_for_positive_gpu_counts(num_gpus: int) -> None:
    adapter = PANNsSEDAdapter(checkpoint_path=_CHECKPOINT)
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("nemo_curator.models.sed.panns.get_model_class") as model_resolver,
        pytest.raises(RuntimeError, match=rf"num_gpus={num_gpus}.*CUDA is not available"),
    ):
        adapter.load_model(num_gpus=num_gpus)
    model_resolver.assert_not_called()


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
