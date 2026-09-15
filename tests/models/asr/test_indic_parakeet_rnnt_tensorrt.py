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

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from nemo_curator.models.asr.indic_parakeet_rnnt_tensorrt import (
    TensorRTParakeetRNNTAdapter,
    load_engine_metadata,
)
from nemo_curator.models.asr.nemo_asr import NeMoASRAdapter


def _metadata() -> dict[str, object]:
    return {
        "schema_version": 1,
        "model_type": "indic_parakeet_rnnt",
        "precision": "fp16",
        "engine_file": "encoder.plan",
        "model_file": "model.nemo",
        "sample_rate": 16000,
        "feature_count": 80,
        "subsampling_factor": 8,
        "vocabulary_size": 958,
        "max_symbols_per_step": 10,
        "input_names": ["audio_signal", "length"],
        "output_names": ["outputs", "encoded_lengths"],
        "profile": {
            "min": {"batch": 1, "feature_frames": 8},
            "opt": {"batch": 8, "feature_frames": 800},
            "max": {"batch": 16, "feature_frames": 4001},
        },
    }


def _engine_bundle(tmp_path: Path) -> Path:
    (tmp_path / "encoder.plan").touch()
    (tmp_path / "model.nemo").touch()
    (tmp_path / "metadata.json").write_text(json.dumps(_metadata()))
    return tmp_path


def _nemo_model() -> SimpleNamespace:
    return SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "encoder": {"feat_in": 80},
                "preprocessor": {"sample_rate": 16000, "window_stride": 0.01},
                "train_ds": {"max_duration": 20},
                "joint": {"num_classes": 958},
                "decoding": {"strategy": "greedy", "greedy": {}},
            }
        ),
        encoder=SimpleNamespace(subsampling_factor=8, _feat_in=80),
        decoder=object(),
        joint=SimpleNamespace(_vocab_size=958),
        eval=MagicMock(),
        to=MagicMock(),
        change_decoding_strategy=MagicMock(),
    )


def test_load_engine_metadata(tmp_path: Path) -> None:
    assert load_engine_metadata(_engine_bundle(tmp_path))["subsampling_factor"] == 8


def test_load_engine_metadata_rejects_wrong_model_type(tmp_path: Path) -> None:
    metadata = _metadata()
    metadata["model_type"] = "canary"
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="model type"):
        load_engine_metadata(tmp_path)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("input_names", None, "input names"),
        ("profile", {"min": None, "opt": {}, "max": {}}, "profile points"),
    ],
)
def test_load_engine_metadata_rejects_malformed_fields(
    tmp_path: Path,
    key: str,
    value: object,
    message: str,
) -> None:
    metadata = _metadata()
    metadata[key] = value
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))

    with pytest.raises((TypeError, ValueError), match=message):
        load_engine_metadata(tmp_path)


def test_adapter_has_no_independent_batch_size_option() -> None:
    parameters = inspect.signature(TensorRTParakeetRNNTAdapter).parameters
    assert "batch_size" not in parameters
    assert "inference_batch_size" not in parameters


def test_download_weights_on_node_validates_local_bundle_directly(tmp_path: Path) -> None:
    engine_dir = _engine_bundle(tmp_path)
    adapter = TensorRTParakeetRNNTAdapter(model_id="ignored.nemo", engine_dir=engine_dir)

    with patch.object(NeMoASRAdapter, "download_weights_on_node") as nemo_download:
        adapter.download_weights_on_node()

    nemo_download.assert_not_called()
    assert adapter.metadata == _metadata()
    assert adapter.model_id == str(engine_dir / "model.nemo")


@pytest.mark.parametrize(("missing", "message"), [("encoder.plan", "engine"), ("model.nemo", "NeMo model")])
def test_download_weights_on_node_rejects_incomplete_bundle(tmp_path: Path, missing: str, message: str) -> None:
    engine_dir = _engine_bundle(tmp_path)
    (engine_dir / missing).unlink()
    adapter = TensorRTParakeetRNNTAdapter(engine_dir=engine_dir)

    with pytest.raises(FileNotFoundError, match=message):
        adapter.download_weights_on_node()


def test_adapter_rejects_invalid_chunking_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="chunking mode"):
        TensorRTParakeetRNNTAdapter(engine_dir=tmp_path, chunking_mode="invalid")  # type: ignore[arg-type]


@pytest.mark.parametrize("num_gpus", [0, 2])
def test_adapter_requires_exactly_one_gpu(tmp_path: Path, num_gpus: int) -> None:
    adapter = TensorRTParakeetRNNTAdapter(engine_dir=tmp_path)

    with pytest.raises(ValueError, match="exactly one GPU"):
        adapter.load_model(num_gpus=num_gpus)


def test_adapter_requires_available_cuda(tmp_path: Path) -> None:
    adapter = TensorRTParakeetRNNTAdapter(engine_dir=tmp_path)

    with patch("torch.cuda.is_available", return_value=False), pytest.raises(RuntimeError, match="requires CUDA"):
        adapter.load_model(num_gpus=1)


def test_adapter_enables_batched_greedy_decoder(tmp_path: Path) -> None:
    adapter = TensorRTParakeetRNNTAdapter(engine_dir=_engine_bundle(tmp_path))
    model = _nemo_model()
    adapter._model = model
    adapter.metadata = _metadata()

    adapter._enable_batched_greedy_decoder()

    assert model.cfg.decoding.strategy == "greedy_batch"
    assert model.cfg.decoding.greedy.max_symbols_per_step == 10
    assert model.cfg.decoding.greedy.use_cuda_graph_decoder is True
    model.change_decoding_strategy.assert_called_once_with(model.cfg.decoding)


def test_adapter_replaces_only_encoder(tmp_path: Path) -> None:
    engine_dir = _engine_bundle(tmp_path)
    adapter = TensorRTParakeetRNNTAdapter(engine_dir=engine_dir)
    model = _nemo_model()
    original_decoder = model.decoder
    original_joint = model.joint
    optimized_encoder = MagicMock()
    optimized_encoder.max_input_shape.return_value = (16, 80, 4001)

    with (
        patch.object(adapter, "_load_checkpoint", return_value=model),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.empty_cache"),
        patch(
            "nemo_curator.models.asr.indic_parakeet_rnnt_tensorrt.TensorRTEncoder",
            return_value=optimized_encoder,
        ) as encoder_type,
    ):
        adapter.load_model(num_gpus=1)

    encoder_type.assert_called_once_with(engine_dir / "encoder.plan", subsampling_factor=8)
    assert model.encoder is optimized_encoder
    assert model.decoder is original_decoder
    assert model.joint is original_joint
    model.to.assert_called_once_with(dtype=torch.float16)
    model.change_decoding_strategy.assert_called_once()
    assert adapter._chunk_duration_sec == 40.0


def test_adapter_rejects_engine_shorter_than_40_seconds(tmp_path: Path) -> None:
    adapter = TensorRTParakeetRNNTAdapter(engine_dir=_engine_bundle(tmp_path))
    model = _nemo_model()
    optimized_encoder = MagicMock()
    optimized_encoder.max_input_shape.return_value = (16, 80, 4000)

    with (
        patch.object(adapter, "_load_checkpoint", return_value=model),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.empty_cache"),
        patch(
            "nemo_curator.models.asr.indic_parakeet_rnnt_tensorrt.TensorRTEncoder",
            return_value=optimized_encoder,
        ),
        pytest.raises(ValueError, match="--max-frames 4001"),
    ):
        adapter.load_model(num_gpus=1)

    optimized_encoder.close.assert_called_once_with()
    assert adapter._model is None


def test_adapter_chunks_long_audio_without_overlap(tmp_path: Path) -> None:
    adapter = TensorRTParakeetRNNTAdapter(engine_dir=_engine_bundle(tmp_path))
    adapter._chunk_duration_sec = 40.0
    adapter._trt_encoder = MagicMock()
    adapter._trt_encoder.max_input_shape.return_value = (16, 80, 4001)
    waveform = np.zeros(45 * 16000, dtype=np.float32)

    with patch.object(
        NeMoASRAdapter,
        "_transcribe_waveforms",
        return_value=["second", "first"],
    ) as transcribe:
        texts = adapter._transcribe_waveforms([waveform])

    prepared = transcribe.call_args.args[0]
    assert [chunk.shape[0] for chunk in prepared] == [5 * 16000, 40 * 16000]
    assert texts == ["first second"]


def test_adapter_can_disable_chunking(tmp_path: Path) -> None:
    adapter = TensorRTParakeetRNNTAdapter(engine_dir=_engine_bundle(tmp_path), chunking_mode="none")
    waveform = np.zeros(45 * 16000, dtype=np.float32)

    with patch.object(NeMoASRAdapter, "_transcribe_waveforms", return_value=["full"]) as transcribe:
        texts = adapter._transcribe_waveforms([waveform])

    transcribe.assert_called_once_with([waveform])
    assert texts == ["full"]


def test_adapter_orders_chunks_by_duration_and_restores_rows(tmp_path: Path) -> None:
    adapter = TensorRTParakeetRNNTAdapter(engine_dir=_engine_bundle(tmp_path))
    adapter._chunk_duration_sec = 40.0
    adapter._trt_encoder = MagicMock()
    adapter._trt_encoder.max_input_shape.return_value = (16, 80, 4001)
    waveforms = [
        np.zeros(8 * 16000, dtype=np.float32),
        np.zeros(6 * 16000, dtype=np.float32),
    ]

    with patch.object(
        NeMoASRAdapter,
        "_transcribe_waveforms",
        return_value=["second", "first"],
    ) as transcribe:
        texts = adapter._transcribe_waveforms(waveforms)

    prepared = transcribe.call_args.args[0]
    assert [chunk.shape[0] for chunk in prepared] == [6 * 16000, 8 * 16000]
    assert texts == ["first", "second"]


def test_adapter_bounds_expanded_chunks_by_engine_profile(tmp_path: Path) -> None:
    adapter = TensorRTParakeetRNNTAdapter(engine_dir=_engine_bundle(tmp_path))
    adapter._chunk_duration_sec = 40.0
    adapter._trt_encoder = MagicMock()
    adapter._trt_encoder.max_input_shape.return_value = (2, 80, 4001)
    waveforms = [np.zeros(45 * 16000, dtype=np.float32) for _ in range(3)]

    transcription_batch_sizes: list[int] = []

    def transcribe(chunks: list[np.ndarray]) -> list[str]:
        transcription_batch_sizes.append(len(chunks))
        return [str(chunk.shape[0]) for chunk in chunks]

    with patch.object(NeMoASRAdapter, "_transcribe_waveforms", side_effect=transcribe):
        texts = adapter._transcribe_waveforms(waveforms)

    assert transcription_batch_sizes == [2, 2, 2]
    assert texts == ["640000 80000"] * 3


def test_unload_model_closes_tensorrt_encoder(tmp_path: Path) -> None:
    adapter = TensorRTParakeetRNNTAdapter(engine_dir=_engine_bundle(tmp_path))
    optimized_encoder = MagicMock()
    adapter._trt_encoder = optimized_encoder
    adapter._model = SimpleNamespace()
    adapter.metadata = _metadata()
    adapter._chunk_duration_sec = 40.0

    with patch("torch.cuda.is_available", return_value=False):
        adapter.unload_model()

    optimized_encoder.close.assert_called_once_with()
    assert adapter._trt_encoder is None
    assert adapter._model is None
    assert adapter.metadata is None
    assert adapter._chunk_duration_sec is None
