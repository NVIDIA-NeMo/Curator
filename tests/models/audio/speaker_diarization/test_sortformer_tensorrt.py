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

"""Tests for the TensorRT implementation of the diarization adapter."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from nemo_curator.models.audio.speaker_diarization.base import DiarizationAdapter, DiarizationInputError
from nemo_curator.models.audio.speaker_diarization.sortformer_tensorrt import (
    TensorRTSortformerAdapter,
    _binarize,
    _create_state_modules,
)

if TYPE_CHECKING:
    from pathlib import Path


_INPUT_NAMES = [
    "chunk",
    "chunk_lengths",
    "spkcache",
    "spkcache_lengths",
    "fifo",
    "fifo_lengths",
]
_OUTPUT_NAMES = ["predictions", "pred_lengths", "chunk_embs", "chunk_emb_lengths"]


class _FakeSession:
    def __init__(self) -> None:
        self.device = torch.device("cuda")
        self.input_names = list(_INPUT_NAMES)
        self.output_names = list(_OUTPUT_NAMES)
        self.closed = False

    @staticmethod
    def input_shape_range(name: str, profile_index: int = 0) -> tuple[tuple[int, ...], ...]:
        assert profile_index == 0
        profiles = {
            "chunk": ((1, 8, 128), (2, 8, 128), (4, 8, 128)),
            "spkcache": ((1, 1, 4), (2, 4, 4), (4, 4, 4)),
            "fifo": ((1, 1, 4), (2, 1, 4), (4, 2, 4)),
            "chunk_lengths": ((1,), (2,), (4,)),
            "spkcache_lengths": ((1,), (2,), (4,)),
            "fifo_lengths": ((1,), (2,), (4,)),
        }
        return profiles[name]

    def close(self) -> None:
        self.closed = True


def _config() -> dict[str, object]:
    return {
        "schema_version": 2,
        "sample_rate": 16_000,
        "n_fft": 4,
        "win_length": 4,
        "hop_length": 2,
        "preemphasis": 0.97,
        "log_guard": 2**-24,
        "normalization": "per_feature",
        "chunk_len": 8,
        "emb_dim": 4,
        "num_speakers": 2,
        "subsampling_factor": 2,
        "spkcache_refresh_rate": 0,
        "spkcache_len": 4,
        "fifo_len": 2,
        "max_batch_size": 4,
        "center_chunk_frames": 4,
        "left_context_frames": 2,
        "right_context_frames": 0,
        "output_step_ms": 80,
        "mel_basis": "mel_basis.npy",
        "learnable_sil_emb": "learnable_sil_emb.npy",
    }


def _bundle(tmp_path: Path, *, config_updates: dict[str, object] | None = None) -> dict[str, str]:
    engine = tmp_path / "sortformer.plan"
    engine.touch()
    runtime_module = tmp_path / "sortformer_modules.py"
    runtime_module.write_text(
        "class SortformerModules:\n"
        "    def __init__(self, learnable_sil_emb=None, **kwargs):\n"
        "        self.learnable_sil_emb = learnable_sil_emb\n",
        encoding="utf-8",
    )
    np.save(tmp_path / "mel_basis.npy", np.ones((128, 3), dtype=np.float32), allow_pickle=False)
    np.save(tmp_path / "learnable_sil_emb.npy", np.arange(4, dtype=np.float32), allow_pickle=False)
    config = _config()
    if config_updates:
        config.update(config_updates)
    config_path = tmp_path / "sortformer.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return {
        "engine_path": str(engine),
        "config_path": str(config_path),
        "runtime_module_path": str(runtime_module),
    }


def _adapter(tmp_path: Path, **kwargs: object) -> TensorRTSortformerAdapter:
    return TensorRTSortformerAdapter(**_bundle(tmp_path), **kwargs)


def _item(samples: int, *, sample_rate: object = 16_000) -> dict[str, object]:
    return {
        "waveform": np.zeros(samples, dtype=np.float32),
        "sample_rate": sample_rate,
    }


def _path_item(path: str = "/audio/example.wav") -> dict[str, object]:
    return {"audio_filepath": path}


def test_adapter_conforms_to_diarization_protocol(tmp_path: Path) -> None:
    assert isinstance(_adapter(tmp_path), DiarizationAdapter)


def test_binarize_keeps_the_full_final_active_frame() -> None:
    actual = _binarize(torch.tensor([0.1, 0.9]), frame_step=0.08, threshold=0.5)

    torch.testing.assert_close(actual, torch.tensor([[0.08, 0.16]]))


def test_binarize_keeps_a_single_active_frame() -> None:
    actual = _binarize(torch.tensor([0.9]), frame_step=0.08, threshold=0.5)

    torch.testing.assert_close(actual, torch.tensor([[0.0, 0.08]]))


@pytest.mark.parametrize("name", ["engine_path", "config_path", "runtime_module_path"])
def test_adapter_requires_every_bundle_path(name: str) -> None:
    kwargs = {
        "engine_path": "/engines/sortformer.plan",
        "config_path": "/engines/sortformer.json",
        "runtime_module_path": "/engines/sortformer_modules.py",
    }
    kwargs[name] = ""
    with pytest.raises(ValueError, match=rf"{name} is required"):
        TensorRTSortformerAdapter(**kwargs)


def test_prefetch_validates_complete_bundle_without_loading_runtime(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    with patch("nemo_curator.models.audio.speaker_diarization.sortformer_tensorrt._load_runtime_module") as loader:
        adapter.download_weights_on_node()
    loader.assert_not_called()


def test_prefetch_rejects_engine_config_sample_rate_mismatch(tmp_path: Path) -> None:
    adapter = TensorRTSortformerAdapter(**_bundle(tmp_path, config_updates={"sample_rate": 8_000}))
    with pytest.raises(ValueError, match="does not match adapter sample_rate"):
        adapter.download_weights_on_node()


@pytest.mark.parametrize("name", ["center_chunk_frames", "left_context_frames"])
def test_prefetch_rejects_unaligned_center_or_left_frames(tmp_path: Path, name: str) -> None:
    adapter = TensorRTSortformerAdapter(**_bundle(tmp_path, config_updates={name: 1}))

    with pytest.raises(ValueError, match="divisible by subsampling_factor=2"):
        adapter.download_weights_on_node()


def test_prefetch_accepts_partial_right_context_frames(tmp_path: Path) -> None:
    adapter = TensorRTSortformerAdapter(**_bundle(tmp_path, config_updates={"right_context_frames": 1}))

    adapter.download_weights_on_node()


def test_prefetch_rejects_fifo_smaller_than_emitted_chunk(tmp_path: Path) -> None:
    adapter = TensorRTSortformerAdapter(**_bundle(tmp_path, config_updates={"fifo_len": 1}))

    with pytest.raises(ValueError, match="must be 0 or at least 2"):
        adapter.download_weights_on_node()


def test_prefetch_rejects_legacy_config_without_normalization_contract(tmp_path: Path) -> None:
    paths = _bundle(tmp_path)
    config_path = tmp_path / "sortformer.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.pop("normalization")
    config["schema_version"] = 1
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="must declare a supported normalization contract"):
        TensorRTSortformerAdapter(**paths).download_weights_on_node()


@pytest.mark.parametrize("normalization", ["NA", "fixed", None, True, {"fixed_mean": [0.0]}])
def test_prefetch_rejects_noncanonical_normalization_contract(
    tmp_path: Path,
    normalization: object,
) -> None:
    adapter = TensorRTSortformerAdapter(**_bundle(tmp_path, config_updates={"normalization": normalization}))

    with pytest.raises(ValueError, match="must declare a supported normalization contract"):
        adapter.download_weights_on_node()


def test_prefetch_validates_mel_basis_shape(tmp_path: Path) -> None:
    paths = _bundle(tmp_path)
    np.save(tmp_path / "mel_basis.npy", np.ones((80, 3), dtype=np.float32), allow_pickle=False)
    with pytest.raises(ValueError, match="mel basis dtype/shape"):
        TensorRTSortformerAdapter(**paths).download_weights_on_node()


def test_prefetch_validates_learned_silence_shape(tmp_path: Path) -> None:
    paths = _bundle(tmp_path)
    np.save(tmp_path / "learnable_sil_emb.npy", np.ones(3, dtype=np.float32), allow_pickle=False)
    with pytest.raises(ValueError, match="learned silence embedding"):
        TensorRTSortformerAdapter(**paths).download_weights_on_node()


@pytest.mark.parametrize("num_gpus", [0, -1, 2, 1.5, True])
def test_load_model_requires_exactly_one_gpu(tmp_path: Path, num_gpus: object) -> None:
    adapter = _adapter(tmp_path)
    with pytest.raises(ValueError, match="requires exactly one GPU"):
        adapter.load_model(num_gpus=num_gpus)  # type: ignore[arg-type]


def test_load_model_reuses_shared_session_and_loads_learned_silence(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, inference_batch_size=2)
    session = _FakeSession()
    mel_tensor = MagicMock()
    silence_tensor = MagicMock()
    runtime_module = SimpleNamespace(SortformerModules=MagicMock())
    state_modules = MagicMock()
    window = MagicMock()

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch(
            "nemo_curator.stages.audio.inference.tensorrt_encoder.TensorRTEncoderSession",
            return_value=session,
        ) as session_cls,
        patch(
            "nemo_curator.models.audio.speaker_diarization.sortformer_tensorrt._load_runtime_module",
            return_value=runtime_module,
        ),
        patch(
            "nemo_curator.models.audio.speaker_diarization.sortformer_tensorrt._create_state_modules",
            return_value=state_modules,
        ) as create_modules,
        patch("torch.from_numpy", side_effect=[mel_tensor, silence_tensor]),
        patch("torch.hann_window", return_value=window),
    ):
        adapter.load_model(num_gpus=1)

    session_cls.assert_called_once_with(adapter._engine_file)
    create_modules.assert_called_once()
    assert create_modules.call_args.args[2] is silence_tensor.to.return_value
    assert adapter._session is session
    assert adapter._modules is state_modules
    assert adapter._window is window


def test_invalid_engine_contract_closes_partial_session(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    session = _FakeSession()
    session.input_names.remove("fifo_lengths")
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch(
            "nemo_curator.stages.audio.inference.tensorrt_encoder.TensorRTEncoderSession",
            return_value=session,
        ),
        pytest.raises(ValueError, match="engine inputs"),
    ):
        adapter.load_model(num_gpus=1)
    assert session.closed


def test_engine_profile_must_support_configured_batch(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, inference_batch_size=4)
    session = _FakeSession()
    session.input_shape_range = MagicMock(return_value=((1, 8, 128), (1, 8, 128), (2, 8, 128)))
    with pytest.raises(ValueError, match=r"does not support 1\.\.4"):
        adapter._validate_session(session, _config())


def test_engine_cache_profiles_must_support_empty_initial_state(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    session = _FakeSession()
    input_shape_range = session.input_shape_range
    session.input_shape_range = MagicMock(
        side_effect=lambda name: ((1, 2, 4), (2, 4, 4), (4, 4, 4)) if name == "spkcache" else input_shape_range(name)
    )

    with pytest.raises(ValueError, match=r"does not support \(\*, 1\.\.4, 4\)"):
        adapter._validate_session(session, _config())


def test_inference_uses_engine_reported_chunk_embedding_lengths(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    adapter._config = _config()
    session = _FakeSession()
    session.device = torch.device("cpu")
    session.infer = MagicMock(
        return_value={
            "predictions": torch.zeros((1, 4, 2)),
            "pred_lengths": torch.tensor([4]),
            "chunk_embs": torch.zeros((1, 4, 4)),
            "chunk_emb_lengths": torch.tensor([3]),
        }
    )
    adapter._session = session
    observed: dict[str, list[int]] = {}

    def streaming_update_batched(**kwargs: object) -> tuple[list[object], torch.Tensor, None]:
        observed["chunk_emb_lengths"] = kwargs["chunk_emb_lengths"]  # type: ignore[assignment]
        return kwargs["batch_states"], kwargs["preds"], None  # type: ignore[return-value]

    adapter._modules = SimpleNamespace(
        sync_pending_compression_batched=lambda _states: None,
        apply_mask_to_preds=lambda predictions, _lengths: predictions,
        streaming_update_batched=streaming_update_batched,
    )
    state = SimpleNamespace(
        spkcache_len_cached=1,
        fifo_len_cached=0,
        spkcache=torch.zeros((1, 1, 4)),
        fifo=torch.zeros((1, 0, 4)),
    )

    _, probabilities = adapter._infer_batch(
        [state],
        [torch.zeros((8, 128))],
        [0],
        [0],
        [1],
    )

    assert observed["chunk_emb_lengths"] == [3]
    assert probabilities[0].shape == (3, 2)


def test_unload_closes_shared_session(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    session = _FakeSession()
    adapter._session = session
    with patch("torch.cuda.is_available", return_value=False):
        adapter.unload_model()
    assert session.closed
    assert adapter._session is None


def _cpu_feature_adapter(tmp_path: Path, *, n_fft: int = 4) -> TensorRTSortformerAdapter:
    adapter = _adapter(tmp_path, stft_block_seconds=1.25)
    adapter._config = _config()
    adapter._config.update({"n_fft": n_fft, "win_length": n_fft})
    adapter._window = torch.hann_window(n_fft, periodic=False)
    adapter._mel_basis = torch.zeros((128, n_fft // 2 + 1))
    adapter._mel_basis[:2, :3] = torch.tensor([[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]])
    adapter._stft_block_samples = 20
    return adapter


@pytest.mark.parametrize("n_fft", [4, 16])
def test_streaming_stft_matches_full_stft_across_blocks(tmp_path: Path, n_fft: int) -> None:
    adapter = _cpu_feature_adapter(tmp_path, n_fft=n_fft)
    waveform = torch.sin(torch.arange(47, dtype=torch.float32) * 0.2)

    expected = adapter._extract_features(waveform, waveform.numel() // 2)
    actual = torch.cat(list(adapter._waveform_feature_blocks(waveform)))

    torch.testing.assert_close(actual, expected)


def test_per_feature_frontend_matches_nemo_preprocessor(tmp_path: Path) -> None:
    nemo_modules = pytest.importorskip("nemo.collections.asr.modules")
    preprocessor = nemo_modules.AudioToMelSpectrogramPreprocessor(
        sample_rate=1600,
        window_size=None,
        window_stride=None,
        n_window_size=16,
        n_window_stride=8,
        n_fft=16,
        features=128,
        normalize="per_feature",
        dither=0.0,
        pad_to=0,
    ).eval()
    adapter = _cpu_feature_adapter(tmp_path)
    adapter._config.update(
        {
            "sample_rate": 1600,
            "n_fft": 16,
            "win_length": 16,
            "hop_length": 8,
            "normalization": "per_feature",
        }
    )
    adapter._window = preprocessor.featurizer.window.detach().clone()
    adapter._mel_basis = preprocessor.featurizer.fb[0].detach().clone()
    waveform = torch.sin(torch.arange(103, dtype=torch.float32) * 0.2)
    expected, expected_length = preprocessor(
        input_signal=waveform.unsqueeze(0),
        length=torch.tensor([waveform.numel()]),
    )

    raw_features = adapter._extract_features(waveform, waveform.numel() // 8)
    actual = adapter._normalize_features(raw_features)

    torch.testing.assert_close(actual, expected[0, :, : expected_length.item()].transpose(0, 1))


@pytest.mark.parametrize("normalization", ["none", "per_feature", "all_features"])
def test_bounded_feature_blocks_are_normalized_without_concatenating(
    tmp_path: Path,
    normalization: str,
) -> None:
    adapter = _cpu_feature_adapter(tmp_path)
    adapter._config["normalization"] = normalization
    first = torch.arange(4 * 128, dtype=torch.float32).reshape(4, 128)
    second = torch.arange(4 * 128, 7 * 128, dtype=torch.float32).reshape(3, 128)
    expected = adapter._normalize_features(torch.cat((first, second)).clone())
    adapter._infer_probabilities = MagicMock(side_effect=lambda features: [features[0].clone()])

    with patch("torch.cat", side_effect=AssertionError("feature blocks must not be concatenated")):
        actual = adapter._infer_feature_blocks(iter((first, second)))

    torch.testing.assert_close(actual, expected)


def test_feature_spool_is_removed_when_inference_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _cpu_feature_adapter(tmp_path)
    adapter._infer_probabilities = MagicMock(side_effect=RuntimeError("inference failed"))
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))

    with pytest.raises(RuntimeError, match="inference failed"):
        adapter._infer_feature_blocks(iter((torch.zeros((2, 128)),)))

    assert list(tmp_path.glob("nemo-curator-sortformer-features-*")) == []


def test_feature_spool_is_float32_independent_of_torch_default(tmp_path: Path) -> None:
    adapter = _cpu_feature_adapter(tmp_path)
    adapter._config["normalization"] = "none"
    adapter._infer_probabilities = MagicMock(side_effect=lambda features: [features[0].clone()])
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        actual = adapter._infer_feature_blocks(iter((torch.ones((2, 128), dtype=torch.float32),)))
    finally:
        torch.set_default_dtype(previous_dtype)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, torch.ones((2, 128), dtype=torch.float32))


@pytest.mark.parametrize("n_fft", [4, 16])
def test_long_file_stft_matches_full_stft_without_loading_whole_file(tmp_path: Path, n_fft: int) -> None:
    soundfile = pytest.importorskip("soundfile")
    adapter = _cpu_feature_adapter(tmp_path, n_fft=n_fft)
    waveform = torch.sin(torch.arange(47, dtype=torch.float32) * 0.2)
    path = tmp_path / "long.wav"
    soundfile.write(path, waveform.numpy(), 16_000, subtype="FLOAT")

    expected = adapter._extract_features(waveform, waveform.numel() // 2)
    actual = torch.cat(list(adapter._file_feature_blocks(str(path))))

    torch.testing.assert_close(actual, expected)


def test_long_file_streaming_requires_engine_sample_rate(tmp_path: Path) -> None:
    soundfile = pytest.importorskip("soundfile")
    adapter = _cpu_feature_adapter(tmp_path)
    path = tmp_path / "wrong-rate.wav"
    soundfile.write(path, np.zeros(32, dtype=np.float32), 8_000, subtype="FLOAT")

    with pytest.raises(DiarizationInputError, match="must use the engine sample rate"):
        list(adapter._file_feature_blocks(str(path)))


def test_file_probe_identifies_soundfile_errors(tmp_path: Path) -> None:
    adapter = _cpu_feature_adapter(tmp_path)
    path = tmp_path / "invalid.wav"
    path.write_text("not audio", encoding="utf-8")

    with pytest.raises(DiarizationInputError, match="could not inspect"):
        adapter._source_duration_and_empty(str(path))


def test_passes_learned_silence_to_updated_runtime_module() -> None:
    class UpdatedModules:
        def __init__(self, learnable_sil_emb: torch.Tensor | None = None, **_kwargs: object) -> None:
            self.learnable_sil_emb = learnable_sil_emb

    learned_silence = torch.arange(4, dtype=torch.float32)
    modules = _create_state_modules(
        SimpleNamespace(SortformerModules=UpdatedModules),
        _config(),
        learned_silence,
    )
    assert modules.learnable_sil_emb is learned_silence


def test_applies_learned_silence_to_legacy_runtime_module() -> None:
    class LegacyModules:
        def __init__(self, **_kwargs: object) -> None:
            pass

    learned_silence = torch.arange(4, dtype=torch.float32)
    modules = _create_state_modules(
        SimpleNamespace(SortformerModules=LegacyModules),
        _config(),
        learned_silence,
    )
    actual = modules._get_silence_profile(torch.zeros((3, 2, 4)), torch.zeros((3, 2, 2)))
    torch.testing.assert_close(actual, learned_silence.expand(3, -1))


def test_rejects_speaker_cache_too_small_for_runtime_silence_frames() -> None:
    class Modules:
        def __init__(self, **_kwargs: object) -> None:
            self.spkcache_sil_frames_per_spk = 3

    with pytest.raises(ValueError, match="expected at least 6"):
        _create_state_modules(SimpleNamespace(SortformerModules=Modules), _config(), None)


@pytest.mark.parametrize(("right_context_frames", "frame_count"), [(1, 11), (0, 12)])
def test_inference_preserves_chunk_grid_context_and_final_flag(
    tmp_path: Path, right_context_frames: int, frame_count: int
) -> None:
    adapter = _adapter(tmp_path)
    adapter._config = _config()
    adapter._config["subsampling_factor"] = 1
    adapter._config["right_context_frames"] = right_context_frames
    adapter._session = _FakeSession()
    state = SimpleNamespace()
    adapter._modules = SimpleNamespace(init_streaming_state=MagicMock(return_value=state))
    calls: list[tuple[list[float], int, int, int]] = []

    def infer_batch(
        batch_states: list[object],
        windows: list[torch.Tensor],
        left_embeddings: list[int],
        right_embeddings: list[int],
        end_flags: list[int],
    ) -> tuple[list[object], list[torch.Tensor]]:
        calls.append(
            (
                windows[0][:, 0].tolist(),
                left_embeddings[0],
                right_embeddings[0],
                end_flags[0],
            )
        )
        output_length = windows[0].shape[0] - left_embeddings[0] - right_embeddings[0]
        return batch_states, [torch.zeros((output_length, 2))]

    adapter._infer_batch = infer_batch  # type: ignore[method-assign]
    features = torch.arange(frame_count, dtype=torch.float32).reshape(-1, 1).repeat(1, 128)

    probabilities = adapter._infer_probabilities([features])[0]

    assert probabilities.shape == (frame_count, 2)
    assert calls == [
        (list(range(4 + right_context_frames)), 0, right_context_frames, 0),
        (list(range(2, 8 + right_context_frames)), 2, right_context_frames, 0),
        (list(range(6, frame_count)), 2, 0, 1),
    ]


def test_aligned_chunk_grid_keeps_long_recording_timestamps_exact(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    adapter._config = _config()
    adapter._config.update(
        {
            "chunk_len": 128,
            "center_chunk_frames": 112,
            "left_context_frames": 16,
            "right_context_frames": 0,
            "subsampling_factor": 8,
            "fifo_len": 80,
            "num_speakers": 1,
            "output_step_ms": 80,
        }
    )
    adapter._session = _FakeSession()
    adapter._modules = SimpleNamespace(init_streaming_state=MagicMock(return_value=SimpleNamespace()))

    def infer_batch(
        batch_states: list[object],
        feature_windows: list[torch.Tensor],
        left_embeddings: list[int],
        right_embeddings: list[int],
        _end_flags: list[int],
    ) -> tuple[list[object], list[torch.Tensor]]:
        probabilities = []
        for feature, left, right in zip(feature_windows, left_embeddings, right_embeddings, strict=True):
            output_length = (feature.shape[0] + 7) // 8 - left - right
            probabilities.append(torch.ones((output_length, 1)))
        return batch_states, probabilities

    adapter._infer_batch = infer_batch  # type: ignore[method-assign]

    probabilities = adapter._infer_probabilities([torch.zeros((11_200, 128))])[0]

    assert probabilities.shape == (1_400, 1)
    assert adapter._segments(probabilities) == [{"start": 0.0, "end": 112.0, "speaker": "speaker_0"}]


def test_diarize_batch_streams_only_long_outliers_and_preserves_empty_positions(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, long_audio_seconds=1)
    adapter._session = _FakeSession()
    adapter._config = _config()
    adapter._features = MagicMock(return_value=[torch.zeros((4, 128))])
    adapter._infer_probabilities = MagicMock(side_effect=[[torch.tensor([[1.0]])], [torch.tensor([[2.0]])]])
    adapter._waveform_feature_blocks = MagicMock(return_value=iter((torch.zeros((4, 128)),)))
    adapter._segments = MagicMock(side_effect=lambda probabilities: [{"value": float(probabilities[0, 0])}])

    results = adapter.diarize_batch([_item(0), _item(8_000), _item(32_000)])

    assert results[0].segments == []
    assert results[1].segments == [{"value": 1.0}]
    assert results[2].segments == [{"value": 2.0}]
    assert adapter._infer_probabilities.call_count == 2


def test_diarize_batch_streams_long_filepaths_and_loads_only_regular_files(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, long_audio_seconds=1)
    adapter._session = _FakeSession()
    adapter._config = _config()
    adapter._source_duration_and_empty = MagicMock(
        side_effect=lambda source: (2.0, False) if source == "long.wav" else (0.5, False)
    )
    regular_waveform = torch.zeros(8_000)
    adapter._regular_waveforms = MagicMock(return_value=[regular_waveform])
    adapter._features = MagicMock(return_value=[torch.zeros((4, 128))])
    adapter._infer_probabilities = MagicMock(side_effect=[[torch.tensor([[1.0]])], [torch.tensor([[2.0]])]])
    file_blocks = iter((torch.zeros((4, 128)),))
    adapter._file_feature_blocks = MagicMock(return_value=file_blocks)
    adapter._segments = MagicMock(side_effect=lambda probabilities: [{"value": float(probabilities[0, 0])}])

    results = adapter.diarize_batch([_path_item("long.wav"), _path_item("short.wav")])

    assert results[0].segments == [{"value": 2.0}]
    assert results[1].segments == [{"value": 1.0}]
    adapter._regular_waveforms.assert_called_once_with(["short.wav"])
    adapter._file_feature_blocks.assert_called_once_with("long.wav")


@pytest.mark.parametrize(
    "item",
    [
        {},
        {"waveform": np.zeros(1, dtype=np.float32), "sample_rate": 16_000, "audio_filepath": "a.wav"},
    ],
)
def test_diarize_batch_requires_exactly_one_input_mode(tmp_path: Path, item: dict[str, object]) -> None:
    adapter = _adapter(tmp_path)
    adapter._session = _FakeSession()

    with pytest.raises(ValueError, match="exactly one"):
        adapter.diarize_batch([item])


def test_diarize_batch_validates_stage_sample_rate(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    adapter._session = _FakeSession()
    with pytest.raises(ValueError, match="must provide 16000 Hz"):
        adapter.diarize_batch([_item(100, sample_rate=8_000)])


def test_public_package_resolves_tensorrt_adapter_lazily() -> None:
    from nemo_curator.models.audio import speaker_diarization

    assert speaker_diarization.TensorRTSortformerAdapter is TensorRTSortformerAdapter


def test_sortformer_stage_constructs_tensorrt_adapter(tmp_path: Path) -> None:
    from nemo_curator.stages.audio.inference.speaker_diarization.stage import InferenceSortformerStage

    stage = InferenceSortformerStage(
        adapter_target=("nemo_curator.models.audio.speaker_diarization.sortformer_tensorrt.TensorRTSortformerAdapter"),
        adapter_kwargs={**_bundle(tmp_path), "inference_batch_size": 1},
    )

    assert isinstance(stage._create_adapter(), TensorRTSortformerAdapter)
