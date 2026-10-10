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

"""Tests for the NeMo implementation of the diarization adapter contract."""

from __future__ import annotations

import builtins
import errno
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from nemo_curator.models.audio.speaker_diarization.base import DiarizationAdapter, DiarizationInputError
from nemo_curator.models.audio.speaker_diarization.sortformer import (
    NeMoSortformerAdapter,
    _extract_nemo_features_in_blocks,
    parse_sortformer_segments,
)

if TYPE_CHECKING:
    from pathlib import Path

_SAMPLE_RATE = 16_000


def _item(samples: int = _SAMPLE_RATE, *, sample_rate: object = _SAMPLE_RATE) -> dict[str, object]:
    return {
        "waveform": np.zeros(samples, dtype=np.float32),
        "sample_rate": sample_rate,
    }


def _path_item(path: str = "/audio/example.wav") -> dict[str, object]:
    return {"audio_filepath": path}


def _mock_model(outputs: object = None) -> MagicMock:
    model = MagicMock()
    model.diarize.return_value = [[]] if outputs is None else outputs
    model.streaming_mode = False
    model.sortformer_modules = SimpleNamespace(
        chunk_len=264,
        chunk_left_context=1,
        chunk_right_context=1,
        fifo_len=0,
        spkcache_update_period=188,
        spkcache_len=264,
        _check_streaming_parameters=MagicMock(),
    )
    model.parameters.return_value = iter([torch.zeros(1)])
    model.encoder = SimpleNamespace(pos_enc=SimpleNamespace(extend_pe=MagicMock()))
    return model


def test_adapter_conforms_to_diarization_protocol() -> None:
    assert isinstance(NeMoSortformerAdapter(), DiarizationAdapter)


def test_package_import_does_not_load_concrete_adapter_or_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__
    blocked: list[str] = []
    blocked_prefixes = (
        "huggingface_hub",
        "nemo.collections",
        "nemo_curator.models.audio.speaker_diarization.sortformer",
    )

    def tracking_import(
        name: str,
        globals_: object | None = None,
        locals_: object | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name.startswith(blocked_prefixes):
            blocked.append(name)
            msg = f"blocked eager import of {name}"
            raise ImportError(msg)
        return original_import(name, globals_, locals_, fromlist, level)

    package_name = "nemo_curator.models.audio.speaker_diarization"
    saved_package = sys.modules.pop(package_name, None)
    monkeypatch.setattr(builtins, "__import__", tracking_import)
    try:
        package = __import__(package_name, fromlist=("speaker_diarization",))
        assert blocked == []
        assert "NeMoSortformerAdapter" not in vars(package)
    finally:
        sys.modules.pop(package_name, None)
        if saved_package is not None:
            sys.modules[package_name] = saved_package


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            ["0.00 2.70 speaker_0"],
            [{"start": 0.0, "end": 2.7, "speaker": "speaker_0"}],
        ),
        (
            [SimpleNamespace(start=1.0, end=3.5, speaker="speaker_1")],
            [{"start": 1.0, "end": 3.5, "speaker": "speaker_1"}],
        ),
        (
            [SimpleNamespace(start=1.0, end=3.5, label="speaker_2")],
            [{"start": 1.0, "end": 3.5, "speaker": "speaker_2"}],
        ),
        (
            [(2.0, 4.0, "speaker_3")],
            [{"start": 2.0, "end": 4.0, "speaker": "speaker_3"}],
        ),
        (
            [{"start": 4, "end": 5, "speaker": "speaker_4"}],
            [{"start": 4.0, "end": 5.0, "speaker": "speaker_4"}],
        ),
    ],
)
def test_segment_parser_supports_nemo_output_shapes(raw: list[object], expected: list[dict]) -> None:
    assert parse_sortformer_segments(raw) == expected


def test_segment_parser_uses_unknown_for_string_without_speaker() -> None:
    assert parse_sortformer_segments(["0.0 1.0"]) == [{"start": 0.0, "end": 1.0, "speaker": "unknown"}]


def test_segment_parser_omits_unknown_and_nonfinite_values() -> None:
    assert parse_sortformer_segments([42, "bad", "0 nan speaker_0"]) == []


def test_existing_model_path_prefetches_without_hub_or_nemo_import(tmp_path: Path) -> None:
    model_path = tmp_path / "sortformer.nemo"
    model_path.touch()
    adapter = NeMoSortformerAdapter(model_path=str(model_path))

    with (
        patch("nemo_curator.models.audio.speaker_diarization.sortformer._snapshot_download") as download,
        patch("nemo_curator.models.audio.speaker_diarization.sortformer._sortformer_model_class") as model_class,
    ):
        adapter.download_weights_on_node()

    download.assert_not_called()
    model_class.assert_not_called()


def test_preloaded_model_prefetch_does_not_resolve_weights() -> None:
    adapter = NeMoSortformerAdapter(preloaded_model=_mock_model())

    with patch.object(adapter, "_resolve_model_path") as resolve:
        adapter.download_weights_on_node()

    resolve.assert_not_called()


def test_hub_prefetch_resolves_nested_nemo_checkpoint(tmp_path: Path) -> None:
    first = tmp_path / "a" / "first.nemo"
    second = tmp_path / "b" / "second.nemo"
    first.parent.mkdir()
    second.parent.mkdir()
    first.touch()
    second.touch()
    adapter = NeMoSortformerAdapter(model_id="nvidia/example", cache_dir="/cache")

    with patch(
        "nemo_curator.models.audio.speaker_diarization.sortformer._snapshot_download",
        return_value=str(tmp_path),
    ) as download:
        assert adapter._resolve_model_path() == first

    download.assert_called_once_with("nvidia/example", "/cache")


def test_missing_explicit_model_path_fails_before_provider_import(tmp_path: Path) -> None:
    adapter = NeMoSortformerAdapter(model_path=str(tmp_path / "missing.nemo"))
    with (
        patch("nemo_curator.models.audio.speaker_diarization.sortformer._snapshot_download") as download,
        pytest.raises(FileNotFoundError, match="checkpoint not found"),
    ):
        adapter.download_weights_on_node()
    download.assert_not_called()


def test_load_model_uses_stage_owned_cpu_count_and_is_idempotent(tmp_path: Path) -> None:
    model_path = tmp_path / "sortformer.nemo"
    model_path.touch()
    model = _mock_model()
    adapter = NeMoSortformerAdapter(
        model_path=str(model_path),
        bounded_stft=False,
        max_positional_encoding_length=None,
    )

    with patch.object(adapter, "_restore_model", return_value=model) as restore:
        adapter.load_model(num_gpus=0)
        adapter.load_model(num_gpus=0)

    assert restore.call_count == 1
    assert restore.call_args.args[1].type == "cpu"
    model.to.assert_called_once_with(torch.device("cpu"))
    model.eval.assert_called_once_with()


def test_load_model_configures_injected_model_without_restoring_checkpoint() -> None:
    model = _mock_model()
    adapter = NeMoSortformerAdapter(
        preloaded_model=model,
        bounded_stft=False,
        max_positional_encoding_length=None,
    )

    with patch.object(adapter, "_restore_model") as restore:
        adapter.load_model(num_gpus=0)

    restore.assert_not_called()
    assert adapter._model is model
    assert adapter._device == torch.device("cpu")
    model.to.assert_not_called()
    model.eval.assert_called_once_with()


def test_load_model_preserves_injected_model_cuda_placement() -> None:
    model = _mock_model()
    parameter = SimpleNamespace(device=torch.device("cuda:1"))
    model.parameters.side_effect = lambda: iter([parameter])
    adapter = NeMoSortformerAdapter(
        preloaded_model=model,
        precision="fp16",
        bounded_stft=False,
        max_positional_encoding_length=None,
    )

    adapter.load_model(num_gpus=0)

    assert adapter._device == torch.device("cuda:1")
    model.to.assert_not_called()


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_injected_cpu_model_rejects_reduced_precision(precision: str) -> None:
    model = _mock_model()
    adapter = NeMoSortformerAdapter(
        preloaded_model=model,
        precision=precision,  # type: ignore[arg-type]
        bounded_stft=False,
        max_positional_encoding_length=None,
    )

    with pytest.raises(ValueError, match="preloaded model to be on CUDA"):
        adapter.load_model(num_gpus=1)

    model.to.assert_not_called()


def test_preloaded_model_can_be_reconfigured_after_unload() -> None:
    model = _mock_model()
    adapter = NeMoSortformerAdapter(
        preloaded_model=model,
        bounded_stft=False,
        max_positional_encoding_length=None,
    )

    adapter.load_model(num_gpus=0)
    adapter.unload_model()
    adapter.load_model(num_gpus=0)

    assert adapter._model is model
    assert model.eval.call_count == 2


@pytest.mark.parametrize("num_gpus", [-1, 1.5, 2, True])
def test_load_model_rejects_invalid_worker_gpu_counts(num_gpus: object) -> None:
    adapter = NeMoSortformerAdapter()
    with pytest.raises(ValueError, match="requires num_gpus to be 0 or 1"):
        adapter.load_model(num_gpus=num_gpus)  # type: ignore[arg-type]


def test_load_model_rejects_unavailable_requested_gpu() -> None:
    adapter = NeMoSortformerAdapter()
    with (
        patch("torch.cuda.is_available", return_value=False),
        pytest.raises(RuntimeError, match="CUDA is not available"),
    ):
        adapter.load_model(num_gpus=1)


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_cpu_load_rejects_reduced_precision(precision: str) -> None:
    adapter = NeMoSortformerAdapter(precision=precision)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires one GPU"):
        adapter.load_model(num_gpus=0)


def test_streaming_overrides_apply_and_validate() -> None:
    model = _mock_model()
    adapter = NeMoSortformerAdapter(
        chunk_len=124,
        chunk_left_context=2,
        chunk_right_context=7,
        fifo_len=16,
        spkcache_update_period=144,
        spkcache_len=200,
    )

    adapter._configure_streaming(model)

    modules = model.sortformer_modules
    assert modules.chunk_len == 124
    assert modules.chunk_left_context == 2
    assert modules.chunk_right_context == 7
    assert modules.fifo_len == 16
    assert modules.spkcache_update_period == 144
    assert modules.spkcache_len == 200
    modules._check_streaming_parameters.assert_called_once_with()


def test_none_streaming_values_preserve_checkpoint_configuration() -> None:
    model = _mock_model()
    adapter = NeMoSortformerAdapter(
        chunk_len=None,
        chunk_left_context=None,
        chunk_right_context=None,
        fifo_len=None,
        spkcache_update_period=None,
        spkcache_len=None,
    )

    adapter._configure_streaming(model)

    assert model.sortformer_modules.chunk_len == 264
    assert model.sortformer_modules.spkcache_len == 264
    model.sortformer_modules._check_streaming_parameters.assert_not_called()


def test_positional_encoding_extension_uses_model_device_and_dtype() -> None:
    model = _mock_model()
    parameter = torch.zeros(1, dtype=torch.float16)
    model.parameters.return_value = iter([parameter])
    adapter = NeMoSortformerAdapter(max_positional_encoding_length=12_345)

    adapter._extend_positional_encoding(model)

    model.encoder.pos_enc.extend_pe.assert_called_once_with(12_345, parameter.device, parameter.dtype)


@pytest.mark.parametrize("error", [MemoryError("out of memory"), torch.cuda.OutOfMemoryError("out of memory")])
def test_positional_encoding_memory_failures_propagate(error: Exception) -> None:
    model = _mock_model()
    model.parameters.return_value = iter([torch.zeros(1)])
    model.encoder.pos_enc.extend_pe.side_effect = error
    adapter = NeMoSortformerAdapter(max_positional_encoding_length=12_345)

    with pytest.raises(type(error), match="out of memory"):
        adapter._extend_positional_encoding(model)


def test_bounded_stft_patch_is_restored_during_unload() -> None:
    model = _mock_model()
    original_process_signal = MagicMock()
    model.streaming_mode = True
    model.process_signal = original_process_signal
    model.preprocessor = MagicMock()
    adapter = NeMoSortformerAdapter(bounded_stft=True, stft_block_seconds=30.0)
    adapter._model = model

    adapter._enable_bounded_stft(model)

    assert model.process_signal is not original_process_signal
    assert model._curator_original_process_signal is original_process_signal
    assert model._curator_stft_block_seconds == 30.0

    adapter.unload_model()

    assert model.process_signal is original_process_signal
    assert "_curator_original_process_signal" not in vars(model)
    assert "_curator_stft_block_seconds" not in vars(model)


def test_compile_encoder_replaces_only_model_encoder() -> None:
    model = _mock_model()
    original_encoder = model.encoder
    compiled_encoder = MagicMock()
    adapter = NeMoSortformerAdapter(compile_encoder=True)

    with patch("torch.compile", return_value=compiled_encoder) as compile_model:
        adapter._compile_encoder(model)

    compile_model.assert_called_once_with(original_encoder, dynamic=False)
    assert model.encoder is compiled_encoder


def test_diarize_batch_uses_one_ordered_nemo_call() -> None:
    model = _mock_model(
        [
            ["0.00 1.00 speaker_0"],
            [{"start": 1.0, "end": 2.0, "speaker": "speaker_1"}],
        ]
    )
    adapter = NeMoSortformerAdapter(inference_batch_size=2)
    adapter._model = model

    results = adapter.diarize_batch([_item(), _item(samples=2 * _SAMPLE_RATE)])

    assert results[0].segments == [{"start": 0.0, "end": 1.0, "speaker": "speaker_0"}]
    assert results[1].segments == [{"start": 1.0, "end": 2.0, "speaker": "speaker_1"}]
    call = model.diarize.call_args.kwargs
    assert call["batch_size"] == 2
    assert call["sample_rate"] == _SAMPLE_RATE
    assert len(call["audio"]) == 2
    assert all(value.flags.c_contiguous for value in call["audio"])


def test_diarize_batch_preserves_empty_waveform_positions() -> None:
    model = _mock_model([["0.0 1.0 speaker_0"]])
    adapter = NeMoSortformerAdapter()
    adapter._model = model

    results = adapter.diarize_batch([_item(samples=0), _item()])

    assert results[0].segments == []
    assert results[1].segments == [{"start": 0.0, "end": 1.0, "speaker": "speaker_0"}]
    assert len(model.diarize.call_args.kwargs["audio"]) == 1


def test_diarize_batch_passes_filepaths_directly_without_sample_rate() -> None:
    model = _mock_model([["0.0 1.0 speaker_0"], ["1.0 2.0 speaker_1"]])
    adapter = NeMoSortformerAdapter(inference_batch_size=2)
    adapter._model = model

    results = adapter.diarize_batch([_path_item("first.wav"), _path_item("second.wav")])

    assert results[0].segments[0]["speaker"] == "speaker_0"
    assert results[1].segments[0]["speaker"] == "speaker_1"
    call = model.diarize.call_args.kwargs
    assert call["audio"] == ["first.wav", "second.wav"]
    assert call["batch_size"] == 2
    assert "sample_rate" not in call


def test_diarize_batch_identifies_nemo_file_loading_errors() -> None:
    class AudioLoadingError(Exception):
        pass

    AudioLoadingError.__module__ = "lhotse.audio.utils"
    model = _mock_model()
    model.diarize.side_effect = AudioLoadingError("decode failed")
    adapter = NeMoSortformerAdapter()
    adapter._model = model

    with pytest.raises(DiarizationInputError, match="could not read"):
        adapter.diarize_batch([_path_item()])


@pytest.mark.parametrize(
    "detail",
    [
        "<class 'MemoryError'>: out of memory",
        "<class 'torch.OutOfMemoryError'>: CUDA out of memory",
        "<class 'OSError'>: [Errno 28] No space left on device",
    ],
)
def test_diarize_batch_preserves_lhotse_wrapped_resource_failures(detail: str) -> None:
    class AudioLoadingError(Exception):
        pass

    AudioLoadingError.__module__ = "lhotse.audio.utils"
    model = _mock_model()
    model.diarize.side_effect = AudioLoadingError(detail)
    adapter = NeMoSortformerAdapter()
    adapter._model = model

    with pytest.raises(AudioLoadingError, match=r"MemoryError|Errno 28"):
        adapter.diarize_batch([_path_item()])


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("provider unavailable"),
        MemoryError("out of memory"),
        torch.cuda.OutOfMemoryError("out of memory"),
        OSError(errno.ENOSPC, "No space left on device"),
    ],
)
def test_diarize_batch_preserves_provider_and_resource_failures(error: Exception) -> None:
    model = _mock_model()
    model.diarize.side_effect = error
    adapter = NeMoSortformerAdapter()
    adapter._model = model

    with pytest.raises(type(error)):
        adapter.diarize_batch([_path_item()])


def test_diarize_batch_supports_mixed_input_modes_without_reordering_results() -> None:
    model = _mock_model()
    model.diarize.side_effect = [
        [["1.0 2.0 waveform_speaker"]],
        [["0.0 1.0 filepath_speaker"]],
    ]
    adapter = NeMoSortformerAdapter()
    adapter._model = model

    results = adapter.diarize_batch([_path_item(), _item()])

    assert results[0].segments[0]["speaker"] == "filepath_speaker"
    assert results[1].segments[0]["speaker"] == "waveform_speaker"
    assert model.diarize.call_count == 2
    assert model.diarize.call_args_list[0].kwargs["sample_rate"] == _SAMPLE_RATE
    assert "sample_rate" not in model.diarize.call_args_list[1].kwargs


@pytest.mark.parametrize(
    "item",
    [
        {},
        {"waveform": np.zeros(1, dtype=np.float32), "sample_rate": _SAMPLE_RATE, "audio_filepath": "a.wav"},
    ],
)
def test_diarize_batch_requires_exactly_one_input_mode(item: dict[str, object]) -> None:
    adapter = NeMoSortformerAdapter()
    adapter._model = _mock_model()

    with pytest.raises(ValueError, match="exactly one"):
        adapter.diarize_batch([item])


def test_diarize_batch_empty_input_does_not_require_loaded_model() -> None:
    assert NeMoSortformerAdapter().diarize_batch([]) == []


def test_diarize_batch_requires_loaded_model() -> None:
    with pytest.raises(RuntimeError, match="call load_model"):
        NeMoSortformerAdapter().diarize_batch([_item()])


def test_diarize_batch_requires_mono_waveform() -> None:
    adapter = NeMoSortformerAdapter()
    adapter._model = _mock_model([])
    item = _item()
    item["waveform"] = np.zeros((1, _SAMPLE_RATE), dtype=np.float32)

    with pytest.raises(ValueError, match="mono 1-D waveform"):
        adapter.diarize_batch([item])


@pytest.mark.parametrize("sample_rate", [8_000, None, True])
def test_diarize_batch_requires_declared_sample_rate(sample_rate: object) -> None:
    adapter = NeMoSortformerAdapter()
    adapter._model = _mock_model([])

    with pytest.raises(ValueError, match="must provide 16000 Hz"):
        adapter.diarize_batch([_item(sample_rate=sample_rate)])


def test_diarize_batch_rejects_wrong_provider_result_count() -> None:
    adapter = NeMoSortformerAdapter(inference_batch_size=2)
    adapter._model = _mock_model([[]])

    with pytest.raises(RuntimeError, match="1 results for 2 valid inputs"):
        adapter.diarize_batch([_item(), _item()])


@pytest.mark.parametrize(("precision", "dtype"), [("fp16", torch.float16), ("bf16", torch.bfloat16)])
def test_reduced_precision_uses_cuda_autocast(precision: str, dtype: torch.dtype) -> None:
    adapter = NeMoSortformerAdapter(precision=precision)  # type: ignore[arg-type]
    adapter._model = _mock_model([[]])
    adapter._device = torch.device("cuda")

    with patch("torch.autocast") as autocast:
        adapter.diarize_batch([_item()])

    autocast.assert_called_once_with(device_type="cuda", dtype=dtype)


@pytest.mark.parametrize("block_seconds", [0.02, 1.0])
def test_bounded_stft_matches_full_nemo_preprocessor(block_seconds: float) -> None:
    nemo_modules = pytest.importorskip("nemo.collections.asr.modules")
    preprocessor = nemo_modules.AudioToMelSpectrogramPreprocessor(
        sample_rate=1600,
        window_size=None,
        window_stride=None,
        n_window_size=16,
        n_window_stride=8,
        n_fft=16,
        features=8,
        normalize="per_feature",
        dither=0.0,
        pad_to=0,
    ).eval()
    waveform = torch.sin(torch.arange(103, dtype=torch.float32) * 0.2)
    expected, expected_length = preprocessor(
        input_signal=waveform.unsqueeze(0),
        length=torch.tensor([waveform.numel()]),
    )

    actual = _extract_nemo_features_in_blocks(
        preprocessor,
        waveform,
        waveform.numel(),
        block_seconds=block_seconds,
    )

    assert actual.shape[1] == expected_length.item()
    torch.testing.assert_close(actual, expected[0, :, : expected_length.item()])
    assert preprocessor.featurizer.normalize == "per_feature"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sample_rate": 0}, "sample_rate"),
        ({"inference_batch_size": 0}, "inference_batch_size"),
        ({"precision": "int8"}, "precision"),
        ({"chunk_len": 0}, "chunk_len"),
        ({"fifo_len": -1}, "fifo_len"),
        ({"max_positional_encoding_length": 0}, "max_positional_encoding_length"),
        ({"stft_block_seconds": float("nan")}, "stft_block_seconds"),
    ],
)
def test_adapter_rejects_invalid_configuration(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        NeMoSortformerAdapter(**kwargs)  # type: ignore[arg-type]
