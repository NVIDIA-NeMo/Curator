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

from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path

import numpy as np
import pytest

from nemo_curator.models.audio.speaker_diarization.build_sortformer_tensorrt_engine import (
    _bundle_paths,
    _checkpoint_normalization,
    _copy_runtime_module,
    _model_config_bytes,
    _model_parameters,
    _profile_shapes,
    _publish_bundle,
    _runtime_config,
    _validate_learned_silence,
    _validate_staged_bundle,
)
from nemo_curator.models.audio.speaker_diarization.sortformer_tensorrt import TensorRTSortformerAdapter


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        num_speakers=None,
        spkcache_len=None,
        fifo_len=80,
        chunk_len=128,
        spkcache_refresh_rate=0,
        max_batch_size=16,
        center_chunk_frames=112,
        left_context_frames=16,
        right_context_frames=0,
        output_step_ms=80,
    )


def test_model_parameters_follow_checkpoint_frontend_and_streaming_defaults() -> None:
    config = {
        "preprocessor": {
            "sample_rate": 16_000,
            "window_size": 0.025,
            "window_stride": 0.01,
            "n_fft": 512,
            "preemph": 0.97,
            "normalize": "per_feature",
        },
        "model_defaults": {"fc_d_model": 512},
        "encoder": {"subsampling_factor": 8},
        "sortformer_modules": {"num_spks": 4, "spkcache_len": 188},
    }

    parameters = _model_parameters(config, _args())

    assert parameters["sample_rate"] == 16_000
    assert parameters["win_length"] == 400
    assert parameters["hop_length"] == 160
    assert parameters["spkcache_len"] == 188
    assert parameters["chunk_len"] == 128
    assert parameters["emb_dim"] == 512
    assert parameters["normalization"] == "per_feature"


@pytest.mark.parametrize(
    ("checkpoint_value", "runtime_value"),
    [
        ("per_feature", "per_feature"),
        ("all_features", "all_features"),
        ("NA", "none"),
        (None, "none"),
        (False, "none"),
    ],
)
def test_checkpoint_normalization_is_canonicalized(checkpoint_value: object, runtime_value: str) -> None:
    assert _checkpoint_normalization(checkpoint_value) == runtime_value


@pytest.mark.parametrize("checkpoint_value", [True, "fixed", {"fixed_mean": [0.0], "fixed_std": [1.0]}])
def test_checkpoint_normalization_rejects_unsupported_forms(checkpoint_value: object) -> None:
    with pytest.raises(ValueError, match="Unsupported Sortformer preprocessor normalization"):
        _checkpoint_normalization(checkpoint_value)


def test_missing_checkpoint_normalization_uses_nemo_default() -> None:
    parameters = _model_parameters({"preprocessor": {}}, _args())

    assert parameters["normalization"] == "per_feature"


def test_runtime_config_persists_normalization_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _bundle_paths(tmp_path / "model.plan", tmp_path / "model.onnx")
    parameters = {
        "max_batch_size": 16,
        "chunk_len": 128,
        "spkcache_len": 188,
        "fifo_len": 80,
        "emb_dim": 512,
        "normalization": "per_feature",
    }
    args = argparse.Namespace(
        opt_batch_size=8,
        precision="fp16",
        nemo_model=tmp_path / "model.nemo",
        optimization_level=5,
        workspace_gib=16,
    )
    monkeypatch.setattr(
        "nemo_curator.models.audio.speaker_diarization.build_sortformer_tensorrt_engine._sha256",
        lambda _path: "sha256",
    )
    monkeypatch.setattr("torch.cuda.get_device_name", lambda: "GPU")
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda: (9, 0))

    config = _runtime_config(
        args,
        paths,
        parameters,
        build_seconds=1.0,
        tensorrt_version="10.9",
        lowering_error=None,
        has_learned_silence=False,
    )

    assert config["schema_version"] == 2
    assert config["normalization"] == "per_feature"


def test_profile_shapes_cover_batch_cache_and_fifo_maxima() -> None:
    parameters = {
        "max_batch_size": 16,
        "chunk_len": 128,
        "spkcache_len": 188,
        "fifo_len": 80,
        "emb_dim": 512,
    }

    profiles = _profile_shapes(parameters, opt_batch_size=8)

    assert profiles["chunk"] == ((1, 128, 128), (8, 128, 128), (16, 128, 128))
    assert profiles["spkcache"][-1] == (16, 188, 512)
    assert profiles["fifo"] == ((1, 1, 512), (8, 40, 512), (16, 80, 512))


def test_model_config_reader_does_not_extract_archive(tmp_path: Path) -> None:
    nemo_path = tmp_path / "model.nemo"
    model_config = b"sample_rate: 16000\n"
    with tarfile.open(nemo_path, "w") as archive:
        info = tarfile.TarInfo("./model_config.yaml")
        info.size = len(model_config)
        archive.addfile(info, io.BytesIO(model_config))

    assert _model_config_bytes(nemo_path) == model_config
    assert list(tmp_path.iterdir()) == [nemo_path]


def test_learned_silence_validation_enforces_float32_embedding_shape(tmp_path: Path) -> None:
    path = tmp_path / "learnable_sil_emb.npy"
    np.save(path, np.ones(3, dtype=np.float32), allow_pickle=False)

    with pytest.raises(ValueError, match="expected float32"):
        _validate_learned_silence(path, {"emb_dim": 4}, required=True)


def test_runtime_module_is_copied_to_bundle_specific_name(tmp_path: Path) -> None:
    source = tmp_path / "riva_state.py"
    source.write_text("class SortformerModules: pass\n", encoding="utf-8")
    output = tmp_path / "bundle"
    output.mkdir()
    destination = output / "meeting.sortformer_modules.py"

    result = _copy_runtime_module(source, destination)

    assert result == destination
    assert result.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")


def test_bundle_assets_are_namespaced_by_engine_stem(tmp_path: Path) -> None:
    first = _bundle_paths(tmp_path / "first.plan", tmp_path / "first.onnx")
    second = _bundle_paths(tmp_path / "second.plan", tmp_path / "second.onnx")

    assert first.mel_basis.name == "first.mel_basis.npy"
    assert first.runtime_module.name == "first.sortformer_modules.py"
    assert first.learned_silence.name == "first.learnable_sil_emb.npy"
    assert {first.mel_basis, first.runtime_module, first.learned_silence}.isdisjoint(
        {second.mel_basis, second.runtime_module, second.learned_silence}
    )


def test_validated_bundle_publish_moves_config_last(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    staging = tmp_path / "staging"
    final_dir = tmp_path / "final"
    staging.mkdir()
    staged = _bundle_paths(staging / "model.plan", staging / "model.onnx")
    final = _bundle_paths(final_dir / "model.plan", final_dir / "model.onnx")
    for path in vars(staged).values():
        path.write_text(path.name, encoding="utf-8")

    calls: list[Path] = []
    original_replace = Path.replace

    def tracking_replace(source: Path, destination: Path) -> Path:
        calls.append(destination)
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", tracking_replace)
    _publish_bundle(staged, final, has_learned_silence=True, publish_onnx=True)

    assert calls[-1] == final.config
    assert all(path.is_file() for path in vars(final).values())


def test_staged_bundle_validation_deserializes_engine_and_always_unloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _bundle_paths(tmp_path / "model.plan", tmp_path / "model.onnx")
    calls: list[tuple[str, int | None]] = []

    def fail_while_loading(_self: TensorRTSortformerAdapter, *, num_gpus: int) -> None:
        calls.append(("load", num_gpus))
        msg = "TensorRT could not deserialize the staged plan"
        raise RuntimeError(msg)

    def record_unload(_self: TensorRTSortformerAdapter) -> None:
        calls.append(("unload", None))

    monkeypatch.setattr(TensorRTSortformerAdapter, "load_model", fail_while_loading)
    monkeypatch.setattr(TensorRTSortformerAdapter, "unload_model", record_unload)

    with pytest.raises(RuntimeError, match="could not deserialize"):
        _validate_staged_bundle(paths, {"sample_rate": 16_000})

    assert calls == [("load", 1), ("unload", None)]
