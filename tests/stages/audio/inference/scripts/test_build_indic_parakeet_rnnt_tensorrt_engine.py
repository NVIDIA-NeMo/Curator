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

import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from nemo_curator.stages.audio.inference.scripts import build_indic_parakeet_rnnt_tensorrt_engine as builder
from nemo_curator.stages.audio.inference.scripts import tensorrt_encoder_utils


def test_builder_imports_packaged_encoder_utility() -> None:
    assert builder.build_encoder_bundle is tensorrt_encoder_utils.build_encoder_bundle


def test_builder_remains_directly_executable_from_a_source_checkout() -> None:
    completed = subprocess.run(  # noqa: S603 - executable and script are trusted test-environment paths
        [sys.executable, str(Path(builder.__file__).resolve()), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Build an FP16 TensorRT encoder bundle" in completed.stdout


def test_parse_args_accepts_valid_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "builder",
            "--model",
            "model.nemo",
            "--output-dir",
            "engine",
            "--min-batch",
            "2",
            "--opt-batch",
            "4",
            "--max-batch",
            "8",
            "--min-frames",
            "10",
            "--opt-frames",
            "100",
            "--max-frames",
            "4001",
            "--max-symbols-per-step",
            "12",
            "--workspace-gb",
            "4",
            "--verbose",
        ],
    )

    args = builder.parse_args()

    assert (args.min_batch, args.opt_batch, args.max_batch) == (2, 4, 8)
    assert (args.min_frames, args.opt_frames, args.max_frames) == (10, 100, 4001)
    assert args.max_symbols_per_step == 12
    assert args.workspace_gb == 4
    assert args.verbose


@pytest.mark.parametrize(
    "invalid_args",
    [
        ["--min-batch", "3", "--opt-batch", "2"],
        ["--min-frames", "10", "--opt-frames", "9"],
        ["--max-frames", "4000"],
        ["--max-symbols-per-step", "0"],
        ["--workspace-gb", "0"],
    ],
)
def test_parse_args_rejects_invalid_build_limits(
    monkeypatch: pytest.MonkeyPatch,
    invalid_args: list[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["builder", "--model", "model.nemo", "--output-dir", "engine", *invalid_args],
    )

    with pytest.raises(SystemExit, match="2"):
        builder.parse_args()


def test_load_model_requires_cuda(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    asr_module = ModuleType("nemo.collections.asr")
    collections_module = ModuleType("nemo.collections")
    collections_module.__path__ = []  # type: ignore[attr-defined]
    collections_module.asr = asr_module  # type: ignore[attr-defined]
    nemo_module = ModuleType("nemo")
    nemo_module.__path__ = []  # type: ignore[attr-defined]
    nemo_module.collections = collections_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nemo", nemo_module)
    monkeypatch.setitem(sys.modules, "nemo.collections", collections_module)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr", asr_module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="requires CUDA"):
        builder._load_model(tmp_path / "model.nemo")


@pytest.mark.parametrize("complete_model", [True, False])
def test_load_model_restores_and_validates_rnnt_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    complete_model: bool,
) -> None:
    calls: dict[str, Any] = {}

    class FakeModel:
        encoder = object()
        decoder = object()

        if complete_model:
            joint = object()

        def eval(self) -> "FakeModel":
            calls["eval"] = True
            return self

        def to(self, **kwargs: object) -> "FakeModel":
            calls["to"] = kwargs
            return self

    model = FakeModel()
    asr_module = ModuleType("nemo.collections.asr")
    asr_module.models = SimpleNamespace(  # type: ignore[attr-defined]
        ASRModel=SimpleNamespace(
            restore_from=lambda **kwargs: calls.update(restore=kwargs) or model,
        )
    )
    collections_module = ModuleType("nemo.collections")
    collections_module.__path__ = []  # type: ignore[attr-defined]
    collections_module.asr = asr_module  # type: ignore[attr-defined]
    nemo_module = ModuleType("nemo")
    nemo_module.__path__ = []  # type: ignore[attr-defined]
    nemo_module.collections = collections_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nemo", nemo_module)
    monkeypatch.setitem(sys.modules, "nemo.collections", collections_module)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr", asr_module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    if complete_model:
        restored = builder._load_model(tmp_path / "model.nemo")
        assert restored is model
        assert calls["restore"]["restore_path"] == str(tmp_path / "model.nemo")
        assert calls["restore"]["map_location"] == torch.device("cuda")
        assert calls["eval"]
        assert calls["to"] == {"dtype": torch.float16}
    else:
        with pytest.raises(TypeError, match="not an RNN-T"):
            builder._load_model(tmp_path / "model.nemo")


def test_build_bundle_validates_model_path(tmp_path: Path) -> None:
    args = SimpleNamespace(model=str(tmp_path / "missing.nemo"))

    with pytest.raises(FileNotFoundError, match="Local NeMo checkpoint not found"):
        builder.build_bundle(args)


def test_build_bundle_passes_model_metadata_to_shared_builder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "indic.nemo"
    model_path.touch()
    output_dir = tmp_path / "bundle"
    model = SimpleNamespace(
        encoder=SimpleNamespace(_feat_in=80, subsampling_factor=8),
        decoder=object(),
        joint=SimpleNamespace(_vocab_size=1024),
        cfg=SimpleNamespace(
            encoder=SimpleNamespace(feat_in=80),
            preprocessor=SimpleNamespace(sample_rate=16_000),
            joint=SimpleNamespace(num_classes=1024),
        ),
    )
    args = SimpleNamespace(
        model=str(model_path),
        output_dir=str(output_dir),
        max_symbols_per_step=10,
    )
    calls: dict[str, Any] = {}
    monkeypatch.setattr(builder, "_load_model", lambda _path: model)
    monkeypatch.setattr(
        builder,
        "build_encoder_bundle",
        lambda *positional, **keyword: calls.update(positional=positional, keyword=keyword),
    )

    builder.build_bundle(args)

    assert calls["positional"] == (model, model_path.resolve(), output_dir.resolve())
    assert calls["keyword"]["args"] is args
    assert calls["keyword"]["metadata"] == {
        "model_type": "indic_parakeet_rnnt",
        "sample_rate": 16_000,
        "feature_count": 80,
        "subsampling_factor": 8,
        "vocabulary_size": 1024,
        "max_symbols_per_step": 10,
    }
    assert calls["keyword"]["temporary_prefix"] == ".indic-parakeet-rnnt-"
    assert calls["keyword"]["parity_message"] == "INDIC_PARAKEET_RNNT_TENSORRT_ENCODER_PARITY_PASSED"
