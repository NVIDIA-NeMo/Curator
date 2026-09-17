# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# ruff: noqa: INP001

from argparse import Namespace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pytest


def _load_encoder_utils() -> ModuleType:
    module_path = Path(__file__).parents[2] / "scripts" / "tensorrt_encoder_utils.py"
    spec = spec_from_file_location("curator_tensorrt_encoder_utils", module_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load {module_path}"
        raise RuntimeError(msg)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


encoder_utils = _load_encoder_utils()


def _args() -> Namespace:
    return Namespace(
        min_batch=1,
        opt_batch=2,
        max_batch=4,
        min_frames=8,
        opt_frames=80,
        max_frames=4001,
    )


def _stub_build_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        encoder_utils,
        "export_encoder",
        lambda _model, path, **_kwargs: path.write_text("onnx"),
    )

    def build_engine(_onnx_path: Path, engine_path: Path, **_kwargs: object) -> str:
        engine_path.write_text("engine")
        return "10.9"

    monkeypatch.setattr(encoder_utils, "build_engine", build_engine)


def test_bundle_publication_is_all_or_nothing_and_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "source.nemo"
    model_path.write_text("model")
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    _stub_build_steps(monkeypatch)
    monkeypatch.setattr(
        encoder_utils,
        "validate_engine",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("validation failed")),
    )

    with pytest.raises(RuntimeError, match="validation failed"):
        encoder_utils.build_encoder_bundle(
            object(),
            model_path,
            output_dir,
            args=_args(),
            metadata={"feature_count": 80},
            temporary_prefix="bundle-test-",
            parity_message="parity passed",
        )

    assert output_dir.is_dir()
    assert list(output_dir.iterdir()) == []

    monkeypatch.setattr(encoder_utils, "validate_engine", lambda *_args, **_kwargs: None)
    encoder_utils.build_encoder_bundle(
        object(),
        model_path,
        output_dir,
        args=_args(),
        metadata={"feature_count": 80},
        temporary_prefix="bundle-test-",
        parity_message="parity passed",
    )

    assert sorted(path.name for path in output_dir.iterdir()) == ["encoder.plan", "metadata.json", "model.nemo"]


def test_bundle_publication_rejects_a_nonempty_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "source.nemo"
    model_path.write_text("model")
    output_dir = tmp_path / "bundle"
    output_dir.mkdir()
    (output_dir / "unrelated.txt").write_text("keep")
    _stub_build_steps(monkeypatch)

    with pytest.raises(FileExistsError, match="must be empty"):
        encoder_utils.build_encoder_bundle(
            object(),
            model_path,
            output_dir,
            args=_args(),
            metadata={"feature_count": 80},
            temporary_prefix="bundle-test-",
            parity_message="parity passed",
        )

    assert (output_dir / "unrelated.txt").read_text() == "keep"
