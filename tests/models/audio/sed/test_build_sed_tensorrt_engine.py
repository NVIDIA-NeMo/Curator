# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemo_curator.models.audio.sed import build_sed_tensorrt_engine as builder
from nemo_curator.models.audio.sed.build_sed_tensorrt_engine import _write_metadata_sidecar
from nemo_curator.utils import atomic_io


def test_metadata_sidecar_failure_preserves_previous_complete_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_path = tmp_path / "cnn14.plan.json"
    previous = {"engine_sha256": "previous"}
    metadata_path.write_text(json.dumps(previous))
    monkeypatch.setattr(atomic_io.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("interrupted")))

    with pytest.raises(OSError, match="interrupted"):
        _write_metadata_sidecar(metadata_path, {"engine_sha256": "new"})

    assert json.loads(metadata_path.read_text()) == previous
    assert list(tmp_path.glob(".*.tmp")) == []


def test_metadata_sidecar_is_published_as_complete_json(tmp_path: Path) -> None:
    metadata_path = tmp_path / "cnn14.plan.json"
    metadata = {"engine_sha256": "new", "profiles": {"logmel": {"max": [32, 1, 4001, 64]}}}

    _write_metadata_sidecar(metadata_path, metadata)

    assert json.loads(metadata_path.read_text()) == metadata
    assert metadata_path.read_text().endswith("\n")


@pytest.mark.parametrize("sample_rate", [16000, 32000])
def test_load_model_uses_requested_sample_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sample_rate: int
) -> None:
    model_class = MagicMock()
    monkeypatch.setitem(
        sys.modules,
        "nemo_curator.models.audio.sed.cnn14",
        SimpleNamespace(Cnn14DecisionLevelMax=model_class),
    )
    checkpoint = {"model": {"weights": "loaded"}}
    loader = MagicMock(return_value=checkpoint)
    monkeypatch.setattr(builder.torch, "load", loader)

    builder._load_model(tmp_path / "cnn14.pth", sample_rate)

    model_class.assert_called_once_with(sample_rate=sample_rate)
    model_class.return_value.load_state_dict.assert_called_once_with(checkpoint["model"])
    loader.assert_called_once_with(tmp_path / "cnn14.pth", map_location="cpu", weights_only=True)


@pytest.mark.parametrize("sample_rate", [None, 16000, 32000])
def test_cli_threads_sample_rate_to_export_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sample_rate: int | None
) -> None:
    checkpoint = tmp_path / "cnn14.pth"
    checkpoint.write_bytes(b"checkpoint")
    engine = tmp_path / "cnn14.plan"
    argv = ["build_sed", "--checkpoint", str(checkpoint), "--output", str(engine)]
    if sample_rate is not None:
        argv.extend(["--sample-rate", str(sample_rate)])
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(builder.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(builder.torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(builder.torch.cuda, "get_device_name", lambda: "test GPU")
    monkeypatch.setattr(builder, "_driver_version", lambda: "test driver")
    exported_rates = []

    def export_onnx(_checkpoint: Path, output: Path, rate: int) -> tuple[None, None, float]:
        exported_rates.append(rate)
        output.write_bytes(b"onnx")
        return None, None, 0.0

    def build_engine(_args: object, _onnx_path: Path) -> tuple[float, str]:
        engine.write_bytes(b"engine")
        return 1.0, "test version"

    monkeypatch.setattr(builder, "_export_onnx", export_onnx)
    monkeypatch.setattr(builder, "_build_engine", build_engine)

    builder.main()

    expected_rate = sample_rate or 32000
    metadata = json.loads(engine.with_suffix(".plan.json").read_text())
    assert exported_rates == [expected_rate]
    assert metadata["sample_rate"] == expected_rate
    assert metadata["frontend"]["sample_rate"] == expected_rate
