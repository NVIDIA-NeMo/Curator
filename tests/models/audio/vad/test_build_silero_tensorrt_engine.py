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

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from nemo_curator.models.audio.vad import build_silero_tensorrt_engine as builder_module
from nemo_curator.models.audio.vad.build_silero_tensorrt_engine import (
    _parse_args,
    _sha256,
    _shape_for_batch,
    _validation_batch_sizes,
    _write_metadata_sidecar,
)
from nemo_curator.utils import atomic_io


def test_shape_for_batch_resolves_dynamic_dimensions() -> None:
    assert _shape_for_batch((-1, 576), 8) == (8, 576)
    assert _shape_for_batch((-1, 2, 128), 64) == (64, 2, 128)


def test_parse_args_has_reference_profile_defaults(tmp_path: Path) -> None:
    args = _parse_args(["--output", str(tmp_path / "silero.plan")])

    assert (args.min_batch, args.opt_batch, args.max_batch) == (1, 16, 64)
    assert args.workspace_gb == 2
    assert args.fp16 is False


def test_validation_batches_stay_inside_small_profiles() -> None:
    assert _validation_batch_sizes(4) == (1, 4)
    assert _validation_batch_sizes(1) == (1,)


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--min-batch", "0"],
        ["--min-batch", "2", "--opt-batch", "2"],
        ["--min-batch", "8", "--opt-batch", "4"],
        ["--opt-batch", "65"],
        ["--workspace-gb", "0"],
    ],
)
def test_parse_args_rejects_invalid_profiles(tmp_path: Path, extra_args: list[str]) -> None:
    with pytest.raises(SystemExit):
        _parse_args(["--output", str(tmp_path / "silero.plan"), *extra_args])


def test_sha256_hashes_complete_file(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    contents = b"silero-tensorrt\x00artifact"
    artifact.write_bytes(contents)

    assert _sha256(artifact) == hashlib.sha256(contents).hexdigest()


def test_metadata_sidecar_failure_preserves_previous_complete_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_path = tmp_path / "silero.plan.json"
    previous = {"engine_sha256": "previous"}
    metadata_path.write_text(json.dumps(previous))
    monkeypatch.setattr(atomic_io.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("interrupted")))

    with pytest.raises(OSError, match="interrupted"):
        _write_metadata_sidecar(metadata_path, {"engine_sha256": "new"})

    assert json.loads(metadata_path.read_text()) == previous
    assert list(tmp_path.glob(".*.tmp")) == []


def test_metadata_sidecar_is_published_as_complete_json(tmp_path: Path) -> None:
    metadata_path = tmp_path / "silero.plan.json"
    metadata = {
        "engine_sha256": "new",
        "profiles": {"input": {"max": [64, 576]}, "state": {"max": [64, 2, 128]}},
    }

    _write_metadata_sidecar(metadata_path, metadata)

    assert json.loads(metadata_path.read_text()) == metadata
    assert metadata_path.read_text().endswith("\n")


@pytest.mark.parametrize("onnx_name", ["silero.plan", "silero.plan.json", "silero.plan.part"])
def test_main_rejects_colliding_output_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, onnx_name: str) -> None:
    args = argparse.Namespace(
        output=tmp_path / "silero.plan",
        onnx_output=tmp_path / onnx_name,
        force=True,
    )
    monkeypatch.setattr(builder_module, "_parse_args", lambda _argv: args)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    with pytest.raises(ValueError, match="output paths must be distinct"):
        builder_module.main([])


@pytest.mark.parametrize("existing_name", ["silero.plan", "silero.onnx", "silero.plan.json", "silero.plan.part"])
def test_main_preserves_existing_bundle_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing_name: str
) -> None:
    (tmp_path / existing_name).touch()
    args = argparse.Namespace(
        output=tmp_path / "silero.plan",
        onnx_output=None,
        force=False,
    )
    monkeypatch.setattr(builder_module, "_parse_args", lambda _argv: args)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    with pytest.raises(RuntimeError, match="Silero artifacts already exist"):
        builder_module.main([])
