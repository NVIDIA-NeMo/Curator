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

"""Unit tests for SIGMOSAdapter."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from nemo_curator.models.audio.sigmos import SIGMOSAdapter, download_sigmos_model
from nemo_curator.stages.audio.filtering.sigmos import SIGMOSFilterStage

if TYPE_CHECKING:
    from pathlib import Path

_DOWNLOAD = "nemo_curator.models.audio.sigmos.download_sigmos_model"


def test_explicit_valid_model_path_returns_without_download(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model")

    with patch(_DOWNLOAD) as download:
        assert SIGMOSAdapter(model_path=str(model_path))._resolve_model_path() == str(model_path)

    download.assert_not_called()


@pytest.mark.parametrize(
    ("name", "content", "match"), [("missing.onnx", None, "model_path"), ("empty.onnx", b"", "empty")]
)
def test_explicit_bad_model_path_does_not_download(
    tmp_path: Path, name: str, content: bytes | None, match: str
) -> None:
    model_path = tmp_path / name
    if content is not None:
        model_path.write_bytes(content)

    with patch(_DOWNLOAD) as download, pytest.raises(FileNotFoundError, match=match):
        SIGMOSAdapter(model_path=str(model_path))._resolve_model_path()

    download.assert_not_called()


def test_default_uncached_model_path_uses_download(tmp_path: Path) -> None:
    with patch(_DOWNLOAD, return_value="/cache/model.onnx") as download:
        assert SIGMOSAdapter(model_dir=str(tmp_path))._resolve_model_path() == "/cache/model.onnx"

    download.assert_called_once_with(str(tmp_path))


def test_cached_model_is_reused_without_network(tmp_path: Path) -> None:
    model_path = tmp_path / "model-sigmos_1697718653_41d092e8-epo-200.onnx"
    model_path.write_bytes(b"cached")

    with patch("nemo_curator.models.audio.sigmos.requests.get") as request:
        assert download_sigmos_model(str(tmp_path)) == str(model_path)

    request.assert_not_called()


def test_stage_setup_on_node_prefetches_only_default_model(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit.onnx"
    explicit.write_bytes(b"model")

    with patch(_DOWNLOAD) as download:
        SIGMOSFilterStage(model_path=str(explicit)).setup_on_node()
        download.assert_not_called()

        SIGMOSFilterStage(model_dir=str(tmp_path)).setup_on_node()
        download.assert_called_once_with(str(tmp_path))


def test_load_model_uses_cpu_without_gpus_and_unload_releases(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model")
    adapter = SIGMOSAdapter(model_path=str(model_path))

    with (
        patch("nemo_curator.utils.gpu_utils.ensure_cudnn_loaded"),
        patch("nemo_curator.models.audio.sigmos.build_sigmos_model") as build,
    ):
        adapter.load_model(num_gpus=0)
        adapter.load_model(num_gpus=0)

    build.assert_called_once_with(force_cpu=True, model_path=str(model_path))
    build.return_value.run.return_value = {"MOS_OVRL": 4.0}
    assert adapter.run(audio=None, sr=16000) == {"MOS_OVRL": 4.0}
    adapter.unload_model()
    with pytest.raises(RuntimeError, match="load_model"):
        adapter.run(audio=None, sr=16000)
