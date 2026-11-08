# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def shared_ray_cluster() -> None:
    """The endpoint freshness check must fail before starting any workers."""


def test_server_refuses_to_replace_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = Path(__file__).resolve().parents[2] / "tutorials/audio/granary_v2/full_english_pipeline/serve.py"
    spec = importlib.util.spec_from_file_location("full_english_serve", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    endpoint = tmp_path / "endpoint.txt"
    endpoint.write_text("http://existing-server:8000/v1\n")
    monkeypatch.setattr(sys, "argv", [str(path), "--endpoint-file", str(endpoint)])
    with pytest.raises(FileExistsError):
        module.main()
    assert endpoint.read_text() == "http://existing-server:8000/v1\n"
