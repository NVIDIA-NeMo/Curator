# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import json
from pathlib import Path

import pytest

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
