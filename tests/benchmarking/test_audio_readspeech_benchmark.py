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

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "benchmarking" / "scripts" / "audio_readspeech_benchmark.py"
sys.path.insert(0, str(_SCRIPT_PATH.parent))
_SPEC = importlib.util.spec_from_file_location("audio_readspeech_benchmark", _SCRIPT_PATH)
assert _SPEC
assert _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_create_partial_manifest_selects_deterministic_prefix(tmp_path: Path) -> None:
    input_manifest = tmp_path / "input.jsonl"
    output_manifest = tmp_path / "scratch" / "partial.jsonl"
    _write_manifest(
        input_manifest,
        [
            {"audio_filepath": "audio/first.flac", "text": "first"},
            {"audio_filepath": "audio/second.flac", "text": "second"},
            {"audio_filepath": "audio/third.flac", "text": "third"},
        ],
    )

    selected = _MODULE._create_partial_manifest(input_manifest, output_manifest, max_samples=2)

    rows = [json.loads(line) for line in output_manifest.read_text(encoding="utf-8").splitlines()]
    assert selected == 2
    assert [row["text"] for row in rows] == ["first", "second"]
    assert rows[0]["audio_filepath"] == str((tmp_path / "audio" / "first.flac").resolve())


def test_create_partial_manifest_requires_requested_rows(tmp_path: Path) -> None:
    input_manifest = tmp_path / "input.jsonl"
    _write_manifest(input_manifest, [{"audio_filepath": "/audio/only.flac"}])

    with pytest.raises(RuntimeError, match="has only 1 rows; requested 2"):
        _MODULE._create_partial_manifest(input_manifest, tmp_path / "partial.jsonl", max_samples=2)


def test_create_partial_manifest_rejects_duplicate_audio_paths(tmp_path: Path) -> None:
    input_manifest = tmp_path / "input.jsonl"
    _write_manifest(
        input_manifest,
        [
            {"audio_filepath": "/audio/duplicate.flac"},
            {"audio_filepath": "/audio/duplicate.flac"},
        ],
    )

    with pytest.raises(RuntimeError, match="Duplicate input audio_filepath"):
        _MODULE._create_partial_manifest(input_manifest, tmp_path / "partial.jsonl", max_samples=-1)


def test_require_output_segments_rejects_empty_output() -> None:
    with pytest.raises(RuntimeError, match="produced no output segments"):
        _MODULE._require_output_segments(0)
