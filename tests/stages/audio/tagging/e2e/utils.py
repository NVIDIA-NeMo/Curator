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

"""Utilities for audio tagging e2e tests."""

from __future__ import annotations

import json
import os
from typing import Any

import pytest


def load_manifest(manifest_file: str, encoding: str | None = None) -> list[dict[str, Any]]:
    with open(manifest_file, encoding=encoding or "utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _approx_value(v: Any) -> Any:  # noqa: ANN401
    """Wrap a single value with pytest.approx if numeric, recursing into containers."""
    if isinstance(v, dict):
        return {k: _approx_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_approx_value(item) for item in v]
    if isinstance(v, float):
        return pytest.approx(v, rel=1e-2)
    return v


def check_output(output_manifest: str, reference_manifest: str, text_key: str = "text") -> None:
    """Compare pipeline output manifest against a reference.

    Validates segment boundaries, text, metrics, and word-level alignment.
    """
    assert os.path.exists(output_manifest), f"Output manifest not found: {output_manifest}"

    output_data = load_manifest(output_manifest)
    reference_data = load_manifest(reference_manifest)

    output_data = sorted(output_data, key=lambda x: x["audio_item_id"])
    reference_data = sorted(reference_data, key=lambda x: x["audio_item_id"])

    assert len(output_data) == len(reference_data), (
        f"Entry count mismatch: output={len(output_data)}, reference={len(reference_data)}"
    )

    for out_entry, ref_entry in zip(output_data, reference_data, strict=True):
        if "segments" in ref_entry:
            assert len(out_entry["segments"]) == len(ref_entry["segments"]), (
                f"Segment count mismatch for {out_entry.get('audio_item_id')}: "
                f"output={len(out_entry['segments'])}, reference={len(ref_entry['segments'])}"
            )
            for out_seg, ref_seg in zip(out_entry["segments"], ref_entry["segments"], strict=True):
                assert out_seg["start"] == pytest.approx(ref_seg["start"], abs=0.5), (
                    f"Segment start mismatch: got {out_seg['start']}, expected {ref_seg['start']}"
                )
                assert out_seg["end"] == pytest.approx(ref_seg["end"], abs=0.5), (
                    f"Segment end mismatch: got {out_seg['end']}, expected {ref_seg['end']}"
                )
                if text_key in ref_seg:
                    assert out_seg[text_key] == ref_seg[text_key], (
                        f"Text mismatch in segment ({ref_seg['start']:.2f}-{ref_seg['end']:.2f})"
                    )
                if "metrics" in ref_seg:
                    assert out_seg["metrics"] == _approx_value(ref_seg["metrics"])

        if "alignment" in ref_entry:
            assert len(out_entry["alignment"]) == len(ref_entry["alignment"]), (
                f"Alignment word count mismatch for {out_entry.get('audio_item_id')}"
            )
            for out_word, ref_word in zip(out_entry["alignment"], ref_entry["alignment"], strict=True):
                assert out_word["word"] == ref_word["word"]
                assert out_word["start"] == pytest.approx(ref_word["start"], abs=0.5), (
                    f"Alignment start mismatch for '{ref_word['word']}': got {out_word['start']}, expected {ref_word['start']}"
                )
                assert out_word["end"] == pytest.approx(ref_word["end"], abs=0.5), (
                    f"Alignment end mismatch for '{ref_word['word']}': got {out_word['end']}, expected {ref_word['end']}"
                )
