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


def load_manifest(manifest_file: str, encoding: str | None = None) -> list[dict[str, Any]]:
    with open(manifest_file, encoding=encoding or "utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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
        assert out_entry[text_key] == ref_entry[text_key], f"Text mismatch for {out_entry.get('audio_item_id')}"

        for out_seg, ref_seg in zip(out_entry["segments"], ref_entry["segments"], strict=True):
            assert out_seg["start"] == ref_seg["start"]
            assert out_seg["end"] == ref_seg["end"]
            assert out_seg["text"] == ref_seg["text"]
            assert out_seg["metrics"] == ref_seg["metrics"]

        for out_word, ref_word in zip(out_entry["alignment"], ref_entry["alignment"], strict=True):
            assert out_word["word"] == ref_word["word"]
            assert round(out_word["start"], 2) == round(ref_word["start"], 2)
            assert round(out_word["end"], 2) == round(ref_word["end"], 2)
