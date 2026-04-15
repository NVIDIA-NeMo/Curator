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

"""Minimal tests for prompt template utilities and TextOnlyLLMRequestStage.

Run: pytest tests/stages/audio/request/test_prompt_template.py -v --noconftest
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Direct import (bypass Curator __init__ chain)
# ---------------------------------------------------------------------------
def _imp(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_base = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
_pt = _imp("_test_pt", os.path.abspath(os.path.join(_base, "nemo_curator", "stages", "audio", "request", "prompt_template.py")))

build_prompt_conversation = _pt.build_prompt_conversation
load_prompt_config = _pt.load_prompt_config


# ---------------------------------------------------------------------------
# Tests: build_prompt_conversation
# ---------------------------------------------------------------------------


class TestBuildPromptConversation:
    def test_simple_text_template(self) -> None:
        cfg = {
            "input_fields": {"pred_text": "qwen3_omni_pred_text"},
            "conversation": [
                {"role": "user", "content": "Fix punctuation: {pred_text}"},
            ],
        }
        sample = {"qwen3_omni_pred_text": "hello world"}
        result = build_prompt_conversation(sample, cfg)
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Fix punctuation: hello world"

    def test_multimodal_content_list(self) -> None:
        cfg = {
            "input_fields": {"audio_filepath": "audio_filepath"},
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": "{audio_filepath}"},
                        {"type": "text", "text": "Transcribe this."},
                    ],
                }
            ],
        }
        sample = {"audio_filepath": "/data/test.wav"}
        result = build_prompt_conversation(sample, cfg)
        parts = result[0]["content"]
        assert parts[0]["audio"] == "/data/test.wav"
        assert parts[1]["text"] == "Transcribe this."

    def test_system_plus_user(self) -> None:
        cfg = {
            "input_fields": {"text": "input_text"},
            "conversation": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Process: {text}"},
            ],
        }
        sample = {"input_text": "test"}
        result = build_prompt_conversation(sample, cfg)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["content"] == "Process: test"

    def test_missing_field_keeps_placeholder(self) -> None:
        cfg = {
            "input_fields": {"missing_key": "nonexistent_field"},
            "conversation": [
                {"role": "user", "content": "Value: {missing_key}"},
            ],
        }
        sample = {}
        result = build_prompt_conversation(sample, cfg)
        # Missing value maps to empty string from .get()
        assert result[0]["content"] == "Value: "


class TestLoadPromptConfig:
    def test_load_yaml(self, tmp_path: Path) -> None:
        yaml_content = """
input_fields:
  text: input_text
output_field: corrected_text
conversation:
  - role: user
    content: "Fix: {text}"
"""
        p = tmp_path / "test_prompt.yaml"
        p.write_text(yaml_content)
        cfg = load_prompt_config(str(p))
        assert cfg["output_field"] == "corrected_text"
        assert cfg["input_fields"]["text"] == "input_text"

    def test_cascade_pass2_format(self, tmp_path: Path) -> None:
        """Test the 2nd-pass format with audio + text content."""
        yaml_content = """
input_fields:
  audio_filepath: audio_filepath
  pred_text: qwen3_omni_pred_text
output_field: qwen3_omni_verified_text
conversation:
  - role: user
    content:
      - type: audio
        audio: "{audio_filepath}"
      - type: text
        text: "Verify against audio: {pred_text}"
"""
        p = tmp_path / "pass2.yaml"
        p.write_text(yaml_content)
        cfg = load_prompt_config(str(p))
        sample = {"audio_filepath": "/data/a.wav", "qwen3_omni_pred_text": "hello"}
        conv = build_prompt_conversation(sample, cfg)
        parts = conv[0]["content"]
        assert parts[0]["audio"] == "/data/a.wav"
        assert "hello" in parts[1]["text"]
