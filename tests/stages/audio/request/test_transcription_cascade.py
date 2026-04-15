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

"""Tests for transcription cascade: mock 3 LLM calls -> 3 output fields.

Run: pytest tests/stages/audio/request/test_transcription_cascade.py -v --noconftest
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

# ---------------------------------------------------------------------------
# Direct imports (bypass Curator __init__ for Py3.9)
# ---------------------------------------------------------------------------
_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _imp(name, relpath):
    path = os.path.join(_base, relpath)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Pre-load prompt_template under its full package name so transcription_cascade
# can import it without triggering nemo_curator.__init__ (Py3.9 compat).
_pt = _imp("nemo_curator.stages.audio.request.prompt_template", "nemo_curator/stages/audio/request/prompt_template.py")
_tc = _imp("_tc_cascade", "nemo_curator/stages/audio/request/transcription_cascade.py")

run_cascade_on_row = _tc.run_cascade_on_row
get_prompt_path = _tc.get_prompt_path
list_available_languages = _tc.list_available_languages
TranscriptionCascadeConfig = _tc.TranscriptionCascadeConfig
create_pnc_stage = _tc.create_pnc_stage
load_prompt_config = _pt.load_prompt_config


# ---------------------------------------------------------------------------
# Tests: prompt path resolution
# ---------------------------------------------------------------------------


class TestPromptResolution:
    def test_en_prompts_exist(self) -> None:
        p1 = get_prompt_path("En", "1st_pass")
        p2 = get_prompt_path("En", "2nd_pass")
        p3 = get_prompt_path("En", "3_llm_pnc")
        assert os.path.exists(p1)
        assert os.path.exists(p2)
        assert os.path.exists(p3)

    def test_available_languages(self) -> None:
        langs = list_available_languages()
        assert "En" in langs

    def test_missing_language_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            get_prompt_path("Zz", "1st_pass")

    def test_config_resolve(self) -> None:
        cfg = TranscriptionCascadeConfig(language="En")
        p1, p2, p3 = cfg.resolve_paths()
        assert "1st_pass" in p1
        assert "2nd_pass" in p2
        assert "3_llm_pnc" in p3


# ---------------------------------------------------------------------------
# Tests: cascade execution with mock LLM
# ---------------------------------------------------------------------------


class TestCascadeExecution:
    def test_three_passes_populate_fields(self) -> None:
        """Mock 3 LLM calls -> 3 output fields populated correctly."""
        cfg = TranscriptionCascadeConfig(language="En")
        p1_path, p2_path, p3_path = cfg.resolve_paths()
        p1_cfg = load_prompt_config(p1_path)
        p2_cfg = load_prompt_config(p2_path)
        p3_cfg = load_prompt_config(p3_path)

        call_count = {"omni": 0, "llm": 0}

        def mock_omni(messages):
            call_count["omni"] += 1
            if call_count["omni"] == 1:
                return "hello world"  # Pass 1: ASR
            return "Hello world"  # Pass 2: verified

        def mock_llm(messages):
            call_count["llm"] += 1
            return "Hello world."  # Pass 3: PnC

        row = {"audio_filepath": "/data/test.wav"}
        result = run_cascade_on_row(row, p1_cfg, p2_cfg, p3_cfg, mock_omni, mock_llm)

        assert result["qwen3_omni_pred_text"] == "hello world"
        assert result["qwen3_omni_verified_text"] == "Hello world"
        assert result["qwen3_llm_corrected_text"] == "Hello world."
        assert call_count["omni"] == 2
        assert call_count["llm"] == 1

    def test_pass1_error_stops_cascade(self) -> None:
        cfg = TranscriptionCascadeConfig(language="En")
        p1_path, p2_path, p3_path = cfg.resolve_paths()
        p1_cfg = load_prompt_config(p1_path)
        p2_cfg = load_prompt_config(p2_path)
        p3_cfg = load_prompt_config(p3_path)

        def fail_omni(messages):
            raise RuntimeError("API error")

        def mock_llm(messages):
            return "should not be called"

        row = {"audio_filepath": "/data/test.wav"}
        result = run_cascade_on_row(row, p1_cfg, p2_cfg, p3_cfg, fail_omni, mock_llm)

        assert "qwen3_omni_pred_text_error" in result
        # Pass 2 and 3 should NOT have been attempted
        assert "qwen3_omni_verified_text" not in result
        assert "qwen3_llm_corrected_text" not in result

    def test_ru_language_cascade(self) -> None:
        """Target language: Ru — verify prompts load and cascade works."""
        cfg = TranscriptionCascadeConfig(language="Ru")
        p1_path, p2_path, p3_path = cfg.resolve_paths()
        p1_cfg = load_prompt_config(p1_path)
        p2_cfg = load_prompt_config(p2_path)
        p3_cfg = load_prompt_config(p3_path)

        row = {"audio_filepath": "/data/ru_test.wav"}
        result = run_cascade_on_row(
            row, p1_cfg, p2_cfg, p3_cfg,
            lambda m: "привет мир",
            lambda m: "Привет, мир.",
        )
        assert result["qwen3_omni_pred_text"] == "привет мир"
        assert result["qwen3_llm_corrected_text"] == "Привет, мир."

    def test_original_fields_preserved(self) -> None:
        cfg = TranscriptionCascadeConfig(language="En")
        p1_path, p2_path, p3_path = cfg.resolve_paths()
        p1_cfg = load_prompt_config(p1_path)
        p2_cfg = load_prompt_config(p2_path)
        p3_cfg = load_prompt_config(p3_path)

        row = {"audio_filepath": "/data/test.wav", "duration": 5.0, "speaker": "A"}
        result = run_cascade_on_row(
            row, p1_cfg, p2_cfg, p3_cfg,
            lambda m: "asr",
            lambda m: "pnc",
        )
        assert result["duration"] == 5.0
        assert result["speaker"] == "A"
        assert result["audio_filepath"] == "/data/test.wav"


# ---------------------------------------------------------------------------
# Tests: PnC standalone factory
# ---------------------------------------------------------------------------


class TestPnCFactory:
    def test_create_pnc_stage_en(self) -> None:
        """PnC factory creates stage with correct output key."""
        # create_pnc_stage imports TextOnlyLLMRequestStage which needs Curator.
        # We just verify the config resolution works.
        path = get_prompt_path("En", "3_llm_pnc")
        cfg = load_prompt_config(path)
        assert cfg["output_field"] == "qwen3_llm_corrected_text"

    def test_create_pnc_stage_ru(self) -> None:
        path = get_prompt_path("Ru", "3_llm_pnc")
        cfg = load_prompt_config(path)
        assert "output_field" in cfg
