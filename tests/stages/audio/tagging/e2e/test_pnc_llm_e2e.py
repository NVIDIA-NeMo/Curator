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

"""End-to-end tests for the PNC with LLM audio tagging pipeline.

Three test scenarios matching generic-sdp test_asr_pnc_with_llm.py:

1. **Simple PNC LLM** -- ManifestReader -> PNC -> Clean -> Write.
2. **2nd-pass ASR** -- mirrors yt_pnc_with_llm_after_second_pass_asr.yaml.
3. **1st-pass ASR** -- mirrors yt_pnc_with_llm_after_first_pass_asr.yaml.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.config.run import create_pipeline_from_yaml
from tests import FIXTURES_DIR

from .conftest import CONFIGS_DIR, REFERENCE_DIR
from .utils import check_output

logger = logging.getLogger(__name__)

TAGGING_FIXTURES_DIR = FIXTURES_DIR / "audio" / "tagging"


def run_and_test_e2e(  # noqa: PLR0913
    tmp_path: Path,
    input_manifest_file: str,
    config_path: Path,
    final_manifest: Path,
    reference_manifest_file: str,
    text_key: str,
    prompt_file: str | None = None,
) -> None:
    cfg = OmegaConf.load(config_path)
    cfg.input_manifest = str(input_manifest_file)
    cfg.final_manifest = str(final_manifest)
    cfg.workspace_dir = str(tmp_path)
    cfg.language_short = "en"

    if prompt_file:
        cfg.prompt_file = str(prompt_file)

    if OmegaConf.is_missing(cfg, "resampled_audio_dir"):
        cfg.resampled_audio_dir = str(tmp_path / "audio_resampled")
    if hasattr(cfg, "hf_token"):
        cfg.hf_token = os.getenv("HF_TOKEN", "")

    pipeline = create_pipeline_from_yaml(cfg)
    executor = XennaExecutor(config={"execution_mode": "batch"})
    pipeline.run(executor)

    ref_path = Path(reference_manifest_file)
    if not ref_path.exists() or ref_path.stat().st_size == 0:
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cfg.final_manifest, ref_path)
        logger.warning("Reference manifest generated: %s", ref_path)
        return

    check_output(cfg.final_manifest, reference_manifest_file, text_key)


@pytest.fixture
def pnc_simple_manifest(tmp_path: Path) -> str:
    """Input manifest with pre-populated text_2 for the simple PNC test (no ASR stage)."""
    import json

    audio_dir = TAGGING_FIXTURES_DIR / "audios"
    manifest_path = tmp_path / "manifest_simple.jsonl"

    sample_texts = [
        "hello world this is a test of the punctuation and capitalization system",
        "the quick brown fox jumps over the lazy dog it was a sunny day",
    ]

    target_files = sorted(["audio_1.opus", "audio_2.opus"])
    with open(manifest_path, "w", encoding="utf-8") as fout:
        for fname, text in zip(target_files, sample_texts, strict=True):
            entry = {
                "audio_filepath": str(audio_dir / fname),
                "audio_item_id": Path(fname).stem,
                "text_2": text,
            }
            fout.write(json.dumps(entry) + "\n")

    return str(manifest_path)


PROMPT_FILE = CONFIGS_DIR / "prompts" / "prompt_en.yaml"


@pytest.mark.gpu
def test_pnc_llm_simple(tmp_path: Path, pnc_simple_manifest: str) -> None:
    """Simple PNC LLM: ManifestReader -> vLLM PNC -> Clean -> Write."""
    config_path = CONFIGS_DIR / "pnc_llm_pipeline.yaml"
    reference_manifest_file = str(REFERENCE_DIR / "pnc_llm" / "test_data_ref_pnc_llm_simple.jsonl")
    final_manifest = tmp_path / "output_manifest_pnc_llm_simple.jsonl"

    run_and_test_e2e(
        tmp_path,
        pnc_simple_manifest,
        config_path,
        final_manifest,
        reference_manifest_file,
        text_key="text_2",
        prompt_file=PROMPT_FILE,
    )


@pytest.mark.gpu
def test_pnc_llm_second_pass(tmp_path: Path) -> None:
    """2nd-pass ASR: Resample -> PrepareSegments -> ASR (segment-only) ->
    PNC -> Clean -> BERT -> WER -> Write.

    Mirrors generic-sdp test_pnc_llm_1 / yt_pnc_with_llm_after_second_pass_asr.yaml.
    """
    config_path = CONFIGS_DIR / "pnc_llm_pipeline_second_pass.yaml"
    reference_manifest_file = str(REFERENCE_DIR / "pnc_llm" / "test_data_ref_pnc_llm_2.jsonl")
    final_manifest = tmp_path / "output_manifest_pnc_llm_2.jsonl"
    input_manifest_file = TAGGING_FIXTURES_DIR / "base_manifest.jsonl"

    run_and_test_e2e(
        tmp_path,
        input_manifest_file,
        config_path,
        final_manifest,
        reference_manifest_file,
        text_key="text_2",
        prompt_file=PROMPT_FILE,
    )


@pytest.mark.gpu
@pytest.mark.skipif(not os.getenv("HF_TOKEN"), reason="HF_TOKEN required for PyAnnote models")
def test_pnc_llm_first_pass(tmp_path: Path, get_input_manifest: str) -> None:
    """1st-pass ASR: Resample -> Diarize -> Split -> ASR Align ->
    PNC (top-level) -> Clean (alignment update) -> BERT -> Join ->
    Merge -> SQUIM -> Bandwidth -> Write.

    Mirrors generic-sdp test_pnc_llm_2 / yt_pnc_with_llm_after_first_pass_asr.yaml.
    """
    config_path = CONFIGS_DIR / "pnc_llm_pipeline_first_pass.yaml"
    reference_manifest_file = str(REFERENCE_DIR / "pnc_llm" / "test_data_ref_pnc_llm_1.jsonl")
    final_manifest = tmp_path / "output_manifest_pnc_llm_1.jsonl"

    run_and_test_e2e(
        tmp_path,
        get_input_manifest,
        config_path,
        final_manifest,
        reference_manifest_file,
        text_key="text_2",
        prompt_file=PROMPT_FILE,
    )
