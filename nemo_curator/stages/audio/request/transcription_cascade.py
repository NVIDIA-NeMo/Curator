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

"""Three-pass transcription cascade: ASR -> Verification -> PnC.

Orchestrates three sequential LLM passes using prompt YAML configs:

- **Pass 1** (Qwen3-Omni): Audio-only ASR + number normalization.
  Output: ``qwen3_omni_pred_text``
- **Pass 2** (Qwen3-Omni): Audio + draft text verification.
  Output: ``qwen3_omni_verified_text``
- **Pass 3** (Qwen3-LLM): Text-only punctuation & capitalization.
  Output: ``qwen3_llm_corrected_text``

Each pass uses a YAML prompt config bundled in ``prompts/{language}/``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from nemo_curator.stages.audio.request.prompt_template import (
    build_prompt_conversation,
    load_prompt_config,
)

# Bundled prompts directory
_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def get_prompt_path(language: str, pass_name: str) -> str:
    """Resolve path to a bundled prompt YAML.

    Args:
        language: Language code (e.g. ``"En"``, ``"Ru"``).
        pass_name: Pass filename (e.g. ``"1st_pass"``, ``"2nd_pass"``, ``"3_llm_pnc"``).

    Returns:
        Absolute path to the YAML file.

    Raises:
        FileNotFoundError: If the prompt file doesn't exist.
    """
    path = os.path.join(_PROMPTS_DIR, language, f"{pass_name}.yaml")
    if not os.path.exists(path):
        msg = f"Prompt config not found: {path}"
        raise FileNotFoundError(msg)
    return path


def list_available_languages() -> list[str]:
    """Return language codes with bundled prompts."""
    if not os.path.isdir(_PROMPTS_DIR):
        return []
    return sorted(
        d for d in os.listdir(_PROMPTS_DIR) if os.path.isdir(os.path.join(_PROMPTS_DIR, d))
    )


@dataclass
class TranscriptionCascadeConfig:
    """Configuration for the 3-pass transcription cascade.

    Args:
        language: Language code for prompt selection (e.g. ``"En"``).
        pass1_prompt_path: Override path for Pass 1 YAML (auto-resolved if empty).
        pass2_prompt_path: Override path for Pass 2 YAML (auto-resolved if empty).
        pass3_prompt_path: Override path for Pass 3 YAML (auto-resolved if empty).
    """

    language: str = "En"
    pass1_prompt_path: str = ""
    pass2_prompt_path: str = ""
    pass3_prompt_path: str = ""

    def resolve_paths(self) -> tuple[str, str, str]:
        """Return resolved paths for all 3 passes."""
        p1 = self.pass1_prompt_path or get_prompt_path(self.language, "1st_pass")
        p2 = self.pass2_prompt_path or get_prompt_path(self.language, "2nd_pass")
        p3 = self.pass3_prompt_path or get_prompt_path(self.language, "3_llm_pnc")
        return p1, p2, p3


def create_pnc_stage(
    client: Any,
    model_name: str,
    language: str = "En",
    generation_config: dict[str, Any] | None = None,
    prompt_config_path: str = "",
) -> Any:
    """Create a standalone Punctuation & Capitalization stage.

    Convenience factory that configures a ``TextOnlyLLMRequestStage`` with the
    Pass 3 (PnC) prompt for the given language.

    Args:
        client: ``LLMClient`` or ``AsyncLLMClient``.
        model_name: LLM model name (e.g. ``"Qwen/Qwen3-30B-A3B-Instruct"``).
        language: Language code (default ``"En"``). ``"Ru"`` also available.
        generation_config: Optional LLM generation params.
        prompt_config_path: Override prompt YAML path (auto-resolved if empty).

    Returns:
        Configured ``TextOnlyLLMRequestStage``.
    """
    from nemo_curator.stages.audio.request.text_only_llm_request import TextOnlyLLMRequestStage

    path = prompt_config_path or get_prompt_path(language, "3_llm_pnc")
    return TextOnlyLLMRequestStage(
        client=client,
        model_name=model_name,
        generation_config=generation_config or {},
        prompt_config_path=path,
    )


def run_cascade_on_row(
    row: dict[str, Any],
    pass1_cfg: dict[str, Any],
    pass2_cfg: dict[str, Any],
    pass3_cfg: dict[str, Any],
    omni_query_fn: Any,
    llm_query_fn: Any,
) -> dict[str, Any]:
    """Run the 3-pass cascade on a single manifest row.

    Args:
        row: Manifest row dict (must contain ``audio_filepath``).
        pass1_cfg: Loaded YAML config for Pass 1.
        pass2_cfg: Loaded YAML config for Pass 2.
        pass3_cfg: Loaded YAML config for Pass 3.
        omni_query_fn: Callable(messages) -> str for Qwen3-Omni (Passes 1 & 2).
        llm_query_fn: Callable(messages) -> str for Qwen3-LLM (Pass 3).

    Returns:
        Updated row dict with all three output fields populated.
    """
    result = dict(row)

    # Pass 1: ASR + number normalization
    p1_messages = build_prompt_conversation(result, pass1_cfg)
    try:
        result[pass1_cfg["output_field"]] = omni_query_fn(p1_messages)
    except Exception as e:
        result[pass1_cfg["output_field"] + "_error"] = str(e)
        logger.warning(f"Pass 1 failed: {e}")
        return result

    # Pass 2: Verification
    p2_messages = build_prompt_conversation(result, pass2_cfg)
    try:
        result[pass2_cfg["output_field"]] = omni_query_fn(p2_messages)
    except Exception as e:
        result[pass2_cfg["output_field"] + "_error"] = str(e)
        logger.warning(f"Pass 2 failed: {e}")
        return result

    # Pass 3: PnC (text-only)
    p3_messages = build_prompt_conversation(result, pass3_cfg)
    try:
        result[pass3_cfg["output_field"]] = llm_query_fn(p3_messages)
    except Exception as e:
        result[pass3_cfg["output_field"] + "_error"] = str(e)
        logger.warning(f"Pass 3 failed: {e}")

    return result
