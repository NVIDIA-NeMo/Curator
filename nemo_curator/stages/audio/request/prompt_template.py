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

"""Prompt template utilities for the transcription cascade.

Implements the same YAML-based prompt config format used by ameister's
hifi_granary ``omni_text_normalization/cascade/``.  A prompt config is a
YAML dict with three keys:

    input_fields:
      prompt_placeholder: manifest_field_name
    output_field: output_column_name
    conversation:
      - role: user
        content:
          - type: text
            text: "Transcribe: {prompt_placeholder}"

``build_prompt_conversation()`` renders ``{placeholders}`` in the
conversation template from per-row field values.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

_PLACEHOLDER_RE = re.compile(r"\{([^{}]+)\}")


def load_prompt_config(path: str) -> dict[str, Any]:
    """Load a YAML prompt config file."""
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_prompt_conversation(
    sample: dict[str, Any],
    prompt_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Render a prompt config into a conversation with per-row field values.

    Args:
        sample: Row data (dict from the manifest).
        prompt_cfg: Loaded YAML prompt config with ``input_fields`` mapping
            and ``conversation`` template.

    Returns:
        OpenAI-style conversation list with all ``{placeholder}`` values
        substituted from ``sample``.
    """
    conversation = deepcopy(prompt_cfg["conversation"])
    prompt_values: dict[str, Any] = {}
    for prompt_field, sample_field in prompt_cfg.get("input_fields", {}).items():
        prompt_values[prompt_field] = sample.get(sample_field, "")

    for turn in conversation:
        content = turn["content"]
        if isinstance(content, str):
            turn["content"] = _render_template(content, prompt_values)
        elif isinstance(content, list):
            for part in content:
                content_type = part.get("type", "")
                if content_type in part and isinstance(part[content_type], str):
                    part[content_type] = _render_template(part[content_type], prompt_values)
    return conversation


def _render_template(template: str, values: dict[str, Any]) -> str:
    """Replace ``{key}`` placeholders in a template string."""
    def _replace(match: re.Match) -> str:
        key = match.group(1)
        if key not in values:
            return match.group(0)
        return str(values[key])

    return _PLACEHOLDER_RE.sub(_replace, template)
