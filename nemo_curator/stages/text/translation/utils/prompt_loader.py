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

"""Shared prompt-loading helpers for translation stages."""

from __future__ import annotations

from pathlib import Path

import yaml

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt_template(filename: str) -> tuple[str, str]:
    """Load a YAML prompt file and return ``(system_prompt, user_template)``.

    Parameters
    ----------
    filename : str
        Name of the YAML file inside the ``prompts/`` directory
        (e.g. ``"translate.yaml"`` or ``"faith_eval.yaml"``).

    Returns
    -------
    tuple[str, str]
        ``(system prompt, user template)``.

    Raises
    ------
    FileNotFoundError
        If the prompt file does not exist.
    ValueError
        If the YAML is malformed or does not contain a top-level mapping.
    KeyError
        If the top-level mapping is missing the ``system`` or ``user`` key.
    """
    prompt_path = _PROMPT_DIR / filename
    try:
        with open(prompt_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed prompt template {prompt_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Prompt template {prompt_path} must contain a top-level mapping, "
            f"got {type(data).__name__}"
        )
    missing = [k for k in ("system", "user") if k not in data]
    if missing:
        raise KeyError(
            f"Prompt template {prompt_path} is missing required keys: {missing}"
        )
    return data["system"], data["user"]
