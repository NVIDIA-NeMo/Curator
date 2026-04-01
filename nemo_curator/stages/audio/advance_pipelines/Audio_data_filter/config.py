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

"""
Configuration loader for the Audio Data Filter pipeline.

Loads pipeline parameters from a YAML config file organised by stage.
Users edit the YAML to override defaults without touching code.

Example:
    from nemo_curator.stages.audio.advance_pipelines.Audio_data_filter.config import (
        load_config,
    )

    # Load defaults
    cfg = load_config()

    # Load user overrides (only specified values override defaults)
    cfg = load_config("/path/to/my_config.yaml")
"""

import copy
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

SUPPORTED_AUDIO_FORMATS: tuple[str, ...] = (
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
    ".opus",
    ".webm",
)

DEFAULT_OUTPUT_FORMAT: str = "wav"

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "default_config.yaml"


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* into *base*, returning a new dict."""
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load pipeline configuration from YAML.

    Loads the shipped default config and deep-merges any user overrides
    on top.  Only the values explicitly set in the user file override
    defaults; everything else keeps its default value.

    Args:
        config_path: Path to a user YAML config file.  When *None*,
            the built-in ``default_config.yaml`` is used as-is.

    Returns:
        A nested dict keyed by stage name with parameter values.

    Raises:
        FileNotFoundError: If *config_path* does not exist.
        yaml.YAMLError: If either YAML file is malformed.
    """
    with open(_DEFAULT_CONFIG_PATH) as fh:
        defaults = yaml.safe_load(fh)

    if config_path is None:
        return defaults

    user_path = Path(config_path)
    if not user_path.is_file():
        msg = f"Config file not found: {user_path}"
        raise FileNotFoundError(msg)

    with open(user_path) as fh:
        user_cfg = yaml.safe_load(fh)

    if not user_cfg:
        logger.warning(f"User config file is empty, using defaults: {user_path}")
        return defaults

    unknown_sections = set(user_cfg) - set(defaults)
    if unknown_sections:
        logger.warning(f"Unknown config sections (ignored): {unknown_sections}")

    merged = _deep_merge(defaults, user_cfg)

    _validate(merged)

    return merged


def _validate(cfg: dict[str, Any]) -> None:
    """Validate cross-field constraints after merge."""
    vad = cfg.get("vad", {})
    if vad.get("enable", True):
        mn = vad.get("min_duration_sec", 0)
        mx = vad.get("max_duration_sec", float("inf"))
        if mn >= mx:
            msg = f"vad.min_duration_sec ({mn}) must be less than vad.max_duration_sec ({mx})"
            raise ValueError(msg)

    concat = cfg.get("concatenation", {})
    silence = concat.get("silence_duration_sec", 0)
    if silence < 0:
        msg = f"concatenation.silence_duration_sec must be non-negative, got {silence}"
        raise ValueError(msg)


def get_enabled_stages(cfg: dict[str, Any]) -> list[str]:
    """Return a list of enabled stage names from a loaded config."""
    stages: list[str] = []
    if cfg.get("vad", {}).get("enable", True):
        stages.append("vad")
    if cfg.get("band_filter", {}).get("enable", True):
        stages.append("band_filter")
    if cfg.get("utmos", {}).get("enable", True):
        stages.append("utmos")
    if cfg.get("sigmos", {}).get("enable", True):
        stages.append("sigmos")
    if cfg.get("speaker_separation", {}).get("enable", True):
        stages.append("speaker_separation")
    return stages
