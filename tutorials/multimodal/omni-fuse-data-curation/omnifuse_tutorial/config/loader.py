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

"""Configuration loading helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from omnifuse_tutorial.config.models import ExperimentConfig


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    _load_dotenv(Path.cwd() / ".env")
    _load_dotenv(config_path.parent / ".env")
    data = _load_mapping(config_path)
    config = ExperimentConfig.from_dict(data)
    return _resolve_relative_paths(config, config_path.parent)


def _load_mapping(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise ValueError(f"Config must be a mapping: {path}")
        return loaded
    raise ValueError(f"Unsupported config suffix: {path.suffix}")


def _resolve_relative_paths(config: ExperimentConfig, base_dir: Path) -> ExperimentConfig:
    if not config.output_dir.is_absolute():
        config.output_dir = (base_dir / config.output_dir).resolve()
    if config.runtime.cache_dir and not config.runtime.cache_dir.is_absolute():
        config.runtime.cache_dir = (base_dir / config.runtime.cache_dir).resolve()
    if config.sns.sns_output_dir and not config.sns.sns_output_dir.is_absolute():
        config.sns.sns_output_dir = (base_dir / config.sns.sns_output_dir).resolve()
    if config.sns.cg_detr_checkpoint and not config.sns.cg_detr_checkpoint.is_absolute():
        config.sns.cg_detr_checkpoint = (base_dir / config.sns.cg_detr_checkpoint).resolve()
    if config.projection.save_weights_path and not config.projection.save_weights_path.is_absolute():
        config.projection.save_weights_path = (base_dir / config.projection.save_weights_path).resolve()
    for pool in config.data_pools:
        if not pool.root_dir.is_absolute():
            pool.root_dir = (base_dir / pool.root_dir).resolve()
    return config


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
