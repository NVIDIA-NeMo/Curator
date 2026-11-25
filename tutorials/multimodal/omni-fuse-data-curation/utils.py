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

"""Shared CLI helpers for the numbered Omni-Fuse tutorial scripts."""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING, Any

from omnifuse_tutorial.config import load_config
from omnifuse_tutorial.data import write_json

if TYPE_CHECKING:
    from omnifuse_tutorial.config import ExperimentConfig


def parse_config(description: str | None) -> ExperimentConfig:
    """Parse `--config`, load the YAML config, and save `config.resolved.json` in the run directory."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to the Omni-Fuse tutorial YAML config")
    config = load_config(parser.parse_args().config)
    write_json(config.run_dir / "config.resolved.json", config.to_dict())
    return config


def print_result(result: dict[str, Any]) -> None:
    """Print a step's output paths and summary metrics as formatted JSON."""
    print(json.dumps(result, indent=2, sort_keys=True))
