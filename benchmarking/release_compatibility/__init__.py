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

"""Explicit compatibility for benchmark scripts, never Curator-under-test."""

from pathlib import Path
from typing import Any


def validate_profile(profile: str | None) -> str | None:
    if profile not in (None, "26.07"):
        message = f"Unknown benchmark compatibility profile: {profile!r}"
        raise ValueError(message)
    return profile


def compatibility_config(profile: str | None = None) -> dict[str, Any]:
    """Load the optional requirements override owned by the selected profile."""
    profile = validate_profile(profile)
    if profile is None:
        return {}
    import yaml

    path = Path(__file__).with_name(f"curator_{profile.replace('.', '_')}.yaml")
    with path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def apply_script_profile(command: str, entry_name: str, profile: str | None) -> str:
    """Pass the selected profile only to scripts covered by its overrides."""
    config = compatibility_config(profile)
    if any(entry["name"] == entry_name for entry in config.get("entries", [])):
        return f"{command} --benchmark-compat-profile {profile}"
    return command
