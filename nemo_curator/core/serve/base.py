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

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, ClassVar


class InferenceBackend(ABC):
    """Base class for inference server backend implementations."""

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...


@dataclass
class BaseModelConfig:
    """Base public model config shared by inference backends."""

    model_identifier: str
    model_name: str | None = None
    runtime_env: dict[str, Any] = field(default_factory=dict)

    @property
    def resolved_model_name(self) -> str:
        return self.model_name or self.model_identifier

    @staticmethod
    def merge_runtime_envs(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
        """Merge two runtime_env dicts while preserving package lists."""
        if not base and not override:
            return {}
        if not override:
            return {**base}
        if not base:
            return {**override}

        merged = {**base, **override}

        base_env_vars = base.get("env_vars", {})
        override_env_vars = override.get("env_vars", {})
        if base_env_vars or override_env_vars:
            merged["env_vars"] = {**base_env_vars, **override_env_vars}

        for key in ("pip", "uv"):
            if key in base or key in override:
                merged[key] = BaseModelConfig._merge_package_runtime_env(key, base.get(key), override.get(key))

        return merged

    @staticmethod
    def _merge_package_runtime_env(
        key: str,
        base: dict[str, Any] | list[str] | None,
        override: dict[str, Any] | list[str] | None,
    ) -> dict[str, Any] | list[str]:
        # Ray accepts pip/uv as either a list of packages or a dict carrying
        # ``packages`` plus ``{pip,uv_pip}_install_options``. A list-form
        # override must append to ``packages`` without dropping the dict-form
        # base's installer options.
        if base is None:
            return deepcopy(override)
        if override is None:
            return deepcopy(base)

        if isinstance(base, list) and isinstance(override, list):
            return [*base, *override]

        option_key = "uv_pip_install_options" if key == "uv" else "pip_install_options"

        if isinstance(base, dict):
            merged: dict[str, Any] = deepcopy(base)
            base_packages = list(base.get("packages", []))
            base_options = list(base.get(option_key, []))
        else:
            merged = {"packages": list(base)}
            base_packages = list(base)
            base_options = []

        if isinstance(override, dict):
            override_packages = list(override.get("packages", []))
            override_options = list(override.get(option_key, []))
            for k, v in override.items():
                if k not in {"packages", option_key}:
                    merged[k] = deepcopy(v)
        else:
            override_packages = list(override)
            override_options = []

        merged["packages"] = [*base_packages, *override_packages]
        if base_options or override_options:
            merged[option_key] = [*base_options, *override_options]
        return merged


@dataclass
class BaseServerConfig:
    """Base server-level config; subclasses declare which model config types they accept."""

    model_configs: ClassVar[tuple[type[BaseModelConfig], ...]] = ()
