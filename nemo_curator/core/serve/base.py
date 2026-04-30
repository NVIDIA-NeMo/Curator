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
            base_packages = base.get(key, [])
            override_packages = override.get(key, [])
            if base_packages and override_packages:
                merged[key] = [*base_packages, *override_packages]

        return merged


@dataclass
class BaseServerConfig:
    """Base server-level config; subclasses declare which model config types they accept."""

    model_configs: ClassVar[tuple[type[BaseModelConfig], ...]] = ()
