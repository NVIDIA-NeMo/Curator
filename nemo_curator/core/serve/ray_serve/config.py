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

from dataclasses import dataclass, field
from typing import Any, ClassVar

from nemo_curator.core.serve.base import BaseModelConfig, BaseServerConfig


@dataclass
class RayServeModelConfig(BaseModelConfig):
    """Ray Serve model config."""

    deployment_config: dict[str, Any] = field(default_factory=dict)
    engine_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class RayServeServerConfig(BaseServerConfig):
    """Server-level Ray Serve config."""

    model_configs: ClassVar[tuple[type[BaseModelConfig], ...]] = (RayServeModelConfig,)
