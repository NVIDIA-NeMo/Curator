# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any
import traceback

from loguru import logger

from runner.sinks.sink import Sink


class MlflowSink(Sink):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.config = config
        self.tracking_uri = config.get("tracking_uri")
        if not self.tracking_uri:
            raise ValueError("MlflowSink: No tracking URI configured")
        self.experiment = config.get("experiment")
        if not self.experiment:
            raise ValueError("MlflowSink: No experiment configured")
        self.enabled = self.config.get("enabled", True)
        self.results: list[dict[str, Any]] = []
        self.session_name: str = None
        self.env_data: dict[str, Any] = None

    def initialize(self, session_name: str, env_data: dict[str, Any]) -> None:
        self.session_name = session_name
        self.env_data = env_data

    def process_result(self, result: dict[str, Any]) -> None:
        self.results.append(result)

    def finalize(self) -> None:
        if self.enabled:
            try:
                self._push(self.results)
            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f"MlflowSink: Error posting to Mlflow: {e}\n{tb}")
        else:
            logger.warning("MlflowSink: Not enabled, skipping post.")

    def _push(self, results: list[dict[str, Any]]) -> None:
        pass
