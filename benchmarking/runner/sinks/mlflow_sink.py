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

import traceback
from typing import Any

from loguru import logger
from runner.entry import Entry
from runner.session import Session
from runner.sinks.sink import Sink


class MlflowSink(Sink):
    def __init__(self, sink_config: dict[str, Any]):
        super().__init__(sink_config)
        self.sink_config = sink_config
        self.tracking_uri = sink_config.get("tracking_uri")
        if not self.tracking_uri:
            msg = "MlflowSink: No tracking URI configured"
            raise ValueError(msg)
        self.experiment = sink_config.get("experiment")
        if not self.experiment:
            msg = "MlflowSink: No experiment configured"
            raise ValueError(msg)
        self.results: list[dict[str, Any]] = []
        self.session_name: str = None
        self.matrix_config: Session = None
        self.env_dict: dict[str, Any] = None

    def initialize(self, session_name: str, matrix_config: Session, env_dict: dict[str, Any]) -> None:
        self.session_name = session_name
        self.matrix_config = matrix_config
        self.env_dict = env_dict

    def process_result(self, result_dict: dict[str, Any], matrix_entry: Entry) -> None:
        # Use the matrix_entry to get any entry-specific settings for the Slack report
        # such as additional metrics to include in the report.
        if matrix_entry:
            additional_metrics = matrix_entry.get_sink_data(self.name).get("additional_metrics", [])
        else:
            additional_metrics = []
        self.results.append((additional_metrics, result_dict))

    def finalize(self) -> None:
        try:
            self._push(self.results)
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            logger.error(f"MlflowSink: Error posting to Mlflow: {e}\n{tb}")

    def _push(self, results: list[dict[str, Any]]) -> None:
        pass
