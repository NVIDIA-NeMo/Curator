# Copyright (c) 2025, NVIDIA CORPORATION.
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
from typing import Any

from runner.entry import Entry
from runner.session import Session


class Sink(ABC):
    """Abstract base class for benchmark result sinks."""

    @abstractmethod
    def __init__(self, sink_config: dict[str, Any]):
        """Initialize the sink with configuration.

        Args:
            sink_config: Configuration dictionary for the sink.
        """

    @abstractmethod
    def initialize(
        self,
        session_name: str,
        session: Session,
        env_dict: dict[str, Any],
    ) -> None:
        """Initialize the sink for a benchmark session.

        Args:
            session_name: Name of the benchmark session.
            session: Session configuration for the session.
            env_dict: Environment dictionary for the session.
        """

    @abstractmethod
    def register_benchmark_entry_starting(self, result_dict: dict[str, Any], benchmark_entry: Entry) -> None:
        """Register that a benchmark entry is starting.

        Args:
            result_dict: Dictionary containing benchmark entry data.
            benchmark_entry: Entry configuration.
        """

    @abstractmethod
    def register_benchmark_entry_finished(self, result_dict: dict[str, Any], benchmark_entry: Entry) -> None:
        """Register that a benchmark entry has finished.

        Args:
            result_dict: Dictionary containing benchmark result data.
            benchmark_entry: Entry configuration.
        """

    @abstractmethod
    def finalize(self) -> None:
        """Finalize the sink after all results have been processed."""
