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


class Sink(ABC):
    """Abstract base class for benchmark result sinks."""

    @abstractmethod
    def __init__(self, config: dict[str, Any]):
        """Initialize the sink with configuration.
        
        Args:
            config: Configuration dictionary for the sink.
        """
        pass

    @abstractmethod
    def initialize(self, session_name: str, env_data: dict[str, Any]) -> None:
        """Initialize the sink for a benchmark session.
        
        Args:
            session_name: Name of the benchmark session.
            env_data: Environment data for the session.
        """
        pass

    @abstractmethod
    def process_result(self, result: dict[str, Any]) -> None:
        """Process an individual benchmark result.
        
        Args:
            result: Dictionary containing benchmark result data.
        """
        pass

    @abstractmethod
    def finalize(self) -> None:
        """Finalize the sink after all results have been processed."""
        pass

