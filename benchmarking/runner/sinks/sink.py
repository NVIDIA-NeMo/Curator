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

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from runner.entry import Entry
    from runner.session import Session

logger = logging.getLogger(__name__)


def _log_sink_exception(sink_name: str, hook_name: str, context_msg: str, error: Exception) -> None:
    try:
        from loguru import logger as loguru_logger
    except ModuleNotFoundError:
        logger.exception(
            "Sink %s.%s failed%s; benchmark execution will continue: %s",
            sink_name,
            hook_name,
            context_msg,
            error,
        )
    else:
        loguru_logger.exception(
            "Sink {}.{} failed{}; benchmark execution will continue: {}",
            sink_name,
            hook_name,
            context_msg,
            error,
        )


def _log_sink_disabled(sink_name: str) -> None:
    try:
        from loguru import logger as loguru_logger
    except ModuleNotFoundError:
        logger.warning("Disabling sink %s for the remainder of this session", sink_name)
    else:
        loguru_logger.warning("Disabling sink {} for the remainder of this session", sink_name)


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


def call_sink_hook(sink: object, hook_name: str, *, context: str = "", **kwargs: object) -> bool:
    """Call a sink hook without letting reporting failures affect benchmarks."""
    sink_name = sink.__class__.__name__
    try:
        getattr(sink, hook_name)(**kwargs)
    except Exception as e:
        context_msg = f" while {context}" if context else ""
        _log_sink_exception(sink_name, hook_name, context_msg, e)
        return False
    return True


def initialize_sinks(
    sinks: Sequence[object], *, session_name: str, session: Session, env_dict: dict[str, Any]
) -> list[object]:
    """Initialize sinks, returning only those safe to use for later hooks."""
    active_sinks = []
    for sink in sinks:
        if call_sink_hook(
            sink,
            "initialize",
            context="initializing benchmark reporting sinks",
            session_name=session_name,
            session=session,
            env_dict=env_dict,
        ):
            active_sinks.append(sink)
        else:
            _log_sink_disabled(sink.__class__.__name__)
    return active_sinks
