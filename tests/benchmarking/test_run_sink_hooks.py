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

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarking"))

from runner.sinks.sink import call_sink_hook, initialize_sinks


class _Sink:
    def __init__(self, fail_hooks: set[str] | None = None) -> None:
        self.fail_hooks = fail_hooks or set()
        self.calls: list[tuple[str, dict[str, object]]] = []

    def _record(self, hook_name: str, kwargs: dict[str, object]) -> None:
        self.calls.append((hook_name, kwargs))
        if hook_name in self.fail_hooks:
            msg = f"{hook_name} failed"
            raise RuntimeError(msg)

    def initialize(self, **kwargs: object) -> None:
        self._record("initialize", kwargs)

    def register_benchmark_entry_starting(self, **kwargs: object) -> None:
        self._record("register_benchmark_entry_starting", kwargs)

    def register_benchmark_entry_finished(self, **kwargs: object) -> None:
        self._record("register_benchmark_entry_finished", kwargs)

    def finalize(self, **kwargs: object) -> None:
        self._record("finalize", kwargs)


def test_call_sink_hook_returns_false_when_sink_raises() -> None:
    sink = _Sink(fail_hooks={"register_benchmark_entry_starting"})

    success = call_sink_hook(
        sink,
        "register_benchmark_entry_starting",
        context="marking an entry as running",
        result_dict={"name": "entry_a"},
        benchmark_entry=object(),
    )

    assert success is False
    assert sink.calls[0][0] == "register_benchmark_entry_starting"


def test_call_sink_hook_returns_true_when_sink_succeeds() -> None:
    sink = _Sink()

    success = call_sink_hook(sink, "finalize", context="finalizing sinks")

    assert success is True
    assert sink.calls == [("finalize", {})]


def test_initialize_sinks_filters_failed_initialization() -> None:
    failing_sink = _Sink(fail_hooks={"initialize"})
    working_sink = _Sink()

    active_sinks = initialize_sinks(
        [failing_sink, working_sink],
        session_name="test-session",
        session=object(),
        env_dict={},
    )

    assert active_sinks == [working_sink]
    assert failing_sink.calls[0][0] == "initialize"
    assert working_sink.calls[0][0] == "initialize"
