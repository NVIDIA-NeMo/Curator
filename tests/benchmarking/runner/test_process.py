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

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarking"))

from runner.process import run_command_with_timeout


def test_run_command_with_timeout_dumps_subprocess_environment(tmp_path: Path) -> None:
    log_path = tmp_path / "stdouterr.log"
    env = {
        **os.environ,
        "BENCHMARK_TEST_ENV": "present",
        "INHERITED_TEST_ENV": "inherited-value",
        "SLACK_BOT_TOKEN": "secret-token",
        "NVIDIA_API_KEY": "secret-key",
        "GITHUB_PAT": "secret-pat",
        "ALERT_WEBHOOK": "secret-webhook",
        "TOKENIZERS_PARALLELISM": "false",
    }

    result = run_command_with_timeout(
        command=f"{sys.executable} -c 'print(\"done\")'",
        timeout=10,
        stdouterr_path=log_path,
        env=env,
        env_value_allowlist={
            "ALERT_WEBHOOK",
            "BENCHMARK_TEST_ENV",
            "GITHUB_PAT",
            "NVIDIA_API_KEY",
            "SLACK_BOT_TOKEN",
            "TOKENIZERS_PARALLELISM",
        },
        fancy=False,
    )

    log_text = log_path.read_text()
    assert result == {"returncode": 0, "timed_out": False}
    assert "--- Subprocess environment ---" in log_text
    assert "BENCHMARK_TEST_ENV=present" in log_text
    assert "INHERITED_TEST_ENV=<redacted>" in log_text
    assert "SLACK_BOT_TOKEN=<redacted>" in log_text
    assert "NVIDIA_API_KEY=<redacted>" in log_text
    assert "GITHUB_PAT=<redacted>" in log_text
    assert "ALERT_WEBHOOK=<redacted>" in log_text
    assert "TOKENIZERS_PARALLELISM=false" in log_text
    assert "inherited-value" not in log_text
    assert "secret-token" not in log_text
    assert "secret-key" not in log_text
    assert "secret-pat" not in log_text
    assert "secret-webhook" not in log_text
    assert "done" in log_text
