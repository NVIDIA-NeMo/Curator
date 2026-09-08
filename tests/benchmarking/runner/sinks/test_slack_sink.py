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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "benchmarking"))

from runner.sinks.slack_sink import SlackParentMessage, SlackSink


def _section_texts(blocks: list[dict[str, Any]]) -> list[str]:
    return [
        block["text"]["text"]
        for block in blocks
        if block.get("type") == "section" and block.get("text", {}).get("type") == "mrkdwn"
    ]


def test_slack_parent_message_reports_entry_status_counts() -> None:
    message = SlackParentMessage(session_name="test-session", env_dict={})
    message.update_entry("waiting_entry", "⏳ waiting to start")
    message.update_entry("running_entry", "▶️ running")
    message.update_entry("passed_entry", "✅ success")
    message.update_entry("failed_entry", "❌ FAILED")

    blocks = message.to_slack_blocks()
    section_texts = _section_texts(blocks)

    assert section_texts[0] == "*Run status:* ▶️ running"
    assert section_texts[1] == (
        "*Total entries:* 4  •  *passed ✅:* 1  •  *failed ❌:* 1  •  *running ▶️:* 1  •  *waiting ⏳:* 1"
    )
    assert all("Overall Status" not in text for text in section_texts)
    assert all(block.get("type") != "table" for block in blocks)


def test_slack_parent_message_labels_viewer_link_with_session_name() -> None:
    message = SlackParentMessage(
        session_name="test-session",
        env_dict={},
        viewer_url="http://viewer.example.com/run?name=test-session",
    )

    section_texts = _section_texts(message.to_slack_blocks())

    assert section_texts[0] == "*Run status:* ✅ complete"
    assert section_texts[2] == "*Results viewer:* <http://viewer.example.com/run?name=test-session|test-session>"


def test_slack_parent_message_fallback_reports_entry_status_counts() -> None:
    message = SlackParentMessage(session_name="test-session", env_dict={})
    message.update_entry("passed_entry", "✅ success")
    message.update_entry("failed_entry", "❌ FAILED")

    fallback_text = message.to_fallback_text()

    assert fallback_text.splitlines()[2] == "Run status: ❌ complete with failures"
    assert "Total entries: 2" in fallback_text
    assert "passed ✅: 1" in fallback_text
    assert "failed ❌: 1" in fallback_text
    assert "running ▶️: 0" in fallback_text
    assert "waiting ⏳: 0" in fallback_text
    assert "Run status: ❌ complete with failures" in fallback_text
    assert "Benchmark Entries:" not in fallback_text
    assert "passed_entry" not in fallback_text


def _slack_sink(monkeypatch: pytest.MonkeyPatch, **config_overrides: object) -> SlackSink:
    monkeypatch.setenv("SLACK_BOT_TOKEN", "test-token")
    return SlackSink(
        {
            "channel_id": "C123",
            "default_metrics": ["exec_time_s"],
            **config_overrides,
        }
    )


def test_slack_sink_posts_entry_replies_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    sink = _slack_sink(monkeypatch)

    assert sink._should_post_benchmark_entry_message({"success": True}, []) is True


def test_slack_sink_can_limit_entry_replies_to_failures_or_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    sink = _slack_sink(monkeypatch, thread_replies_for_failures_or_warnings_only=True)

    assert sink._should_post_benchmark_entry_message({"success": True}, []) is False
    assert sink._should_post_benchmark_entry_message({"success": True}, ["warning"])
    assert sink._should_post_benchmark_entry_message({"success": False}, [])
