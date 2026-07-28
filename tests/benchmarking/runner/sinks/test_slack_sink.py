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
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "benchmarking"))

from runner.sinks.slack_sink import SlackParentMessage


def _section_texts(blocks: list[dict[str, Any]]) -> list[str]:
    return [
        block["text"]["text"]
        for block in blocks
        if block.get("type") == "section" and block.get("text", {}).get("type") == "mrkdwn"
    ]


def _table_rows(blocks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    return next(block["rows"] for block in blocks if block.get("type") == "table")


def _row_texts(row: list[dict[str, Any]]) -> list[str]:
    return [cell["elements"][0]["elements"][0]["text"] for cell in row]


def test_slack_parent_message_reports_entry_status_counts() -> None:
    message = SlackParentMessage(session_name="test-session", env_dict={})
    time_taken_s = SlackParentMessage.format_time_taken_s({"metrics": {"time_taken_s": 12.345}})
    message.update_entry("waiting_entry", "⏳ waiting to start")
    message.update_entry("running_entry", "▶️ running")
    message.update_entry("passed_entry", "✅ success", time_taken_s=time_taken_s)
    message.update_entry("failed_entry", "❌ FAILED")

    blocks = message.to_slack_blocks()
    section_texts = _section_texts(blocks)
    table_rows = _table_rows(blocks)

    assert section_texts[0] == (
        "*Total entries:* 4  •  *passed ✅:* 1  •  *failed ❌:* 1  •  *running ▶️:* 1  •  *waiting ⏳:* 1"
    )
    assert all("Overall Status" not in text for text in section_texts)
    assert all(len(row) == 3 for row in table_rows)
    assert _row_texts(table_rows[0]) == ["waiting_entry", "⏳ waiting to start", " "]
    assert _row_texts(table_rows[1]) == ["running_entry", "▶️ running", " "]
    assert _row_texts(table_rows[2]) == ["passed_entry", "✅ success", "12.35s"]
    assert _row_texts(table_rows[3]) == ["failed_entry", "❌ FAILED", " "]


def test_slack_parent_message_labels_viewer_link_with_session_name() -> None:
    message = SlackParentMessage(
        session_name="test-session",
        env_dict={},
        viewer_url="http://viewer.example.com/run?name=test-session",
    )

    section_texts = _section_texts(message.to_slack_blocks())

    assert section_texts[1] == "*Results viewer:* <http://viewer.example.com/run?name=test-session|test-session>"


def test_slack_parent_message_fallback_reports_entry_status_counts() -> None:
    message = SlackParentMessage(session_name="test-session", env_dict={})
    message.update_entry("passed_entry", "✅ success", time_taken_s="12.35s")
    message.update_entry("failed_entry", "❌ FAILED")

    fallback_text = message.to_fallback_text()

    assert "Total entries: 2" in fallback_text
    assert "passed ✅: 1" in fallback_text
    assert "failed ❌: 1" in fallback_text
    assert "running ▶️: 0" in fallback_text
    assert "waiting ⏳: 0" in fallback_text
    assert "passed_entry: ✅ success (12.35s)" in fallback_text
