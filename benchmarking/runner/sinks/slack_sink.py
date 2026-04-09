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

import fcntl
import json
import os
import random
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any, ClassVar

from loguru import logger
from runner.entry import Entry
from runner.session import Session
from runner.sinks.sink import Sink
from runner.utils import find_result, human_readable_bytes_repr
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

_SLACK_STATE_POLL_INTERVAL_S: float = 0.5
_SLACK_STATE_POLL_TIMEOUT_S: float = 120.0


class SlackMessageBase:
    """Base class for Slack messages."""

    # Constant for creating blank rows in Slack rich text tables
    _TWO_COL_BLANK_ROW: ClassVar[list[dict[str, Any]]] = [
        {
            "type": "rich_text",
            "elements": [{"type": "rich_text_section", "elements": [{"type": "text", "text": " "}]}],
        },
        {
            "type": "rich_text",
            "elements": [{"type": "rich_text_section", "elements": [{"type": "text", "text": " "}]}],
        },
    ]

    def __init__(self):
        """Initialize the base message."""
        self.response: dict[str, Any] | None = None  # Response from Slack API chat_postMessage

    def to_slack_blocks(self) -> list[dict[str, Any]]:
        """Convert the message data to Slack blocks format.

        Returns:
            List of Slack block dictionaries for use with chat_postMessage API.
        """
        msg = "SlackMessageBase: Subclasses must implement to_slack_blocks()"
        raise NotImplementedError(msg)

    def to_fallback_text(self) -> str:
        """Convert the message data to a fallback text string.

        Returns:
            Fallback text string for use with chat_postMessage API.
        """
        msg = "SlackMessageBase: Subclasses must implement to_fallback_text()"
        raise NotImplementedError(msg)

    def set_response(self, response: dict[str, Any]) -> None:
        """Store the response from Slack API chat_postMessage.

        Args:
            response: Response dictionary from Slack API.
        """
        self.response = response

    def was_posted(self) -> bool:
        """Check if the message was posted to Slack.

        Returns:
            True if the message has been posted (has a response), False otherwise.
        """
        return self.response is not None

    def get_timestamp(self) -> str | None:
        """Get the message timestamp from the response, which is needed for threaded replies and message updates.

        Returns:
            Message timestamp string or None if not available.
        """
        if self.response:
            return self.response.get("ts")
        return None

    ####################################################################################################################
    # Helper methods for creating Slack rich text tables
    @staticmethod
    def _get_two_column_row(left_text: str, right_text: str) -> list[dict[str, Any]]:
        """Create a two-column row for Slack rich text tables.

        Args:
            left_text: Text for the left column.
            right_text: Text for the right column.

        Returns:
            List of Slack block dictionaries representing a two-column row.
        """
        return [
            {
                "type": "rich_text",
                "elements": [{"type": "rich_text_section", "elements": [{"type": "text", "text": left_text}]}],
            },
            {
                "type": "rich_text",
                "elements": [{"type": "rich_text_section", "elements": [{"type": "text", "text": right_text}]}],
            },
        ]

    @staticmethod
    def _get_two_column_row_bold(left_text: str, right_text: str) -> list[dict[str, Any]]:
        """Create a two-column row with bold left column for Slack rich text tables.

        Args:
            left_text: Text for the left column (will be bold).
            right_text: Text for the right column.

        Returns:
            List of Slack block dictionaries representing a two-column row with bold left text.
        """
        return [
            {
                "type": "rich_text",
                "elements": [
                    {
                        "type": "rich_text_section",
                        "elements": [{"type": "text", "text": left_text, "style": {"bold": True}}],
                    }
                ],
            },
            {
                "type": "rich_text",
                "elements": [{"type": "rich_text_section", "elements": [{"type": "text", "text": right_text}]}],
            },
        ]

    @staticmethod
    def _get_formatted_metric_value_tuple(metric: str, result: Any) -> tuple[str, str]:  # noqa: ANN401
        """Format a metric value for display in Slack.

        Args:
            metric: The metric name.
            result: The metric value.

        Returns:
            Tuple of (metric_name, formatted_value).
        """
        # time metrics
        if metric.endswith("_s"):
            try:
                hours = int(result // 3600)
                minutes = int((result % 3600) // 60)
                seconds = result % 60
            except (ValueError, TypeError):
                return (metric, str(result))
            else:
                formatted_str = f"{result:.2f}s"
                if hours > 0 or minutes > 0:
                    formatted_str += " ("
                    if hours > 0:
                        formatted_str += f"{hours:02}h : "
                    formatted_str += f"{minutes:02}m : {seconds:05.2f}s)"
                return (metric, formatted_str)
        # memory metrics
        elif metric.endswith("_bytes"):
            try:
                return (metric, f"{human_readable_bytes_repr(int(result))}  ({result} bytes)")
            except (ValueError, TypeError):
                return (metric, str(result))
        # all other metrics
        else:
            return (metric, str(result))


class SlackParentMessage(SlackMessageBase):
    """Represents a parent message in a Slack channel containing benchmark run summary.

    Maintains a list of benchmark entries and their status, along with metadata about the
    benchmark run such as session name and environment information.
    """

    def __init__(self, session_name: str, env_dict: dict[str, Any]):
        """Initialize a SlackParentMessage.

        Args:
            session_name: Name of the benchmark session.
            env_dict: Environment dictionary for the session.
        """
        super().__init__()
        self.session_name = session_name
        self.env_dict = env_dict
        self.entries: dict[str, str] = {}  # Dictionary mapping entry_name to status_string
        self._has_updates: bool = False  # Track if entries have changed since last post

    def update_entry(self, entry_name: str, status: str) -> None:
        """Add or update a benchmark entry status.

        Args:
            entry_name: Name of the benchmark entry.
            status: Status string (e.g., "✅ success", "❌ FAILED", "▶️ running", "⏳ waiting to start").
        """
        # Check if this is actually a change
        if entry_name not in self.entries or self.entries[entry_name] != status:
            self.entries[entry_name] = status
            self._has_updates = True

    def to_slack_blocks(self) -> list[dict[str, Any]]:
        """Convert the parent message data to Slack blocks format.

        Returns:
            List of Slack block dictionaries for use with chat_postMessage API.
        """
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"Curator Benchmark Summary - {self.session_name}",
                },
            },
            {"type": "divider"},
        ]

        # Overall status
        all_statuses = self.entries.values()
        if any("⏳" in status or "▶️" in status for status in all_statuses):
            overall_status = "⏳ In progress"
        elif any("❌" in status for status in all_statuses):
            overall_status = "❌ one or more FAILED"
        else:
            overall_status = "✅ All passed"
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Overall Status:* {overall_status}",
                },
            }
        )

        # Table of benchmark entries, their status, and the environment
        blocks.append({"type": "divider"})
        rows = []
        indent = "-    "  # start with a dash since leading whitespace is stripped
        for entry_name, status in self.entries.items():
            rows.append(self._get_two_column_row_bold(entry_name, status))
        rows.append(self._TWO_COL_BLANK_ROW)
        rows.append(self._get_two_column_row_bold("ENVIRONMENT", " "))
        for var, val in self.env_dict.items():
            if var in {"pip_freeze_txt", "conda_explicit_txt"}:
                continue
            (fvar, fval) = self._get_formatted_metric_value_tuple(var, val)
            rows.append(self._get_two_column_row(f"{indent}{fvar}", fval))
        rows.append(self._TWO_COL_BLANK_ROW)

        blocks.append(
            {
                "type": "table",
                "rows": rows,
            }
        )

        return blocks

    def to_fallback_text(self) -> str:
        """Convert the message data to a fallback text string.

        Returns:
            Fallback text string for use with chat_postMessage API.
        """
        lines = [f"Curator Benchmark Summary - {self.session_name}"]
        if self.entries:
            lines.append("\nBenchmark Entries:")
            for entry_name, status in self.entries.items():
                lines.append(f"  • {entry_name}: {status}")
        return "\n".join(lines)

    def get_channel_id(self) -> str | None:
        """Get the channel ID from the response for posting threaded replies.

        Returns:
            Channel ID string or None if not available.
        """
        if self.response:
            return self.response.get("channel")
        return None

    def has_updates(self) -> bool:
        """Check if the message has updates that need to be posted.

        Returns:
            True if the message needs to be updated, False otherwise.
        """
        # Return True if message hasn't been posted yet, or if there are pending updates
        return not self.was_posted() or self._has_updates

    def set_response(self, response: dict[str, Any]) -> None:
        """Store the response from Slack API and clear the updates flag if successful.

        Args:
            response: Response dictionary from Slack API.
        """
        super().set_response(response)
        # Clear the updates flag only if the response indicates success
        if response and response.get("ok", False):
            self._has_updates = False


class SlackMessage(SlackMessageBase):
    """Represents a message for an individual benchmark entry in Slack.

    Can be posted as a standalone message in a channel or as a threaded reply
    under a SlackParentMessage.
    """

    def __init__(self, entry_name: str, result_dict: dict[str, Any], metrics: list[str], pings: list[str]):
        """Initialize a SlackMessage.

        Args:
            entry_name: Name of the benchmark entry.
            result_dict: Dictionary containing benchmark result data.
            metrics: List of metric names to include in the message.
            pings: List of Slack user IDs (e.g. U01234567) to mention; each becomes <@ID> so the user is notified.
        """
        super().__init__()
        self.entry_name = entry_name
        self.result_dict = result_dict
        self.metrics = metrics
        self.pings = pings

    def _format_ping_mentions(self) -> list[str]:
        """Format ping strings as Slack @ mentions so the user gets notified.

        Each ping string must be a Slack user ID (e.g. U01234567).
        """
        return [f"<@{slack_id}>" for slack_id in [sid.strip() for sid in self.pings] if slack_id]

    def to_slack_blocks(self) -> list[dict[str, Any]]:
        """Convert the message data to Slack blocks format.

        Returns:
            List of Slack block dictionaries for use with chat_postMessage API.
        """
        success = find_result(self.result_dict, "success")
        status_text = "✅ Success" if success else "❌ Failed"
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{self.entry_name}: {status_text}",
                },
            },
        ]
        # Table of metrics and values
        rows = []
        for metric in self.metrics:
            result = find_result(self.result_dict, metric, 0)
            m, v = self._get_formatted_metric_value_tuple(metric, result)
            rows.append(self._get_two_column_row(m, str(v)))

        # Requirements checks - add a row for each requirement that was not met
        if "requirements_not_met" in self.result_dict:
            all_requirements_met = True
            for metric_name, reason_not_met in self.result_dict["requirements_not_met"].items():
                rows.append(
                    self._get_two_column_row(f"Requirement for {metric_name} was not met", f"{reason_not_met}")
                )
                all_requirements_met = False
            if all_requirements_met:
                rows.append(self._get_two_column_row("All requirements met", "✅"))
            else:
                rows.append(self._get_two_column_row("All requirements met", "❌"))
        blocks.append({"type": "table", "rows": rows})
        if self.pings:
            mentions = self._format_ping_mentions()
            if mentions:
                blocks.append(
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": " ".join(mentions)},
                    }
                )
        blocks.append({"type": "divider"})
        return blocks

    def to_fallback_text(self) -> str:
        """Convert the message data to a fallback text string.

        Returns:
            Fallback text string for use with chat_postMessage API.
        """
        success = find_result(self.result_dict, "success")
        status_text = "Success" if success else "Failed"
        lines = [f"{self.entry_name}: {status_text}"]

        if self.metrics:
            lines.append("\nMetrics:")
            for metric in self.metrics:
                value = find_result(self.result_dict, metric, "N/A")
                lines.append(f"  • {metric}: {value}")

        if "requirements_not_met" in self.result_dict:
            requirements_not_met = self.result_dict["requirements_not_met"]
            if requirements_not_met:
                lines.append("\nRequirements Not Met:")
                for metric_name, reason in requirements_not_met.items():
                    lines.append(f"  • {metric_name}: {reason}")
            else:
                lines.append("\nAll Requirements Met ✅")

        if self.pings:
            mentions = self._format_ping_mentions()
            if mentions:
                lines.append("\n" + " ".join(mentions))

        return "\n".join(lines)


class SlackSink(Sink):
    name: str = "slack"

    def __init__(self, sink_config: dict[str, Any]):
        super().__init__(sink_config)
        self.sink_config = sink_config
        self.session_name: str | None = None
        self.session: Session | None = None
        self.env_dict: dict[str, Any] | None = None

        self._parent_message: SlackParentMessage | None = None
        self._child_messages: list[SlackMessage] = []

        self.live_updates: bool = sink_config.get("live_updates", False)

        self.default_metrics: list[str] = sink_config.get("default_metrics", [])
        if not self.default_metrics:
            msg = "SlackSink: No default metrics configured"
            raise ValueError(msg)

        # needed by Slack API
        self.channel_id: str | None = sink_config.get("channel_id")
        if self.channel_id is None:
            msg = "SlackSink: No channel ID configured"
            raise ValueError(msg)
        self._slack_bot_token: str | None = os.environ.get("SLACK_BOT_TOKEN")
        if self._slack_bot_token is None:
            msg = "SlackSink: SLACK_BOT_TOKEN environment variable is not set"
            raise ValueError(msg)

        # Parallel-run coordination state
        self._state_path: Path | None = None  # Set in initialize()
        self._is_winner: bool = False

    def _get_state_path(self) -> Path:
        return Path(self.session.results_path) / self.session_name / ".slack_state.json"

    def _wait_for_session_state(self, state_path: Path) -> dict[str, Any]:
        deadline = time.monotonic() + _SLACK_STATE_POLL_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                with open(state_path) as f:
                    data = json.load(f)
                if data.get("ts"):
                    return data
            except (OSError, json.JSONDecodeError):
                pass
            time.sleep(_SLACK_STATE_POLL_INTERVAL_S)
        msg = f"SlackSink follower: timed out waiting for session state at {state_path}"
        raise TimeoutError(msg)

    def initialize(self, session_name: str, session: Session, env_dict: dict[str, Any]) -> None:
        self.session_name = session_name
        self.env_dict = env_dict
        self.session = session
        self._child_messages = []
        self._state_path = self._get_state_path()
        self._state_path.parent.mkdir(parents=True, exist_ok=True)

        fd: int | None = None
        try:
            try:
                # Open the state file for writing with the following flags:
                # - os.O_CREAT: create the file if it does not exist
                # - os.O_EXCL: fail if the file already exists (ensures "winner" for the current process)
                # - os.O_WRONLY: open for write-only access
                # This lets us atomically determine which process was first to create the session state file,
                # coordinating parallel benchmarking runs.
                fd = os.open(str(self._state_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                self._is_winner = True
            except FileExistsError:
                self._is_winner = False

            if self._is_winner:
                self._parent_message = self._create_session_summary_message(env_dict)
                self._post_message(self._parent_message)
                initial_state = {
                    "ts": self._parent_message.get_timestamp(),
                    "channel": self._parent_message.get_channel_id(),
                    "entries": dict(self._parent_message.entries),
                }
                # NOTE: This is the only time the state file is created.
                # If the benchmark session is re-run using the same session name
                # (resulting in the same state file path), the file will already exist and
                # all benchmarking info will be added to the previous Slack parent message.
                # This is by design. New benchmark runs are assumed to use new session names,
                # and therefore will generate new/unique state file paths.
                payload = json.dumps(initial_state).encode()
                os.write(fd, payload)
            else:
                state = self._wait_for_session_state(self._state_path)
                self._parent_message = SlackParentMessage(session_name=session_name, env_dict=env_dict)
                self._parent_message.set_response({"ts": state["ts"], "channel": state["channel"], "ok": True})
                for entry_name, entry_status in state["entries"].items():
                    self._parent_message.entries[entry_name] = entry_status
        finally:
            if fd is not None:
                os.close(fd)

    def register_benchmark_entry_starting(self, result_dict: dict[str, Any], benchmark_entry: Entry) -> None:  # noqa: ARG002
        # Register that a benchmark entry is starting.
        # In live mode, this could be used to post an initial status message.
        # For now, this is a no-op as we only post when the entry finishes.
        if self.live_updates:
            if self._parent_message is None:
                logger.warning(
                    "SlackSink: Warning: Ignoring attempt to post an entry starting message without a session summary message. Was initialize() called?"
                )
                return
            self._update_parent_entry(benchmark_entry.name, "▶️ running")

    def register_benchmark_entry_finished(self, result_dict: dict[str, Any], benchmark_entry: Entry) -> None:
        if self._parent_message is None:
            logger.warning(
                "SlackSink: Warning: Ignoring attempt to post an entry finished message without a session summary message. Was initialize() called?"
            )
            return
        # Use the benchmark_entry to get any entry-specific settings for the Slack report
        # such as additional metrics to include in the report, pings, etc.
        sink_data = benchmark_entry.get_sink_data(self.name)
        additional_metrics = sink_data.get("additional_metrics", [])
        pings = [] if result_dict["success"] else sink_data.get("ping_on_failure", [])
        status_text = "✅ success" if result_dict["success"] else "❌ FAILED"

        # Create a new message for the entry to post in the thread.
        msg = self._create_benchmark_entry_message(
            benchmark_entry,
            (self.default_metrics + additional_metrics, result_dict),
            pings,
        )
        self._child_messages.append(msg)
        # Update the session summary message with the new entry status.
        self._update_parent_entry(benchmark_entry.name, status_text)

        if self.live_updates:
            self._post_updates()

    def finalize(self) -> None:
        if self._parent_message is None:
            logger.warning(
                "SlackSink: Warning: Ignoring attempt to finalize without a session summary message. Was initialize() called?"
            )
            return
        self._post_updates()

    def _create_session_summary_message(self, env_dict: dict[str, Any]) -> SlackParentMessage:
        """Create the parent message that summarizes the benchmark session.

        Args:
            env_dict: Environment dictionary.

        Returns:
            SlackParentMessage instance for the session summary.
        """
        msg = SlackParentMessage(session_name=self.session_name, env_dict=env_dict)
        for entry in self.session.entries:
            msg.update_entry(entry.name, "⏳ waiting to start")
        return msg

    def _create_benchmark_entry_message(
        self, benchmark_entry: Entry, data: tuple[list[str], dict[str, Any]], pings: list[str]
    ) -> SlackMessage:
        """Create a message for an individual benchmark entry.

        Args:
            benchmark_entry: The benchmark entry.
            data: Tuple of (metrics, result_dict).
            pings: List of user IDs to ping to make them aware of this message.

        Returns:
            SlackMessage instance for the benchmark entry.
        """
        metrics, result_dict = data
        return SlackMessage(entry_name=benchmark_entry.name, result_dict=result_dict, metrics=metrics, pings=pings)

    def _update_parent_entry(self, entry_name: str, status: str) -> None:
        """Update a single entry's status in the shared state file and post the update to Slack.

        Acquires an exclusive file lock for the duration of the read-modify-write cycle and
        the Slack API call so that concurrent processes do not overwrite each other's updates.

        Args:
            entry_name: Name of the benchmark entry to update.
            status: New status string for the entry.
        """
        if self._state_path is None:
            logger.error("SlackSink: Cannot update parent entry — state path not set. Was initialize() called?")
            return
        try:
            f = open(self._state_path, "r+")  # noqa: SIM115
        except OSError:
            logger.error(f"SlackSink: Cannot open state file {self._state_path} for update")
            return
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            state = json.load(f)
            state["entries"][entry_name] = status
            for name, st in state["entries"].items():
                self._parent_message.update_entry(name, st)
            try:
                self._update_message(self._parent_message)
            finally:
                # Always persist state after attempting Slack update (even if _update_message raises SlackApiError).
                f.seek(0)
                json.dump(state, f)
                f.truncate()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            f.close()

    def _post_updates(self) -> None:
        for msg in self._child_messages:
            if not msg.was_posted():
                self._post_message(msg)

    def _post_message(self, message: SlackMessageBase) -> None:
        """Post a message to Slack.

        Args:
            message: SlackMessageBase instance to post.
        """
        try:
            client = WebClient(token=self._slack_bot_token)

            # Determine the channel ID and thread timestamp based on if the parent message
            # has been posted. All posts will be threaded replies if the parent was posted.
            thread_ts = None
            channel_id = self.channel_id
            if self._parent_message.was_posted():
                if isinstance(message, SlackParentMessage):
                    msg = "SlackSink: ERROR: Attempt to post a parent message more than once"
                    raise ValueError(msg)
                # Best practice states to always use the channel ID from the parent message if available,
                # even if the original intended channel is the same as the parent. While the parent was
                # originally posted to self.channel_id, the response from that post will include the channel
                # to post threaded replies and message updates to. This is often different if the channel is
                # a user ID (starting with 'U'); slack will require followups post to a DM channel ID
                # (starting with 'D') and will provide that in the parent message post response.
                channel_id = self._parent_message.get_channel_id()
                thread_ts = self._parent_message.get_timestamp()

            response = client.chat_postMessage(
                channel=channel_id,
                blocks=message.to_slack_blocks(),
                text=message.to_fallback_text(),
                thread_ts=thread_ts,
            )
            # Save the response for future updates and/or threaded replies (only SlackParentMessage types can
            # have threaded replies). This also sets was_posted to return True for the message.
            message.set_response(response.data)
            logger.debug(f"Posted message to Slack: {response.data.get('ts')}")
        except SlackApiError as e:
            logger.error(f"Error posting message to Slack: {e.response['error']}")
            raise

    def _update_message(self, message: SlackMessageBase) -> None:
        """Update an existing message in Slack.

        Args:
            message: SlackMessageBase instance to update.
        """
        client = WebClient(token=self._slack_bot_token)

        # Get the channel ID from the parent message.
        # This is often updated as part of the initial parent message post response and
        # may differ from the original self.channel_id, esp. if the original channel is a
        # user ID. This applies to both parent and child messages when calling chat_update.
        channel_id = self._parent_message.get_channel_id()

        if not message.was_posted() or channel_id is None:
            logger.warning("Cannot update message that hasn't been posted yet")
            return
        try:
            response = client.chat_update(
                channel=channel_id,
                ts=message.get_timestamp(),
                blocks=message.to_slack_blocks(),
                text=message.to_fallback_text(),
            )
            logger.debug(f"Updated message in Slack: {response.data.get('ts')}")
        except SlackApiError as e:
            logger.error(f"Error updating message in Slack: {e.response['error']}")
            raise


# Run SlackSink from the command line to post a report for existing results.
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Post benchmark results to Slack.")
    parser.add_argument(
        "--benchmark-run-results-dir",
        required=True,
        help="Path to the directory containing results from a benchmark run",
    )
    parser.add_argument("--channel-id", default=os.getenv("SLACK_CHANNEL_ID"), help="Slack channel ID")
    parser.add_argument("--add-additional-metrics", action="store_true", help="Add additional metrics to the report")
    parser.add_argument(
        "--test-live-updates",
        action="store_true",
        help="Simulates 'live updates' which will post results in a thread as benchmarks are running",
    )
    args = parser.parse_args()

    channel_id = args.channel_id
    results_root_path = Path(args.benchmark_run_results_dir)

    def collect_results_from_dir(results_root_path: Path) -> Generator[dict[str, Any], None, None]:
        """Generator: yields dicts loaded from results.json files in subdirectories."""
        for subdir in results_root_path.iterdir():
            if (subdir / "results.json").exists():
                results_json_path = subdir / "results.json"
                with open(results_json_path) as f:
                    yield json.load(f)

    # Create the session and all entries from the results
    entries = []
    for result in collect_results_from_dir(results_root_path):
        if args.add_additional_metrics:
            # Get the keys from result["metrics"], or empty list if not present
            metric_keys = list(result.get("metrics", {}).keys())
            # Pick a random number in 1-5, not greater than the available keys
            n_metrics = min(len(metric_keys), random.randint(1, 5))  # noqa: S311
            additional_metrics = random.sample(metric_keys, n_metrics) if n_metrics > 0 else []
        else:
            additional_metrics = []
        entry = Entry(name=result["name"], sink_data=[{"name": "slack", "additional_metrics": additional_metrics}])
        # Add the result dict to the entry for testing.
        # In a real run.py process, the result dict would be passed to the entry.
        entry.result_dict = result
        entries.append(entry)

    sink_config = {
        "channel_id": channel_id,
        "default_metrics": ["exec_time_s"],
        "live_updates": args.test_live_updates,
    }
    session = Session(results_path=results_root_path, entries=entries)

    env_json_path = results_root_path / "env.json"
    with open(env_json_path) as f:
        env_data = json.load(f)

    # Create a standalone Slack sink to post the results to Slack.
    slack_sink = SlackSink(sink_config=sink_config)
    slack_sink.initialize(session_name="test", session=session, env_dict=env_data)

    # Simulate a run.py process running the entries and posting the results to Slack.
    for entry in entries:
        slack_sink.register_benchmark_entry_starting(result_dict=entry.result_dict, benchmark_entry=entry)
        if args.test_live_updates:
            time.sleep(3)  # simulate a delay between benchmark runs
        slack_sink.register_benchmark_entry_finished(result_dict=entry.result_dict, benchmark_entry=entry)
    slack_sink.finalize()
