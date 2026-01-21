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

import json
import re
import traceback
from collections.abc import Generator
from typing import Any

import requests
from loguru import logger
from runner.entry import Entry
from runner.session import Session
from runner.sinks.sink import Sink
from runner.utils import find_result, get_obj_for_json, human_readable_bytes_repr

_post_template = """
{
  "username": "Curator Benchmark Runner",
  "icon_emoji": ":robot_face:",
  "blocks": [
    {
      "type": "header",
      "text": {
        "type": "plain_text",
        "text": "Curator Benchmark Summary$SESSION_NAME"
      }
    },
    {
      "type": "section",
      "text": {
        "type": "mrkdwn",
        "text": "$EXECUTIVE_SUMMARY"
      }
    },
    {
      "type": "divider"
    },
    $REPORT_JSON_TEXT,
    {
      "type": "actions",
      "elements": [
        {
          "type": "button",
          "text": {
            "type": "plain_text",
            "text": "Logs"
          },
          "url": "$GOOGLE_DRIVE_LINK"
        }
      ]
    }
  ]
}
"""
_blank_row = [
    {"type": "rich_text", "elements": [{"type": "rich_text_section", "elements": [{"type": "text", "text": " "}]}]},
    {"type": "rich_text", "elements": [{"type": "rich_text_section", "elements": [{"type": "text", "text": " "}]}]},
]


class SlackSink(Sink):
    name: str = "slack"

    def __init__(self, sink_config: dict[str, Any]):
        super().__init__(sink_config)
        self.sink_config = sink_config
        self.session_name: str | None = None
        self.matrix_config: Session = None
        self.env_dict: dict[str, Any] = None

        self.results_to_report: list[tuple[list[str], dict[str, Any]]] = []  # list of tuples of (metrics, result_dict)
        self.webhook_url = sink_config.get("webhook_url")
        if not self.webhook_url:
            msg = "SlackSink: No webhook URL configured"
            raise ValueError(msg)
        self.default_metrics = sink_config.get("default_metrics", [])
        if not self.default_metrics:
            msg = "SlackSink: No default metrics configured"
            raise ValueError(msg)

    def initialize(self, session_name: str, matrix_config: Session, env_dict: dict[str, Any]) -> None:
        # Initializes the sink for the session.
        self.session_name = session_name
        self.env_dict = env_dict
        self.matrix_config = matrix_config

    def process_result(self, result_dict: dict[str, Any], matrix_entry: Entry) -> None:
        # Use the matrix_entry to get any entry-specific settings for the Slack report
        # such as additional metrics to include in the report.
        if matrix_entry:
            additional_metrics = matrix_entry.get_sink_data(self.name).get("additional_metrics", [])
        else:
            additional_metrics = []
        # Queues the individual result for posting as a final report during finalize.
        self.results_to_report.append((self.default_metrics + additional_metrics, result_dict))

    def finalize(self) -> None:
        # Posts the queued results to slack as a final report.
        try:
            self._post()
        except Exception as e:  # noqa: BLE001
            # Optionally, log or handle posting errors
            tb = traceback.format_exc()
            logger.error(f"SlackSink: Error posting to Slack: {e}\n{tb}")

    def _post(self) -> None:  # noqa: C901
        message_text_values = {
            "REPORT_JSON_TEXT": "REPORT_JSON_TEXT",
            "GOOGLE_DRIVE_LINK": "https://google.com",
            "EXECUTIVE_SUMMARY": " ",
            "SESSION_NAME": f" - {self.session_name}" if self.session_name else "",
        }
        indent = "-    "  # start with a dash since leading whitespace is stripped

        # Create REPORT_JSON_TEXT: Build the report data as a Python data structure which maps to JSON,
        # then call json.dumps() to convert to a string.
        report_data = []
        table_dict = {"type": "table", "rows": []}
        rows = []
        # Summary rows - list overall status, each individual entry and its success status
        overall_status = (
            "✅ success"
            if all(find_result(results, "success") for _, results in self.results_to_report)
            else "❌ one or more FAILED"
        )
        rows.append(self._two_column_row_bold("OVERALL STATUS", overall_status))
        for _, results in self.results_to_report:
            # Name and success icon row
            entry_name = f"{indent}{find_result(results, 'name')}"
            success_str = "✅ success" if find_result(results, "success") else "❌ FAILED"
            rows.append(self._two_column_row_bold(entry_name, success_str))

        rows.append(_blank_row)

        # Environment header row
        rows.append(self._two_column_row_bold("ENVIRONMENT", " "))
        # Environment rows
        for var, val in self.env_dict.items():
            if var in {"pip_freeze_txt", "conda_explicit_txt"}:
                continue
            (fvar, fval) = self._get_formatted_metric_value_tuple(var, val)
            rows.append(self._two_column_row(f"{indent}{fvar}", fval))

        rows.append(_blank_row)
        # Results header row
        rows.append(self._two_column_row_bold("RESULTS", " "))
        # Results rows
        for metrics, results in self.results_to_report:
            # Name and success icon row
            entry_name = find_result(results, "name")
            success_str = "✅ success" if find_result(results, "success") else "❌ FAILED"
            rows.append(self._two_column_row_bold(entry_name, success_str))

            # Remaining rows are metrics and values
            data = []
            for metric in metrics:
                result = find_result(results, metric, 0)
                data.append(self._get_formatted_metric_value_tuple(metric, result))

            # Requirements checks - add a row for each requirement that was not met
            if "requirements_not_met" in results:
                all_requirements_met = True
                for metric_name, reason_not_met in results["requirements_not_met"].items():
                    data.append((f"Requirement for {metric_name} was not met", f"{reason_not_met}"))
                    all_requirements_met = False
                if all_requirements_met:
                    data.append(("All requirements met", "✅"))
                else:
                    data.append(("All requirements met", "❌"))

            for var, val in data:
                rows.append(self._two_column_row(f"{indent}{var}", str(val)))
            # Add a blank row between entry results
            rows.append(_blank_row)

        # Remove the last blank row added in the loop above
        if len(self.results_to_report) > 0:
            rows.pop(-1)

        table_dict["rows"] = rows
        report_data.append(table_dict)
        # Add a comma to separate each item to be added to the "blocks" array in the template.
        message_text_values["REPORT_JSON_TEXT"] = ",".join(
            [json.dumps(get_obj_for_json(item), indent=2, sort_keys=True) for item in report_data]
        )

        payload = self.substitute_template_placeholders(_post_template, message_text_values).strip()
        response = requests.post(
            self.webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            timeout=100,
        )
        if not response.ok:
            logger.error(f"SlackSink: Failed to send Slack message (status={response.status_code}): {response.text}")

    @staticmethod
    def substitute_template_placeholders(template_str: str, values: dict[str, str]) -> str:
        """
        Substitute variables in template_str of the form $VAR with values from the dictionary { "VAR": ... }.
        The variables to substitute are those in _post_template above, and must occur as $VAR in the string.
        """

        def replacer(match: re.Match[str]) -> str:
            var_with_dollar = match.group(0)
            varname = var_with_dollar[1:]  # strip initial $
            return str(values.get(varname, var_with_dollar))

        # Substitute variables matching $VAR
        return re.sub(r"\$[A-Za-z0-9_]+", replacer, template_str)

    @staticmethod
    def _two_column_row(left_text: str, right_text: str) -> list[dict[str, Any]]:
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
    def _two_column_row_bold(left_text: str, right_text: str) -> list[dict[str, Any]]:
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


# Run SlackSink from the command line to post a summary of the results to Slack.
if __name__ == "__main__":
    import argparse
    import os
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Post benchmark results to Slack via webhook.")
    parser.add_argument(
        "--results-root-dir", required=True, help="Path to the directory containing result subdirectories"
    )
    parser.add_argument("--webhook-url", default=os.getenv("SLACK_WEBHOOK_URL"), help="Slack webhook URL")
    parser.add_argument(
        "--additional-metrics", default=[], help="Additional metrics to include in the report", nargs="+"
    )
    args = parser.parse_args()

    webhook_url = args.webhook_url
    results_root_path = Path(args.results_root_dir)

    def collect_results_from_dir(results_root_path: Path) -> Generator[dict[str, Any], None, None]:
        """Generator: yields dicts loaded from results.json files in subdirectories."""
        for subdir in results_root_path.iterdir():
            if (subdir / "results.json").exists():
                results_json_path = subdir / "results.json"
                with open(results_json_path) as f:
                    yield json.load(f)

    sink_config = {"webhook_url": webhook_url, "default_metrics": ["exec_time_s"]}
    matrix_config = Session(results_path=results_root_path)
    env_json_path = results_root_path / "env.json"
    with open(env_json_path) as f:
        env_data = json.load(f)

    slack_sink = SlackSink(sink_config=sink_config)
    slack_sink.initialize(session_name="test", matrix_config=matrix_config, env_dict=env_data)

    matrix_entry = Entry(name="test", sink_data=[{"name": "slack", "additional_metrics": args.additional_metrics}])
    for result in collect_results_from_dir(results_root_path):
        slack_sink.process_result(result_dict=result, matrix_entry=matrix_entry)
    slack_sink.finalize()
