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
from runner.matrix import MatrixConfig
from runner.sinks.sink import Sink
from runner.utils import get_obj_for_json

_post_template = """
{
  "username": "Curator Benchmark Runner",
  "icon_emoji": ":robot_face:",
  "blocks": [
    {
      "type": "header",
      "text": {
        "type": "plain_text",
        "text": "Curator Benchmark Summary"
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
    def __init__(self, sink_config: dict[str, Any]):
        super().__init__(sink_config)
        self.sink_config = sink_config
        self.webhook_url = sink_config.get("webhook_url")
        if not self.webhook_url:
            msg = "SlackSink: No webhook URL configured"
            raise ValueError(msg)
        self.enabled = self.sink_config.get("enabled", True)
        self.results: list[dict[str, Any]] = []
        self.session_name: str = None
        self.matrix_config: MatrixConfig = None
        self.env_dict: dict[str, Any] = None

    def initialize(self, session_name: str, matrix_config: MatrixConfig, env_dict: dict[str, Any]) -> None:
        # Initializes the sink for the session.
        self.session_name = session_name
        self.matrix_config = matrix_config
        self.env_dict = env_dict

    def process_result(self, result: dict[str, Any]) -> None:
        # Queues the individual result for posting as a final report during finalize.
        self.results.append(result)

    def finalize(self) -> None:
        # Posts the queued results to slack as a final report.
        if self.enabled:
            try:
                self._post_style2()
            except Exception as e:  # noqa: BLE001
                # Optionally, log or handle posting errors
                tb = traceback.format_exc()
                logger.error(f"SlackSink: Error posting to Slack: {e}\n{tb}")
        else:
            logger.warning("SlackSink: Not enabled, skipping post.")

    def _post_style1(self) -> None:
        message_text_values = {
            "REPORT_JSON_TEXT": "REPORT_JSON_TEXT",
            "GOOGLE_DRIVE_LINK": "https://google.com",
            "EXECUTIVE_SUMMARY": " ",
        }
        # Create REPORT_JSON_TEXT: Build the report data as a Python data structure which maps to JSON,
        # then call json.dumps() to convert to a string.
        report_data = []
        report_data.append({"type": "section", "text": {"type": "mrkdwn", "text": "*Environment*"}})
        table_dict = {"type": "table", "rows": []}
        rows = []
        for var, val in self.env_dict.items():
            row = [
                {
                    "type": "rich_text",
                    "elements": [{"type": "rich_text_section", "elements": [{"type": "text", "text": str(var)}]}],
                },
                {
                    "type": "rich_text",
                    "elements": [{"type": "rich_text_section", "elements": [{"type": "text", "text": str(val)}]}],
                },
            ]
            rows.append(row)
        table_dict["rows"] = rows
        report_data.append(table_dict)

        report_data.append({"type": "section", "text": {"type": "mrkdwn", "text": "*Results*"}})
        # Use text fields for results
        for result in self.results:
            fields_dict = {"type": "section", "fields": []}
            data = [
                ("name", result["name"]),
                ("success", result["success"]),
                ("runtime", f"{result.get('exec_time_s', 0):.2f} s"),
            ]
            left, right = zip(*data, strict=False)
            right = [str(val) for val in right]
            fields = [
                {"type": "mrkdwn", "text": "*" + "*\n*".join(left) + "*"},
                {"type": "mrkdwn", "text": "\n".join(right)},
            ]
            fields_dict["fields"] = fields
            report_data.append({"type": "divider"})
            report_data.append(fields_dict)

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

    def _post_style2(self) -> None:
        message_text_values = {
            "REPORT_JSON_TEXT": "REPORT_JSON_TEXT",
            "GOOGLE_DRIVE_LINK": "https://google.com",
            "EXECUTIVE_SUMMARY": " ",
        }

        # Create REPORT_JSON_TEXT: Build the report data as a Python data structure which maps to JSON,
        # then call json.dumps() to convert to a string.
        report_data = []
        table_dict = {"type": "table", "rows": []}
        rows = []
        rows.append(
            [
                {
                    "type": "rich_text",
                    "elements": [
                        {
                            "type": "rich_text_section",
                            "elements": [{"type": "text", "text": "Environment", "style": {"bold": True}}],
                        }
                    ],
                },
                {
                    "type": "rich_text",
                    "elements": [{"type": "rich_text_section", "elements": [{"type": "text", "text": " "}]}],
                },
            ]
        )
        for var, val in self.env_dict.items():
            if var in {"pip_freeze_txt", "conda_explicit_txt"}:
                continue
            row = [
                {
                    "type": "rich_text",
                    "elements": [{"type": "rich_text_section", "elements": [{"type": "text", "text": str(var)}]}],
                },
                {
                    "type": "rich_text",
                    "elements": [{"type": "rich_text_section", "elements": [{"type": "text", "text": str(val)}]}],
                },
            ]
            rows.append(row)

        rows.append(_blank_row)
        rows.append(
            [
                {
                    "type": "rich_text",
                    "elements": [
                        {
                            "type": "rich_text_section",
                            "elements": [{"type": "text", "text": "Results", "style": {"bold": True}}],
                        }
                    ],
                },
                {
                    "type": "rich_text",
                    "elements": [{"type": "rich_text_section", "elements": [{"type": "text", "text": " "}]}],
                },
            ]
        )
        for result in self.results:
            data = [
                ("name", result["name"]),
                ("success", result["success"]),
                ("runtime", f"{result.get('exec_time_s', 0):.2f} s"),
            ]
            for var, val in data:
                row = [
                    {
                        "type": "rich_text",
                        "elements": [{"type": "rich_text_section", "elements": [{"type": "text", "text": str(var)}]}],
                    },
                    {
                        "type": "rich_text",
                        "elements": [{"type": "rich_text_section", "elements": [{"type": "text", "text": str(val)}]}],
                    },
                ]
                rows.append(row)
            rows.append(_blank_row)

        if len(self.results) > 0:
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


# Run SlackSink from the command line to post a summary of the results to Slack.
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Post benchmark results to Slack via webhook.")
    parser.add_argument("webhook_url", help="Slack webhook URL")
    parser.add_argument("results_root_dir", help="Path to the directory containing result subdirectories")
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

    sink_config = {"webhook_url": webhook_url}
    matrix_config = MatrixConfig(results_dir=results_root_path, artifacts_dir=results_root_path)
    env_json_path = results_root_path / "env.json"
    with open(env_json_path) as f:
        env_data = json.load(f)

    slack_sink = SlackSink(sink_config=sink_config)
    slack_sink.initialize(session_name="test", matrix_config=matrix_config, env_dict=env_data)
    for result in collect_results_from_dir(results_root_path):
        slack_sink.process_result(result=result)
    slack_sink.finalize()
