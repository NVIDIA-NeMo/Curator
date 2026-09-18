# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Lightweight CLI wrapper for benchmark execution.

This module intentionally contains only argument parsing and dispatch. It must
remain importable in restricted host environments that use `curator-benchmark`
only to start or exec into a Docker target. The actual benchmark implementation
imports Curator, Ray-related runner helpers, and optional reporting dependencies,
so it lives in `run_impl.py` and is imported only after CLI parsing succeeds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build the parser for running benchmarks in the current environment."""
    parser = argparse.ArgumentParser(description="Runs the benchmarking application")
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        required=True,
        help=(
            "Path to YAML config for benchmark entries, data setups, machine paths, etc. Can be "
            "specified multiple times to merge configs."
        ),
    )
    parser.add_argument(
        "--session-name",
        default=None,
        help=("Optional human-readable session name. Default is benchmark-run__<timestamp>."),
    )
    parser.add_argument(
        "--entries",
        default=None,
        help=(
            "Expression to filter entries to run. Example: 'foo and not foobar' will include "
            "all entries with 'foo' in the name but not 'foobar'. If not specified, all "
            "enabled entries will be run."
        ),
    )
    parser.add_argument(
        "--entries-exact",
        default=None,
        help=(
            "Comma-separated list of exact entry names to run. Unlike --entries (a pytest "
            "'-k' style substring expression), names here must match entry names exactly. "
            "Every supplied name must correspond to a configured (enabled) entry; otherwise "
            "the run fails with an error listing the unknown names. Useful for both "
            "automated callers (e.g. CI per-job invocations) and users targeting a specific "
            "set of entries by exact name. Mutually exclusive with --entries."
        ),
    )
    parser.add_argument(
        "--list",
        default=False,
        action="store_true",
        help="List entries to run and exit.",
    )
    parser.add_argument(
        "--strict-config-check",
        default=False,
        action="store_true",
        help=(
            "If set, fail with an error when an environment variable referenced in the "
            "config is undefined or empty. By default, undefined env var references are "
            "replaced with an empty string and a warning is logged."
        ),
    )
    parser.add_argument(
        "--path-mode",
        choices=["auto", "host", "container"],
        default=None,
        help=(
            "Select whether configured paths resolve to host_path or container_path. "
            "Defaults to CURATOR_BENCHMARK_PATH_MODE, then auto."
        ),
    )
    viewer_url_group = parser.add_mutually_exclusive_group()
    viewer_url_group.add_argument(
        "--viewer-url",
        default=None,
        help=("Resolved run-viewer URL to surface in sinks. Overrides viewer_url_template from YAML config when set."),
    )
    viewer_url_group.add_argument(
        "--viewer-url-template",
        default=None,
        help=(
            "Run-viewer URL template to render after the session path is known. Supports "
            "{results_path}, {results_path_url}, {session_name}, {session_name_url}, "
            "{session_path}, and {session_path_url}. Mutually exclusive with --viewer-url."
        ),
    )
    parser.add_argument(
        "--reason",
        default=None,
        help=(
            "Free-text reason for this run, recorded in env.json and surfaced in the Slack "
            "environment block. Useful for audit trails on ad-hoc runs."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Keep this module importable in minimal host environments used only to
    # launch Docker targets. The implementation imports Curator/runtime deps.
    from curator_benchmarking.commands.run_impl import run_benchmarks

    return run_benchmarks(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
