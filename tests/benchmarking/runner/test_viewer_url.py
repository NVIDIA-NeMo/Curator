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

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarking"))

from runner.viewer_url import render_viewer_url_template, resolve_viewer_url


def test_render_viewer_url_template_substitutes_supported_placeholders() -> None:
    rendered = render_viewer_url_template(
        "http://viewer/run?dir={session_path_url}&run={session_name_url}&root={results_path_url}",
        results_path=Path("/benchmark-results/results branch"),
        session_name="rc run 1",
        session_path=Path("/benchmark-results/results branch/rc run 1"),
    )

    assert (
        rendered
        == "http://viewer/run?dir=%2Fbenchmark-results%2Fresults%20branch%2Frc%20run%201&run=rc%20run%201&root=%2Fbenchmark-results%2Fresults%20branch"
    )


def test_render_viewer_url_template_rejects_unknown_placeholder() -> None:
    with pytest.raises(ValueError, match="Unsupported viewer_url_template placeholder 'unknown'"):
        render_viewer_url_template(
            "http://viewer/run?dir={unknown}",
            results_path=Path("/benchmark-results/results"),
            session_name="run",
            session_path=Path("/benchmark-results/results/run"),
        )


def test_render_viewer_url_template_rejects_format_specifier() -> None:
    with pytest.raises(ValueError, match="do not support conversions or format specifiers"):
        render_viewer_url_template(
            "http://viewer/run?name={session_name!r}",
            results_path=Path("/benchmark-results/results"),
            session_name="run",
            session_path=Path("/benchmark-results/results/run"),
        )


def test_resolve_viewer_url_prefers_explicit_url() -> None:
    resolved = resolve_viewer_url(
        viewer_url="http://viewer/explicit",
        viewer_url_template="http://viewer/template?dir={session_path_url}",
        results_path=Path("/benchmark-results/results"),
        session_name="run",
        session_path=Path("/benchmark-results/results/run"),
    )

    assert resolved == "http://viewer/explicit"


def test_resolve_viewer_url_returns_none_without_url_or_template() -> None:
    resolved = resolve_viewer_url(
        viewer_url=None,
        viewer_url_template=None,
        results_path=Path("/benchmark-results/results"),
        session_name="run",
        session_path=Path("/benchmark-results/results/run"),
    )

    assert resolved is None
