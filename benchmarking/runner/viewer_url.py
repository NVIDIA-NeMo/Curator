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

from string import Formatter
from typing import TYPE_CHECKING
from urllib.parse import quote

if TYPE_CHECKING:
    from pathlib import Path

_VIEWER_URL_TEMPLATE_PLACEHOLDERS = (
    "results_path",
    "results_path_url",
    "session_name",
    "session_name_url",
    "session_path",
    "session_path_url",
)


def _quote_url_component(value: str) -> str:
    return quote(value, safe="")


def render_viewer_url_template(
    template: str,
    *,
    results_path: Path,
    session_name: str,
    session_path: Path,
) -> str:
    """Render a benchmark viewer URL template using a small fixed placeholder set."""
    values = {
        "results_path": str(results_path),
        "results_path_url": _quote_url_component(str(results_path)),
        "session_name": session_name,
        "session_name_url": _quote_url_component(session_name),
        "session_path": str(session_path),
        "session_path_url": _quote_url_component(str(session_path)),
    }

    try:
        parsed_template = list(Formatter().parse(template))
    except ValueError as e:
        msg = f"Invalid viewer_url_template: {e}"
        raise ValueError(msg) from e

    rendered_parts: list[str] = []
    for literal_text, field_name, format_spec, conversion in parsed_template:
        rendered_parts.append(literal_text)
        if field_name is None:
            continue
        if conversion is not None or format_spec:
            msg = (
                "viewer_url_template placeholders do not support conversions or format specifiers. "
                f"Use one of: {', '.join(_VIEWER_URL_TEMPLATE_PLACEHOLDERS)}."
            )
            raise ValueError(msg)
        if field_name not in values:
            msg = (
                f"Unsupported viewer_url_template placeholder '{field_name}'. "
                f"Supported placeholders: {', '.join(_VIEWER_URL_TEMPLATE_PLACEHOLDERS)}."
            )
            raise ValueError(msg)
        rendered_parts.append(values[field_name])

    return "".join(rendered_parts)


def resolve_viewer_url(
    *,
    viewer_url: str | None,
    viewer_url_template: str | None,
    results_path: Path,
    session_name: str,
    session_path: Path,
) -> str | None:
    """Return the final viewer URL, preferring an explicitly supplied URL."""
    if viewer_url is not None:
        return viewer_url
    if viewer_url_template is None:
        return None
    return render_viewer_url_template(
        viewer_url_template,
        results_path=results_path,
        session_name=session_name,
        session_path=session_path,
    )
