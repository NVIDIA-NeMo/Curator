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

"""Ordered image references from MinerU JSON, using the existing text scanner."""

import json
from collections import Counter
from collections.abc import Iterator
from typing import Any

from nemo_curator.stages.interleaved.image_references import _resolve_url, extract_image_references


def _structured_reference(value: dict, path: str, page_url: str | None) -> dict:
    content = value.get("content")
    raw = content.get("url") if isinstance(content, dict) else None
    if raw is not None and not isinstance(raw, str):
        raw, url, status = json.dumps(raw, ensure_ascii=False), None, "invalid_type"
    else:
        url, status = _resolve_url(raw, page_url)
    if not isinstance(content, dict):
        status = "invalid_image_content"
    return {
        "raw": raw,
        "url": url,
        "status": status,
        "kind": "mineru_image",
        "source_path": path + "/content/url",
        "start": None,
        "end": None,
    }


def _walk(
    value: object, path: str, in_content: bool, page_url: str | None, diagnostics: Counter
) -> Iterator[dict[str, Any]]:
    if isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}/{index}", in_content, page_url, diagnostics)
    elif isinstance(value, dict):
        if value.get("type") == "image":
            diagnostics["structured_occurrences"] += 1
            yield _structured_reference(value, path, page_url)
        for key, child in value.items():
            if key in {"url", "data", "type", "bbox", "t", "s", "level"}:
                continue
            escaped = key.replace("~", "~0").replace("/", "~1")
            yield from _walk(child, path + "/" + escaped, in_content or key == "content", page_url, diagnostics)
    elif isinstance(value, str) and in_content:
        references, flags = extract_image_references(value, page_url)
        for key, count in flags.pop("html_attribute_counts").items():
            diagnostics["html_attribute_" + key] += count
        diagnostics.update(flags)
        diagnostics["text_fields_scanned"] += 1
        diagnostics["text_occurrences"] += len(references)
        for reference in references:
            yield {**reference, "source_path": path}


def iter_mineru_image_references(
    text: str | None, page_url: str | None, diagnostics: Counter
) -> Iterator[dict[str, Any]]:
    """Yield structured references and text mentions in decoded source order.

    source_path is a JSON pointer. Text start/end are Unicode offsets within
    that field. Structured references point at their URL field without text
    offsets. URL/data and structural metadata are not document text. Malformed
    documents raise rather than silently dropping their image references.
    """
    if text is None or not text.strip():
        diagnostics["blank_text_rows"] += 1
        return
    node = json.loads(text)
    if isinstance(node, str):
        node = json.loads(node)
        diagnostics["double_encoded_rows"] += 1
    if not isinstance(node, (list, dict)):
        msg = "MinerU JSON root must be a list or object"
        raise TypeError(msg)
    yield from _walk(node, "", False, page_url, diagnostics)
