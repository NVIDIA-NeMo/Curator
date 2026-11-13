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

import json
from collections import Counter

import pytest

from nemo_curator.stages.interleaved.mineru_image_references import iter_mineru_image_references


def extract(node: object):
    counters = Counter()
    rows = list(iter_mineru_image_references(json.dumps(node), "https://example.org/dir/page", counters))
    return rows, counters


def test_mixed_content_preserves_mentions_order_and_positions():
    node = [
        [
            {"type": "image", "content": {"url": " ../same.png "}},
            {"type": "paragraph", "content": [{"c": "![again](../same.png) <img src='other.png'>", "t": "text"}]},
            {"type": "code", "content": {"code_content": "[img]../same.png[/img]"}},
            {"type": "list", "content": {"items": [{"type": "image", "content": {"url": None}}]}},
        ]
    ]
    rows, counts = extract(node)
    assert [r["url"] for r in rows] == [
        "https://example.org/same.png",
        "https://example.org/same.png",
        "https://example.org/dir/other.png",
        "https://example.org/same.png",
        None,
    ]
    assert rows[0]["raw"] == " ../same.png "
    assert rows[1]["source_path"] == "/0/1/content/0/c"
    assert rows[1]["start"] == 0
    assert rows[1]["end"] == 21
    assert rows[-1]["status"] == "missing_url"
    assert counts["structured_occurrences"] == 2
    assert counts["text_occurrences"] == 3


def test_does_not_scan_image_url_or_data_twice():
    rows, _ = extract(
        [
            {
                "type": "image",
                "content": {
                    "url": "https://example.org/![x](fake.png)",
                    "data": "<img src='fake.png'>",
                    "caption": [{"c": "![caption](real.png)"}],
                },
            }
        ]
    )
    assert len(rows) == 2
    assert rows[1]["url"] == "https://example.org/dir/real.png"


def test_raw_resolution_and_unusable_references():
    rows, _ = extract(
        [
            {"type": "image", "content": {"url": u}}
            for u in [
                "https://EXAMPLE.org/a?b=2&a=1&amp;c=3",
                "data:image/png;base64,AA",
                42,
                "http://e:bad/x",
            ]
        ]
    )
    assert rows[0]["url"] == "https://EXAMPLE.org/a?b=2&a=1&amp;c=3"
    assert [r["status"] for r in rows] == ["absolute", "unsupported_scheme", "invalid_type", "invalid_url"]


def test_empty_and_malformed_documents():
    assert extract([])[0] == []
    assert list(iter_mineru_image_references(None, None, Counter())) == []
    with pytest.raises(json.JSONDecodeError):
        list(iter_mineru_image_references("{broken", None, Counter()))
    with pytest.raises(TypeError, match="root must"):
        extract(42)
