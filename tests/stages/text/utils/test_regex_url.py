# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""Regression tests for ``regex_url`` (issue #1601).

The previous expression contained the character class ``[$-_…]``,
which the regex engine interprets as a *range* from ``$`` (U+0024)
through ``_`` (U+005F). That range silently includes ``<``, ``>``,
``;``, ``:``, ``=``, and other characters, so matches bled past the
end of a URL into surrounding HTML/punctuation. The fix escapes the
``-`` so it is treated as a literal character.
"""

from nemo_curator.stages.text.utils.constants import regex_url


def test_url_does_not_swallow_html_tags() -> None:
    # `<bad>` follows a normal URL. The previous regex captured the
    # angle-bracketed token because `<` falls inside the [$-_] range.
    text = "see http://x.com<bad> for details"

    matches = regex_url.findall(text)

    assert matches == ["http://x.com"], (
        "URL match must stop at the closing of x.com — angle brackets are "
        "not URL characters and only matched because of the buggy range."
    )


def test_url_does_not_extend_into_trailing_semicolon_html_entity() -> None:
    # HTML entities like `&amp;` and trailing `;` punctuation also got
    # absorbed by the old range.
    text = "click http://example.com;next"

    matches = regex_url.findall(text)

    assert matches == ["http://example.com"], (
        "URL match must terminate at `;` — semicolons are not URL "
        "characters under RFC 3986 unreserved/reserved sets."
    )


def test_url_still_matches_explicitly_allowed_characters() -> None:
    # The fix must not regress on characters that the original
    # character class *intended* to allow: letters, digits, `$`, `_`,
    # `@`, `.`, `&`, `+`, `-`, `!`, `*`, `(`, `)`, `,`, and
    # percent-encoded escapes.
    text = "ref https://A.B-C_D+E&f!*(g),h%2F end"

    matches = regex_url.findall(text)

    assert matches == ["https://A.B-C_D+E&f!*(g),h%2F"]


def test_https_and_http_both_match() -> None:
    text = "first http://a.example then https://b.example."

    matches = regex_url.findall(text)

    # Trailing `.` is still inside the legal class, matching prior
    # behavior — this test pins that we did not accidentally tighten
    # the regex beyond removing the buggy range.
    assert matches == ["http://a.example", "https://b.example."]
