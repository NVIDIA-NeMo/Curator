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

import re

from nemo_curator.stages.text.modifiers.doc_modifier import DocumentModifier

URL_REGEX = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
# Punctuation right after a URL usually belongs to the surrounding text. The escapes are the ellipsis,
# closing curly quotes, closing guillemets and CJK sentence punctuation.
TRAILING_PUNCTUATION = ".,;:!?'\"\u2026\u201d\u2019\u00bb\u203a\u3002\uff0c\u3001\uff1b\uff1a\uff01\uff1f"
CLOSING_BRACKETS = {
    ")": "(",
    "]": "[",
    "}": "{",
    ">": "<",
    # Full-width and CJK brackets
    "\uff09": "\uff08",
    "\u300d": "\u300c",
    "\u300f": "\u300e",
    "\u3011": "\u3010",
    "\u300b": "\u300a",
    "\u3009": "\u3008",
}


def _url_length(url: str) -> int:
    """Return the length of ``url`` without the trailing punctuation that follows it in the text.

    A closing bracket stays part of the URL when the URL contains the matching opening bracket,
    as in ``https://en.wikipedia.org/wiki/Python_(programming_language)``.
    """
    end = len(url)
    # For each closing bracket seen at the end: how many more closing than opening brackets url[:end] has.
    unmatched: dict[str, int] = {}
    while end > 0:
        char = url[end - 1]
        if char in TRAILING_PUNCTUATION:
            end -= 1
            continue
        opening = CLOSING_BRACKETS.get(char)
        if opening is None:
            break
        if char not in unmatched:
            unmatched[char] = url.count(char, 0, end) - url.count(opening, 0, end)
        if unmatched[char] <= 0:
            break
        unmatched[char] -= 1
        end -= 1
    return end


class UrlRemover(DocumentModifier):
    """
    Removes all URLs in a document.

    Punctuation right after a URL, such as a closing parenthesis, a comma or a period, is kept.
    """

    def __init__(self):
        super().__init__()

    def modify_document(self, text: str) -> str:
        return URL_REGEX.sub(lambda match: match.group()[_url_length(match.group()) :], text)
