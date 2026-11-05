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

# Characters that belong to the surrounding text when they end a URL match.
_TRAILING_PUNCTUATION = frozenset(".,;:!?'\">")
_CLOSING_BRACKETS = {")": "(", "]": "[", "}": "{"}


def _url_length(url: str) -> int:
    """Return the length of ``url`` without trailing punctuation from the surrounding text.

    A closing bracket is kept when it balances an opening bracket inside the URL, as in
    ``https://en.wikipedia.org/wiki/Python_(programming_language)`` or ``http://[::1]``.
    """
    opened = {opener: url.count(opener) for opener in _CLOSING_BRACKETS.values()}
    closed = {closer: url.count(closer) for closer in _CLOSING_BRACKETS}
    end = len(url)
    while end > 0:
        char = url[end - 1]
        if char in _CLOSING_BRACKETS:
            if closed[char] <= opened[_CLOSING_BRACKETS[char]]:
                break
            closed[char] -= 1
        elif char not in _TRAILING_PUNCTUATION:
            break
        end -= 1
    return end


class UrlRemover(DocumentModifier):
    """
    Removes all URLs in a document, keeping punctuation that follows them.
    """

    def __init__(self):
        super().__init__()

    def modify_document(self, text: str) -> str:
        return URL_REGEX.sub(lambda match: match.group(0)[_url_length(match.group(0)) :], text)
