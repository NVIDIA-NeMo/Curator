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

_RAW_URL = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
_TRAIL_PUNCT = re.compile(r'[.,;:!?\)\]\}>"\']+$')


class UrlRemover(DocumentModifier):
    """
    Removes all URLs in a document.
    """

    def __init__(self):
        super().__init__()

    def modify_document(self, text: str) -> str:
        parts: list[str] = []
        last = 0
        for m in _RAW_URL.finditer(text):
            url = m.group(0)
            clean = _TRAIL_PUNCT.sub("", url)
            # Restore closing parens that balance opens inside the URL
            # (e.g. Wikipedia URLs like .../Python_(programming_language))
            if clean != url:
                removed = url[len(clean):]
                opens = clean.count("(") - clean.count(")")
                i = 0
                while opens > 0 and i < len(removed):
                    if removed[i] == ")":
                        clean += ")"
                        opens -= 1
                        i += 1
                    else:
                        break
            parts.append(text[last:m.start()])
            parts.append(url[len(clean):])
            last = m.end()
        parts.append(text[last:])
        return "".join(parts)
