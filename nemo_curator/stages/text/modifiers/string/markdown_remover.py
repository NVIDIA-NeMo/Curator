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

MARKDOWN_BOLD_REGEX = r"\*\*(.*?)\*\*"
MARKDOWN_ITALIC_REGEX = r"\*(.*?)\*"
# As in CommonMark, an underscore inside a word (snake_case, file_name.py, first_last@example.com)
# can't open or close emphasis (_text_ or __text__). The emphasized text can't contain another
# underscore that could open emphasis (one after a non-word character and before a non-space), so a
# long line of unmatched ones is scanned once, not once per underscore. The alternatives for the
# emphasized text are mutually exclusive, so a failed match can't backtrack exponentially.
MARKDOWN_UNDERLINE_REGEX = r"(?<!\w)(__?)(?=\S)((?:[^_]|(?<=\w)_|(?<!\w)_(?!\S))+?)(?<=\S)\1(?!\w)"
MARKDOWN_LINK_REGEX = r"\[.*?\]\((.*?)\)"


class MarkdownRemover(DocumentModifier):
    """
    Removes Markdown formatting in a document including bold, italic, underline, and URL text.
    """

    def __init__(self):
        super().__init__()

    def modify_document(self, text: str) -> str:
        lines = text.split("\n")
        new_lines = []
        for line in lines:
            line = re.sub(MARKDOWN_BOLD_REGEX, r"\1", line)  # **text** #noqa: PLW2901
            line = re.sub(MARKDOWN_ITALIC_REGEX, r"\1", line)  # *text* #noqa: PLW2901
            # The second pass removes emphasis nested in the first (___text___, __bold _italic_ bold__)
            for _ in range(2):
                line = re.sub(MARKDOWN_UNDERLINE_REGEX, r"\2", line)  # _text_ or __text__ #noqa: PLW2901
            line = re.sub(MARKDOWN_LINK_REGEX, r"\1", line)  # [text](url) #noqa: PLW2901
            new_lines.append(line)

        return "\n".join(new_lines)
