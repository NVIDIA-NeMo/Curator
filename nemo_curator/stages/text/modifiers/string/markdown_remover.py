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
MARKDOWN_UNDERLINE_REGEX = r"_(.*?)_"
MARKDOWN_LINK_REGEX = r"\[.*?\]\((.*?)\)"


class MarkdownRemover(DocumentModifier):
    """
    Removes Markdown formatting in a document including bold, italic, underline, and URL text.
    """

    def __init__(
        self,
        markdown_bold_regex: str = MARKDOWN_BOLD_REGEX,
        markdown_italic_regex: str = MARKDOWN_ITALIC_REGEX,
        markdown_underline_regex: str = MARKDOWN_UNDERLINE_REGEX,
        markdown_link_regex: str = MARKDOWN_LINK_REGEX,
    ):
        """
        Args:
            markdown_bold_regex (str): Pattern for bold text. Each match is replaced with its first group.
            markdown_italic_regex (str): Pattern for italic text. Each match is replaced with its first group.
            markdown_underline_regex (str): Pattern for underlined text. Each match is replaced with its
                first group.
            markdown_link_regex (str): Pattern for links. Each match is replaced with its first group.
        """
        super().__init__()
        self.markdown_bold_regex = markdown_bold_regex
        self.markdown_italic_regex = markdown_italic_regex
        self.markdown_underline_regex = markdown_underline_regex
        self.markdown_link_regex = markdown_link_regex

    def modify_document(self, text: str) -> str:
        lines = text.split("\n")
        new_lines = []
        for line in lines:
            line = re.sub(self.markdown_bold_regex, r"\1", line)  # **text** #noqa: PLW2901
            line = re.sub(self.markdown_italic_regex, r"\1", line)  # *text* #noqa: PLW2901
            line = re.sub(self.markdown_underline_regex, r"\1", line)  # _text_ #noqa: PLW2901
            line = re.sub(self.markdown_link_regex, r"\1", line)  # [text](url) #noqa: PLW2901
            new_lines.append(line)

        return "\n".join(new_lines)
