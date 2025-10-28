# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re


def escape_markdown(text: str, *, is_col0: bool = True) -> str:
    """
    Escape plain text as markdown with minimal escaping.
    Only escapes characters when they would be interpreted as markdown formatting.
    Args:
        text: The text to escape.
        is_col0: Whether the text starts a new line.
    """
    if not text:
        return text

    result = []
    lines = text.split("\n")

    for i, line in enumerate(lines):
        if is_col0 or i > 0:
            # Handle headers (# at beginning of line)
            if line.lstrip().startswith("#"):
                index = line.find("#")
                line = line[:index] + "\\" + line[index:]

            # Handle blockquotes (> at beginning of line)
            if line.lstrip().startswith(">"):
                index = line.find(">")
                line = line[:index] + "\\" + line[index:]

            # Handle list items (* - + : at beginning of line)
            if re.match(r"^\s*[\*\-\+:]\s", line):
                index = re.search(r"[\*\-\+:]", line).start()
                line = line[:index] + "\\" + line[index:]

        # Handle alternative header syntax (=== or --- on a line after text)
        if i > 0 and line.strip() and all(c == line.strip()[0] for c in line.strip()):
            if line.strip()[0] in "=-":
                line = "\\" + line

        # Handle horizontal rules (---, ***, ___)
        if line.strip() and all(c == line.strip()[0] for c in line.strip()):
            if line.strip()[0] in "-*_" and len(line.strip()) >= 3:
                line = "\\" + line

        # Handle angle brackets for HTML-like tags (<tag>)
        line = re.sub(r"<([a-zA-Z/])", r"\\<\1", line)

        # Process emphasis patterns (*word* or _word_)
        line = re.sub(r"(\s|^)(\*|_)([^\s])", r"\1\\\2\3", line)

        # Handle markdown links [text](link)
        line = re.sub(r"(\[.*?\])(\()", r"\1\\\2", line)

        # Handle image reference links ![Alt text][ref]
        line = re.sub(r"(!\[.*?\])(\[)", r"\1\\\2", line)

        # Handle reference link definitions
        if re.match(r"^\s*\[.*?\]:\s+.*$", line):
            index = line.find("[")
            line = line[:index] + "\\" + line[index:]

        # Handle backticks for inline code (escape opening backticks)
        line = re.sub(r"(^|\s)(`+)(?=\S)", r"\1\\\2", line)

        # Handle code blocks at start/end of line
        if line.startswith("```"):
            line = "\\" + line

        # Handle strikethrough - only escape the opening sequence
        line = re.sub(r"(^|\s)(~~)(?=\S)", r"\1\\\2", line)

        result.append(line)

    return "\n".join(result)
