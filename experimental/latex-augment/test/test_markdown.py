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

from latex_augment.convert.markdown import escape_markdown


def test_bracket_escaping():
    text = "This is plain text with [brackets] and (parentheses) and [brackets](parentheses)."
    expected = "This is plain text with [brackets] and (parentheses) and [brackets]\\(parentheses)."
    assert escape_markdown(text) == expected


def test_angle_bracket_escaping():
    text = "Testing <brackets>, < testing >, 2 < 5, 2<5>3, a<b, c>d"
    expected = "Testing \\<brackets>, < testing >, 2 < 5, 2<5>3, a\\<b, c>d"
    assert escape_markdown(text) == expected


def test_header_escaping():
    text = "# This is a header"
    expected = "\\# This is a header"
    assert escape_markdown(text) == expected


def test_list_escaping():
    text = "* List item\n- Another item"
    expected = "\\* List item\n\\- Another item"
    assert escape_markdown(text) == expected


def test_asterisk_emphasis():
    text = "This has *emphasis* **even more** but not random asterisk * here"
    expected = "This has \\*emphasis* \\**even more** but not random asterisk * here"
    assert escape_markdown(text) == expected


def test_underscore_emphasis():
    text = "This has _emphasis_ **even more** but not random underscore _ here"
    expected = "This has \\_emphasis_ \\**even more** but not random underscore _ here"
    assert escape_markdown(text) == expected


def test_mixed_content():
    text = "# Header\n* List item\nThis has *emphasis* and _also this_\nRegular [brackets] (parentheses) remain *unchanged*"
    expected = "\\# Header\n\\* List item\nThis has \\*emphasis* and \\_also this_\nRegular [brackets] (parentheses) remain \\*unchanged*"
    assert escape_markdown(text) == expected


def test_multiple_headers():
    text = "# Header 1\n## Header 2\n### Header 3"
    expected = "\\# Header 1\n\\## Header 2\n\\### Header 3"
    assert escape_markdown(text) == expected


def test_indented_headers():
    text = "   # Indented header\n  ## Another indented"
    expected = "   \\# Indented header\n  \\## Another indented"
    assert escape_markdown(text) == expected


def test_bullet_list_variations():
    text = "* Bullet\n+ Plus item\n- Minus item\n  * Indented bullet"
    expected = "\\* Bullet\n\\+ Plus item\n\\- Minus item\n  \\* Indented bullet"
    assert escape_markdown(text) == expected


def test_nested_emphasis():
    text = "*Nested _emphasis_ within* and **_bold italic_**"
    expected = "\\*Nested \\_emphasis_ within* and \\**_bold italic_**"
    assert escape_markdown(text) == expected


def test_emphasis_edge_cases():
    text = "*single asterisk\n_single underscore\n*asterisk with space after* \n_underscore with space after_ "
    expected = "\\*single asterisk\n\\_single underscore\n\\*asterisk with space after* \n\\_underscore with space after_ "
    assert escape_markdown(text) == expected


def test_multiple_links():
    text = "[Link 1](url1) and [Link 2](url2) and [Link with [nested] brackets](url3)"
    expected = "[Link 1]\\(url1) and [Link 2]\\(url2) and [Link with [nested] brackets]\\(url3)"
    assert escape_markdown(text) == expected


def test_code_blocks():
    text = "```python\nprint('hello')\n```"
    expected = "\\```python\nprint('hello')\n\\```"
    assert escape_markdown(text) == expected


def test_html_mixed():
    text = "<div>HTML tag</div> and <span>inline</span> with <p>paragraph</p>"
    expected = "\\<div>HTML tag\\</div> and \\<span>inline\\</span> with \\<p>paragraph\\</p>"
    assert escape_markdown(text) == expected


def test_backslash_escaping():
    text = "This already has some \\* escaped characters \\# like \\ this \\\\"
    expected = "This already has some \\* escaped characters \\# like \\ this \\\\"
    assert escape_markdown(text) == expected


def test_alternative_header_syntax():
    text = "Header 1\n=========\nHeader 2\n---------"
    expected = "Header 1\n\\=========\nHeader 2\n\\---------"
    assert escape_markdown(text) == expected


def test_blockquotes():
    text = "> This is a blockquote\n> with multiple lines\n>> And a nested blockquote"
    expected = "\\> This is a blockquote\n\\> with multiple lines\n\\>> And a nested blockquote"
    assert escape_markdown(text) == expected


def test_horizontal_rules():
    text = "Above\n---\nBetween\n***\nBelow\n___"
    expected = "Above\n\\---\nBetween\n\\***\nBelow\n\\___"
    assert escape_markdown(text) == expected


def test_backtick_escaping():
    text = "Use `code` in your text and ``nested `backticks` `` here."
    expected = "Use \\`code` in your text and \\``nested \\`backticks` \\`` here."
    assert escape_markdown(text) == expected


def test_strikethrough():
    text = "This is ~~strikethrough~~ text"
    expected = "This is \\~~strikethrough~~ text"
    assert escape_markdown(text) == expected


def test_inline_links_with_title():
    text = "[Link with title](https://example.com \"Title here\")"
    expected = "[Link with title]\\(https://example.com \"Title here\")"
    assert escape_markdown(text) == expected


def test_image_syntax():
    text = "![Alt text](image.jpg) and ![Alt text][image]\n\n[image]: image2.jpg"
    expected = "![Alt text]\\(image.jpg) and ![Alt text]\\[image]\n\n\\[image]: image2.jpg"
    assert escape_markdown(text) == expected


def test_definition_lists():
    text = "Term\n: Definition"
    expected = "Term\n\\: Definition"
    assert escape_markdown(text) == expected
