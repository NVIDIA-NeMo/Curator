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

import codecs
import re
from pathlib import Path

import pytest

from latex_augment.latex_translator import translate_latex

TESTDATA_DIR = Path(__file__).parent / "testdata"


testcase_re = re.compile(r"### (.+?)\n(.*?)\n---\n(.*?)(?=\n###|$)", re.DOTALL)


def load_testdata():
    """Load tests from .tex files."""
    tests = []
    paths = sorted(TESTDATA_DIR.glob("*.tex"))
    assert paths, "No test files found"
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        for match in testcase_re.findall(content):
            title, input, output = map(str.strip, match)
            lang = (
                "concat"
                if "[concat]" in title
                else "empty"
                if "[empty]" in title
                else "rot13"
            )
            xfail = "[xfail]" in title
            marks = [pytest.mark.xfail] if xfail else []
            param = pytest.param(lang, input, output, id=f"{path}:{title}", marks=marks)
            tests.append(param)
    return tests


@pytest.mark.parametrize("lang, input, expected", load_testdata())
def test_latex_translator(lang, input, expected):
    """Test LaTeX translation."""

    def rot13(sentence):
        return codecs.encode(sentence, "rot_13").replace("<n", "<a")

    def concat(sentence):
        return "".join(sentence.split())

    def empty(sentence):
        return ""

    translate = {"concat": concat, "rot13": rot13, "empty": empty}
    output = translate_latex(input, translator=translate[lang], keep_tags=True)
    assert output.rstrip("\n") == expected.rstrip("\n")


def test_latex_escape():
    def translate(txt):
        return r"asdf\{foo}%$&#_^"

    output = translate_latex("jeejee", translator=translate)
    expected = r"asdf\\\{foo\}\%\$\&\#\_\^"
    assert output == expected


def test_no_chunks():
    text = "\\usepackage[utf8]{inputenc}\n\\begin{document}\\end{document}"
    output = translate_latex(text, translator=lambda text: text)
    assert output == text


@pytest.mark.parametrize("encoding", ["latin1", "cp1250", "cp1251", "cp1252", "cp1253"])
def test_encodings(encoding):
    text = b"\xe4\xf6\xfc"  # some high-ascii characters
    doc = (
        b"\\usepackage["
        + encoding.encode()
        + b"]{inputenc}\n"
        + b"\\begin{document}\n"
        + text
        + b"\n\\end{document}"
    )
    output = translate_latex(doc, translator=lambda text: text)
    expected = doc.decode(encoding).replace(encoding, "utf8")
    assert output == expected


def test_line_endings():
    source = b"""
\\documentclass{article}
\\usepackage[utf8]{inputenc}
\\begin{document}
Content
\\end{document}
"""

    # Test Windows-style line endings (CRLF)
    output = translate_latex(source.replace(b"\n", b"\r\n"), translator=lambda text: text)
    assert output == source.decode("utf8")

    # Test Mac-style line endings (CR)
    output = translate_latex(source.replace(b"\n", b"\r"), translator=lambda text: text)
    assert output == source.decode("utf8")

    # Test Unix-style line endings (LF)
    output = translate_latex(source, translator=lambda text: text)
    assert output == source.decode("utf8")

    # Test mixed line endings
    mixed_source = b"\\documentclass{article}\r\n\\usepackage[utf8]{inputenc}\n\\begin{document}\rContent\n\\end{document}"
    output = translate_latex(mixed_source, translator=lambda text: text)
    expected = mixed_source.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    assert output == expected.decode("utf8")


def test_move_title_and_author():
    """Test moving title and author commands from preamble to body."""
    input = (
        "\\documentclass{article}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\title{My Title}\n"
        "\\author{Author Name}\n"
        "\\more{foo}\n"
        "\\begin{document}\n"
        "\\stuff{bar}\n"
        "\\maketitle\n"
        "Some text\n"
        "\\end{document}"
    )
    expected = (
        "\\documentclass{article}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\more{foo}\n"
        "\\begin{document}\n"
        "\\stuff{bar}\n"
        "\\title{My Title}\n"
        "\\author{Author Name}\n"
        "\\maketitle\n"
        "Some text\n"
        "\\end{document}"
    )
    output = translate_latex(input, translator=lambda text: text)
    assert output == expected


def test_move_nested_title_commands():
    """Test moving nested title commands from preamble to body."""
    input = (
        "\\documentclass{article}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\title{My \\thanks{Funded by X} Title}\n"
        "\\author{Author \\thanks{University Y} Name}\n"
        "\\begin{document}\n"
        "\\maketitle\n"
        "Some text\n"
        "\\end{document}"
    )
    expected = (
        "\\documentclass{article}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\begin{document}\n"
        "\\title{My \\thanks{Funded by X} Title}\n"
        "\\author{Author \\thanks{University Y} Name}\n"
        "\\maketitle\n"
        "Some text\n"
        "\\end{document}"
    )
    output = translate_latex(input, translator=lambda text: text, verbose=True)
    assert output == expected
