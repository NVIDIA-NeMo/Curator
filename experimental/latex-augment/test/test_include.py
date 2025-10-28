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

import pytest
import tempfile

from latex_augment.latex_parser import LatexError, load_includes


def test_include():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(tmpdir + "/foo.tex", "w") as f:
            f.write("yadda yadda\n")
        with open(tmpdir + "/foo", "w") as f:
            f.write("nope nope\n")
        output = load_includes(b"\\input{foo}\n", tmpdir)
        expected = b"yadda yadda"
        assert output.strip() == expected


def test_recursive_include():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(tmpdir + "/foo.tex", "w") as f:
            f.write("\\input{bar}\n")
        with open(tmpdir + "/bar.tex", "w") as f:
            f.write("yadda yadda\n")
        output = load_includes(b"\\input{foo}\n", tmpdir)
        expected = b"yadda yadda"
        assert output.strip() == expected


def test_load_includes_with_files_dict():
    main_latex = b"""\\documentclass{article}
\\begin{document}
Main content
\\input{chapter1}
More content
\\end{document}"""
    
    files = {
        "chapter1.tex": b"Chapter 1 content\n"
    }
    
    result = load_includes(main_latex, files=files)
    expected = b"""\\documentclass{article}
\\begin{document}
Main content
Chapter 1 content
More content
\\end{document}"""
    
    assert result == expected


def test_load_includes_auto_tex_extension():
    main_latex = b"""\\documentclass{article}
\\input{chapter1}
\\end{document}"""
    
    files = {
        "chapter1.tex": b"Content with auto extension\n"
    }
    
    result = load_includes(main_latex, files=files)
    assert b"Content with auto extension" in result


def test_load_includes_recursive():
    main_latex = b"""\\documentclass{article}
\\input{level1}
\\end{document}"""
    
    files = {
        "level1.tex": b"Level 1 start\n\\input{level2}\nLevel 1 end\n",
        "level2.tex": b"Level 2 content\n"
    }
    
    result = load_includes(main_latex, files=files)
    expected = b"""\\documentclass{article}
Level 1 start
Level 2 content
Level 1 end
\\end{document}"""
    
    assert result == expected


def test_load_includes_missing_file_strict():
    main_latex = b"""\\documentclass{article}
\\input{missing}
\\end{document}"""
    
    files = {}
    
    with pytest.raises(LatexError, match="File not found: missing"):
        load_includes(main_latex, files=files, strict=True)


def test_load_includes_missing_file_non_strict():
    main_latex = b"""\\documentclass{article}
\\input{missing}
\\end{document}"""
    
    files = {}
    
    result = load_includes(main_latex, files=files, strict=False)
    assert result == main_latex


def test_load_includes_path_normalization():
    main_latex = b"""\\documentclass{article}
\\input{./subdir/../chapter1}
\\end{document}"""
    
    files = {
        "chapter1.tex": b"Normalized path content\n"
    }
    
    result = load_includes(main_latex, files=files)
    assert b"Normalized path content" in result


def test_load_includes_multiple_files():
    main_latex = b"""\\documentclass{article}
\\begin{document}
\\input{intro}
\\input{main}
\\input{conclusion}
\\end{document}"""
    
    files = {
        "intro.tex": b"Introduction\n",
        "main.tex": b"Main content\n\n", 
        "conclusion.tex": b"Conclusion\n"
    }
    
    result = load_includes(main_latex, files=files)
    expected = b"""\\documentclass{article}
\\begin{document}
Introduction
Main content

Conclusion
\\end{document}"""
    
    assert result == expected


def test_load_includes_with_include_command():
    main_latex = b"""\\documentclass{article}
\\begin{document}
\\include{chapter}
\\end{document}"""
    
    files = {
        "chapter.tex": b"Included chapter content\n"
    }
    
    result = load_includes(main_latex, files=files)
    assert b"Included chapter content" in result


def test_load_includes_no_includes():
    main_latex = b"""\\documentclass{article}
\\begin{document}
No includes here
\\end{document}"""
    
    result = load_includes(main_latex, files={})
    assert result == main_latex


@pytest.mark.parametrize("end_newline", [True, False])
def test_input_with_text_same_line(end_newline):
    main_latex = b"""\\documentclass{article}
\\input{chapter1.tex}text
more text
\\end{document}"""
    
    files = {
        "chapter1.tex": b"Chapter content" + (b"\n" if end_newline else b"")
    }
    
    result = load_includes(main_latex, files=files)
    expected = b"""\\documentclass{article}
Chapter content
text
more text
\\end{document}"""
    
    assert result == expected


def test_input_with_paragraph():
    main_latex = b"""\\documentclass{article}
\\input{chapter1.tex}text
more text
\\end{document}"""
    
    files = {
        "chapter1.tex": b"Chapter content\n\n"
    }
    
    result = load_includes(main_latex, files=files)
    expected = b"""\\documentclass{article}
Chapter content

text
more text
\\end{document}"""
    
    assert result == expected


@pytest.mark.parametrize("end_newline", [True, False])
def test_input_with_text_next_line(end_newline):
    main_latex = b"""\\documentclass{article}
\\input{chapter1.tex}
more text
\\end{document}"""
    
    files = {
        "chapter1.tex": b"Chapter content" + (b"\n" if end_newline else b"")
    }
    
    result = load_includes(main_latex, files=files)
    expected = b"""\\documentclass{article}
Chapter content
more text
\\end{document}"""
    
    assert result == expected


@pytest.mark.parametrize("end_newline", [True, False])
def test_input_with_blank_line(end_newline):
    main_latex = b"""\\documentclass{article}
\\input{chapter1.tex}

more text
\\end{document}"""
    
    files = {
        "chapter1.tex": b"Chapter content" + (b"\n" if end_newline else b"")
    }
    
    result = load_includes(main_latex, files=files)
    expected = b"""\\documentclass{article}
Chapter content

more text
\\end{document}"""
    
    assert result == expected


@pytest.mark.parametrize("end_newline", [True, False])
def test_input_multiline_content(end_newline):
    main_latex = b"""\\documentclass{article}
\\input{chapter1.tex}text
\\end{document}"""
    
    files = {
        "chapter1.tex": b"Line 1\nLine 2\nLine 3" + (b"\n" if end_newline else b"")
    }
    
    result = load_includes(main_latex, files=files)
    expected = b"""\\documentclass{article}
Line 1
Line 2
Line 3
text
\\end{document}"""
    
    assert result == expected


@pytest.mark.parametrize("end_newline", [True, False])
def test_input_with_trailing_whitespace(end_newline):
    main_latex = b"""\\documentclass{article}
\\input{chapter1.tex}   

more text
\\end{document}"""
    
    files = {
        "chapter1.tex": b"Chapter content" + (b"\n" if end_newline else b"")
    }
    
    result = load_includes(main_latex, files=files)
    expected = b"""\\documentclass{article}
Chapter content

more text
\\end{document}"""
    
    assert result == expected


@pytest.mark.parametrize("end_newline", [True, False])
def test_input_with_trailing_comment(end_newline):
    main_latex = b"""\\documentclass{article}
\\input{chapter1.tex}   % asdf

more text
\\end{document}"""
    
    files = {
        "chapter1.tex": b"Chapter content" + (b"\n" if end_newline else b"")
    }
    
    result = load_includes(main_latex, files=files)
    expected = b"""\\documentclass{article}
Chapter content
% asdf

more text
\\end{document}"""
    
    assert result == expected
