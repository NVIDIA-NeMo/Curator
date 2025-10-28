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

from latex_augment.latex_parser import LatexDocument, LatexError, convert_line_endings


def test_basic_latex_document():
    source = b"""\\documentclass{article}
\\usepackage{amsmath}
\\begin{document}
Hello world
\\end{document}"""
    doc = LatexDocument(source)
    assert doc.documentclass == ("article", [])
    assert list(doc.packages) == ["amsmath"]
    assert doc.body == b"Hello world\n"
    assert doc.source == source


def test_documentclass_with_spaces():
    source = b"""\\documentclass {article}
\\begin {document}
Content
\\end   {document}"""
    doc = LatexDocument(source)
    assert doc.documentclass == ("article", [])
    assert doc.body == b"Content\n"


def test_documentclass_with_options():
    source = b"""\\documentclass[12pt,a4paper]{article}
\\begin{document}
Content
\\end{document}"""
    doc = LatexDocument(source)

    assert doc.documentclass == ("article", ["12pt", "a4paper"])


def test_multiline_documentclass():
    source = b"""\\documentclass[%
aps,
superscriptaddress,
%twocolumn,
10pt,
]{revtex4-1}
\\begin{document}
Content
\\end{document}"""
    doc = LatexDocument(source)
    assert doc.documentclass == ("revtex4-1", ["aps", "superscriptaddress", "10pt"])


def test_multiple_packages():
    source = b"""\\documentclass{article}
\\usepackage{lmodern,babel,adjustbox,booktabs}
\\begin{document}
\\end{document}"""
    doc = LatexDocument(source)
    assert list(doc.packages) == ["lmodern", "babel", "adjustbox", "booktabs"]


def test_with_package():
    source = b"""\\documentclass{article}
\\begin{document}
Content
\\end{document}"""
    doc = LatexDocument(source)

    new_doc = doc.with_package("hyperref")
    assert "hyperref" in new_doc.packages

    # Test adding package that's already present
    new_doc2 = new_doc.with_package("hyperref")
    assert new_doc2.source == new_doc.source


def test_wrap_body():
    source = b"""\\documentclass{article}
\\begin{document}
Content
\\end{document}"""
    doc = LatexDocument(source)

    new_doc = doc.wrap_body("center", "center")
    assert new_doc.body == b"\\begin{center}\nContent\n\\end{center}\n"


def test_no_begin_document():
    source = b"""\\documentclass{article}
Content"""
    with pytest.raises(LatexError, match=r"No \\begin{document} found."):
        LatexDocument(source).body


def test_with_documentclass():
    source = b"""\\documentclass{article}
\\begin{document}
Content
\\end{document}"""
    doc = LatexDocument(source)

    new_doc = doc.with_documentclass("book", ["12pt"])
    assert new_doc.documentclass == ("book", ["12pt"])


def test_preamble():
    source = b"""\\documentclass{article}
\\usepackage{amsmath}
\\begin{document}
Content
\\end{document}"""
    doc = LatexDocument(source)

    expected_preamble = b"""\\documentclass{article}
\\usepackage{amsmath}
"""
    assert doc.preamble == expected_preamble

    source = b"""\\documentclass{article}
\\usepackage{amsmath}
%\\begin{document} hehe
\\begin{document}
Content
\\end{document}"""
    doc = LatexDocument(source)

    expected_preamble = b"""\\documentclass{article}
\\usepackage{amsmath}
%\\begin{document} hehe
"""
    assert doc.preamble == expected_preamble


def test_replace_preamble():
    source = b"""\\documentclass{article}
\\begin{document}
Content
\\end{document}"""
    doc = LatexDocument(source)

    new_preamble = b"""\\documentclass{book}
\\usepackage{amsmath}
"""
    new_doc = doc.replace_preamble(new_preamble)
    assert new_doc.preamble == new_preamble


def test_subsections():
    source = b"""\\documentclass{article}
\\begin{document}
\\section{First}
Content 1
\\subsection{Sub}
Content 2
\\section{Second}
Content 3
\\end{document}"""
    doc = LatexDocument(source)

    sections = doc.subsections
    assert len(sections) > 1
    # Test that sections split the content appropriately
    assert all(isinstance(section, slice) for section in sections)


def test_insert_at():
    source = b"""\\documentclass{article}
\\begin{document}
Content
\\end{document}"""
    doc = LatexDocument(source)

    # Test inserting at single position
    new_doc = doc.editor().insert_at(doc.body_start, b"Inserted text\n").execute()
    assert new_doc.body == b"Inserted text\nContent\n"

    # Test inserting at multiple positions
    editor = doc.editor()
    editor.insert_at(doc.body_start, b"Start insert\n")
    editor.insert_at(doc.body_end, b"End insert\n")
    new_doc = editor.execute()
    assert new_doc.body == b"Start insert\nContent\nEnd insert\n"

    # Test inserting at beginning and end
    editor = doc.editor()
    editor.insert_at(0, b"Start\n")
    editor.insert_at(len(doc.source), b"End\n")
    new_doc = editor.execute()
    assert new_doc.source.startswith(b"Start\n")
    assert new_doc.source.endswith(b"End\n")

    # Test inserting twice at same position
    editor = doc.editor()
    editor.insert_at(doc.body_start, b"First insert\n")
    editor.insert_at(doc.body_start, b"Second insert\n")
    new_doc = editor.execute()
    assert new_doc.body == b"First insert\nSecond insert\nContent\n"

    # Test inserting empty string
    editor = doc.editor()
    editor.insert_at(doc.body_start, b"")
    new_doc = editor.execute()
    assert new_doc.body == doc.body


def test_replace_at():
    source = b"""\\documentclass{article}
\\begin{document}
First line
Second line
Third line
\\end{document}"""
    doc = LatexDocument(source)

    # Find the position of "Second line"
    start = doc.source.index(b"Second line")
    end = start + len(b"Second line")

    # Test replacing single span
    editor = doc.editor()
    editor.replace_at(slice(start, end), b"Replacement line")
    new_doc = editor.execute()
    assert new_doc.body == b"First line\nReplacement line\nThird line\n"

    # Test replacing multiple spans
    start1 = doc.source.index(b"First line")
    end1 = start1 + len(b"First line")
    start2 = doc.source.index(b"Third line")
    end2 = start2 + len(b"Third line")

    editor = doc.editor()
    editor.replace_at(slice(start1, end1), b"New first")
    editor.replace_at(slice(start2, end2), b"New third")
    new_doc = editor.execute()
    assert new_doc.body == b"New first\nSecond line\nNew third\n"

    # Test replacing with empty string
    editor = doc.editor()
    editor.replace_at(slice(doc.body_start, doc.body_start + 11), b"")
    new_doc = editor.execute()
    assert new_doc.body == b"Second line\nThird line\n"

    # Test replacing many times at same position
    editor = doc.editor()
    editor.replace_at(slice(start, end), b"Replacement line")
    editor.replace_at(slice(start, start), b"Added line\n")
    new_doc = editor.execute()
    assert new_doc.body == b"First line\nAdded line\nReplacement line\nThird line\n"


def test_invalid_insert_replace():
    source = b"""\\documentclass{article}
\\begin{document}
Content
\\end{document}"""
    doc = LatexDocument(source)

    # Test overlapping spans
    with pytest.raises(AssertionError, match="Spans must not overlap"):
        editor = doc.editor()
        editor.replace_at(slice(5, 15), b"text1")
        editor.replace_at(slice(10, 20), b"text2")
        editor.execute()

    # Test span with step
    with pytest.raises(AssertionError, match="Spans must not have a step"):
        editor = doc.editor()
        editor.replace_at(slice(0, 10, 2), b"text")
        editor.execute()


def test_line_ending_positions():
    source = b"\\documentclass{article}\r\n\\begin{document}\r\nasdf\r\n\\end{document}"
    source = convert_line_endings(source)
    doc = LatexDocument(source)
    assert doc.body_start == source.index(b"asdf")
    assert doc.body_end == source.index(b"asdf") + 5
    begin = doc.find(lambda node: node.text == b"\\begin{document}")
    assert source[begin.start_byte : begin.end_byte] == b"\\begin{document}"


def test_index_of_single_line_multiple_commands():
    source = b"\\textbf{a}\\textbf{b}\\textbf{c}"
    doc = LatexDocument(source)
    
    pattern = b"\\\\textbf\\{"
    
    first_pos = doc.index_of(pattern, start=0)
    second_pos = doc.index_of(pattern, start=5)
    third_pos = doc.index_of(pattern, start=15)
    
    assert first_pos == 0
    assert second_pos == 10
    assert third_pos == 20


def test_index_of_with_comments():
    """Test index_of correctly ignores patterns in comments."""
    source = b"""First line
% This comment has \\textbf{} which should be ignored
\\textbf{real command}
Third line"""
    doc = LatexDocument(source)
    
    pattern = b"\\\\textbf\\{"
    
    # Should find the real command, not the one in the comment
    pos = doc.index_of(pattern, start=0)
    found_text = doc.source[pos:pos+8]
    assert found_text == b"\\textbf{"
    
    # The position should be after the comment line
    comment_end = doc.source.index(b"\n", doc.source.index(b"% This comment"))
    assert pos > comment_end


def test_index_of_multiline_behavior():
    source = b"""\\textbf{line1}
\\textbf{line2}
\\textbf{line3}"""
    doc = LatexDocument(source)
    
    pattern = b"\\\\textbf\\{"
    
    first_line = doc.index_of(pattern, start=0)
    assert first_line == 0
    
    second_line_start = doc.source.index(b"\n") + 1
    second_line = doc.index_of(pattern, start=second_line_start)
    assert second_line == second_line_start
    
    third_line_start = doc.source.rindex(b"\n") + 1  
    third_line = doc.index_of(pattern, start=third_line_start)
    assert third_line == third_line_start


def test_find_command_basic():
    source = b"\\textbf{hello} some text \\textbf{world}"
    doc = LatexDocument(source)
    
    commands = list(doc.find_command("textbf"))
    assert len(commands) == 2
    
    first_cmd = doc.source[commands[0]]
    assert first_cmd == b"\\textbf{hello} "
    
    second_cmd = doc.source[commands[1]]
    assert second_cmd == b"\\textbf{world}"


def test_find_command_nested_braces():
    """Test find_command with nested braces."""
    source = b"\\textbf{outer {inner} text}"
    doc = LatexDocument(source)
    
    commands = list(doc.find_command("textbf"))
    assert len(commands) == 1
    
    cmd = doc.source[commands[0]]
    assert cmd == b"\\textbf{outer {inner} text}"


def test_find_command_escaped_braces():
    """Test find_command with escaped braces."""
    source = b"\\textbf{foo \\{ bar}"
    doc = LatexDocument(source)

    commands = list(doc.find_command("textbf"))
    assert len(commands) == 1

    cmd = doc.source[commands[0]]
    assert cmd == b"\\textbf{foo \\{ bar}"


def test_find_command_multiple_on_same_line():
    source = b"\\textbf{a}\\textbf{b}\\textbf{c}"
    doc = LatexDocument(source)
    
    commands = list(doc.find_command("textbf"))
    assert len(commands) == 3
    
    expected_commands = [b"\\textbf{a}", b"\\textbf{b}", b"\\textbf{c}"]
    actual_commands = [doc.source[cmd] for cmd in commands]
    
    assert actual_commands == expected_commands
    
    # Test adjacent empty commands
    source = b"\\textbf{}\\textbf{}"
    doc = LatexDocument(source)
    
    commands = list(doc.find_command("textbf"))
    assert len(commands) == 2
    
    first_cmd = doc.source[commands[0]]
    assert first_cmd == b"\\textbf{}"
    
    second_cmd = doc.source[commands[1]]
    assert second_cmd == b"\\textbf{}"


def test_find_command_with_whitespace():
    """Test find_command captures trailing whitespace correctly."""
    source = b"\\textbf{test}   \n\\section{next}"
    doc = LatexDocument(source)
    
    commands = list(doc.find_command("textbf"))
    assert len(commands) == 1
    
    # Command should include trailing whitespace
    cmd = doc.source[commands[0]]
    assert cmd == b"\\textbf{test}   \n"


def test_find_command_mismatched_braces():
    """Test find_command raises error for mismatched braces."""
    source = b"\\textbf{unclosed brace"
    doc = LatexDocument(source)
    
    with pytest.raises(LatexError, match="Mismatched braces"):
        list(doc.find_command("textbf"))


def test_find_command_no_matches():
    """Test find_command with no matching commands."""
    source = b"\\section{title} some text"
    doc = LatexDocument(source)
    
    commands = list(doc.find_command("textbf"))
    assert len(commands) == 0


def test_find_command_with_range():
    source = b"\\textbf{first}\\textbf{second}\\textbf{third}"
    doc = LatexDocument(source)
    
    start = 14
    end = 30
    
    commands = list(doc.find_command("textbf", start=start, end=end))
    assert len(commands) == 1
    
    cmd = doc.source[commands[0]]
    assert cmd == b"\\textbf{second}"
