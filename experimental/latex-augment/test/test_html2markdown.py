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

from latex_augment.convert.html2markdown import html2markdown, convert_table_cell
from bs4 import BeautifulSoup, Tag


def test_headings():
    html = "<h1>Title</h1><h2>Subtitle</h2><h3>Section</h3>"
    expected = "# Title\n\n## Subtitle\n\n### Section"
    assert html2markdown(html) == expected


def test_text_formatting():
    html = "<p>Plain <b>bold</b> <i>italic</i> <code>code</code> <s>strike</s> text</p>"
    expected = "Plain **bold** _italic_ `code` ~~strike~~ text"
    assert html2markdown(html) == expected


def test_css_font_weight_bold():
    html = '<p>Plain text <span style="font-weight:bold">bold CSS</span> more text</p>'
    expected = "Plain text **bold CSS** more text"
    assert html2markdown(html) == expected


def test_css_font_style_italic():
    html = '<p>Plain text <span style="font-style:italic">italic CSS</span> more text</p>'
    expected = "Plain text _italic CSS_ more text"
    assert html2markdown(html) == expected


def test_css_font_styles_combined():
    html = '<p>Text <span style="font-weight:bold; font-style:italic">bold italic</span> end</p>'
    expected = "Text **_bold italic_** end"
    assert html2markdown(html) == expected


def test_lists():
    html = (
        "<ul><li>Item 1</li><li>Item 2</li></ul><ol><li>First</li><li>Second</li></ol>"
    )
    expected = "* Item 1\n* Item 2\n\n1. First\n1. Second"
    assert html2markdown(html) == expected


def test_blockquote():
    html = "<blockquote>This is a quote</blockquote>"
    expected = "> This is a quote"
    assert html2markdown(html) == expected


def test_misc_elements():
    html = "<img><hr>Text after break"
    expected = "---\n\nText after break"
    assert html2markdown(html) == expected


def test_table_formatting():
    html = """<table>
<tr>
<th>Normal</th>
<th><b>Bold</b></th>
<th><i>Italic</i></th>
<th><b><i>Bold Italic</i></b></th>
<th><b><strong>Double Bold</strong></b></th>
<th><i><em>Double Italic</em></i></th>
<th><s><del>Double Strike</del></s></th>
</tr>
<tr>
<td>Plain text</td>
<td><b>Bold text</b></td>
<td><i>Italic text</i></td>
<td><b><i>Bold italic text</i></b></td>
<td><b><strong>Double bold</strong></b></td>
<td><i><em>Double italic</em></i></td>
<td><s><del>Double strike</del></s></td>
</tr>
<tr>
<td>More plain</td>
<td><strong>Strong text</strong></td>
<td><em>Emphasized text</em></td>
<td><strong><em>Strong emphasized text</em></strong></td>
<td><strong><b>Strong bold</b></strong></td>
<td><em><i>Em italic</i></em></td>
<td><del><s>Del strike</s></del></td>
</tr>
</table>"""

    expected = """\\begin{tabular}{ccccccc}
**Normal** & **Bold** & **_Italic_** & **_Bold Italic_** & **Double Bold** & **_Double Italic_** & **~~Double Strike~~** \\\\
Plain text & **Bold text** & _Italic text_ & **_Bold italic text_** & **Double bold** & _Double italic_ & ~~Double strike~~ \\\\
More plain & **Strong text** & _Emphasized text_ & **_Strong emphasized text_** & **Strong bold** & _Em italic_ & ~~Del strike~~ \\\\
\\end{tabular}"""

    assert html2markdown(html) == expected


def test_table_escaped_characters():
    html = """<table>
<tr>
<th>Ampersand</th>
<th>Less Than</th>
<th>Greater Than</th>
</tr>
<tr>
<td>Rock &amp; Roll</td>
<td>5 &lt; 10</td>
<td>10 &gt; 5</td>
</tr>
<tr>
<td>A &amp; B &amp; C</td>
<td>&lt;html&gt;</td>
<td>Test &lt;&amp;&gt;</td>
</tr>
</table>"""

    expected = """\\begin{tabular}{ccc}
**Ampersand** & **Less Than** & **Greater Than** \\\\
Rock \\& Roll & 5 \\< 10 & 10 \\> 5 \\\\
A \\& B \\& C & \\<html\\> & Test \\<\\&\\> \\\\
\\end{tabular}"""

    assert html2markdown(html) == expected


def test_table_line_breaks():
    html = """<table>
<tr>
<th>Address</th>
<th>Contact</th>
</tr>
<tr>
<td>123 Main Street<br>New York, NY<br>10001</td>
<td>Phone: 555-1234<br>Email: test@example.com</td>
</tr>
</table>"""

    expected = """\\begin{tabular}{cc}
**Address** & **Contact** \\\\
123 Main Street<br>New York, NY<br>10001 & Phone: 555-1234<br>Email: test@example.com \\\\
\\end{tabular}"""

    assert html2markdown(html) == expected


def test_table_lists():
    html = """<table>
<tr>
<th>Regular List</th>
<th>Horizontal List</th>
</tr>
<tr>
<td><ul><li>Item 1</li><li>Item 2</li><li>Item 3</li></ul></td>
<td><div><ul><li data-after=" · ">Item A</li><li data-after=" · ">Item B</li><li>Item C</li></ul></div></td>
</tr>
<tr>
<td><ul><li>Berry Field</li><li>Birmingham, AL</li></ul></td>
<td><div><ul><li data-after=" · ">Berry Field</li><li>Birmingham, AL</li></ul></div></td>
</tr>
</table>"""

    expected = """\\begin{tabular}{cc}
**Regular List** & **Horizontal List** \\\\
Item 1 <br> Item 2 <br> Item 3 & Item A · Item B · Item C \\\\
Berry Field <br> Birmingham, AL & Berry Field · Birmingham, AL \\\\
\\end{tabular}"""

    assert html2markdown(html) == expected


def test_convert_table_cell_basic_text():
    result = convert_table_cell(get_td("<td>Plain text</td>"))
    assert result == "Plain text"


def test_convert_table_cell_bold_formatting():
    result = convert_table_cell(get_td("<td><b>Bold text</b></td>"))
    assert result == "**Bold text**"

    result = convert_table_cell(get_td("<td><strong>Strong text</strong></td>"))
    assert result == "**Strong text**"

    result = convert_table_cell(
        get_td('<td><span style="font-weight:bold">Bold span</span></td>')
    )
    assert result == "**Bold span**"


def test_convert_table_cell_italic_formatting():
    result = convert_table_cell(get_td("<td><i>Italic text</i></td>"))
    assert result == "_Italic text_"

    result = convert_table_cell(get_td("<td><em>Emphasized text</em></td>"))
    assert result == "_Emphasized text_"

    result = convert_table_cell(
        get_td('<td><span style="font-style:italic">Italic span</span></td>')
    )
    assert result == "_Italic span_"


def test_convert_table_cell_css_styles_combined():
    result = convert_table_cell(
        get_td('<td><span style="font-weight:bold; font-style:italic">Bold italic span</span></td>')
    )
    assert result == "**_Bold italic span_**"

    result = convert_table_cell(
        get_td('<td><span style="font-style:italic; font-weight:bold">Italic bold span</span></td>')
    )
    assert result == "**_Italic bold span_**"


def test_convert_table_cell_line_breaks():
    result = convert_table_cell(get_td("<td>Line 1<br>Line 2<br>Line 3</td>"))
    assert result == "Line 1<br>Line 2<br>Line 3"


def test_convert_table_cell_lists():
    result = convert_table_cell(
        get_td("<td><ul><li>Item 1</li>" "<li>Item 2</li>" "<li>Item 3</li></ul></td>")
    )
    assert result == "Item 1 <br> Item 2 <br> Item 3"

    # Test with data-after attribute
    result = convert_table_cell(
        get_td(
            '<td><ul><li data-after=" · ">Item 1</li>'
            '<li data-after=" · ">Item 2</li>'
            "<li>Item 3</li></ul></td>"
        )
    )
    assert result == "Item 1 · Item 2 · Item 3"


def test_convert_table_cell_escaping():
    result = convert_table_cell(get_td("<td>Rock &amp; Roll</td>"))
    assert result == "Rock \\& Roll"

    result = convert_table_cell(get_td("<td>5 &lt; 10 &gt; 3</td>"))
    assert result == "5 \\< 10 \\> 3"

    result = convert_table_cell(get_td("<td>Price: $50</td>"))
    assert result == "Price: \\$50"


def test_convert_table_cell_non_breaking_space():
    result = convert_table_cell(get_td("<td>Non\xa0breaking\xa0space</td>"))
    assert result == "Non breaking space"
    result = convert_table_cell(get_td("<td>4<span>&nbsp;</span>November<span>&nbsp;</span>2009 foo</td>"))
    assert result == "4 November 2009 foo"


def test_convert_table_cell_tag_removal():
    result = convert_table_cell(
        get_td("<td><div><span>Text in nested tags</span></div></td>")
    )
    assert result == "Text in nested tags"


def test_convert_table_cell_complex_formatting():
    result = convert_table_cell(
        get_td(
            "<td><b><i>Bold and italic</i></b> with <br><strong>strong</strong> text</td>"
        )
    )
    assert result == "**_Bold and italic_** with <br>**strong** text"


def test_convert_table_cell_empty():
    assert convert_table_cell(get_td("<td></td>")) == ""
    assert convert_table_cell(get_td("<td>   </td>")) == ""


def test_convert_table_cell_whitespace_handling():
    assert (
        convert_table_cell(get_td("<td>  Text with  extra   spaces  </td>"))
        == "Text with extra spaces"
    )


def test_html_entities_text():
    html = '<p>Quotes: &quot;Hello&quot; and &apos;World&apos;</p>'
    expected = 'Quotes: "Hello" and \'World\''
    assert html2markdown(html) == expected


def test_html_entities_symbols():
    html = '<p>Copyright &copy; 2024, Registered &reg;, Trademark &trade;</p>'
    expected = 'Copyright © 2024, Registered ®, Trademark ™'
    assert html2markdown(html) == expected


def test_html_entities_math():
    html = '<p>5 &times; 3 = 15, 20 &divide; 4 = 5, &plusmn; 10 degrees &deg;</p>'
    expected = '5 × 3 = 15, 20 ÷ 4 = 5, ± 10 degrees °'
    assert html2markdown(html) == expected


def test_html_entities_punctuation():
    html = '<p>En dash &ndash; Em dash &mdash; Ellipsis &hellip;</p>'
    expected = 'En dash – Em dash — Ellipsis …'
    assert html2markdown(html) == expected


def test_html_entities_spaces():
    html = '<p>Regular space, en space&ensp;here, em space&emsp;here, thin space&thinsp;here</p>'
    expected = 'Regular space, en space\u2002here, em space\u2003here, thin space\u2009here'
    assert html2markdown(html) == expected


def test_html_entities_numeric():
    html = '<p>Numeric: &#65; &#8212; &#8230;</p>'
    expected = 'Numeric: A — …'
    assert html2markdown(html) == expected


def test_html_entities_hex():
    html = '<p>Hex: &#x41; &#x2014; &#x2026;</p>'
    expected = 'Hex: A — …'
    assert html2markdown(html) == expected


def test_html_entities_mixed():
    html = '<p>&quot;Price: &pound;50 &times; 2 = &pound;100&quot; &mdash; John &amp; Jane</p>'
    expected = '"Price: £50 × 2 = £100" — John & Jane'
    assert html2markdown(html) == expected


def test_table_html_entities_comprehensive():
    html = """<table>
<tr>
<th>Text Entities</th>
<th>Symbol Entities</th>
<th>Math &amp; Punctuation</th>
</tr>
<tr>
<td>&quot;Quotes&quot; &amp; &apos;Apostrophes&apos;</td>
<td>&copy; &reg; &trade; &deg;</td>
<td>&times; &divide; &plusmn; &ndash; &mdash; &hellip;</td>
</tr>
<tr>
<td>Spaces: word&nbsp;nbsp&ensp;ensp&emsp;emsp&thinsp;thin</td>
<td>Currency: &pound;50 &euro;100 &yen;200</td>
<td>Fractions: &frac14; &frac12; &frac34;</td>
</tr>
</table>"""

    expected = """\\begin{tabular}{ccc}
**Text Entities** & **Symbol Entities** & **Math \\& Punctuation** \\\\
"Quotes" \\& 'Apostrophes' & © ® ™ ° & × ÷ ± – — … \\\\
Spaces: word nbsp\u2002ensp\u2003emsp\u2009thin & Currency: £50 €100 ¥200 & Fractions: ¼ ½ ¾ \\\\
\\end{tabular}"""

    assert html2markdown(html) == expected


def test_convert_table_cell_html_entities():
    result = convert_table_cell(get_td('<td>&quot;Hello&quot; &amp; &apos;World&apos;</td>'))
    assert result == '"Hello" \\& \'World\''
    
    result = convert_table_cell(get_td('<td>&copy; &reg; &trade; &deg;</td>'))
    assert result == '© ® ™ °'
    
    result = convert_table_cell(get_td('<td>&times; &divide; &plusmn;</td>'))
    assert result == '× ÷ ±'
    
    result = convert_table_cell(get_td('<td>&ndash; &mdash; &hellip;</td>'))
    assert result == '– — …'


def test_convert_table_cell_space_entities():
    result = convert_table_cell(get_td('<td>word&nbsp;nbsp word&ensp;ensp word&emsp;emsp word&thinsp;thin</td>'))
    assert result == 'word nbsp word\u2002ensp word\u2003emsp word\u2009thin'


def test_convert_table_cell_numeric_entities():
    result = convert_table_cell(get_td('<td>&#65; &#8212; &#8230;</td>'))
    assert result == 'A — …'
    
    result = convert_table_cell(get_td('<td>&#x41; &#x2014; &#x2026;</td>'))
    assert result == 'A — …'


def test_nbsp_conversion_regular_text():
    """Test that &nbsp; is converted to regular spaces in regular text."""
    html = '<p>Word&nbsp;with&nbsp;non-breaking&nbsp;spaces</p>'
    result = html2markdown(html)
    expected = 'Word with non-breaking spaces'
    assert result == expected
    assert '\xa0' not in result  # Should not contain Unicode non-breaking spaces


def get_td(html: str) -> Tag:
    return BeautifulSoup(html, "html.parser").find("td")
