# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import dask.dataframe as dd
import pandas as pd
from dask.dataframe.utils import assert_eq

from nemo_curator import Modify
from nemo_curator.datasets import DocumentDataset
from nemo_curator.modifiers import (
    LineRemover,
    MarkdownRemover,
    NewlineNormalizer,
    QuotationRemover,
    Slicer,
    UnicodeReformatter,
    UrlRemover,
)


def list_to_dataset(documents, col_name="text", npartitions=2):
    data = {col_name: documents}
    pdf = pd.DataFrame(data)

    return DocumentDataset.from_pandas(pdf, npartitions=npartitions)


class TestUnicodeReformatter:
    def test_reformatting(self):
        # Examples taken from ftfy documentation:
        # https://ftfy.readthedocs.io/en/latest/
        dataset = list_to_dataset(
            [
                "âœ” No problems",
                "The Mona Lisa doesnÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢t have eyebrows.",
                "l’humanitÃ©",
                "Ã perturber la rÃ©flexion",
                "Clean document already.",
            ]
        )
        expected_results = [
            "✔ No problems",
            "The Mona Lisa doesn't have eyebrows.",
            "l'humanité",
            "à perturber la réflexion",
            "Clean document already.",
        ]
        expected_results.sort()

        modifier = Modify(UnicodeReformatter(uncurl_quotes=True))
        fixed_dataset = modifier(dataset)
        actual_results = fixed_dataset.df.compute()["text"].to_list()
        actual_results.sort()

        assert (
            expected_results == actual_results
        ), f"Expected: {expected_results}, but got: {actual_results}"


class TestNewlineNormalizer:
    def test_just_newlines(self):
        dataset = list_to_dataset(
            [
                "The quick brown fox jumps over the lazy dog",
                "The quick\nbrown fox jumps \nover the lazy dog",
                "The quick\n\nbrown fox jumps \n\nover the lazy dog",
                "The quick\n\n\nbrown fox jumps \n\n\nover the lazy dog",
                "The quick\n\n\nbrown fox jumps \nover the lazy dog",
            ]
        )
        expected_results = [
            "The quick brown fox jumps over the lazy dog",
            "The quick\nbrown fox jumps \nover the lazy dog",
            "The quick\n\nbrown fox jumps \n\nover the lazy dog",
            "The quick\n\nbrown fox jumps \n\nover the lazy dog",
            "The quick\n\nbrown fox jumps \nover the lazy dog",
        ]
        expected_results.sort()

        modifier = Modify(NewlineNormalizer())
        fixed_dataset = modifier(dataset)
        actual_results = fixed_dataset.df.compute()["text"].to_list()
        actual_results.sort()

        assert (
            expected_results == actual_results
        ), f"Expected: {expected_results}, but got: {actual_results}"

    def test_newlines_and_carriage_returns(self):
        dataset = list_to_dataset(
            [
                "The quick brown fox jumps over the lazy dog",
                "The quick\r\nbrown fox jumps \r\nover the lazy dog",
                "The quick\r\n\r\nbrown fox jumps \r\n\r\nover the lazy dog",
                "The quick\r\n\r\n\r\nbrown fox jumps \r\n\r\n\r\nover the lazy dog",
                "The quick\r\n\r\n\r\nbrown fox jumps \r\nover the lazy dog",
            ]
        )
        expected_results = [
            "The quick brown fox jumps over the lazy dog",
            "The quick\r\nbrown fox jumps \r\nover the lazy dog",
            "The quick\r\n\r\nbrown fox jumps \r\n\r\nover the lazy dog",
            "The quick\r\n\r\nbrown fox jumps \r\n\r\nover the lazy dog",
            "The quick\r\n\r\nbrown fox jumps \r\nover the lazy dog",
        ]
        expected_results.sort()

        modifier = Modify(NewlineNormalizer())
        fixed_dataset = modifier(dataset)
        actual_results = fixed_dataset.df.compute()["text"].to_list()
        actual_results.sort()

        assert (
            expected_results == actual_results
        ), f"Expected: {expected_results}, but got: {actual_results}"


class TestUrlRemover:
    def test_urls(self):
        dataset = list_to_dataset(
            [
                "This is a url: www.nvidia.com",
                "This is a url: http://www.nvidia.com",
                "This is a url: https://www.nvidia.com",
                "This is a url: https://www.nvidia.gov",
                "This is a url: https://nvidia.com",
                "This is a url: HTTPS://WWW.NVIDIA.COM",
                "This is not a url: git@github.com:NVIDIA/NeMo-Curator.git",
            ]
        )
        expected_results = [
            "This is a url: ",
            "This is a url: ",
            "This is a url: ",
            "This is a url: ",
            "This is a url: ",
            "This is a url: ",
            "This is not a url: git@github.com:NVIDIA/NeMo-Curator.git",
        ]
        expected_results.sort()

        modifier = Modify(UrlRemover())
        fixed_dataset = modifier(dataset)
        actual_results = fixed_dataset.df.compute()["text"].to_list()
        actual_results.sort()

        assert (
            expected_results == actual_results
        ), f"Expected: {expected_results}, but got: {actual_results}"


class TestLineRemover:
    def test_remove_exact_match(self):
        text = "Keep this\nRemove me\nAlso keep this\nRemove me"
        patterns = ["Remove me"]
        remover = LineRemover(patterns)
        result = remover.modify_document(text)
        expected = "Keep this\nAlso keep this"
        assert result == expected

    def test_no_removal_when_partial_match(self):
        text = (
            "Keep this line\nThis line contains Remove me as a part of it\nAnother line"
        )
        patterns = ["Remove me"]
        remover = LineRemover(patterns)
        # Only lines that exactly match "Remove me" are removed.
        assert remover.modify_document(text) == text

    def test_empty_input(self):
        text = ""
        patterns = ["Remove me"]
        remover = LineRemover(patterns)
        result = remover.modify_document(text)
        assert result == ""

    def test_multiple_patterns(self):
        text = "Line one\nDelete\nLine two\nRemove\nLine three\nDelete"
        patterns = ["Delete", "Remove"]
        remover = LineRemover(patterns)
        result = remover.modify_document(text)
        expected = "Line one\nLine two\nLine three"
        assert result == expected

    def test_whitespace_sensitivity(self):
        # Exact match requires identical string content.
        text = "Remove me \nRemove me\n  Remove me"
        patterns = ["Remove me"]
        remover = LineRemover(patterns)
        result = remover.modify_document(text)
        # Only the line that exactly equals "Remove me" is removed.
        expected = "Remove me \n  Remove me"
        assert result == expected

    def test_dataset_modification(self):
        docs = [
            "Keep this\nRemove me\nKeep that",
            "Remove me\nDon't remove\nRemove me",
            "No removal here",
            "Remove me",
        ]
        expected_results = [
            "Keep this\nKeep that",
            "Don't remove",
            "No removal here",
            "",
        ]
        dataset = list_to_dataset(docs)
        modifier = Modify(LineRemover(["Remove me"]))
        fixed_dataset = modifier(dataset)
        expected_dataset = list_to_dataset(expected_results)
        assert_eq(fixed_dataset.df, expected_dataset.df)


class TestQuotationRemover:
    def test_remove_quotes_no_newline(self):
        text = '"Hello, World!"'
        remover = QuotationRemover()
        result = remover.modify_document(text)
        expected = "Hello, World!"
        assert result == expected

    def test_no_removal_when_quotes_not_enclosing(self):
        text = 'Hello, "World!"'
        remover = QuotationRemover()
        result = remover.modify_document(text)
        # The text does not start and end with a quotation mark.
        assert result == text

    def test_remove_quotes_with_newline_removal(self):
        text = '"Hello,\nWorld!"'
        remover = QuotationRemover()
        result = remover.modify_document(text)
        # Since there is a newline and the first line does not end with a quote,
        # the quotes are removed.
        expected = "Hello,\nWorld!"
        assert result == expected

    def test_no_removal_with_newline_preserved(self):
        text = '"Hello,"\nWorld!"'
        remover = QuotationRemover()
        result = remover.modify_document(text)
        # The first line ends with a quote so the removal does not occur.
        assert result == text

    def test_short_text_no_removal(self):
        text = '""'
        remover = QuotationRemover()
        result = remover.modify_document(text)
        # With text length not greater than 2 (after stripping), nothing changes.
        assert result == text

    def test_extra_whitespace_prevents_removal(self):
        # If leading/trailing whitespace prevents the text from starting with a quote,
        # nothing is changed.
        text = '   "Test Message"   '
        remover = QuotationRemover()
        result = remover.modify_document(text)
        assert result == text

    def test_dataset_modification(self):
        import pandas as pd
        from dask.dataframe.utils import assert_eq

        docs = ['"Document one"', 'Start "Document two" End', '"Document\nthree"', '""']
        expected_results = [
            "Document one",
            'Start "Document two" End',
            "Document\nthree",
            '""',
        ]
        dataset = list_to_dataset(docs)
        modifier = Modify(QuotationRemover())
        fixed_dataset = modifier(dataset)
        expected_dataset = list_to_dataset(expected_results)
        assert_eq(fixed_dataset.df, expected_dataset.df)


class TestSlicer:
    def test_integer_indices(self):
        text = "Hello, world!"
        slicer = Slicer(left=7, right=12)
        result = slicer.modify_document(text)
        expected = "world"
        assert result == expected

    def test_left_string_including(self):
        text = "abcXYZdef"
        slicer = Slicer(left="XYZ", include_left=True)
        result = slicer.modify_document(text)
        expected = "XYZdef"
        assert result == expected

    def test_left_string_excluding(self):
        text = "abcXYZdef"
        slicer = Slicer(left="XYZ", include_left=False)
        result = slicer.modify_document(text)
        expected = "def"
        assert result == expected

    def test_right_string_including(self):
        text = "abcXYZdef"
        slicer = Slicer(right="XYZ", include_right=True)
        result = slicer.modify_document(text)
        expected = "abcXYZ"
        assert result == expected

    def test_right_string_excluding(self):
        text = "abcXYZdef"
        slicer = Slicer(right="XYZ", include_right=False)
        result = slicer.modify_document(text)
        expected = "abc"
        assert result == expected

    def test_both_left_and_right_with_strings(self):
        text = "start middle end"
        slicer = Slicer(
            left="start", right="end", include_left=False, include_right=False
        )
        result = slicer.modify_document(text)
        # "start" is removed and "end" is excluded; extra spaces are stripped.
        expected = "middle"
        assert result == expected

    def test_non_existing_left(self):
        text = "abcdef"
        slicer = Slicer(left="nonexistent")
        result = slicer.modify_document(text)
        assert result == ""

    def test_non_existing_right(self):
        text = "abcdef"
        slicer = Slicer(right="nonexistent")
        result = slicer.modify_document(text)
        assert result == ""

    def test_no_left_no_right(self):
        text = "   some text with spaces   "
        slicer = Slicer()
        result = slicer.modify_document(text)
        # With no boundaries specified, the entire text is returned (stripped).
        expected = "some text with spaces"
        assert result == expected

    def test_integer_out_of_range(self):
        text = "short"
        slicer = Slicer(left=10)
        result = slicer.modify_document(text)
        # Slicing starting beyond the text length yields an empty string.
        assert result == ""

    def test_multiple_occurrences(self):
        text = "abc__def__ghi"
        # Testing when markers appear multiple times.
        slicer = Slicer(left="__", right="__", include_left=True, include_right=True)
        result = slicer.modify_document(text)
        # left: first occurrence at index 3; right: last occurrence at index 8, include_right adds len("__")
        expected = "__def__"
        assert result == expected

    def test_dataset_modification(self):
        import pandas as pd
        from dask.dataframe.utils import assert_eq

        docs = ["abcdef", "0123456789", "Hello", "Slicer"]
        expected_results = [
            "cde",  # "abcdef" sliced from index 2 to 5
            "234",  # "0123456789" sliced from index 2 to 5
            "llo",  # "Hello" sliced from index 2 to 5
            "ice",  # "Slicer" sliced from index 2 to 5
        ]
        dataset = list_to_dataset(docs)
        modifier = Modify(Slicer(left=2, right=5))
        fixed_dataset = modifier(dataset)
        expected_dataset = list_to_dataset(expected_results)
        assert_eq(fixed_dataset.df, expected_dataset.df)


class TestMarkdownRemover:
    def test_bold_removal(self):
        text = "This is **bold** text."
        remover = MarkdownRemover()
        result = remover.modify_document(text)
        expected = "This is bold text."
        assert result == expected

    def test_italic_removal(self):
        text = "This is *italic* text."
        remover = MarkdownRemover()
        result = remover.modify_document(text)
        expected = "This is italic text."
        assert result == expected

    def test_underline_removal(self):
        text = "This is _underlined_ text."
        remover = MarkdownRemover()
        result = remover.modify_document(text)
        expected = "This is underlined text."
        assert result == expected

    def test_link_removal(self):
        text = "Link: [Google](https://google.com)"
        remover = MarkdownRemover()
        result = remover.modify_document(text)
        expected = "Link: https://google.com"
        assert result == expected

    def test_multiple_markdown(self):
        text = "This is **bold**, *italic*, and _underline_, check [Example](https://example.com)"
        remover = MarkdownRemover()
        result = remover.modify_document(text)
        expected = "This is bold, italic, and underline, check https://example.com"
        assert result == expected

    def test_no_markdown(self):
        text = "This line has no markdown."
        remover = MarkdownRemover()
        result = remover.modify_document(text)
        assert result == text

    def test_incomplete_markdown(self):
        text = "This is *italic text"
        remover = MarkdownRemover()
        result = remover.modify_document(text)
        # Without a closing '*', the text remains unchanged.
        assert result == text

    def test_nested_markdown(self):
        text = "This is **bold and *italic* inside** text."
        remover = MarkdownRemover()
        result = remover.modify_document(text)
        # Bold formatting is removed first, then italics in the resulting string.
        expected = "This is bold and italic inside text."
        assert result == expected

    def test_multiple_lines(self):
        text = "**Bold line**\n*Italic line*\n_Normal line_"
        remover = MarkdownRemover()
        result = remover.modify_document(text)
        expected = "Bold line\nItalic line\nNormal line"
        assert result == expected

    def test_adjacent_markdown(self):
        text = "**Bold****MoreBold**"
        remover = MarkdownRemover()
        result = remover.modify_document(text)
        expected = "BoldMoreBold"
        assert result == expected

    def test_dataset_modification(self):
        import pandas as pd
        from dask.dataframe.utils import assert_eq

        docs = [
            "This is **bold**",
            "This is *italic*",
            "Check [Link](https://example.com)",
            "No markdown here",
        ]
        expected_results = [
            "This is bold",
            "This is italic",
            "Check https://example.com",
            "No markdown here",
        ]
        dataset = list_to_dataset(docs)
        modifier = Modify(MarkdownRemover())
        fixed_dataset = modifier(dataset)
        expected_dataset = list_to_dataset(expected_results)
        assert_eq(fixed_dataset.df, expected_dataset.df)
