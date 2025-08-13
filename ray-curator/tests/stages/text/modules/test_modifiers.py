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

import pandas as pd
import pytest

from ray_curator.stages.text.modifiers import (
    DocumentModifier,
    LineRemover,
    MarkdownRemover,
    NewlineNormalizer,
    QuotationRemover,
    Slicer,
    UnicodeReformatter,
    UrlRemover,
)
from ray_curator.stages.text.modules import Modify
from ray_curator.tasks import DocumentBatch


def list_to_doc_batch(documents: list[str], col_name: str = "text") -> DocumentBatch:
    df = pd.DataFrame({col_name: documents})
    return DocumentBatch(data=df, task_id="test_id", dataset_name="test_ds")


def run_modify(modifier: DocumentModifier, doc_batch: DocumentBatch) -> DocumentBatch:
    m = Modify(modifier)
    m.setup()
    return m.process(doc_batch)


class TestUnicodeReformatter:
    def test_reformatting(self) -> None:
        # Examples taken from ftfy documentation:
        # https://ftfy.readthedocs.io/en/latest/
        doc_batch = list_to_doc_batch(
            [
                "âœ” No problems",
                "The Mona Lisa doesnÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢t have eyebrows.",
                "l’humanitÃ©",  # noqa: RUF001
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
        output = run_modify(UnicodeReformatter(uncurl_quotes=True), doc_batch)
        actual_results = output.data["text"].to_list()
        actual_results.sort()

        assert expected_results == actual_results, f"Expected: {expected_results}, but got: {actual_results}"


class TestNewlineNormalizer:
    def test_just_newlines(self) -> None:
        doc_batch = list_to_doc_batch(
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
        output = run_modify(NewlineNormalizer(), doc_batch)
        actual_results = output.data["text"].to_list()
        actual_results.sort()

        assert expected_results == actual_results, f"Expected: {expected_results}, but got: {actual_results}"

    def test_newlines_and_carriage_returns(self) -> None:
        doc_batch = list_to_doc_batch(
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
        output = run_modify(NewlineNormalizer(), doc_batch)
        actual_results = output.data["text"].to_list()
        actual_results.sort()

        assert expected_results == actual_results, f"Expected: {expected_results}, but got: {actual_results}"


class TestUrlRemover:
    def test_urls(self) -> None:
        doc_batch = list_to_doc_batch(
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
        output = run_modify(UrlRemover(), doc_batch)
        actual_results = output.data["text"].to_list()
        actual_results.sort()

        assert expected_results == actual_results, f"Expected: {expected_results}, but got: {actual_results}"


class TestLineRemover:
    def test_remove_exact_match(self) -> None:
        text = "Keep this\nRemove me\nAlso keep this\nRemove me"
        patterns = ["Remove me"]
        remover = LineRemover(patterns)
        result = remover.modify_document(text)
        expected = "Keep this\nAlso keep this"
        assert result == expected

    def test_no_removal_when_partial_match(self) -> None:
        text = "Keep this line\nThis line contains Remove me as a part of it\nAnother line"
        patterns = ["Remove me"]
        remover = LineRemover(patterns)
        # Only lines that exactly match "Remove me" are removed.
        assert remover.modify_document(text) == text

    def test_empty_input(self) -> None:
        text = ""
        patterns = ["Remove me"]
        remover = LineRemover(patterns)
        result = remover.modify_document(text)
        assert result == ""

    def test_multiple_patterns(self) -> None:
        text = "Line one\nDelete\nLine two\nRemove\nLine three\nDelete"
        patterns = ["Delete", "Remove"]
        remover = LineRemover(patterns)
        result = remover.modify_document(text)
        expected = "Line one\nLine two\nLine three"
        assert result == expected

    def test_whitespace_sensitivity(self) -> None:
        # Exact match requires identical string content.
        text = "Remove me \nRemove me\n  Remove me"
        patterns = ["Remove me"]
        remover = LineRemover(patterns)
        result = remover.modify_document(text)
        # Only the line that exactly equals "Remove me" is removed.
        expected = "Remove me \n  Remove me"
        assert result == expected

    def test_dataset_modification(self) -> None:
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
        doc_batch = list_to_doc_batch(docs)
        output = run_modify(LineRemover(["Remove me"]), doc_batch)
        expected_df = pd.DataFrame({"text": expected_results})
        pd.testing.assert_frame_equal(output.data.reset_index(drop=True), expected_df.reset_index(drop=True))


class TestQuotationRemover:
    def test_remove_quotes_no_newline(self) -> None:
        text = '"Hello, World!"'
        remover = QuotationRemover()
        result = remover.modify_document(text)
        expected = "Hello, World!"
        assert result == expected

    def test_no_removal_when_quotes_not_enclosing(self) -> None:
        text = 'Hello, "World!"'
        remover = QuotationRemover()
        result = remover.modify_document(text)
        # The text does not start and end with a quotation mark.
        assert result == text

    def test_remove_quotes_with_newline_removal(self) -> None:
        text = '"Hello,\nWorld!"'
        remover = QuotationRemover()
        result = remover.modify_document(text)
        # Since there is a newline and the first line does not end with a quote,
        # the quotes are removed.
        expected = "Hello,\nWorld!"
        assert result == expected

    def test_no_removal_with_newline_preserved(self) -> None:
        text = '"Hello,"\nWorld!"'
        remover = QuotationRemover()
        result = remover.modify_document(text)
        # The first line ends with a quote so the removal does not occur.
        assert result == text

    def test_short_text_no_removal(self) -> None:
        text = '""'
        remover = QuotationRemover()
        result = remover.modify_document(text)
        # With text length not greater than 2 (after stripping), nothing changes.
        assert result == text

    def test_extra_whitespace_prevents_removal(self) -> None:
        # If leading/trailing whitespace prevents the text from starting with a quote,
        # nothing is changed.
        text = '   "Test Message"   '
        remover = QuotationRemover()
        result = remover.modify_document(text)
        assert result == text

    def test_dataset_modification(self) -> None:
        docs = ['"Document one"', 'Start "Document two" End', '"Document\nthree"', '""']
        expected_results = [
            "Document one",
            'Start "Document two" End',
            "Document\nthree",
            '""',
        ]
        doc_batch = list_to_doc_batch(docs)
        output = run_modify(QuotationRemover(), doc_batch)
        expected_df = pd.DataFrame({"text": expected_results})
        pd.testing.assert_frame_equal(output.data.reset_index(drop=True), expected_df.reset_index(drop=True))


class TestSlicer:
    def test_integer_indices(self) -> None:
        text = "Hello, world!"
        slicer = Slicer(left=7, right=12)
        result = slicer.modify_document(text)
        expected = "world"
        assert result == expected

    def test_left_string_including(self) -> None:
        text = "abcXYZdef"
        slicer = Slicer(left="XYZ", include_left=True)
        result = slicer.modify_document(text)
        expected = "XYZdef"
        assert result == expected

    def test_left_string_excluding(self) -> None:
        text = "abcXYZdef"
        slicer = Slicer(left="XYZ", include_left=False)
        result = slicer.modify_document(text)
        expected = "def"
        assert result == expected

    def test_right_string_including(self) -> None:
        text = "abcXYZdef"
        slicer = Slicer(right="XYZ", include_right=True)
        result = slicer.modify_document(text)
        expected = "abcXYZ"
        assert result == expected

    def test_right_string_excluding(self) -> None:
        text = "abcXYZdef"
        slicer = Slicer(right="XYZ", include_right=False)
        result = slicer.modify_document(text)
        expected = "abc"
        assert result == expected

    def test_both_left_and_right_with_strings(self) -> None:
        text = "start middle end"
        slicer = Slicer(left="start", right="end", include_left=False, include_right=False)
        result = slicer.modify_document(text)
        # "start" is removed and "end" is excluded; extra spaces are stripped.
        expected = "middle"
        assert result == expected

    def test_non_existing_left(self) -> None:
        text = "abcdef"
        slicer = Slicer(left="nonexistent")
        result = slicer.modify_document(text)
        assert result == ""

    def test_non_existing_right(self) -> None:
        text = "abcdef"
        slicer = Slicer(right="nonexistent")
        result = slicer.modify_document(text)
        assert result == ""

    def test_no_left_no_right(self) -> None:
        text = "   some text with spaces   "
        slicer = Slicer()
        result = slicer.modify_document(text)
        # With no boundaries specified, the entire text is returned (stripped).
        expected = "some text with spaces"
        assert result == expected

    def test_integer_out_of_range(self) -> None:
        text = "short"
        slicer = Slicer(left=10)
        result = slicer.modify_document(text)
        # Slicing starting beyond the text length yields an empty string.
        assert result == ""

    def test_multiple_occurrences(self) -> None:
        text = "abc__def__ghi"
        # Testing when markers appear multiple times.
        slicer = Slicer(left="__", right="__", include_left=True, include_right=True)
        result = slicer.modify_document(text)
        # left: first occurrence at index 3; right: last occurrence at index 8, include_right adds len("__")
        expected = "__def__"
        assert result == expected

    def test_dataset_modification(self) -> None:
        docs = ["abcdef", "0123456789", "Hello", "Slicer"]
        expected_results = [
            "cde",  # "abcdef" sliced from index 2 to 5
            "234",  # "0123456789" sliced from index 2 to 5
            "llo",  # "Hello" sliced from index 2 to 5
            "ice",  # "Slicer" sliced from index 2 to 5
        ]
        doc_batch = list_to_doc_batch(docs)
        output = run_modify(Slicer(left=2, right=5), doc_batch)
        expected_df = pd.DataFrame({"text": expected_results})
        pd.testing.assert_frame_equal(output.data.reset_index(drop=True), expected_df.reset_index(drop=True))


class TestMarkdownRemover:
    def test_bold_removal(self) -> None:
        text = "This is **bold** text."
        remover = MarkdownRemover()
        result = remover.modify_document(text)
        expected = "This is bold text."
        assert result == expected

    def test_italic_removal(self) -> None:
        text = "This is *italic* text."
        remover = MarkdownRemover()
        result = remover.modify_document(text)
        expected = "This is italic text."
        assert result == expected

    def test_underline_removal(self) -> None:
        text = "This is _underlined_ text."
        remover = MarkdownRemover()
        result = remover.modify_document(text)
        expected = "This is underlined text."
        assert result == expected

    def test_link_removal(self) -> None:
        text = "Link: [Google](https://google.com)"
        remover = MarkdownRemover()
        result = remover.modify_document(text)
        expected = "Link: https://google.com"
        assert result == expected

    def test_multiple_markdown(self) -> None:
        text = "This is **bold**, *italic*, and _underline_, check [Example](https://example.com)"
        remover = MarkdownRemover()
        result = remover.modify_document(text)
        expected = "This is bold, italic, and underline, check https://example.com"
        assert result == expected

    def test_no_markdown(self) -> None:
        text = "This line has no markdown."
        remover = MarkdownRemover()
        result = remover.modify_document(text)
        assert result == text

    def test_incomplete_markdown(self) -> None:
        text = "This is *italic text"
        remover = MarkdownRemover()
        result = remover.modify_document(text)
        # Without a closing '*', the text remains unchanged.
        assert result == text

    def test_nested_markdown(self) -> None:
        text = "This is **bold and *italic* inside** text."
        remover = MarkdownRemover()
        result = remover.modify_document(text)
        # Bold formatting is removed first, then italics in the resulting string.
        expected = "This is bold and italic inside text."
        assert result == expected

    def test_multiple_lines(self) -> None:
        text = "**Bold line**\n*Italic line*\n_Normal line_"
        remover = MarkdownRemover()
        result = remover.modify_document(text)
        expected = "Bold line\nItalic line\nNormal line"
        assert result == expected

    def test_adjacent_markdown(self) -> None:
        text = "**Bold****MoreBold**"
        remover = MarkdownRemover()
        result = remover.modify_document(text)
        expected = "BoldMoreBold"
        assert result == expected

    def test_dataset_modification(self) -> None:
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
        doc_batch = list_to_doc_batch(docs)
        output = run_modify(MarkdownRemover(), doc_batch)
        expected_df = pd.DataFrame({"text": expected_results})
        pd.testing.assert_frame_equal(output.data.reset_index(drop=True), expected_df.reset_index(drop=True))


class TestModify:
    def test_callable_single_normalization_and_name(self) -> None:
        def inner(x: str) -> str:
            return x.upper()

        m = Modify(inner)
        assert isinstance(m.modifier_fn, list)
        assert len(m.modifier_fn) == 1
        assert m.modifier_fn[0] is inner
        assert m.text_field == ["text"]
        assert m.name == inner.__name__

    def test_multiple_callables(self) -> None:
        def fn1(s: str) -> str:
            return s.strip()

        def fn2(s: str) -> str:
            return s + "!"

        m = Modify([fn1, fn2])
        assert m.text_field == ["text", "text"]
        assert m.modifier_fn[0] is fn1
        assert m.modifier_fn[1] is fn2
        expected_name = f"modifier_chain_of_{fn1.__name__}_{fn2.__name__}"
        assert m.name == expected_name

        # Validate that values are modified as expected (strip then append "!").
        doc_batch = list_to_doc_batch(["  hello  ", "world  "])
        m.setup()
        out = m.process(doc_batch)
        assert out.data["text"].tolist() == ["hello!", "world!"]

    def test_docmodifier_single_normalization_and_name(self) -> None:
        mod = MarkdownRemover()
        m = Modify(mod)
        assert isinstance(m.modifier_fn, list)
        assert len(m.modifier_fn) == 1
        assert m.modifier_fn[0] is mod
        assert m.text_field == ["text"]
        assert m.name == "MarkdownRemover"

    def test_mixed_modifiers_with_text_fields_preserved(self) -> None:
        def fn(s: str) -> str:
            return s[::-1]

        mod = MarkdownRemover()
        m = Modify([fn, mod], text_field=["a", "b"])
        assert m.text_field == ["a", "b"]
        assert m.modifier_fn[0] is fn
        assert m.modifier_fn[1] is mod
        expected_name = f"modifier_chain_of_{fn.__name__}_MarkdownRemover"
        assert m.name == expected_name

    def test_raises_when_single_modifier_with_multiple_text_fields(self) -> None:
        mod = MarkdownRemover()
        with pytest.raises(ValueError, match=r"More text fields than modifiers provided"):
            Modify(mod, text_field=["a", "b"])

    def test_raises_when_modifier_fn_list_empty(self) -> None:
        with pytest.raises(ValueError, match=r"^modifier_fn list cannot be empty$"):
            Modify(modifier_fn=[], text_field="text")

    def test_raises_when_text_field_is_none(self) -> None:
        with pytest.raises(ValueError, match=r"^Text field cannot be None$"):
            Modify(modifier_fn=str.upper, text_field=None)

    def test_raises_when_modifier_is_not_callable_or_document_modifier(self) -> None:
        with pytest.raises(TypeError, match=r"^Each modifier must be a DocumentModifier or callable$"):
            Modify(modifier_fn=123, text_field="text")
