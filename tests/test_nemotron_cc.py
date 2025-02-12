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
import random

import dask.dataframe as dd
import pandas as pd
import pytest

from nemo_curator.datasets import DocumentDataset
from nemo_curator.synthetic.nemotron_cc import (
    NemotronCCDiverseQAPostprocessor,
    NemotronCCKnowledgeListPostprocessor,
)


# A dummy tokenizer that simply splits text by whitespace.
class DummyTokenizer:
    def tokenize(self, text):
        return text.split()


# Helper function to create a DocumentDataset from provided data.
def create_dataset(data):
    pdf = pd.DataFrame(data)
    return DocumentDataset.from_pandas(pdf)


class TestDiverseQAPostprocessor:
    def test_valid_response_without_tokenizer(self, monkeypatch):
        # Patch randomness so that the ordering and sampling is deterministic.
        monkeypatch.setattr(random, "shuffle", lambda x: None)
        # In the branch without a tokenizer, random.randint(1, max_num_pairs)
        # will be forced to return the upper bound.
        monkeypatch.setattr(random, "randint", lambda lo, hi: hi)

        text = "Document text"
        llm_response = (
            "Here are the questions and answers based on the provided text:\n"
            "- Question: What is this?\n"
            "Answer: It is a test.\n"
            "- Question: How does it work?\n"
            "Answer: By magic."
        )
        # Create a dataset with one row containing both the document and the LLM response.
        ds = create_dataset({"text": [text], "response": [llm_response]})

        # Use no tokenizer so that the branch using max_num_pairs (here, 2) is used.
        processor = NemotronCCDiverseQAPostprocessor(tokenizer=None, max_num_pairs=2)
        result_ds = processor(ds)
        result_df = result_ds.df.compute()

        # Expected processing:
        # 1. Split into lines and remove the leading "- " prefix.
        # 2. Remove the prefix line ("Here are...") if it matches.
        # 3. Merge lines: the first QA pair becomes:
        #       "Question: What is this?\nAnswer: It is a test."
        #    and the second:
        #       "Question: How does it work?\nAnswer: By magic."
        # 4. With our patched randint, both QA pairs are kept.
        expected_qa = (
            "Question: What is this?\nAnswer: It is a test.\n\n"
            "Question: How does it work?\nAnswer: By magic."
        )
        expected_response = f"{text}\n\n{expected_qa}"

        assert not result_df.empty, "Expected non-empty dataset"
        actual_response = result_df.iloc[0]["response"]
        assert (
            actual_response == expected_response
        ), f"Expected: {expected_response}, got: {actual_response}"

    def test_valid_response_with_tokenizer(self, monkeypatch):
        # Using a dummy tokenizer.
        dummy_tokenizer = DummyTokenizer()
        monkeypatch.setattr(random, "shuffle", lambda x: None)
        # For the branch with a tokenizer, the number of tokens is determined by:
        # num_tokens = len(dummy_tokenizer.tokenize(text)). For "Document text" this yields 2.
        # Then max_num = max(1, int(max_num_pairs * num_tokens / 150)) becomes max(1, int(4/150)) -> 1.
        monkeypatch.setattr(random, "randint", lambda lo, hi: hi)

        text = "Document text"
        llm_response = (
            "Here are the questions and answers based on the provided text:\n"
            "- Question: What is this?\n"
            "Answer: It is a test.\n"
            "- Question: How does it work?\n"
            "Answer: By magic."
        )
        ds = create_dataset({"text": [text], "response": [llm_response]})
        processor = NemotronCCDiverseQAPostprocessor(
            tokenizer=dummy_tokenizer, max_num_pairs=2
        )
        result_ds = processor(ds)
        result_df = result_ds.df.compute()

        # In the tokenizer branch only one QA pair is selected (the first one).
        expected_qa = "Question: What is this?\nAnswer: It is a test."
        expected_response = f"{text}\n\n{expected_qa}"

        assert not result_df.empty, "Expected non-empty dataset"
        actual_response = result_df.iloc[0]["response"]
        assert (
            actual_response == expected_response
        ), f"Expected: {expected_response}, got: {actual_response}"

    def test_invalid_response_format(self, monkeypatch):
        # Test a response with an invalid QA format (missing a "Question:" line).
        monkeypatch.setattr(random, "shuffle", lambda x: None)
        monkeypatch.setattr(random, "randint", lambda lo, hi: hi)

        text = "Doc"
        # The response only has an answer line.
        llm_response = (
            "Here are the questions and answers based on the provided text:\n"
            "- Answer: Missing question."
        )
        ds = create_dataset({"text": [text], "response": [llm_response]})
        processor = NemotronCCDiverseQAPostprocessor(tokenizer=None, max_num_pairs=2)
        result_ds = processor(ds)
        result_df = result_ds.df.compute()

        # Since the response format is invalid (no "Question:" to start a QA pair),
        # the postprocessing should return an empty string; the __call__ method then
        # drops that row.
        assert (
            result_df.empty
        ), "Expected dataset to be empty due to invalid response format"

    def test_empty_response(self):
        # Test when the LLM response is empty.
        text = "Doc"
        llm_response = ""
        ds = create_dataset({"text": [text], "response": [llm_response]})
        processor = NemotronCCDiverseQAPostprocessor(tokenizer=None, max_num_pairs=2)
        result_ds = processor(ds)
        result_df = result_ds.df.compute()

        # The empty LLM response should lead to an empty processed text and get filtered out.
        assert result_df.empty, "Expected dataset to be empty for an empty LLM response"

    def test_more_qa_than_max(self, monkeypatch):
        # Test when there are more QA pairs than max_num_pairs.
        monkeypatch.setattr(random, "shuffle", lambda x: None)
        monkeypatch.setattr(random, "randint", lambda lo, hi: hi)

        text = "Document text"
        llm_response = (
            "Here are the questions and answers based on the provided text:\n"
            "- Question: Q1?\n"
            "Answer: A1.\n"
            "- Question: Q2?\n"
            "Answer: A2.\n"
            "- Question: Q3?\n"
            "Answer: A3.\n"
            "- Question: Q4?\n"
            "Answer: A4."
        )
        ds = create_dataset({"text": [text], "response": [llm_response]})
        processor = NemotronCCDiverseQAPostprocessor(tokenizer=None, max_num_pairs=2)
        result_ds = processor(ds)
        result_df = result_ds.df.compute()

        # With max_num_pairs set to 2 and patched randint returning the upper bound,
        # only the first two QA pairs should be selected.
        expected_qa = "Question: Q1?\nAnswer: A1.\n\n" "Question: Q2?\nAnswer: A2."
        expected_response = f"{text}\n\n{expected_qa}"

        assert not result_df.empty, "Expected non-empty dataset"
        actual_response = result_df.iloc[0]["response"]
        assert (
            actual_response == expected_response
        ), f"Expected: {expected_response}, got: {actual_response}"


class TestKnowledgeListPostprocessor:
    def test_basic_formatting(self):
        # Test that a response with an initial non-bullet line (to skip) and bullet lines
        # is correctly cleaned.
        input_response = (
            "Not a bullet line to skip\n"
            "- Fact one: This is the first fact.\n"
            "  Continued fact one.\n"
            "- Fact two: This is the second fact."
        )
        ds = create_dataset({"text": [input_response]})
        processor = NemotronCCKnowledgeListPostprocessor(text_field="text")
        result_ds = processor(ds)
        result_df = result_ds.df.compute()

        # Expected:
        # - First line is skipped (since it does not start with "-").
        # - Bullet lines have the leading "- " or "  " removed.
        expected_output = (
            "Fact one: This is the first fact.\n"
            "Continued fact one.\n"
            "Fact two: This is the second fact."
        )
        actual_output = result_df.iloc[0]["text"]
        assert (
            actual_output == expected_output
        ), f"Expected: {expected_output}, got: {actual_output}"

    def test_all_bullet_lines(self):
        # Test when every line starts with a bullet prefix.
        input_response = "- Item one\n" "- Item two\n" "- Item three"
        ds = create_dataset({"text": [input_response]})
        processor = NemotronCCKnowledgeListPostprocessor(text_field="text")
        result_ds = processor(ds)
        result_df = result_ds.df.compute()

        # Each line should be cleaned by removing the leading bullet.
        expected_output = "Item one\nItem two\nItem three"
        actual_output = result_df.iloc[0]["text"]
        assert (
            actual_output == expected_output
        ), f"Expected: {expected_output}, got: {actual_output}"

    def test_no_bullet_lines(self):
        # If the response contains no bullet lines, then the first line is
        # skipped and no text remains.
        input_response = "This is just plain text without any bullet."
        ds = create_dataset({"text": [input_response]})
        processor = NemotronCCKnowledgeListPostprocessor(text_field="text")
        result_ds = processor(ds)
        result_df = result_ds.df.compute()

        expected_output = ""
        actual_output = result_df.iloc[0]["text"]
        assert (
            actual_output == expected_output
        ), f"Expected an empty string, got: {actual_output}"

    def test_mixed_indentation(self):
        # Test mixed bullet prefixes and additional non-bullet lines.
        input_response = (
            "- Bullet one\n"
            "Some extra text\n"
            "  Indented line\n"
            "- Bullet two\n"
            "  Continuation of bullet two\n"
            "Another standalone line"
        )
        ds = create_dataset({"text": [input_response]})
        processor = NemotronCCKnowledgeListPostprocessor(text_field="text")
        result_ds = processor(ds)
        result_df = result_ds.df.compute()

        # Note: Only the very first line is conditionally skipped if it doesn't start with '-'.
        # Here, since the first line starts with "-", nothing is skipped.
        # Each line that starts with "- " or "  " should have those two characters removed.
        expected_output = (
            "Bullet one\n"
            "Some extra text\n"
            "Indented line\n"
            "Bullet two\n"
            "Continuation of bullet two\n"
            "Another standalone line"
        )
        actual_output = result_df.iloc[0]["text"]
        assert (
            actual_output == expected_output
        ), f"Expected: {expected_output}, got: {actual_output}"

    def test_empty_input(self):
        # Test that an empty input returns an empty string.
        input_response = ""
        ds = create_dataset({"text": [input_response]})
        processor = NemotronCCKnowledgeListPostprocessor(text_field="text")
        result_ds = processor(ds)
        result_df = result_ds.df.compute()

        expected_output = ""
        actual_output = result_df.iloc[0]["text"]
        assert (
            actual_output == expected_output
        ), f"Expected empty string, got: {actual_output}"
