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
from unittest.mock import MagicMock, call

import dask.dataframe as dd
import pandas as pd
import pytest

from nemo_curator.datasets import DocumentDataset
from nemo_curator.synthetic.nemotron_cc import (
    NemotronCCDiverseQAPostprocessor,
    NemotronCCGenerator,
    NemotronCCKnowledgeListPostprocessor,
)
from nemo_curator.synthetic.prompts import (
    DISTILL_PROMPT_TEMPLATE,
    DIVERSE_QA_PROMPT_TEMPLATE,
    EXTRACT_KNOWLEDGE_PROMPT_TEMPLATE,
    KNOWLEDGE_LIST_PROMPT_TEMPLATE,
    NEMOTRON_CC_DISTILL_SYSTEM_PROMPT,
    NEMOTRON_CC_SYSTEM_PROMPT,
    WIKIPEDIA_REPHRASING_PROMPT_TEMPLATE,
)


# A dummy tokenizer that simply splits text by whitespace.
class DummyTokenizer:
    def tokenize(self, text):
        return text.split()


# Helper function to create a DocumentDataset from provided data.
def create_dataset(data):
    pdf = pd.DataFrame(data)
    return DocumentDataset.from_pandas(pdf)


class TestNemotronCCGenerator:
    @pytest.fixture
    def mock_llm_client(self):
        mock_client = MagicMock()
        mock_client.query_model.return_value = ["This is a mock response"]
        return mock_client

    def test_init(self, mock_llm_client):
        """Test the constructor of NemotronCCGenerator."""
        generator = NemotronCCGenerator(mock_llm_client)
        assert generator.client == mock_llm_client

    def test_prompt(self, mock_llm_client):
        """Test the internal _prompt method."""
        generator = NemotronCCGenerator(mock_llm_client)

        document = "Test document content"
        prompt_template = "Test prompt for {document}."
        system_prompt = "System instruction"
        prompt_kwargs = {"extra_param": "test"}
        model_kwargs = {"temperature": 0.7}

        result = generator._prompt(
            model="test_model",
            document=document,
            prompt_template=prompt_template,
            system_prompt=system_prompt,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        # Check if query_model was called with the right parameters
        mock_llm_client.query_model.assert_called_once()
        call_args = mock_llm_client.query_model.call_args[1]
        assert call_args["model"] == "test_model"
        assert call_args["temperature"] == 0.7
        assert len(call_args["messages"]) == 2
        assert call_args["messages"][0]["role"] == "system"
        assert call_args["messages"][0]["content"] == "System instruction"
        assert call_args["messages"][1]["role"] == "user"
        assert (
            call_args["messages"][1]["content"]
            == "Test prompt for Test document content."
        )

        # Check return value
        assert result == ["This is a mock response"]

    def test_rewrite_to_wikipedia_style(self, mock_llm_client):
        """Test rewrite_to_wikipedia_style method."""
        generator = NemotronCCGenerator(mock_llm_client)

        document = "Original document text"
        result = generator.rewrite_to_wikipedia_style(
            document=document,
            model="test_model",
        )

        # Check the right parameters were passed to query_model
        mock_llm_client.query_model.assert_called_once()
        messages = mock_llm_client.query_model.call_args[1]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == NEMOTRON_CC_SYSTEM_PROMPT
        assert messages[1]["role"] == "user"
        assert document in messages[1]["content"]
        assert "test_model" == mock_llm_client.query_model.call_args[1]["model"]

        # Check the result
        assert result == ["This is a mock response"]

    def test_generate_diverse_qa(self, mock_llm_client):
        """Test generate_diverse_qa method."""
        generator = NemotronCCGenerator(mock_llm_client)

        document = "Document text for QA generation"
        result = generator.generate_diverse_qa(
            document=document,
            model="test_model",
        )

        # Check the right parameters were passed to query_model
        mock_llm_client.query_model.assert_called_once()
        messages = mock_llm_client.query_model.call_args[1]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == NEMOTRON_CC_SYSTEM_PROMPT
        assert messages[1]["role"] == "user"
        assert document in messages[1]["content"]
        assert (
            DIVERSE_QA_PROMPT_TEMPLATE.format(document=document)
            == messages[1]["content"]
        )
        assert "test_model" == mock_llm_client.query_model.call_args[1]["model"]

        # Check the result
        assert result == ["This is a mock response"]

    def test_distill(self, mock_llm_client):
        """Test distill method."""
        generator = NemotronCCGenerator(mock_llm_client)

        document = "Document text to distill"
        result = generator.distill(
            document=document,
            model="test_model",
        )

        # Check the right parameters were passed to query_model
        mock_llm_client.query_model.assert_called_once()
        messages = mock_llm_client.query_model.call_args[1]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == NEMOTRON_CC_DISTILL_SYSTEM_PROMPT
        assert messages[1]["role"] == "user"
        assert document in messages[1]["content"]
        assert (
            DISTILL_PROMPT_TEMPLATE.format(document=document) == messages[1]["content"]
        )
        assert "test_model" == mock_llm_client.query_model.call_args[1]["model"]

        # Check the result
        assert result == ["This is a mock response"]

    def test_extract_knowledge(self, mock_llm_client):
        """Test extract_knowledge method."""
        generator = NemotronCCGenerator(mock_llm_client)

        document = "Document text for knowledge extraction"
        result = generator.extract_knowledge(
            document=document,
            model="test_model",
        )

        # Check the right parameters were passed to query_model
        mock_llm_client.query_model.assert_called_once()
        messages = mock_llm_client.query_model.call_args[1]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == NEMOTRON_CC_SYSTEM_PROMPT
        assert messages[1]["role"] == "user"
        assert document in messages[1]["content"]
        assert (
            EXTRACT_KNOWLEDGE_PROMPT_TEMPLATE.format(document=document)
            == messages[1]["content"]
        )
        assert "test_model" == mock_llm_client.query_model.call_args[1]["model"]

        # Check the result
        assert result == ["This is a mock response"]

    def test_generate_knowledge_list(self, mock_llm_client):
        """Test generate_knowledge_list method."""
        generator = NemotronCCGenerator(mock_llm_client)

        document = "Document text for knowledge list generation"
        result = generator.generate_knowledge_list(
            document=document,
            model="test_model",
        )

        # Check the right parameters were passed to query_model
        mock_llm_client.query_model.assert_called_once()
        messages = mock_llm_client.query_model.call_args[1]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == NEMOTRON_CC_SYSTEM_PROMPT
        assert messages[1]["role"] == "user"
        assert document in messages[1]["content"]
        assert (
            KNOWLEDGE_LIST_PROMPT_TEMPLATE.format(document=document)
            == messages[1]["content"]
        )
        assert "test_model" == mock_llm_client.query_model.call_args[1]["model"]

        # Check the result
        assert result == ["This is a mock response"]

    def test_custom_prompt_and_model_kwargs(self, mock_llm_client):
        """Test methods with custom prompt template and model kwargs."""
        generator = NemotronCCGenerator(mock_llm_client)

        document = "Test document"
        custom_prompt = "Custom prompt for {document} with {extra_param}"
        custom_system_prompt = "Custom system prompt"
        prompt_kwargs = {"extra_param": "additional context"}
        model_kwargs = {"temperature": 0.5, "top_p": 0.9}

        # Test with rewrite_to_wikipedia_style
        generator.rewrite_to_wikipedia_style(
            document=document,
            model="test_model",
            prompt_template=custom_prompt,
            system_prompt=custom_system_prompt,
            prompt_kwargs=prompt_kwargs,
            model_kwargs=model_kwargs,
        )

        # Check that custom parameters were used
        call_args = mock_llm_client.query_model.call_args[1]
        assert call_args["temperature"] == 0.5
        assert call_args["top_p"] == 0.9
        assert call_args["messages"][0]["content"] == custom_system_prompt
        expected_prompt = custom_prompt.format(
            document=document, extra_param="additional context"
        )
        assert call_args["messages"][1]["content"] == expected_prompt


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

    def test_no_qa_pairs(self):
        """Test case where len(qa_pairs) == 0, which happens when there are no lines
        starting with 'Question:' in the response."""
        text = "Document text"
        # A response with only text but no lines starting with "Question:"
        llm_response = (
            "Here are the questions and answers based on the provided text:\n"
            "- This is a response without any question lines\n"
            "- Just some random text that doesn't start with Question:"
        )
        ds = create_dataset({"text": [text], "response": [llm_response]})
        processor = NemotronCCDiverseQAPostprocessor(tokenizer=None, max_num_pairs=2)
        result_ds = processor(ds)
        result_df = result_ds.df.compute()

        # Since there are no valid QA pairs, we expect the dataset to be empty
        # because _postprocess_llm_response returns an empty string
        assert (
            result_df.empty
        ), "Expected dataset to be empty when no QA pairs are found"


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
