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

from unittest.mock import AsyncMock, call, patch

import pytest
import yaml

from nemo_curator.synthetic.async_nemotron_cc import AsyncNemotronCCGenerator
from nemo_curator.synthetic.prompts import (
    DISTILL_PROMPT_TEMPLATE,
    DIVERSE_QA_PROMPT_TEMPLATE,
    EXTRACT_KNOWLEDGE_PROMPT_TEMPLATE,
    KNOWLEDGE_LIST_PROMPT_TEMPLATE,
    NEMOTRON_CC_DISTILL_SYSTEM_PROMPT,
    NEMOTRON_CC_SYSTEM_PROMPT,
    WIKIPEDIA_REPHRASING_PROMPT_TEMPLATE,
)


class TestAsyncNemotronCCGenerator:
    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock for the LLM client."""
        mock_client = AsyncMock()
        mock_client.query_model.return_value = ["This is a mock response"]
        return mock_client

    @pytest.mark.asyncio
    async def test_init(self, mock_llm_client):
        """Test the constructor of AsyncNemotronCCGenerator."""
        generator = AsyncNemotronCCGenerator(mock_llm_client)
        assert generator.client == mock_llm_client

    @pytest.mark.asyncio
    async def test_prompt(self, mock_llm_client):
        """Test the internal _prompt method."""
        generator = AsyncNemotronCCGenerator(mock_llm_client)

        document = "Test document"
        prompt_template = "Test prompt with {document}."
        system_prompt = "Test system prompt"
        prompt_kwargs = {"extra_param": "value"}
        model_kwargs = {"temperature": 0.7}

        result = await generator._prompt(
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

        # Check that messages were properly constructed
        messages = call_args["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == system_prompt
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Test prompt with Test document."

        # Check that the result is returned properly
        assert result == ["This is a mock response"]

    @pytest.mark.asyncio
    async def test_rewrite_to_wikipedia_style(self, mock_llm_client):
        """Test the rewrite_to_wikipedia_style method."""
        generator = AsyncNemotronCCGenerator(mock_llm_client)

        document = "This is a test document about a topic."
        result = await generator.rewrite_to_wikipedia_style(
            document=document,
            model="test_model",
        )

        # Check if _prompt was called with the right parameters
        mock_llm_client.query_model.assert_called_once()
        call_args = mock_llm_client.query_model.call_args[1]

        # Check the messages structure
        messages = call_args["messages"]
        assert len(messages) == 2

        # System message should have the correct prompt
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == NEMOTRON_CC_SYSTEM_PROMPT

        # User message should contain the document
        assert messages[1]["role"] == "user"
        assert document in messages[1]["content"]

        # Check that the result is returned properly
        assert result == ["This is a mock response"]

    @pytest.mark.asyncio
    async def test_generate_diverse_qa(self, mock_llm_client):
        """Test the generate_diverse_qa method."""
        generator = AsyncNemotronCCGenerator(mock_llm_client)

        document = "This is a test document about AI technology."
        result = await generator.generate_diverse_qa(
            document=document,
            model="test_model",
        )

        # Check if _prompt was called with the right parameters
        mock_llm_client.query_model.assert_called_once()
        call_args = mock_llm_client.query_model.call_args[1]

        # Check the messages structure
        messages = call_args["messages"]
        assert len(messages) == 2

        # System message should have the correct prompt
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == NEMOTRON_CC_SYSTEM_PROMPT

        # User message should contain the document and use the correct prompt template
        assert messages[1]["role"] == "user"
        assert document in messages[1]["content"]

        # Check that the result is returned properly
        assert result == ["This is a mock response"]

    @pytest.mark.asyncio
    async def test_distill(self, mock_llm_client):
        """Test the distill method."""
        generator = AsyncNemotronCCGenerator(mock_llm_client)

        document = "This is a test document with lots of information to distill."
        result = await generator.distill(
            document=document,
            model="test_model",
        )

        # Check if _prompt was called with the right parameters
        mock_llm_client.query_model.assert_called_once()
        call_args = mock_llm_client.query_model.call_args[1]

        # Check the messages structure
        messages = call_args["messages"]
        assert len(messages) == 2

        # System message should have the correct prompt - for distill it uses a special system prompt
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == NEMOTRON_CC_DISTILL_SYSTEM_PROMPT

        # User message should contain the document and use the correct prompt template
        assert messages[1]["role"] == "user"
        assert document in messages[1]["content"]

        # Check that the result is returned properly
        assert result == ["This is a mock response"]

    @pytest.mark.asyncio
    async def test_extract_knowledge(self, mock_llm_client):
        """Test the extract_knowledge method."""
        generator = AsyncNemotronCCGenerator(mock_llm_client)

        document = "This is a test document containing knowledge to extract."
        result = await generator.extract_knowledge(
            document=document,
            model="test_model",
        )

        # Check if _prompt was called with the right parameters
        mock_llm_client.query_model.assert_called_once()
        call_args = mock_llm_client.query_model.call_args[1]

        # Check the messages structure
        messages = call_args["messages"]
        assert len(messages) == 2

        # System message should have the correct prompt
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == NEMOTRON_CC_SYSTEM_PROMPT

        # User message should contain the document and use the correct prompt template
        assert messages[1]["role"] == "user"
        assert document in messages[1]["content"]

        # Check that the result is returned properly
        assert result == ["This is a mock response"]

    @pytest.mark.asyncio
    async def test_generate_knowledge_list(self, mock_llm_client):
        """Test the generate_knowledge_list method."""
        generator = AsyncNemotronCCGenerator(mock_llm_client)

        document = "This is a test document for generating a knowledge list."
        result = await generator.generate_knowledge_list(
            document=document,
            model="test_model",
        )

        # Check if _prompt was called with the right parameters
        mock_llm_client.query_model.assert_called_once()
        call_args = mock_llm_client.query_model.call_args[1]

        # Check the messages structure
        messages = call_args["messages"]
        assert len(messages) == 2

        # System message should have the correct prompt
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == NEMOTRON_CC_SYSTEM_PROMPT

        # User message should contain the document and use the correct prompt template
        assert messages[1]["role"] == "user"
        assert document in messages[1]["content"]

        # Check that the result is returned properly
        assert result == ["This is a mock response"]

    @pytest.mark.asyncio
    async def test_with_custom_prompt_and_system_prompt(self, mock_llm_client):
        """Test using custom prompt templates and system prompts."""
        generator = AsyncNemotronCCGenerator(mock_llm_client)

        document = "Test document"
        custom_prompt = "Custom prompt with {document}"
        custom_system_prompt = "Custom system prompt"

        result = await generator.generate_diverse_qa(
            document=document,
            model="test_model",
            prompt_template=custom_prompt,
            system_prompt=custom_system_prompt,
        )

        # Check if the custom prompts were used
        messages = mock_llm_client.query_model.call_args[1]["messages"]
        assert messages[0]["content"] == custom_system_prompt
        assert messages[1]["content"] == "Custom prompt with Test document"

        # Check that the result is returned properly
        assert result == ["This is a mock response"]

    @pytest.mark.asyncio
    async def test_with_additional_prompt_kwargs(self, mock_llm_client):
        """Test passing additional kwargs to the prompt template."""
        generator = AsyncNemotronCCGenerator(mock_llm_client)

        document = "Test document"
        custom_prompt = "Custom prompt with {document} and {extra_param}"

        result = await generator.generate_diverse_qa(
            document=document,
            model="test_model",
            prompt_template=custom_prompt,
            prompt_kwargs={"extra_param": "extra value"},
        )

        # Check if the extra parameter was used in formatting
        user_content = mock_llm_client.query_model.call_args[1]["messages"][1][
            "content"
        ]
        assert "extra value" in user_content

        # Check that the result is returned properly
        assert result == ["This is a mock response"]

    @pytest.mark.asyncio
    async def test_with_additional_model_kwargs(self, mock_llm_client):
        """Test passing additional kwargs to the model."""
        generator = AsyncNemotronCCGenerator(mock_llm_client)

        document = "Test document"
        model_kwargs = {
            "temperature": 0.8,
            "max_tokens": 500,
            "top_p": 0.95,
        }

        result = await generator.generate_diverse_qa(
            document=document,
            model="test_model",
            model_kwargs=model_kwargs,
        )

        # Check if the model kwargs were passed to query_model
        call_kwargs = mock_llm_client.query_model.call_args[1]
        assert call_kwargs["temperature"] == 0.8
        assert call_kwargs["max_tokens"] == 500
        assert call_kwargs["top_p"] == 0.95

        # Check that the result is returned properly
        assert result == ["This is a mock response"]
