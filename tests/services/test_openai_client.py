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

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai._types import NOT_GIVEN

from nemo_curator.services.openai_client import AsyncOpenAIClient, OpenAIClient


class TestOpenAIClient:
    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client for testing."""
        mock_client = MagicMock()
        # Configure the mock response structure to match what OpenAI returns
        completion_response = MagicMock()
        choice = MagicMock()
        choice.message.content = "This is a mock response"
        completion_response.choices = [choice]
        mock_client.chat.completions.create.return_value = completion_response
        return mock_client

    def test_init(self, mock_openai_client):
        """Test the constructor of OpenAIClient."""
        client = OpenAIClient(mock_openai_client)
        assert client.client == mock_openai_client

    def test_query_model_basic(self, mock_openai_client):
        """Test the query_model method with basic parameters."""
        client = OpenAIClient(mock_openai_client)
        messages = [{"role": "user", "content": "Hello"}]

        result = client.query_model(
            messages=messages,
            model="gpt-4",
        )

        # Check if OpenAI client was called with the right parameters
        mock_openai_client.chat.completions.create.assert_called_once()
        call_args = mock_openai_client.chat.completions.create.call_args[1]
        assert call_args["messages"] == messages
        assert call_args["model"] == "gpt-4"
        assert call_args["max_tokens"] is NOT_GIVEN
        assert call_args["temperature"] is NOT_GIVEN

        # Check return value
        assert result == ["This is a mock response"]

    def test_query_model_with_all_parameters(self, mock_openai_client):
        """Test the query_model method with all possible parameters."""
        client = OpenAIClient(mock_openai_client)
        messages = [{"role": "user", "content": "Hello"}]

        result = client.query_model(
            messages=messages,
            model="gpt-4",
            max_tokens=100,
            n=1,
            seed=42,
            stop=["stop_token"],
            stream=False,
            temperature=0.7,
            top_p=0.9,
            top_k=5,  # This should raise a warning but still work
            conversation_formatter=MagicMock(),  # This should raise a warning but still work
        )

        # Check if OpenAI client was called with the right parameters
        mock_openai_client.chat.completions.create.assert_called_once()
        call_args = mock_openai_client.chat.completions.create.call_args[1]
        assert call_args["messages"] == messages
        assert call_args["model"] == "gpt-4"
        assert call_args["max_tokens"] == 100
        assert call_args["n"] == 1
        assert call_args["seed"] == 42
        assert call_args["stop"] == ["stop_token"]
        assert call_args["stream"] is False
        assert call_args["temperature"] == 0.7
        assert call_args["top_p"] == 0.9

        # Check return value
        assert result == ["This is a mock response"]

    def test_query_model_with_warnings(self, mock_openai_client):
        """Test that warnings are raised appropriately."""
        client = OpenAIClient(mock_openai_client)
        messages = [{"role": "user", "content": "Hello"}]

        with pytest.warns(UserWarning, match="conversation_formatter is not used"):
            client.query_model(
                messages=messages,
                model="gpt-4",
                conversation_formatter=MagicMock(),
            )

        with pytest.warns(UserWarning, match="top_k is not used"):
            client.query_model(
                messages=messages,
                model="gpt-4",
                top_k=5,
            )

    def test_query_model_multiple_choices(self, mock_openai_client):
        """Test handling of multiple choices in response."""
        client = OpenAIClient(mock_openai_client)
        messages = [{"role": "user", "content": "Hello"}]

        # Configure mock to return multiple choices
        choice1 = MagicMock()
        choice1.message.content = "First response"
        choice2 = MagicMock()
        choice2.message.content = "Second response"
        completion_response = MagicMock()
        completion_response.choices = [choice1, choice2]
        mock_openai_client.chat.completions.create.return_value = completion_response

        result = client.query_model(
            messages=messages,
            model="gpt-4",
            n=2,
        )

        # Check that all choices are returned
        assert result == ["First response", "Second response"]

    def test_query_reward_model(self, mock_openai_client):
        """Test the query_reward_model method."""
        client = OpenAIClient(mock_openai_client)
        messages = [
            {"role": "user", "content": "Write a sentence"},
            {"role": "assistant", "content": "This is a sentence"},
        ]

        # Configure mock to return logprobs
        choice = MagicMock()
        logprob_token = MagicMock()
        logprob_token.token = "GOOD"
        logprob_token.logprob = -0.5
        choice.logprobs.content = [logprob_token]
        completion_response = MagicMock()
        completion_response.choices = [choice]
        mock_openai_client.chat.completions.create.return_value = completion_response

        result = client.query_reward_model(
            messages=messages,
            model="reward-model",
        )

        # Check if OpenAI client was called with the right parameters
        mock_openai_client.chat.completions.create.assert_called_once_with(
            messages=messages, model="reward-model"
        )

        # Check return value
        assert result == {"GOOD": -0.5}

    def test_query_reward_model_error(self, mock_openai_client):
        """Test error handling when logprobs are not found."""
        client = OpenAIClient(mock_openai_client)
        messages = [
            {"role": "user", "content": "Write a sentence"},
            {"role": "assistant", "content": "This is a sentence"},
        ]

        # Configure mock to return a response without logprobs
        choice = MagicMock()
        choice.logprobs = None
        completion_response = MagicMock()
        completion_response.choices = [choice]
        mock_openai_client.chat.completions.create.return_value = completion_response

        with pytest.raises(ValueError, match="Logprobs not found"):
            client.query_reward_model(
                messages=messages,
                model="not-a-reward-model",
            )


class TestAsyncOpenAIClient:
    @pytest.fixture
    def mock_async_openai_client(self):
        """Create a mock AsyncOpenAI client for testing."""
        mock_client = MagicMock()
        # Configure the mock response structure to match what AsyncOpenAI returns
        completion_response = MagicMock()
        choice = MagicMock()
        choice.message.content = "This is a mock async response"
        completion_response.choices = [choice]

        # Use AsyncMock for asynchronous method
        mock_client.chat.completions.create = AsyncMock()
        mock_client.chat.completions.create.return_value = completion_response
        return mock_client

    def test_init(self, mock_async_openai_client):
        """Test the constructor of AsyncOpenAIClient."""
        client = AsyncOpenAIClient(mock_async_openai_client)
        assert client.client == mock_async_openai_client

    @pytest.mark.asyncio
    async def test_query_model_basic(self, mock_async_openai_client):
        """Test the query_model method with basic parameters."""
        client = AsyncOpenAIClient(mock_async_openai_client)
        messages = [{"role": "user", "content": "Hello"}]

        result = await client.query_model(
            messages=messages,
            model="gpt-4",
        )

        # Check if AsyncOpenAI client was called with the right parameters
        mock_async_openai_client.chat.completions.create.assert_called_once()
        call_args = mock_async_openai_client.chat.completions.create.call_args[1]
        assert call_args["messages"] == messages
        assert call_args["model"] == "gpt-4"
        assert call_args["max_tokens"] is NOT_GIVEN
        assert call_args["temperature"] is NOT_GIVEN

        # Check return value
        assert result == ["This is a mock async response"]

    @pytest.mark.asyncio
    async def test_query_model_with_all_parameters(self, mock_async_openai_client):
        """Test the query_model method with all possible parameters."""
        client = AsyncOpenAIClient(mock_async_openai_client)
        messages = [{"role": "user", "content": "Hello"}]

        result = await client.query_model(
            messages=messages,
            model="gpt-4",
            max_tokens=100,
            n=1,
            seed=42,
            stop=["stop_token"],
            stream=False,
            temperature=0.7,
            top_p=0.9,
            top_k=5,  # This should raise a warning but still work
            conversation_formatter=MagicMock(),  # This should raise a warning but still work
        )

        # Check if AsyncOpenAI client was called with the right parameters
        mock_async_openai_client.chat.completions.create.assert_called_once()
        call_args = mock_async_openai_client.chat.completions.create.call_args[1]
        assert call_args["messages"] == messages
        assert call_args["model"] == "gpt-4"
        assert call_args["max_tokens"] == 100
        assert call_args["n"] == 1
        assert call_args["seed"] == 42
        assert call_args["stop"] == ["stop_token"]
        assert call_args["stream"] is False
        assert call_args["temperature"] == 0.7
        assert call_args["top_p"] == 0.9

        # Check return value
        assert result == ["This is a mock async response"]

    @pytest.mark.asyncio
    async def test_query_model_with_warnings(self, mock_async_openai_client):
        """Test that warnings are raised appropriately."""
        client = AsyncOpenAIClient(mock_async_openai_client)
        messages = [{"role": "user", "content": "Hello"}]

        with pytest.warns(UserWarning, match="conversation_formatter is not used"):
            await client.query_model(
                messages=messages,
                model="gpt-4",
                conversation_formatter=MagicMock(),
            )

        with pytest.warns(UserWarning, match="top_k is not used"):
            await client.query_model(
                messages=messages,
                model="gpt-4",
                top_k=5,
            )

    @pytest.mark.asyncio
    async def test_query_reward_model(self, mock_async_openai_client):
        """Test the query_reward_model method."""
        client = AsyncOpenAIClient(mock_async_openai_client)
        messages = [
            {"role": "user", "content": "Write a sentence"},
            {"role": "assistant", "content": "This is a sentence"},
        ]

        # Configure mock to return logprobs
        choice = MagicMock()
        logprob_token = MagicMock()
        logprob_token.token = "GOOD"
        logprob_token.logprob = -0.5
        choice.logprobs.content = [logprob_token]
        completion_response = MagicMock()
        completion_response.choices = [choice]
        mock_async_openai_client.chat.completions.create.return_value = (
            completion_response
        )

        result = await client.query_reward_model(
            messages=messages,
            model="reward-model",
        )

        # Check if AsyncOpenAI client was called with the right parameters
        mock_async_openai_client.chat.completions.create.assert_called_once_with(
            messages=messages, model="reward-model"
        )

        # Check return value
        assert result == {"GOOD": -0.5}

    @pytest.mark.asyncio
    async def test_query_reward_model_error(self, mock_async_openai_client):
        """Test error handling when logprobs are not found."""
        client = AsyncOpenAIClient(mock_async_openai_client)
        messages = [
            {"role": "user", "content": "Write a sentence"},
            {"role": "assistant", "content": "This is a sentence"},
        ]

        # Configure mock to return a response without logprobs
        choice = MagicMock()
        choice.logprobs = None
        completion_response = MagicMock()
        completion_response.choices = [choice]
        mock_async_openai_client.chat.completions.create.return_value = (
            completion_response
        )

        with pytest.raises(ValueError, match="Logprobs not found"):
            await client.query_reward_model(
                messages=messages,
                model="not-a-reward-model",
            )
