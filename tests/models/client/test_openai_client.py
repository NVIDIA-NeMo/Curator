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
# See the specific language for the License for the specific language governing permissions and
# limitations under the License.

"""
Unit tests for ray_curator.stages.services.openai_client module.
"""

import asyncio
import warnings
from unittest.mock import AsyncMock, Mock, patch

import pytest

from nemo_curator.models.client.llm_client import ConversationFormatter, GenerationConfig
from nemo_curator.models.client.openai_client import AsyncOpenAIClient, OpenAIClient


class TestOpenAIClient:
    """Test cases for the OpenAIClient class."""

    @patch("nemo_curator.models.client.openai_client.OpenAI")
    def test_setup(self, mock_openai: Mock) -> None:
        """Test setup method creates OpenAI client with kwargs."""
        mock_client = Mock()
        mock_openai.return_value = mock_client

        client = OpenAIClient(api_key="test-key")
        client.setup()

        mock_openai.assert_called_once_with(api_key="test-key")
        assert client.client == mock_client

    @patch("nemo_curator.models.client.openai_client.OpenAI")
    def test_query_model_success(self, mock_openai: Mock) -> None:
        """Test successful query_model execution."""
        # Setup mock response
        mock_choice = Mock()
        mock_choice.message.content = "Test response content"
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        client = OpenAIClient()
        client.setup()

        config = GenerationConfig(max_tokens=1024, temperature=0.7)
        result = client.query_model(
            messages=[{"role": "user", "content": "test"}], model="gpt-4", generation_config=config
        )

        assert result == ["Test response content"]
        mock_client.chat.completions.create.assert_called_once_with(
            messages=[{"role": "user", "content": "test"}],
            model="gpt-4",
            max_tokens=1024,
            n=1,
            seed=0,
            stop=None,
            stream=False,
            temperature=0.7,
            top_p=0.95,
            timeout=120,
        )

    @patch("nemo_curator.models.client.openai_client.OpenAI")
    def test_query_model_multiple_choices(self, mock_openai: Mock) -> None:
        """Test query_model with multiple response choices."""
        # Setup mock response with multiple choices
        mock_choice1 = Mock()
        mock_choice1.message.content = "Response 1"
        mock_choice2 = Mock()
        mock_choice2.message.content = "Response 2"
        mock_response = Mock()
        mock_response.choices = [mock_choice1, mock_choice2]

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        client = OpenAIClient()
        client.setup()

        config = GenerationConfig(n=2)
        result = client.query_model(
            messages=[{"role": "user", "content": "test"}], model="gpt-4", generation_config=config
        )

        assert result == ["Response 1", "Response 2"]

    @patch("nemo_curator.models.client.openai_client.OpenAI")
    def test_query_model_with_conversation_formatter_warning(self, mock_openai: Mock) -> None:
        """Test query_model warns when conversation_formatter is provided."""
        # Setup proper mock response
        mock_choice = Mock()
        mock_choice.message.content = "Test response"
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        client = OpenAIClient()
        client.setup()

        formatter = Mock(spec=ConversationFormatter)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            client.query_model(
                messages=[{"role": "user", "content": "test"}], model="gpt-4", conversation_formatter=formatter
            )

            assert len(w) == 1
            assert "conversation_formatter is not used in an OpenAIClient" in str(w[0].message)

    @patch("nemo_curator.models.client.openai_client.OpenAI")
    def test_query_model_with_top_k_warning(self, mock_openai: Mock) -> None:
        """Test query_model warns when top_k is provided."""
        # Setup proper mock response
        mock_choice = Mock()
        mock_choice.message.content = "Test response"
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        client = OpenAIClient()
        client.setup()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = GenerationConfig(top_k=10)
            client.query_model(messages=[{"role": "user", "content": "test"}], model="gpt-4", generation_config=config)

            assert len(w) == 1
            assert "top_k is not used in an OpenAIClient" in str(w[0].message)


class TestAsyncOpenAIClient:
    """Test cases for the AsyncOpenAIClient class."""

    @patch("nemo_curator.models.client.openai_client.AsyncOpenAI")
    def test_setup(self, mock_async_openai: Mock) -> None:
        """Test setup method creates AsyncOpenAI client with kwargs."""
        mock_client = AsyncMock()
        mock_async_openai.return_value = mock_client

        client = AsyncOpenAIClient(api_key="test-key")
        client.setup()

        mock_async_openai.assert_called_once_with(api_key="test-key")
        assert client.client == mock_client

    @pytest.mark.asyncio
    @patch("nemo_curator.models.client.openai_client.AsyncOpenAI")
    async def test_query_model_impl_success(self, mock_async_openai: Mock) -> None:
        """Test successful _query_model_impl execution."""
        # Setup mock response
        mock_choice = Mock()
        mock_choice.message.content = "Test response content"
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_async_openai.return_value = mock_client

        client = AsyncOpenAIClient()
        client.setup()

        config = GenerationConfig(max_tokens=1024, temperature=0.7)
        result = await client._query_model_impl(
            messages=[{"role": "user", "content": "test"}], model="gpt-4", generation_config=config
        )

        assert result == ["Test response content"]
        mock_client.chat.completions.create.assert_called_once_with(
            messages=[{"role": "user", "content": "test"}],
            model="gpt-4",
            max_tokens=1024,  # Match the value passed in the test call
            n=1,
            seed=0,
            stop=None,
            stream=False,
            temperature=0.7,
            top_p=0.95,
            timeout=120,
        )

    @pytest.mark.asyncio
    @patch("nemo_curator.models.client.openai_client.AsyncOpenAI")
    async def test_query_model_impl_multiple_choices(self, mock_async_openai: Mock) -> None:
        """Test _query_model_impl with multiple response choices."""
        # Setup mock response with multiple choices
        mock_choice1 = Mock()
        mock_choice1.message.content = "Response 1"
        mock_choice2 = Mock()
        mock_choice2.message.content = "Response 2"
        mock_response = Mock()
        mock_response.choices = [mock_choice1, mock_choice2]

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_async_openai.return_value = mock_client

        client = AsyncOpenAIClient()
        client.setup()

        config = GenerationConfig(n=2)
        result = await client._query_model_impl(
            messages=[{"role": "user", "content": "test"}], model="gpt-4", generation_config=config
        )

        assert result == ["Response 1", "Response 2"]

    @pytest.mark.asyncio
    @patch("nemo_curator.models.client.openai_client.AsyncOpenAI")
    async def test_query_model_impl_with_conversation_formatter_warning(self, mock_async_openai: Mock) -> None:
        """Test _query_model_impl warns when conversation_formatter is provided."""
        # Setup proper mock response
        mock_choice = Mock()
        mock_choice.message.content = "Test response"
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_async_openai.return_value = mock_client

        client = AsyncOpenAIClient()
        client.setup()

        formatter = Mock(spec=ConversationFormatter)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            await client._query_model_impl(
                messages=[{"role": "user", "content": "test"}], model="gpt-4", conversation_formatter=formatter
            )

            assert len(w) == 1
            assert "conversation_formatter is not used in an AsyncOpenAIClient" in str(w[0].message)

    @pytest.mark.asyncio
    @patch("nemo_curator.models.client.openai_client.AsyncOpenAI")
    async def test_query_model_impl_with_top_k_warning(self, mock_async_openai: Mock) -> None:
        """Test _query_model_impl warns when top_k is provided."""
        # Setup proper mock response
        mock_choice = Mock()
        mock_choice.message.content = "Test response"
        mock_response = Mock()
        mock_response.choices = [mock_choice]

        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_async_openai.return_value = mock_client

        client = AsyncOpenAIClient()
        client.setup()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = GenerationConfig(top_k=10)
            await client._query_model_impl(
                messages=[{"role": "user", "content": "test"}], model="gpt-4", generation_config=config
            )

            assert len(w) == 1
            assert "top_k is not used in an AsyncOpenAIClient" in str(w[0].message)

    @pytest.mark.asyncio
    @patch("nemo_curator.models.client.openai_client.AsyncOpenAI")
    async def test_integration_with_parent_retry_logic(self, mock_async_openai: Mock) -> None:
        """Test that AsyncOpenAIClient works with parent class retry logic."""
        # Setup mock to fail first two times, succeed on third
        call_count = 0

        class RateLimitError(Exception):
            """Custom exception for rate limit errors."""

        def side_effect(*_args: object, **_kwargs: object) -> Mock:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                error_msg = "429 Rate limit exceeded"
                raise RateLimitError(error_msg)

            mock_choice = Mock()
            mock_choice.message.content = "Success after retry"
            mock_response = Mock()
            mock_response.choices = [mock_choice]
            return mock_response

        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = side_effect
        mock_async_openai.return_value = mock_client

        client = AsyncOpenAIClient(max_retries=3, base_delay=0.01)  # Fast test
        client.setup()

        with patch("nemo_curator.models.client.llm_client.logger"):  # Suppress warning logs
            result = await client.query_model(messages=[{"role": "user", "content": "test"}], model="gpt-4")

        assert result == ["Success after retry"]
        assert call_count == 3  # Should have tried 3 times

    @pytest.mark.asyncio
    @patch("nemo_curator.models.client.openai_client.AsyncOpenAI")
    async def test_concurrent_request_limiting(self, mock_async_openai: Mock) -> None:
        """Test that concurrent requests are properly limited."""
        active_requests = 0
        max_active = 0

        async def mock_create(*_args: object, **_kwargs: object) -> Mock:
            nonlocal active_requests, max_active
            active_requests += 1
            max_active = max(max_active, active_requests)
            await asyncio.sleep(0.1)  # Simulate API call delay
            active_requests -= 1

            mock_choice = Mock()
            mock_choice.message.content = "Test response"
            mock_response = Mock()
            mock_response.choices = [mock_choice]
            return mock_response

        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = mock_create
        mock_async_openai.return_value = mock_client

        client = AsyncOpenAIClient(max_concurrent_requests=2)
        client.setup()

        # Start 5 concurrent requests
        tasks = [
            client.query_model(messages=[{"role": "user", "content": f"test{i}"}], model="gpt-4") for i in range(5)
        ]

        results = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results) == 5
        assert all(r == ["Test response"] for r in results)

        # But max concurrent should be limited to 2
        assert max_active <= 2
