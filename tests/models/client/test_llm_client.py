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
Unit tests for ray_curator.stages.services.model_client module.
"""

import asyncio
from collections.abc import Iterable
from unittest.mock import patch

import pytest

from nemo_curator.models.client.llm_client import AsyncLLMClient, GenerationConfig, LLMClient


class TestLLMClient:
    """Test cases for the LLMClient abstract base class."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        """Test that LLMClient cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LLMClient()

    def test_abstract_methods_raise_not_implemented_error(self) -> None:
        """Test that abstract methods raise NotImplementedError when called."""

        class ConcreteLLMClient(LLMClient):
            pass

        # Should not be able to instantiate without implementing abstract methods
        with pytest.raises(TypeError):
            ConcreteLLMClient()

    def test_concrete_implementation_works(self) -> None:
        """Test that a concrete implementation can be instantiated and used."""

        class TestLLMClient(LLMClient):
            def setup(self) -> None:
                pass

            def query_model(self, *, messages: Iterable, model: str, **kwargs: object) -> list[str]:
                return ["test response"]

        client = TestLLMClient()
        client.setup()

        # Test query_model
        result = client.query_model(messages=[{"role": "user", "content": "test"}], model="test-model")
        assert result == ["test response"]


class TestAsyncLLMClient:
    """Test cases for the AsyncLLMClient abstract base class."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        """Test that AsyncLLMClient cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AsyncLLMClient()

    @pytest.mark.asyncio
    async def test_query_model_success(self) -> None:
        """Test successful query_model execution."""

        class TestAsyncLLMClient(AsyncLLMClient):
            def setup(self) -> None:
                pass

            async def _query_model_impl(self, *, messages: Iterable, model: str, **kwargs: object) -> list[str]:
                return ["test response"]

        client = TestAsyncLLMClient()
        result = await client.query_model(messages=[{"role": "user", "content": "test"}], model="test-model")
        assert result == ["test response"]

    @pytest.mark.asyncio
    async def test_query_model_dict_generation_config(self) -> None:
        """Test that generation_config dict is properly converted to GenerationConfig."""

        class TestAsyncLLMClient(AsyncLLMClient):
            def setup(self) -> None:
                pass

            async def _query_model_impl(
                self,
                *,
                messages: Iterable,
                model: str,
                generation_config: GenerationConfig | None = None,
                **kwargs: object,
            ) -> list[str]:
                # Verify that generation_config was converted from dict to GenerationConfig
                assert isinstance(generation_config, GenerationConfig)
                assert generation_config.max_tokens == 512
                assert generation_config.temperature == 0.7
                assert generation_config.top_p == 0.9
                return ["test response"]

        client = TestAsyncLLMClient()

        # Pass generation_config as a dictionary
        config_dict = {"max_tokens": 512, "temperature": 0.7, "top_p": 0.9}
        result = await client.query_model(
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
            generation_config=config_dict,
        )
        assert result == ["test response"]

    @pytest.mark.asyncio
    async def test_query_model_rate_limit_retry(self) -> None:
        """Test query_model retry logic for rate limit errors."""

        class RateLimitError(Exception):
            """Custom exception for rate limit errors."""

        class TestAsyncLLMClient(AsyncLLMClient):
            def __init__(self) -> None:
                super().__init__(max_retries=2, base_delay=0.01)  # Fast test
                self.attempt_count = 0

            def setup(self) -> None:
                pass

            async def _query_model_impl(self, *, messages: Iterable, model: str, **kwargs: object) -> list[str]:
                self.attempt_count += 1
                if self.attempt_count <= 2:
                    error_msg = "429 Rate limit exceeded"
                    raise RateLimitError(error_msg)
                return ["success after retry"]

        client = TestAsyncLLMClient()

        with patch("nemo_curator.models.client.llm_client.logger"):  # Suppress warning logs
            result = await client.query_model(messages=[{"role": "user", "content": "test"}], model="test-model")

        assert result == ["success after retry"]
        assert client.attempt_count == 3  # Should have tried 3 times

    @pytest.mark.asyncio
    async def test_query_model_non_rate_limit_error(self) -> None:
        """Test query_model with non-rate-limit errors (should not retry)."""

        class TestAsyncLLMClient(AsyncLLMClient):
            def setup(self) -> None:
                pass

            async def _query_model_impl(self, *, messages: Iterable, model: str, **kwargs: object) -> list[str]:
                error_msg = "Some other error"
                raise ValueError(error_msg)

        client = TestAsyncLLMClient()

        with pytest.raises(ValueError, match="Some other error"):
            await client.query_model(messages=[{"role": "user", "content": "test"}], model="test-model")

    @pytest.mark.asyncio
    async def test_query_model_max_retries_exceeded(self) -> None:
        """Test query_model when max retries are exceeded."""

        class RateLimitError(Exception):
            """Custom exception for rate limit errors."""

        class TestAsyncLLMClient(AsyncLLMClient):
            def __init__(self) -> None:
                super().__init__(max_retries=1, base_delay=0.01)  # Fast test

            def setup(self) -> None:
                pass

            async def _query_model_impl(self, *, messages: Iterable, model: str, **kwargs: object) -> list[str]:
                error_msg = "429 Rate limit exceeded"
                raise RateLimitError(error_msg)

        client = TestAsyncLLMClient()

        with (
            patch("nemo_curator.models.client.llm_client.logger"),
            pytest.raises(RateLimitError, match="429 Rate limit exceeded"),
        ):
            await client.query_model(messages=[{"role": "user", "content": "test"}], model="test-model")

    @pytest.mark.asyncio
    async def test_semaphore_initialization_and_reuse(self) -> None:
        """Test that semaphore is properly initialized and reused."""

        class TestAsyncLLMClient(AsyncLLMClient):
            def setup(self) -> None:
                pass

            async def _query_model_impl(self, *, messages: Iterable, model: str, **kwargs: object) -> list[str]:
                return ["test response"]

        client = TestAsyncLLMClient(max_concurrent_requests=3)

        # First call should initialize semaphore
        await client.query_model(messages=[{"role": "user", "content": "test"}], model="test-model")

        assert client._semaphore is not None
        assert client._semaphore._value == 3  # max_concurrent_requests
        assert client._semaphore_loop is not None

        # Store references to verify reuse
        original_semaphore = client._semaphore
        original_loop = client._semaphore_loop

        # Second call should reuse semaphore
        await client.query_model(messages=[{"role": "user", "content": "test2"}], model="test-model")

        assert client._semaphore is original_semaphore
        assert client._semaphore_loop is original_loop

    @pytest.mark.asyncio
    async def test_concurrent_requests_limited_by_semaphore(self) -> None:
        """Test that concurrent requests are properly limited by semaphore."""

        class TestAsyncLLMClient(AsyncLLMClient):
            def __init__(self) -> None:
                super().__init__(max_concurrent_requests=2)
                self.active_requests = 0
                self.max_active = 0

            def setup(self) -> None:
                pass

            async def _query_model_impl(self, *, messages: Iterable, model: str, **kwargs: object) -> list[str]:
                self.active_requests += 1
                self.max_active = max(self.max_active, self.active_requests)
                await asyncio.sleep(0.1)  # Simulate work
                self.active_requests -= 1
                return ["test response"]

        client = TestAsyncLLMClient()

        # Start 5 concurrent requests
        tasks = [
            client.query_model(messages=[{"role": "user", "content": f"test{i}"}], model="test-model")
            for i in range(5)
        ]

        results = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results) == 5
        assert all(r == ["test response"] for r in results)

        # But max concurrent should be limited to 2
        assert client.max_active <= 2
