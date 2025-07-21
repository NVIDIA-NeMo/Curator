"""
Unit tests for ray_curator.stages.services.model_client module.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, patch
from collections.abc import Iterable

from ray_curator.stages.services.model_client import LLMClient, AsyncLLMClient
from ray_curator.stages.services.conversation_formatter import ConversationFormatter


class TestLLMClient:
    """Test cases for the LLMClient abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that LLMClient cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LLMClient()

    def test_abstract_methods_raise_not_implemented_error(self):
        """Test that abstract methods raise NotImplementedError when called."""
        
        class ConcreteLLMClient(LLMClient):
            pass
        
        # Should not be able to instantiate without implementing abstract methods
        with pytest.raises(TypeError):
            ConcreteLLMClient()

    def test_concrete_implementation_works(self):
        """Test that a concrete implementation can be instantiated and used."""
        
        class TestLLMClient(LLMClient):
            def setup(self):
                pass
            
            def query_model(self, *, messages: Iterable, model: str, **kwargs) -> list[str]:
                return ["test response"]
            
            def query_reward_model(self, *, messages: Iterable, model: str, **kwargs) -> dict:
                return {"score": 0.5}
        
        client = TestLLMClient()
        client.setup()
        
        # Test query_model
        result = client.query_model(messages=[{"role": "user", "content": "test"}], model="test-model")
        assert result == ["test response"]
        
        # Test query_reward_model
        result = client.query_reward_model(messages=[{"role": "user", "content": "test"}], model="test-model")
        assert result == {"score": 0.5}


class TestAsyncLLMClient:
    """Test cases for the AsyncLLMClient abstract base class."""

    def test_init_with_defaults(self):
        """Test AsyncLLMClient initialization with default parameters."""
        
        class TestAsyncLLMClient(AsyncLLMClient):
            def setup(self):
                pass
            
            async def _query_model_impl(self, *, messages: Iterable, model: str, **kwargs) -> list[str]:
                return ["test response"]
            
            async def query_reward_model(self, *, messages: Iterable, model: str, **kwargs) -> dict:
                return {"score": 0.5}
        
        client = TestAsyncLLMClient()
        assert client.max_concurrent_requests == 5
        assert client.max_retries == 3
        assert client.base_delay == 1.0
        assert client._semaphore is None
        assert client._semaphore_loop is None

    def test_init_with_custom_parameters(self):
        """Test AsyncLLMClient initialization with custom parameters."""
        
        class TestAsyncLLMClient(AsyncLLMClient):
            def setup(self):
                pass
            
            async def _query_model_impl(self, *, messages: Iterable, model: str, **kwargs) -> list[str]:
                return ["test response"]
            
            async def query_reward_model(self, *, messages: Iterable, model: str, **kwargs) -> dict:
                return {"score": 0.5}
        
        client = TestAsyncLLMClient(
            max_concurrent_requests=10,
            max_retries=5,
            base_delay=2.0
        )
        assert client.max_concurrent_requests == 10
        assert client.max_retries == 5
        assert client.base_delay == 2.0

    def test_cannot_instantiate_abstract_class(self):
        """Test that AsyncLLMClient cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AsyncLLMClient()

    @pytest.mark.asyncio
    async def test_query_model_success(self):
        """Test successful query_model execution."""
        
        class TestAsyncLLMClient(AsyncLLMClient):
            def setup(self):
                pass
            
            async def _query_model_impl(self, *, messages: Iterable, model: str, **kwargs) -> list[str]:
                return ["test response"]
            
            async def query_reward_model(self, *, messages: Iterable, model: str, **kwargs) -> dict:
                return {"score": 0.5}
        
        client = TestAsyncLLMClient()
        result = await client.query_model(
            messages=[{"role": "user", "content": "test"}],
            model="test-model"
        )
        assert result == ["test response"]

    @pytest.mark.asyncio
    async def test_query_model_with_custom_parameters(self):
        """Test query_model with all custom parameters."""
        
        class TestAsyncLLMClient(AsyncLLMClient):
            def setup(self):
                pass
            
            async def _query_model_impl(self, *, messages: Iterable, model: str, **kwargs) -> list[str]:
                # Verify parameters are passed through
                assert kwargs.get("max_tokens") == 1024
                assert kwargs.get("temperature") == 0.5
                assert kwargs.get("seed") == 42
                return ["test response"]
            
            async def query_reward_model(self, *, messages: Iterable, model: str, **kwargs) -> dict:
                return {"score": 0.5}
        
        client = TestAsyncLLMClient()
        result = await client.query_model(
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
            max_tokens=1024,
            temperature=0.5,
            seed=42
        )
        assert result == ["test response"]

    @pytest.mark.asyncio
    async def test_query_model_rate_limit_retry(self):
        """Test query_model retry logic for rate limit errors."""
        
        class TestAsyncLLMClient(AsyncLLMClient):
            def __init__(self):
                super().__init__(max_retries=2, base_delay=0.01)  # Fast test
                self.attempt_count = 0
            
            def setup(self):
                pass
            
            async def _query_model_impl(self, *, messages: Iterable, model: str, **kwargs) -> list[str]:
                self.attempt_count += 1
                if self.attempt_count <= 2:
                    raise Exception("429 Rate limit exceeded")
                return ["success after retry"]
            
            async def query_reward_model(self, *, messages: Iterable, model: str, **kwargs) -> dict:
                return {"score": 0.5}
        
        client = TestAsyncLLMClient()
        
        with patch('builtins.print'):  # Suppress warning prints
            result = await client.query_model(
                messages=[{"role": "user", "content": "test"}],
                model="test-model"
            )
        
        assert result == ["success after retry"]
        assert client.attempt_count == 3  # Should have tried 3 times

    @pytest.mark.asyncio
    async def test_query_model_non_rate_limit_error(self):
        """Test query_model with non-rate-limit errors (should not retry)."""
        
        class TestAsyncLLMClient(AsyncLLMClient):
            def setup(self):
                pass
            
            async def _query_model_impl(self, *, messages: Iterable, model: str, **kwargs) -> list[str]:
                raise ValueError("Some other error")
            
            async def query_reward_model(self, *, messages: Iterable, model: str, **kwargs) -> dict:
                return {"score": 0.5}
        
        client = TestAsyncLLMClient()
        
        with pytest.raises(ValueError, match="Some other error"):
            await client.query_model(
                messages=[{"role": "user", "content": "test"}],
                model="test-model"
            )

    @pytest.mark.asyncio
    async def test_query_model_max_retries_exceeded(self):
        """Test query_model when max retries are exceeded."""
        
        class TestAsyncLLMClient(AsyncLLMClient):
            def __init__(self):
                super().__init__(max_retries=1, base_delay=0.01)  # Fast test
            
            def setup(self):
                pass
            
            async def _query_model_impl(self, *, messages: Iterable, model: str, **kwargs) -> list[str]:
                raise Exception("429 Rate limit exceeded")
            
            async def query_reward_model(self, *, messages: Iterable, model: str, **kwargs) -> dict:
                return {"score": 0.5}
        
        client = TestAsyncLLMClient()
        
        with patch('builtins.print'):  # Suppress warning prints
            with pytest.raises(Exception, match="429 Rate limit exceeded"):
                await client.query_model(
                    messages=[{"role": "user", "content": "test"}],
                    model="test-model"
                )

    @pytest.mark.asyncio
    async def test_semaphore_initialization_and_reuse(self):
        """Test that semaphore is properly initialized and reused."""
        
        class TestAsyncLLMClient(AsyncLLMClient):
            def setup(self):
                pass
            
            async def _query_model_impl(self, *, messages: Iterable, model: str, **kwargs) -> list[str]:
                return ["test response"]
            
            async def query_reward_model(self, *, messages: Iterable, model: str, **kwargs) -> dict:
                return {"score": 0.5}
        
        client = TestAsyncLLMClient(max_concurrent_requests=3)
        
        # First call should initialize semaphore
        await client.query_model(
            messages=[{"role": "user", "content": "test"}],
            model="test-model"
        )
        
        assert client._semaphore is not None
        assert client._semaphore._value == 3  # max_concurrent_requests
        assert client._semaphore_loop is not None
        
        # Store references to verify reuse
        original_semaphore = client._semaphore
        original_loop = client._semaphore_loop
        
        # Second call should reuse semaphore
        await client.query_model(
            messages=[{"role": "user", "content": "test2"}],
            model="test-model"
        )
        
        assert client._semaphore is original_semaphore
        assert client._semaphore_loop is original_loop

    @pytest.mark.asyncio
    async def test_concurrent_requests_limited_by_semaphore(self):
        """Test that concurrent requests are properly limited by semaphore."""
        
        class TestAsyncLLMClient(AsyncLLMClient):
            def __init__(self):
                super().__init__(max_concurrent_requests=2)
                self.active_requests = 0
                self.max_active = 0
            
            def setup(self):
                pass
            
            async def _query_model_impl(self, *, messages: Iterable, model: str, **kwargs) -> list[str]:
                self.active_requests += 1
                self.max_active = max(self.max_active, self.active_requests)
                await asyncio.sleep(0.1)  # Simulate work
                self.active_requests -= 1
                return ["test response"]
            
            async def query_reward_model(self, *, messages: Iterable, model: str, **kwargs) -> dict:
                return {"score": 0.5}
        
        client = TestAsyncLLMClient()
        
        # Start 5 concurrent requests
        tasks = [
            client.query_model(
                messages=[{"role": "user", "content": f"test{i}"}],
                model="test-model"
            )
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(results) == 5
        assert all(r == ["test response"] for r in results)
        
        # But max concurrent should be limited to 2
        assert client.max_active <= 2 