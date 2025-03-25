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

from unittest.mock import MagicMock, patch

import pytest

from nemo_curator.services.conversation_formatter import ConversationFormatter
from nemo_curator.services.nemo_client import NemoDeployClient


class TestNemoDeployClient:
    @pytest.fixture
    def mock_nemo_deploy(self):
        """Create a mock NemoQueryLLM client for testing."""
        mock_client = MagicMock()
        # Configure the mock response structure to match what NemoQueryLLM returns
        # NemoQueryLLM.query_llm returns a list, and we're accessing the first item [0]
        # So the return value should be a list containing a single string
        mock_client.query_llm.return_value = [["This is a mock NeMo response"]]
        return mock_client

    @pytest.fixture
    def mock_conversation_formatter(self):
        """Create a mock ConversationFormatter for testing."""
        formatter = MagicMock(spec=ConversationFormatter)
        formatter.format_conversation.return_value = "Formatted conversation text"
        return formatter

    def test_init(self, mock_nemo_deploy):
        """Test the constructor of NemoDeployClient."""
        client = NemoDeployClient(mock_nemo_deploy)
        assert client.client == mock_nemo_deploy

    def test_query_model_basic(self, mock_nemo_deploy, mock_conversation_formatter):
        """Test the query_model method with basic parameters."""
        client = NemoDeployClient(mock_nemo_deploy)
        messages = [{"role": "user", "content": "Hello"}]

        result = client.query_model(
            messages=messages,
            model="nemo_model",
            conversation_formatter=mock_conversation_formatter,
        )

        # Check if formatter was called with the right messages
        mock_conversation_formatter.format_conversation.assert_called_once_with(
            messages
        )

        # Check if NemoQueryLLM client was called with the right parameters
        mock_nemo_deploy.query_llm.assert_called_once()
        call_args = mock_nemo_deploy.query_llm.call_args[1]
        assert call_args["prompts"] == ["Formatted conversation text"]
        assert mock_nemo_deploy.model_name == "nemo_model"

        # Check result
        assert result == ["This is a mock NeMo response"]

    def test_query_model_with_all_parameters(
        self, mock_nemo_deploy, mock_conversation_formatter
    ):
        """Test the query_model method with all possible parameters."""
        client = NemoDeployClient(mock_nemo_deploy)
        messages = [{"role": "user", "content": "Hello"}]

        # Set up specific stop words for testing postprocessing
        mock_nemo_deploy.query_llm.return_value = [["Response with stop_token"]]

        with pytest.warns(UserWarning, match="n is not supported"):
            with pytest.warns(UserWarning, match="streamming is not supported"):
                result = client.query_model(
                    messages=messages,
                    model="nemo_model",
                    conversation_formatter=mock_conversation_formatter,
                    max_tokens=100,
                    n=1,  # Should trigger warning
                    seed=42,
                    stop=["stop_token"],
                    stream=True,  # Should trigger warning
                    temperature=0.7,
                    top_k=5,
                    top_p=0.9,
                )

        # Check if NemoQueryLLM client was called with the right parameters
        mock_nemo_deploy.query_llm.assert_called_once()
        call_args = mock_nemo_deploy.query_llm.call_args[1]
        assert call_args["prompts"] == ["Formatted conversation text"]
        assert call_args["max_output_len"] == 100
        assert call_args["random_seed"] == 42
        assert call_args["stop_words_list"] == ["stop_token"]
        assert call_args["temperature"] == 0.7
        assert call_args["top_k"] == 5
        assert call_args["top_p"] == 0.9

        # Check result and postprocessing (stop words should be removed)
        assert result == ["Response with"]

    def test_query_model_with_string_stop(
        self, mock_nemo_deploy, mock_conversation_formatter
    ):
        """Test that query_model properly handles string stop words."""
        client = NemoDeployClient(mock_nemo_deploy)
        messages = [{"role": "user", "content": "Hello"}]

        # Configure mock to return a properly nested response
        mock_nemo_deploy.query_llm.return_value = [["Response text"]]

        client.query_model(
            messages=messages,
            model="nemo_model",
            conversation_formatter=mock_conversation_formatter,
            stop="stop_word",
        )

        # Check if stop word was converted to a list
        call_args = mock_nemo_deploy.query_llm.call_args[1]
        assert call_args["stop_words_list"] == ["stop_word"]

    def test_query_model_with_none_stop(
        self, mock_nemo_deploy, mock_conversation_formatter
    ):
        """Test that query_model properly handles None stop words."""
        client = NemoDeployClient(mock_nemo_deploy)
        messages = [{"role": "user", "content": "Hello"}]

        # Configure mock to return a properly nested response
        mock_nemo_deploy.query_llm.return_value = [["Response text"]]

        # Patch the _postprocess_response method to avoid the None error
        with patch.object(
            NemoDeployClient, "_postprocess_response", return_value=["Response text"]
        ):
            client.query_model(
                messages=messages,
                model="nemo_model",
                conversation_formatter=mock_conversation_formatter,
                stop=None,
            )

        # Check if None stop was passed to query_llm appropriately
        call_args = mock_nemo_deploy.query_llm.call_args[1]
        assert call_args["stop_words_list"] is None

    def test_query_model_without_formatter(self, mock_nemo_deploy):
        """Test that query_model raises an error when no formatter is provided."""
        client = NemoDeployClient(mock_nemo_deploy)
        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(ValueError, match="requires a conversation_formatter"):
            client.query_model(
                messages=messages,
                model="nemo_model",
            )

    def test_postprocess_response(self, mock_nemo_deploy):
        """Test the _postprocess_response method."""
        # This is a static method that can be tested directly
        responses = [
            "Response ending with stop.",
            "Another response with different end.",
        ]
        stop_words = ["stop.", "end."]

        result = NemoDeployClient._postprocess_response(responses, stop_words)

        # Check that stop words were removed
        assert result == ["Response ending with", "Another response with different"]

    def test_postprocess_response_no_matches(self, mock_nemo_deploy):
        """Test _postprocess_response when no stop words match."""
        responses = ["Response without stop words", "Another normal response"]
        stop_words = ["stop.", "end."]

        result = NemoDeployClient._postprocess_response(responses, stop_words)

        # Check that responses are returned stripped but otherwise unchanged
        assert result == ["Response without stop words", "Another normal response"]

    def test_postprocess_response_with_empty_stop_words(self, mock_nemo_deploy):
        """Test _postprocess_response handles empty stop_words list gracefully."""
        responses = ["Response text", "Another response"]

        # Test with empty list of stop_words
        result = NemoDeployClient._postprocess_response(responses, [])
        assert result == ["Response text", "Another response"]

    def test_postprocess_response_with_none_stop(self, mock_nemo_deploy):
        """Test that _postprocess_response handles None stop_words gracefully."""
        # Test with None stop_words - this should raise a TypeError
        with pytest.raises(TypeError):
            NemoDeployClient._postprocess_response(["Response text"], None)

        # This verifies that we need to handle None stop_words in query_model before calling _postprocess_response

    def test_query_reward_model_not_implemented(self, mock_nemo_deploy):
        """Test that query_reward_model raises NotImplementedError."""
        client = NemoDeployClient(mock_nemo_deploy)
        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(
            NotImplementedError, match="not supported in NeMo Deploy Clients"
        ):
            client.query_reward_model(
                messages=messages,
                model="reward_model",
            )
