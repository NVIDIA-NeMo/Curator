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

from typing import Any, List, Union
from unittest.mock import MagicMock, patch

import pytest

from nemo_curator.synthetic.generator import SyntheticDataGenerator


class SimpleSyntheticGenerator(SyntheticDataGenerator):
    """
    A simple implementation of SyntheticDataGenerator for testing purposes.
    """

    def __init__(self, client=None):
        super().__init__()
        self.client = client

    def generate(self, llm_prompt: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Generate a response using the provided prompt.

        Args:
            llm_prompt: A prompt or list of prompts to send to the LLM.

        Returns:
            The LLM's response(s).
        """
        if self.client is None:
            # If no client is provided, just echo the prompt as a mock response
            return (
                f"Response to: {llm_prompt}"
                if isinstance(llm_prompt, str)
                else [f"Response to: {p}" for p in llm_prompt]
            )

        # Use the client to generate a response
        if isinstance(llm_prompt, str):
            return self.client.query_model(llm_prompt)
        return [self.client.query_model(p) for p in llm_prompt]

    def parse_response(self, llm_response: Union[str, List[str]]) -> Any:
        """
        Parse the response from the LLM.

        Args:
            llm_response: The response(s) from the LLM.

        Returns:
            The parsed response.
        """
        if isinstance(llm_response, str):
            # Simply remove the "Response to: " prefix if it exists
            return llm_response.replace("Response to: ", "")
        return [resp.replace("Response to: ", "") for resp in llm_response]


class IncompleteGenerator(SyntheticDataGenerator):
    """
    An incomplete implementation that doesn't implement the abstract methods.
    This class is used to test that the abstract methods are enforced.
    """

    pass


class TestSyntheticDataGenerator:
    def test_init(self):
        """Test the constructor of SyntheticDataGenerator."""
        generator = SimpleSyntheticGenerator()
        assert generator._name == "SimpleSyntheticGenerator"

    def test_abstract_methods_required(self):
        """Test that concrete classes must implement abstract methods."""
        with pytest.raises(TypeError) as excinfo:
            IncompleteGenerator()
        assert "Can't instantiate abstract class" in str(excinfo.value)
        assert "generate" in str(excinfo.value)
        assert "parse_response" in str(excinfo.value)

    def test_generate_string_input(self):
        """Test generate method with a string input."""
        generator = SimpleSyntheticGenerator()
        prompt = "Hello, world!"
        result = generator.generate(prompt)
        assert result == f"Response to: {prompt}"

    def test_generate_list_input(self):
        """Test generate method with a list input."""
        generator = SimpleSyntheticGenerator()
        prompts = ["Hello", "World"]
        result = generator.generate(prompts)
        assert result == [f"Response to: {p}" for p in prompts]

    def test_parse_response_string_input(self):
        """Test parse_response method with a string input."""
        generator = SimpleSyntheticGenerator()
        response = "Response to: Hello, world!"
        result = generator.parse_response(response)
        assert result == "Hello, world!"

    def test_parse_response_list_input(self):
        """Test parse_response method with a list input."""
        generator = SimpleSyntheticGenerator()
        responses = ["Response to: Hello", "Response to: World"]
        result = generator.parse_response(responses)
        assert result == ["Hello", "World"]

    def test_with_mock_client(self):
        """Test with a mock client instead of the default echo behavior."""
        mock_client = MagicMock()
        mock_client.query_model.return_value = "Client response"

        generator = SimpleSyntheticGenerator(client=mock_client)
        result = generator.generate("Test prompt")

        # Check if client was called correctly
        mock_client.query_model.assert_called_once_with("Test prompt")
        assert result == "Client response"

    def test_generate_and_parse(self):
        """Test the full pipeline of generate and parse."""
        generator = SimpleSyntheticGenerator()
        prompt = "Test prompt"

        # Generate response
        response = generator.generate(prompt)
        assert response == f"Response to: {prompt}"

        # Parse the response
        parsed = generator.parse_response(response)
        assert parsed == prompt

        # This should return the original prompt demonstrating a full round trip
        assert parsed == prompt
