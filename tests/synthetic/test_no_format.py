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

import pytest

from nemo_curator.synthetic.no_format import NoFormat


class TestNoFormat:
    def test_format_conversation_single_turn(self):
        """Test formatting with a single user turn, which is the expected use case."""
        conv = [{"role": "user", "content": "Hello world"}]

        # With NoFormat, the output should be exactly the content of the user's message
        expected_output = "Hello world"

        result = NoFormat().format_conversation(conv)
        assert result == expected_output

    def test_format_conversation_multiple_turns(self):
        """Test error handling when conversation has multiple turns."""
        conv = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        with pytest.raises(ValueError) as excinfo:
            NoFormat().format_conversation(conv)
        assert "There must be exactly one turn" in str(excinfo.value)

    def test_format_conversation_empty(self):
        """Test error handling when conversation is empty."""
        conv = []

        with pytest.raises(ValueError) as excinfo:
            NoFormat().format_conversation(conv)
        assert "There must be exactly one turn" in str(excinfo.value)

    def test_format_conversation_invalid_role(self):
        """Test error handling when the single turn is not from a user."""
        conv = [{"role": "assistant", "content": "Hello, how can I help?"}]

        with pytest.raises(ValueError) as excinfo:
            NoFormat().format_conversation(conv)
        assert "Conversation turn 0 is not 'user'" in str(excinfo.value)

    def test_format_conversation_with_special_characters(self):
        """Test formatting with special characters and formatting in the content."""
        conv = [
            {"role": "user", "content": "# Heading\n* bullet point\n```code block```"}
        ]

        expected_output = "# Heading\n* bullet point\n```code block```"

        result = NoFormat().format_conversation(conv)
        assert result == expected_output
