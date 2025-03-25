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

from nemo_curator.synthetic.mixtral import Mixtral8x7BFormatter


class TestMixtral8x7BFormatter:
    def test_format_conversation_basic(self):
        """Test basic conversation formatting with alternating user and assistant turns."""
        conv = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
        ]

        expected_output = (
            "<s> [INST] \n"
            "Hello [/INST]"
            "Hi there!</s>[INST] "
            "How are you? [/INST]"
            "I'm doing well, thank you!</s>[INST] "
        )

        result = Mixtral8x7BFormatter.format_conversation(conv)
        assert result == expected_output

    def test_format_conversation_single_turn(self):
        """Test formatting with just a single user turn."""
        conv = [{"role": "user", "content": "Hello"}]

        expected_output = "<s> [INST] \nHello [/INST]"

        result = Mixtral8x7BFormatter.format_conversation(conv)
        assert result == expected_output

    def test_format_conversation_invalid_roles(self):
        """Test error handling when conversation has invalid role sequence."""
        # User followed by user
        conv = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "Are you there?"},
        ]

        with pytest.raises(ValueError) as excinfo:
            Mixtral8x7BFormatter.format_conversation(conv)
        assert "Conversation turn 1 is not 'assistant'" in str(excinfo.value)

        # Assistant as first turn
        conv = [
            {"role": "assistant", "content": "Hello, how can I help?"},
            {"role": "user", "content": "Hi there!"},
        ]

        with pytest.raises(ValueError) as excinfo:
            Mixtral8x7BFormatter.format_conversation(conv)
        assert "Conversation turn 0 is not 'user'" in str(excinfo.value)

    def test_format_conversation_multiple_turns(self):
        """Test formatting with multiple turns between user and assistant."""
        conv = [
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is a subset of AI..."},
            {"role": "user", "content": "Can you give examples?"},
            {"role": "assistant", "content": "Sure! Examples include..."},
            {"role": "user", "content": "Thanks for explaining!"},
        ]

        expected_output = (
            "<s> [INST] \n"
            "What is machine learning? [/INST]"
            "Machine learning is a subset of AI...</s>[INST] "
            "Can you give examples? [/INST]"
            "Sure! Examples include...</s>[INST] "
            "Thanks for explaining! [/INST]"
        )

        result = Mixtral8x7BFormatter.format_conversation(conv)
        assert result == expected_output
