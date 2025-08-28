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

from unittest.mock import Mock, patch

import pytest
import torch

from nemo_curator.models.prompt_formatter import VARIANT_MAPPING, PromptFormatter


class TestPromptFormatter:
    """Test cases for PromptFormatter class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        with patch("nemo_curator.models.prompt_formatter.AutoProcessor") as mock_processor:
            mock_processor_instance = Mock()
            mock_processor.from_pretrained.return_value = mock_processor_instance
            self.formatter = PromptFormatter(prompt_variant="qwen")
            self.mock_processor = mock_processor_instance

    def test_variant_mapping_constants(self) -> None:
        """Test that variant mapping constants are correctly defined."""
        assert "qwen" in VARIANT_MAPPING
        assert VARIANT_MAPPING["qwen"] == "Qwen/Qwen2.5-VL-7B-Instruct"

    def test_initialization_valid_variant(self) -> None:
        """Test initialization with valid prompt variant."""
        with patch("nemo_curator.models.prompt_formatter.AutoProcessor") as mock_processor:
            mock_processor_instance = Mock()
            mock_processor.from_pretrained.return_value = mock_processor_instance

            formatter = PromptFormatter(prompt_variant="qwen")

            assert formatter.prompt_variant == "qwen"
            assert formatter.text_prompt is None
            assert formatter.processor == mock_processor_instance
            mock_processor.from_pretrained.assert_called_once_with(VARIANT_MAPPING["qwen"])

    def test_initialization_invalid_variant(self) -> None:
        """Test initialization with invalid prompt variant raises ValueError."""
        with pytest.raises(ValueError, match="Invalid prompt variant: invalid_variant"):
            PromptFormatter(prompt_variant="invalid_variant")

    @patch("nemo_curator.models.prompt_formatter.AutoProcessor")
    def test_generate_inputs_first_time(self, mock_processor_class: Mock) -> None:
        """Test generate_inputs method when text_prompt is None (first time)."""
        # Setup mock processor
        mock_processor_instance = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor_instance
        mock_processor_instance.apply_chat_template.return_value = "formatted_prompt"

        formatter = PromptFormatter(prompt_variant="qwen")

        # Create mock video tensor
        video_tensor = torch.randn(1, 3, 224, 224)

        result = formatter.generate_inputs(prompt="Test prompt", video_inputs=video_tensor)

        # Verify the result structure
        assert isinstance(result, dict)
        assert "prompt" in result
        assert "multi_modal_data" in result
        assert result["prompt"] == "formatted_prompt"
        assert result["multi_modal_data"]["video"] is video_tensor

        # Verify processor was called correctly
        expected_message = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "Test prompt"}]}]
        mock_processor_instance.apply_chat_template.assert_called_once_with(
            expected_message, tokenizer=False, add_generation_prompt=True
        )

        # Verify text_prompt was cached
        assert formatter.text_prompt == "formatted_prompt"

    @patch("nemo_curator.models.prompt_formatter.AutoProcessor")
    def test_generate_inputs_cached_prompt(self, mock_processor_class: Mock) -> None:
        """Test generate_inputs method when text_prompt is already cached."""
        # Setup mock processor
        mock_processor_instance = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor_instance

        formatter = PromptFormatter(prompt_variant="qwen")
        formatter.text_prompt = "cached_prompt"  # Pre-set cached prompt

        video_tensor = torch.randn(1, 3, 224, 224)

        result = formatter.generate_inputs(prompt="Test prompt", video_inputs=video_tensor)

        # Verify cached prompt is used
        assert result["prompt"] == "cached_prompt"
        assert result["multi_modal_data"]["video"] is video_tensor

        # Verify processor was NOT called since prompt is cached
        mock_processor_instance.apply_chat_template.assert_not_called()

    @patch("nemo_curator.models.prompt_formatter.AutoProcessor")
    def test_generate_inputs_override_text_prompt(self, mock_processor_class: Mock) -> None:
        """Test generate_inputs method with override_text_prompt=True."""
        # Setup mock processor
        mock_processor_instance = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor_instance
        mock_processor_instance.apply_chat_template.return_value = "new_formatted_prompt"

        formatter = PromptFormatter(prompt_variant="qwen")
        formatter.text_prompt = "old_cached_prompt"  # Pre-set cached prompt

        video_tensor = torch.randn(1, 3, 224, 224)

        result = formatter.generate_inputs(prompt="Test prompt", video_inputs=video_tensor, override_text_prompt=True)

        # Verify new prompt is generated and cached
        assert result["prompt"] == "new_formatted_prompt"
        assert formatter.text_prompt == "new_formatted_prompt"

        # Verify processor was called even though prompt was cached
        mock_processor_instance.apply_chat_template.assert_called_once()

    @patch("nemo_curator.models.prompt_formatter.AutoProcessor")
    def test_generate_inputs_no_video_inputs(self, mock_processor_class: Mock) -> None:
        """Test generate_inputs method with no video inputs."""
        mock_processor_instance = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor_instance
        mock_processor_instance.apply_chat_template.return_value = "formatted_prompt"

        formatter = PromptFormatter(prompt_variant="qwen")

        result = formatter.generate_inputs(prompt="Test prompt")

        assert result["prompt"] == "formatted_prompt"
        assert result["multi_modal_data"]["video"] is None

    def test_create_message(self) -> None:
        """Test create_message method creates correct message structure."""
        result = self.formatter.create_message("Test prompt text")

        expected_message = [
            {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "Test prompt text"}]}
        ]

        assert result == expected_message
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0]["type"] == "video"
        assert result[0]["content"][1]["type"] == "text"
        assert result[0]["content"][1]["text"] == "Test prompt text"

    def test_create_message_empty_prompt(self) -> None:
        """Test create_message method with empty prompt."""
        result = self.formatter.create_message("")

        assert result[0]["content"][1]["text"] == ""

    def test_create_message_special_characters(self) -> None:
        """Test create_message method with special characters in prompt."""
        special_prompt = "Test with 特殊字符 and émojis 🎉"
        result = self.formatter.create_message(special_prompt)

        assert result[0]["content"][1]["text"] == special_prompt

    @patch("nemo_curator.models.prompt_formatter.AutoProcessor")
    def test_processor_interaction(self, mock_processor_class: Mock) -> None:
        """Test that the processor is correctly initialized and used."""
        mock_processor_instance = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor_instance

        # Create formatter and verify processor initialization
        formatter = PromptFormatter(prompt_variant="qwen")
        mock_processor_class.from_pretrained.assert_called_once_with(VARIANT_MAPPING["qwen"])

        # Test processor method call
        mock_processor_instance.apply_chat_template.return_value = "test_output"
        formatter.generate_inputs("test prompt")

        # Verify apply_chat_template was called with correct parameters
        mock_processor_instance.apply_chat_template.assert_called_once_with(
            formatter.create_message("test prompt"), tokenizer=False, add_generation_prompt=True
        )
