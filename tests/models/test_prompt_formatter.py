# modality: video

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

import numpy as np
import pytest
import torch

from nemo_curator.models.prompt_formatter import VARIANT_MAPPING, PromptFormatter


class TestPromptFormatterVariantMapping:
    """Test cases for variant mapping constants."""

    def test_variant_mapping_contains_all_variants(self) -> None:
        """Test that all expected variants are in mapping."""
        expected_variants = {"qwen", "nemotron", "nemotron-bf16", "nemotron-fp8", "nemotron-nvfp4"}
        assert set(VARIANT_MAPPING.keys()) == expected_variants

    def test_variant_mapping_qwen_hf_id(self) -> None:
        """Test that Qwen variant has correct HuggingFace ID."""
        assert VARIANT_MAPPING["qwen"] == "Qwen/Qwen2.5-VL-7B-Instruct"

    def test_variant_mapping_nemotron_hf_ids(self) -> None:
        """Test that Nemotron variants have correct HuggingFace IDs."""
        assert VARIANT_MAPPING["nemotron"] == "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
        assert VARIANT_MAPPING["nemotron-bf16"] == "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
        assert VARIANT_MAPPING["nemotron-fp8"] == "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8"
        assert VARIANT_MAPPING["nemotron-nvfp4"] == "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD"


class TestPromptFormatterQwen:
    """Test cases for PromptFormatter with Qwen variant."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        with patch("nemo_curator.models.prompt_formatter.AutoProcessor") as mock_processor:
            mock_processor_instance = Mock()
            mock_processor.from_pretrained.return_value = mock_processor_instance
            self.formatter = PromptFormatter(prompt_variant="qwen")
            self.mock_processor = mock_processor_instance

    def test_initialization_valid_variant(self) -> None:
        """Test initialization with valid prompt variant."""
        with patch("nemo_curator.models.prompt_formatter.AutoProcessor") as mock_processor:
            mock_processor_instance = Mock()
            mock_processor.from_pretrained.return_value = mock_processor_instance

            formatter = PromptFormatter(prompt_variant="qwen")

            assert formatter.prompt_variant == "qwen"
            assert formatter.text_prompt is None
            assert formatter.processor == mock_processor_instance
            mock_processor.from_pretrained.assert_called_once_with(VARIANT_MAPPING["qwen"], trust_remote_code=True)

    def test_initialization_invalid_variant(self) -> None:
        """Test initialization with invalid prompt variant raises ValueError."""
        with pytest.raises(ValueError, match="Invalid prompt variant: invalid_variant"):
            PromptFormatter(prompt_variant="invalid_variant")

    @patch("nemo_curator.models.prompt_formatter.AutoProcessor")
    def test_generate_inputs_first_time(self, mock_processor_class: Mock) -> None:
        """Test generate_inputs method when text_prompt is None (first time)."""
        mock_processor_instance = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor_instance
        mock_processor_instance.apply_chat_template.return_value = "formatted_prompt"

        formatter = PromptFormatter(prompt_variant="qwen")

        video_tensor = torch.randn(1, 3, 224, 224)

        result = formatter.generate_inputs(prompt="Test prompt", video_inputs=video_tensor)

        assert isinstance(result, dict)
        assert "prompt" in result
        assert "multi_modal_data" in result
        assert result["prompt"] == "formatted_prompt"
        assert result["multi_modal_data"]["video"] is video_tensor

        # Verify processor was called correctly
        expected_message = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "Test prompt"}]}]
        mock_processor_instance.apply_chat_template.assert_called_once_with(
            expected_message, tokenize=False, add_generation_prompt=True
        )

        # Verify text_prompt was cached
        assert formatter.text_prompt == "formatted_prompt"

    @patch("nemo_curator.models.prompt_formatter.AutoProcessor")
    def test_generate_inputs_cached_prompt(self, mock_processor_class: Mock) -> None:
        """Test generate_inputs method when text_prompt is already cached."""
        mock_processor_instance = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor_instance

        formatter = PromptFormatter(prompt_variant="qwen")
        formatter.text_prompt = "cached_prompt"

        video_tensor = torch.randn(1, 3, 224, 224)

        result = formatter.generate_inputs(prompt="Test prompt", video_inputs=video_tensor)

        assert result["prompt"] == "cached_prompt"
        assert result["multi_modal_data"]["video"] is video_tensor
        mock_processor_instance.apply_chat_template.assert_not_called()

    @patch("nemo_curator.models.prompt_formatter.AutoProcessor")
    def test_generate_inputs_override_text_prompt(self, mock_processor_class: Mock) -> None:
        """Test generate_inputs method with override_text_prompt=True."""
        mock_processor_instance = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor_instance
        mock_processor_instance.apply_chat_template.return_value = "new_formatted_prompt"

        formatter = PromptFormatter(prompt_variant="qwen")
        formatter.text_prompt = "old_cached_prompt"

        video_tensor = torch.randn(1, 3, 224, 224)

        result = formatter.generate_inputs(prompt="Test prompt", video_inputs=video_tensor, override_text_prompt=True)

        assert result["prompt"] == "new_formatted_prompt"
        assert formatter.text_prompt == "new_formatted_prompt"
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

    def test_create_qwen_message(self) -> None:
        """Test _create_qwen_message method creates correct message structure."""
        result = self.formatter._create_qwen_message("Test prompt text")

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

    def test_create_qwen_message_empty_prompt(self) -> None:
        """Test _create_qwen_message method with empty prompt."""
        result = self.formatter._create_qwen_message("")
        assert result[0]["content"][1]["text"] == ""

    def test_create_qwen_message_special_characters(self) -> None:
        """Test _create_qwen_message method with special characters in prompt."""
        special_prompt = "Test with 特殊字符 and émojis 🎉"
        result = self.formatter._create_qwen_message(special_prompt)
        assert result[0]["content"][1]["text"] == special_prompt


class TestPromptFormatterNemotron:
    """Test cases for PromptFormatter with Nemotron variants."""

    @patch("nemo_curator.models.prompt_formatter.AutoProcessor")
    def test_initialization_nemotron(self, mock_processor_class: Mock) -> None:
        """Test Nemotron initialization uses AutoProcessor from HuggingFace."""
        mock_processor_instance = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor_instance

        formatter = PromptFormatter(prompt_variant="nemotron")

        assert formatter.prompt_variant == "nemotron"
        assert formatter.processor == mock_processor_instance
        mock_processor_class.from_pretrained.assert_called_once_with(
            VARIANT_MAPPING["nemotron"], trust_remote_code=True
        )

    @patch("nemo_curator.models.prompt_formatter.AutoProcessor")
    def test_initialization_nemotron_fp8(self, mock_processor_class: Mock) -> None:
        """Test nemotron-fp8 initialization uses correct HuggingFace model ID."""
        mock_processor_instance = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor_instance

        formatter = PromptFormatter(prompt_variant="nemotron-fp8")

        assert formatter.prompt_variant == "nemotron-fp8"
        mock_processor_class.from_pretrained.assert_called_once_with(
            VARIANT_MAPPING["nemotron-fp8"], trust_remote_code=True
        )

    @patch("nemo_curator.models.prompt_formatter.AutoProcessor")
    def test_generate_inputs_nemotron(self, mock_processor_class: Mock) -> None:
        """Test generate_inputs for Nemotron variant with video metadata."""
        mock_processor_instance = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor_instance
        mock_processor_instance.apply_chat_template.return_value = "nemotron_formatted_prompt"

        formatter = PromptFormatter(prompt_variant="nemotron")

        # Create video tensor (T, C, H, W)
        video_tensor = torch.randint(0, 255, (10, 3, 224, 224), dtype=torch.uint8)

        result = formatter.generate_inputs(prompt="Test prompt", video_inputs=video_tensor, fps=2.0)

        assert result["prompt"] == "nemotron_formatted_prompt"

        # Verify video is returned as tuple with metadata
        video_data = result["multi_modal_data"]["video"]
        assert isinstance(video_data, tuple)
        assert len(video_data) == 2

        video_np, metadata = video_data
        assert isinstance(video_np, np.ndarray)
        assert video_np.shape == (10, 224, 224, 3)  # Converted to (T, H, W, C)
        assert metadata["fps"] == 2.0
        assert metadata["frames_indices"] == list(range(10))

    @patch("nemo_curator.models.prompt_formatter.AutoProcessor")
    def test_generate_inputs_nemotron_no_video(self, mock_processor_class: Mock) -> None:
        """Test generate_inputs for Nemotron variant without video."""
        mock_processor_instance = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor_instance
        mock_processor_instance.apply_chat_template.return_value = "prompt"

        formatter = PromptFormatter(prompt_variant="nemotron")

        result = formatter.generate_inputs(prompt="Test prompt", video_inputs=None)

        assert result["multi_modal_data"]["video"] is None

    @patch("nemo_curator.models.prompt_formatter.AutoProcessor")
    def test_generate_inputs_nemotron_numpy_video(self, mock_processor_class: Mock) -> None:
        """Test generate_inputs for Nemotron variant with numpy video."""
        mock_processor_instance = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor_instance
        mock_processor_instance.apply_chat_template.return_value = "prompt"

        formatter = PromptFormatter(prompt_variant="nemotron")

        # Create numpy video (T, H, W, C) - already in correct format
        rng = np.random.default_rng(42)
        video_np = rng.integers(0, 255, (10, 224, 224, 3), dtype=np.uint8)

        result = formatter.generate_inputs(prompt="Test", video_inputs=video_np, fps=4.0)

        video_data = result["multi_modal_data"]["video"]
        assert isinstance(video_data, tuple)
        assert video_data[1]["fps"] == 4.0

    @patch("nemo_curator.models.prompt_formatter.AutoProcessor")
    def test_generate_inputs_nemotron_message_format(self, mock_processor_class: Mock) -> None:
        """Test that Nemotron uses correct message format with system prompt."""
        mock_processor_instance = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor_instance
        mock_processor_instance.apply_chat_template.return_value = "prompt"

        formatter = PromptFormatter(prompt_variant="nemotron")
        formatter.generate_inputs(prompt="Describe this video")

        # Verify message format passed to apply_chat_template
        call_args = mock_processor_instance.apply_chat_template.call_args
        messages = call_args[0][0]

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."
        assert messages[1]["role"] == "user"
        assert "<video>\nDescribe this video" in messages[1]["content"][0]["text"]


class TestPromptFormatterConvertToNumpy:
    """Test cases for _convert_to_numpy method."""

    @patch("nemo_curator.models.prompt_formatter.AutoProcessor")
    def test_convert_tensor_to_numpy(self, mock_processor_class: Mock) -> None:
        """Test converting torch tensor to numpy array."""
        mock_processor_class.from_pretrained.return_value = Mock()
        formatter = PromptFormatter(prompt_variant="nemotron")

        # Tensor in (T, C, H, W) format
        tensor = torch.randint(0, 255, (10, 3, 224, 224), dtype=torch.uint8)
        result = formatter._convert_to_numpy(tensor)

        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 224, 224, 3)  # Converted to (T, H, W, C)
        assert result.dtype == np.uint8

    @patch("nemo_curator.models.prompt_formatter.AutoProcessor")
    def test_convert_float_tensor_to_uint8(self, mock_processor_class: Mock) -> None:
        """Test converting float tensor (0-1 range) to uint8."""
        mock_processor_class.from_pretrained.return_value = Mock()
        formatter = PromptFormatter(prompt_variant="nemotron")

        # Float tensor in 0-1 range
        tensor = torch.rand(10, 3, 224, 224, dtype=torch.float32)
        result = formatter._convert_to_numpy(tensor)

        assert result.dtype == np.uint8
        assert result.max() <= 255

    @patch("nemo_curator.models.prompt_formatter.AutoProcessor")
    def test_passthrough_numpy_array(self, mock_processor_class: Mock) -> None:
        """Test passing through numpy array."""
        mock_processor_class.from_pretrained.return_value = Mock()
        formatter = PromptFormatter(prompt_variant="nemotron")

        rng = np.random.default_rng(42)
        arr = rng.integers(0, 255, (10, 224, 224, 3), dtype=np.uint8)
        result = formatter._convert_to_numpy(arr)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
