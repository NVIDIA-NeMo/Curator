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

import pathlib
import re
from typing import Any
from unittest.mock import Mock, patch

import pytest

from nemo_curator.models.qwen_vl import _QWEN2_5_VL_MODEL_ID, _QWEN_VARIANTS_INFO, QwenVL


class TestQwenVL:
    """Test cases for QwenVL model class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Mock VLLM_AVAILABLE to True so tests can run without vllm installed
        self.vllm_patcher = patch("nemo_curator.models.qwen_vl.VLLM_AVAILABLE", True)
        self.vllm_patcher.start()

        self.model_dir = "/test/model/dir"
        self.model_variant = "qwen"
        self.caption_batch_size = 4
        self.qwen_vl = QwenVL(
            model_dir=self.model_dir,
            model_variant=self.model_variant,
            caption_batch_size=self.caption_batch_size,
            fp8=True,
            max_output_tokens=512,
            model_does_preprocess=False,
            disable_mmcache=False,
            stage2_prompt_text="Stage 2 prompt: ",
            verbose=False,
        )

    def teardown_method(self) -> None:
        """Clean up test fixtures."""
        self.vllm_patcher.stop()

    def test_constants(self) -> None:
        """Test that module constants are correctly defined."""
        assert _QWEN2_5_VL_MODEL_ID == "Qwen/Qwen2.5-VL-7B-Instruct"
        assert "qwen" in _QWEN_VARIANTS_INFO
        assert _QWEN_VARIANTS_INFO["qwen"] == _QWEN2_5_VL_MODEL_ID

    def test_initialization_default_parameters(self) -> None:
        """Test initialization with default parameters."""
        qwen_vl = QwenVL(
            model_dir=self.model_dir, model_variant=self.model_variant, caption_batch_size=self.caption_batch_size
        )

        assert qwen_vl.model_dir == self.model_dir
        assert qwen_vl.model_variant == self.model_variant
        assert qwen_vl.caption_batch_size == self.caption_batch_size
        assert qwen_vl.fp8 is True
        assert qwen_vl.max_output_tokens == 512
        assert qwen_vl.model_does_preprocess is False
        assert qwen_vl.disable_mmcache is False
        assert qwen_vl.stage2_prompt is None
        assert qwen_vl.verbose is False

        expected_weight_file = str(pathlib.Path(self.model_dir) / _QWEN_VARIANTS_INFO[self.model_variant])
        assert qwen_vl.weight_file == expected_weight_file

    def test_initialization_custom_parameters(self) -> None:
        """Test initialization with custom parameters."""
        assert self.qwen_vl.model_dir == self.model_dir
        assert self.qwen_vl.model_variant == self.model_variant
        assert self.qwen_vl.caption_batch_size == self.caption_batch_size
        assert self.qwen_vl.fp8 is True
        assert self.qwen_vl.max_output_tokens == 512
        assert self.qwen_vl.model_does_preprocess is False
        assert self.qwen_vl.disable_mmcache is False
        assert self.qwen_vl.stage2_prompt == "Stage 2 prompt: "
        assert self.qwen_vl.verbose is False

    def test_initialization_different_variant(self) -> None:
        """Test initialization with different model variant."""
        # Note: This test assumes the variant exists in _QWEN_VARIANTS_INFO
        qwen_vl = QwenVL(model_dir="/another/path", model_variant="qwen", caption_batch_size=8)

        expected_weight_file = str(pathlib.Path("/another/path") / _QWEN_VARIANTS_INFO["qwen"])
        assert qwen_vl.weight_file == expected_weight_file

    def test_model_id_names_property(self) -> None:
        """Test model_id_names property returns correct list."""
        model_ids = self.qwen_vl.model_id_names

        assert isinstance(model_ids, list)
        assert len(model_ids) == 1
        assert model_ids[0] == _QWEN_VARIANTS_INFO[self.model_variant]
        assert model_ids[0] == _QWEN2_5_VL_MODEL_ID

    @patch("nemo_curator.models.qwen_vl.LLM")
    @patch("nemo_curator.models.qwen_vl.SamplingParams")
    @patch("nemo_curator.models.qwen_vl.logger")
    def test_setup_with_fp8(self, mock_logger: Mock, mock_sampling_params: Mock, mock_llm: Mock) -> None:
        """Test setup method with fp8 quantization enabled."""
        # Mock the LLM and SamplingParams
        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance
        mock_sampling_params_instance = Mock()
        mock_sampling_params.return_value = mock_sampling_params_instance

        # Call setup
        self.qwen_vl.setup()

        # Verify LLM initialization
        expected_mm_processor_kwargs = {
            "do_resize": False,
            "do_rescale": False,
            "do_normalize": False,
        }
        mock_llm.assert_called_once_with(
            model=self.qwen_vl.weight_file,
            limit_mm_per_prompt={"image": 0, "video": 1},
            quantization="fp8",
            max_seq_len_to_capture=32768,
            max_model_len=32768,
            gpu_memory_utilization=0.85,
            mm_processor_kwargs=expected_mm_processor_kwargs,
            disable_mm_preprocessor_cache=False,
            max_num_batched_tokens=32768,
        )

        # Verify SamplingParams initialization
        mock_sampling_params.assert_called_once_with(
            temperature=0.1,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=512,
            stop_token_ids=[],
        )

        # Verify model and sampling_params are set
        assert self.qwen_vl.model == mock_llm_instance
        assert self.qwen_vl.sampling_params == mock_sampling_params_instance

        # Verify logger was called
        mock_logger.info.assert_called_once_with(
            "CUDA graph enabled for sequences smaller than 16k tokens; adjust accordingly for even longer sequences"
        )

    @patch("nemo_curator.models.qwen_vl.LLM")
    @patch("nemo_curator.models.qwen_vl.SamplingParams")
    def test_setup_without_fp8(self, mock_sampling_params: Mock, mock_llm: Mock) -> None:  # noqa: ARG002
        """Test setup method with fp8 quantization disabled."""
        # Create QwenVL instance with fp8=False
        qwen_vl = QwenVL(
            model_dir=self.model_dir,
            model_variant=self.model_variant,
            caption_batch_size=self.caption_batch_size,
            fp8=False,
        )

        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance

        qwen_vl.setup()

        # Verify quantization is None when fp8=False
        call_args = mock_llm.call_args
        assert call_args[1]["quantization"] is None

    @patch("nemo_curator.models.qwen_vl.LLM")
    @patch("nemo_curator.models.qwen_vl.SamplingParams")
    def test_setup_with_model_preprocessing(self, mock_sampling_params: Mock, mock_llm: Mock) -> None:  # noqa: ARG002
        """Test setup method with model preprocessing enabled."""
        qwen_vl = QwenVL(
            model_dir=self.model_dir,
            model_variant=self.model_variant,
            caption_batch_size=self.caption_batch_size,
            model_does_preprocess=True,
            disable_mmcache=True,
        )

        qwen_vl.setup()

        # Verify mm_processor_kwargs with preprocessing enabled
        call_args = mock_llm.call_args
        expected_mm_processor_kwargs = {
            "do_resize": True,
            "do_rescale": True,
            "do_normalize": True,
        }
        assert call_args[1]["mm_processor_kwargs"] == expected_mm_processor_kwargs
        assert call_args[1]["disable_mm_preprocessor_cache"] is True

    @patch("nemo_curator.models.qwen_vl.grouping.split_by_chunk_size")
    def test_generate_simple_case(self, mock_split_by_chunk_size: Mock) -> None:
        """Test generate method with simple case (no stage2)."""
        # Setup mocks
        mock_model = Mock()
        self.qwen_vl.model = mock_model
        self.qwen_vl.sampling_params = Mock()

        # Mock input videos
        videos = [
            {"prompt": "Describe this video", "multi_modal_data": {"video": "video1"}},
            {"prompt": "What is happening?", "multi_modal_data": {"video": "video2"}},
        ]

        # Mock grouping function to return one batch
        mock_split_by_chunk_size.return_value = [videos]

        # Mock model outputs
        mock_output1 = Mock()
        mock_output1.outputs = [Mock(text="Generated text 1")]
        mock_output2 = Mock()
        mock_output2.outputs = [Mock(text="Generated text 2")]
        mock_model.generate.return_value = [mock_output1, mock_output2]

        # Call generate
        result = self.qwen_vl.generate(videos, generate_stage2_caption=False, batch_size=16)

        # Verify results
        assert result == ["Generated text 1", "Generated text 2"]

        # Verify grouping was called correctly
        mock_split_by_chunk_size.assert_called_once_with(videos, 16)

        # Verify model.generate was called once
        assert mock_model.generate.call_count == 1
        call_args = mock_model.generate.call_args
        assert call_args[0][0] == list(videos)  # model_inputs
        assert call_args[1]["sampling_params"] == self.qwen_vl.sampling_params
        assert call_args[1]["use_tqdm"] is False

    @patch("nemo_curator.models.qwen_vl.grouping.split_by_chunk_size")
    def test_generate_multiple_batches(self, mock_split_by_chunk_size: Mock) -> None:
        """Test generate method with multiple batches."""
        mock_model = Mock()
        self.qwen_vl.model = mock_model
        self.qwen_vl.sampling_params = Mock()

        # Mock input videos
        videos = [{"prompt": f"Video {i}", "multi_modal_data": {"video": f"video{i}"}} for i in range(4)]

        # Mock grouping to return two batches
        batch1 = videos[:2]
        batch2 = videos[2:]
        mock_split_by_chunk_size.return_value = [batch1, batch2]

        # Track call count to distinguish between batches
        call_count = 0

        # Mock model outputs for each batch
        def mock_generate_side_effect(inputs: Any, **kwargs: Any) -> list[Mock]:  # noqa: ARG001, ANN401
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # First batch
                mock_output1 = Mock()
                mock_output1.outputs = [Mock(text="Batch 1 Text 1")]
                mock_output2 = Mock()
                mock_output2.outputs = [Mock(text="Batch 1 Text 2")]
                return [mock_output1, mock_output2]
            else:  # Second batch
                mock_output1 = Mock()
                mock_output1.outputs = [Mock(text="Batch 2 Text 1")]
                mock_output2 = Mock()
                mock_output2.outputs = [Mock(text="Batch 2 Text 2")]
                return [mock_output1, mock_output2]

        mock_model.generate.side_effect = mock_generate_side_effect

        # Call generate
        result = self.qwen_vl.generate(videos, generate_stage2_caption=False, batch_size=2)

        # Verify results from both batches
        expected_result = ["Batch 1 Text 1", "Batch 1 Text 2", "Batch 2 Text 1", "Batch 2 Text 2"]
        assert result == expected_result

        # Verify model.generate was called twice (once per batch)
        assert mock_model.generate.call_count == 2

    @patch("nemo_curator.models.qwen_vl.grouping.split_by_chunk_size")
    @patch("nemo_curator.models.qwen_vl.re.sub")
    def test_generate_with_stage2_caption(self, mock_re_sub: Mock, mock_split_by_chunk_size: Mock) -> None:
        """Test generate method with stage2 caption generation."""
        # Setup mocks
        mock_model = Mock()
        self.qwen_vl.model = mock_model
        self.qwen_vl.sampling_params = Mock()
        self.qwen_vl.pattern = r"(.*)(user_prompt)(.*)"  # Mock pattern for stage2

        # Mock input videos
        videos = [{"prompt": "Initial prompt", "multi_modal_data": {"video": "video1"}}]
        mock_split_by_chunk_size.return_value = [videos]

        # Mock first generation (stage 1)
        mock_output_stage1 = Mock()
        mock_output_stage1.outputs = [Mock(text="Stage 1 caption")]

        # Mock second generation (stage 2)
        mock_output_stage2 = Mock()
        mock_output_stage2.outputs = [Mock(text="Stage 2 final caption")]

        # Setup side effect for two generate calls
        mock_model.generate.side_effect = [[mock_output_stage1], [mock_output_stage2]]

        # Mock re.sub to simulate prompt updating
        mock_re_sub.return_value = "Updated prompt with stage 2"

        # Call generate with stage2 enabled
        result = self.qwen_vl.generate(videos, generate_stage2_caption=True, batch_size=16)

        # Verify final result
        assert result == ["Stage 2 final caption"]

        # Verify model.generate was called twice
        assert mock_model.generate.call_count == 2

        # Verify re.sub was called to update the prompt
        expected_updated_prompt = self.qwen_vl.stage2_prompt + "Stage 1 caption"
        mock_re_sub.assert_called_once_with(
            self.qwen_vl.pattern,
            rf"\1{expected_updated_prompt}\3",
            "Initial prompt",
            flags=re.DOTALL,
        )

    @patch("nemo_curator.models.qwen_vl.grouping.split_by_chunk_size")
    @patch("nemo_curator.models.qwen_vl.logger")
    def test_generate_exception_handling(self, mock_logger: Mock, mock_split_by_chunk_size: Mock) -> None:
        """Test generate method exception handling."""
        # Setup mocks
        mock_model = Mock()
        self.qwen_vl.model = mock_model
        self.qwen_vl.sampling_params = Mock()

        # Mock input videos
        videos = [{"prompt": "Test prompt", "multi_modal_data": {"video": "video1"}}]
        mock_split_by_chunk_size.return_value = [videos]

        # Mock model to raise exception
        test_exception = Exception("Test error")
        mock_model.generate.side_effect = test_exception

        # Verify exception is raised and logged
        with pytest.raises(Exception, match="Test error"):
            self.qwen_vl.generate(videos)

        # Verify error was logged
        mock_logger.error.assert_called_once_with("Error generating caption for batch: Test error")

    def test_generate_empty_videos(self) -> None:
        """Test generate method with empty videos list."""
        mock_model = Mock()
        self.qwen_vl.model = mock_model
        self.qwen_vl.sampling_params = Mock()

        result = self.qwen_vl.generate([])

        assert result == []
        # Model should not be called with empty input
        mock_model.generate.assert_not_called()

    def test_weight_file_path_construction(self) -> None:
        """Test that weight_file path is constructed correctly."""
        expected_path = str(pathlib.Path(self.model_dir) / _QWEN_VARIANTS_INFO[self.model_variant])
        assert self.qwen_vl.weight_file == expected_path

        # Test with different paths
        qwen_vl2 = QwenVL(model_dir="/different/path", model_variant="qwen", caption_batch_size=1)
        expected_path2 = str(pathlib.Path("/different/path") / _QWEN_VARIANTS_INFO["qwen"])
        assert qwen_vl2.weight_file == expected_path2

    def test_max_output_tokens_parameter(self) -> None:
        """Test that max_output_tokens parameter is properly handled."""
        custom_tokens = 1024
        qwen_vl = QwenVL(
            model_dir=self.model_dir,
            model_variant=self.model_variant,
            caption_batch_size=self.caption_batch_size,
            max_output_tokens=custom_tokens,
        )

        assert qwen_vl.max_output_tokens == custom_tokens

    @patch("nemo_curator.models.qwen_vl.LLM")
    @patch("nemo_curator.models.qwen_vl.SamplingParams")
    def test_setup_sampling_params_with_custom_tokens(self, mock_sampling_params: Mock, mock_llm: Mock) -> None:  # noqa: ARG002
        """Test that SamplingParams uses the custom max_output_tokens."""
        custom_tokens = 256
        qwen_vl = QwenVL(
            model_dir=self.model_dir,
            model_variant=self.model_variant,
            caption_batch_size=self.caption_batch_size,
            max_output_tokens=custom_tokens,
        )

        qwen_vl.setup()

        # Verify SamplingParams was called with custom max_tokens
        mock_sampling_params.assert_called_once_with(
            temperature=0.1,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=custom_tokens,
            stop_token_ids=[],
        )
