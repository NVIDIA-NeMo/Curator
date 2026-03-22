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

import pathlib
import re
from typing import Any
from unittest.mock import Mock, patch

import pytest

from nemo_curator.models.qwen_vl import _QWEN_VL_MODEL_ID, QwenVL


class TestQwenVL:
    """Test cases for QwenVL model class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Mock VLLM_AVAILABLE to True so tests can run without vllm installed
        self.vllm_patcher = patch("nemo_curator.models.qwen_vl.VLLM_AVAILABLE", True)
        self.vllm_patcher.start()

        self.model_dir = "/test/model/dir"
        self.caption_batch_size = 4
        self.qwen_vl = QwenVL(
            model_dir=self.model_dir,
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
        assert _QWEN_VL_MODEL_ID == "Qwen/Qwen3-VL-8B-Instruct"

    def test_initialization_default_parameters(self) -> None:
        """Test initialization with default parameters."""
        qwen_vl = QwenVL(model_dir=self.model_dir, caption_batch_size=self.caption_batch_size)

        assert qwen_vl.model_dir == self.model_dir
        assert qwen_vl.model_name == _QWEN_VL_MODEL_ID
        assert qwen_vl.caption_batch_size == self.caption_batch_size
        assert qwen_vl.fp8 is True
        assert qwen_vl.max_output_tokens == 512
        assert qwen_vl.model_does_preprocess is False
        assert qwen_vl.disable_mmcache is False
        assert qwen_vl.stage2_prompt is None
        assert qwen_vl.verbose is False

        expected_weight_file = str(pathlib.Path(self.model_dir) / _QWEN_VL_MODEL_ID)
        assert qwen_vl.weight_file == expected_weight_file

    def test_initialization_custom_parameters(self) -> None:
        """Test initialization with custom parameters."""
        assert self.qwen_vl.model_dir == self.model_dir
        assert self.qwen_vl.model_name == _QWEN_VL_MODEL_ID
        assert self.qwen_vl.caption_batch_size == self.caption_batch_size
        assert self.qwen_vl.fp8 is True
        assert self.qwen_vl.max_output_tokens == 512
        assert self.qwen_vl.model_does_preprocess is False
        assert self.qwen_vl.disable_mmcache is False
        assert self.qwen_vl.stage2_prompt == "Stage 2 prompt: "
        assert self.qwen_vl.verbose is False

    def test_custom_model_name(self) -> None:
        """Test initialization with a custom Qwen model name."""
        custom_model = "Qwen/Qwen2.5-VL-72B-Instruct"
        qwen_vl = QwenVL(model_dir=self.model_dir, caption_batch_size=self.caption_batch_size, model_name=custom_model)
        assert qwen_vl.model_name == custom_model
        assert qwen_vl.model_id_names == [custom_model]
        assert qwen_vl.weight_file == str(pathlib.Path(self.model_dir) / custom_model)

    def test_invalid_model_name_raises(self) -> None:
        """Test that non-Qwen model names raise ValueError."""
        with pytest.raises(ValueError, match="must be a Qwen model"):
            QwenVL(
                model_dir=self.model_dir,
                caption_batch_size=self.caption_batch_size,
                model_name="mistralai/Mistral-7B-Instruct",
            )

    def test_model_id_names_property(self) -> None:
        """Test model_id_names property returns correct list."""
        model_ids = self.qwen_vl.model_id_names

        assert isinstance(model_ids, list)
        assert len(model_ids) == 1
        assert model_ids[0] == _QWEN_VL_MODEL_ID

    @patch("nemo_curator.models.qwen_vl._check_vllm_supports_model")
    @patch("nemo_curator.models.qwen_vl.LLM")
    @patch("nemo_curator.models.qwen_vl.SamplingParams")
    @patch("nemo_curator.models.qwen_vl.logger")
    def test_setup_with_fp8(
        self, mock_logger: Mock, mock_sampling_params: Mock, mock_llm: Mock, mock_check_vllm: Mock
    ) -> None:
        """Test setup method with fp8 quantization enabled."""
        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance
        mock_sampling_params_instance = Mock()
        mock_sampling_params.return_value = mock_sampling_params_instance

        # Simulate weights already present
        with patch("nemo_curator.models.qwen_vl.Path") as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.glob.return_value = ["model.safetensors"]
            mock_path.return_value = mock_path_instance

            self.qwen_vl.setup()

        mock_check_vllm.assert_called_once_with(self.qwen_vl.model_name)
        expected_mm_processor_kwargs = {
            "do_resize": False,
            "do_rescale": False,
            "do_normalize": False,
        }
        mock_llm.assert_called_once_with(
            model=self.qwen_vl.weight_file,
            limit_mm_per_prompt={"image": 0, "video": 1},
            quantization="fp8",
            max_model_len=32768,
            gpu_memory_utilization=0.85,
            mm_processor_kwargs=expected_mm_processor_kwargs,
            mm_processor_cache_gb=4,
            max_num_batched_tokens=32768,
        )
        mock_sampling_params.assert_called_once_with(
            temperature=0.1,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=512,
            stop_token_ids=[],
        )
        assert self.qwen_vl.model == mock_llm_instance
        assert self.qwen_vl.sampling_params == mock_sampling_params_instance
        mock_logger.info.assert_called_once_with(
            "CUDA graph enabled for sequences smaller than 16k tokens; adjust accordingly for even longer sequences"
        )

    @patch("nemo_curator.models.qwen_vl._check_vllm_supports_model")
    @patch("nemo_curator.models.qwen_vl.LLM")
    @patch("nemo_curator.models.qwen_vl.SamplingParams")
    def test_setup_without_fp8(self, mock_sampling_params: Mock, mock_llm: Mock, mock_check_vllm: Mock) -> None:
        """Test setup method with fp8 quantization disabled."""
        qwen_vl = QwenVL(model_dir=self.model_dir, caption_batch_size=self.caption_batch_size, fp8=False)
        mock_llm.return_value = Mock()

        with patch("nemo_curator.models.qwen_vl.Path") as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.glob.return_value = ["model.safetensors"]
            mock_path.return_value = mock_path_instance
            qwen_vl.setup()

        assert mock_llm.call_args[1]["quantization"] is None

    @patch("nemo_curator.models.qwen_vl._check_vllm_supports_model")
    @patch("nemo_curator.models.qwen_vl.LLM")
    @patch("nemo_curator.models.qwen_vl.SamplingParams")
    def test_setup_with_model_preprocessing(
        self, mock_sampling_params: Mock, mock_llm: Mock, mock_check_vllm: Mock
    ) -> None:
        """Test setup method with model preprocessing enabled."""
        qwen_vl = QwenVL(
            model_dir=self.model_dir,
            caption_batch_size=self.caption_batch_size,
            model_does_preprocess=True,
            disable_mmcache=True,
        )
        mock_llm.return_value = Mock()

        with patch("nemo_curator.models.qwen_vl.Path") as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.glob.return_value = ["model.safetensors"]
            mock_path.return_value = mock_path_instance
            qwen_vl.setup()

        call_args = mock_llm.call_args
        expected_mm_processor_kwargs = {"do_resize": True, "do_rescale": True, "do_normalize": True}
        assert call_args[1]["mm_processor_kwargs"] == expected_mm_processor_kwargs
        assert call_args[1]["mm_processor_cache_gb"] == 0

    @patch("nemo_curator.models.qwen_vl._check_vllm_supports_model")
    @patch("nemo_curator.models.qwen_vl.download_model_from_hf")
    @patch("nemo_curator.models.qwen_vl.LLM")
    @patch("nemo_curator.models.qwen_vl.SamplingParams")
    def test_setup_auto_downloads_if_missing(
        self, mock_sampling_params: Mock, mock_llm: Mock, mock_download: Mock, mock_check_vllm: Mock
    ) -> None:
        """Test that setup downloads weights when not present."""
        mock_llm.return_value = Mock()

        with patch("nemo_curator.models.qwen_vl.Path") as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = False
            mock_path_instance.glob.return_value = []
            mock_path_instance.mkdir = Mock()
            mock_path.return_value = mock_path_instance

            self.qwen_vl.setup()

        mock_download.assert_called_once()

    @patch("nemo_curator.models.qwen_vl._check_vllm_supports_model")
    @patch("nemo_curator.models.qwen_vl.download_model_from_hf")
    @patch("nemo_curator.models.qwen_vl.LLM")
    @patch("nemo_curator.models.qwen_vl.SamplingParams")
    def test_setup_skips_download_if_weights_present(
        self, mock_sampling_params: Mock, mock_llm: Mock, mock_download: Mock, mock_check_vllm: Mock
    ) -> None:
        """Test that setup skips download when weights are already present."""
        mock_llm.return_value = Mock()

        with patch("nemo_curator.models.qwen_vl.Path") as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.glob.return_value = ["model.safetensors"]
            mock_path.return_value = mock_path_instance

            self.qwen_vl.setup()

        mock_download.assert_not_called()

    def test_setup_raises_if_vllm_unsupported(self) -> None:
        """Test that setup raises ValueError for unsupported models."""
        with (
            patch(
                "nemo_curator.models.qwen_vl._check_vllm_supports_model",
                side_effect=ValueError("not supported by vLLM"),
            ),
            pytest.raises(ValueError, match="not supported by vLLM"),
        ):
            self.qwen_vl.setup()

    @patch("nemo_curator.models.qwen_vl.grouping.split_by_chunk_size")
    def test_generate_simple_case(self, mock_split_by_chunk_size: Mock) -> None:
        """Test generate method with simple case (no stage2)."""
        mock_model = Mock()
        self.qwen_vl.model = mock_model
        self.qwen_vl.sampling_params = Mock()

        videos = [
            {"prompt": "Describe this video", "multi_modal_data": {"video": "video1"}},
            {"prompt": "What is happening?", "multi_modal_data": {"video": "video2"}},
        ]
        mock_split_by_chunk_size.return_value = [videos]

        mock_output1 = Mock()
        mock_output1.outputs = [Mock(text="Generated text 1")]
        mock_output2 = Mock()
        mock_output2.outputs = [Mock(text="Generated text 2")]
        mock_model.generate.return_value = [mock_output1, mock_output2]

        result = self.qwen_vl.generate(videos, generate_stage2_caption=False, batch_size=16)

        assert result == ["Generated text 1", "Generated text 2"]
        mock_split_by_chunk_size.assert_called_once_with(videos, 16)
        assert mock_model.generate.call_count == 1
        call_args = mock_model.generate.call_args
        assert call_args[0][0] == list(videos)
        assert call_args[1]["sampling_params"] == self.qwen_vl.sampling_params
        assert call_args[1]["use_tqdm"] is False

    @patch("nemo_curator.models.qwen_vl.grouping.split_by_chunk_size")
    def test_generate_multiple_batches(self, mock_split_by_chunk_size: Mock) -> None:
        """Test generate method with multiple batches."""
        mock_model = Mock()
        self.qwen_vl.model = mock_model
        self.qwen_vl.sampling_params = Mock()

        videos = [{"prompt": f"Video {i}", "multi_modal_data": {"video": f"video{i}"}} for i in range(4)]
        batch1 = videos[:2]
        batch2 = videos[2:]
        mock_split_by_chunk_size.return_value = [batch1, batch2]

        call_count = 0

        def mock_generate_side_effect(inputs: Any, **kwargs: Any) -> list[Mock]:  # noqa: ARG001, ANN401
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [Mock(outputs=[Mock(text="Batch 1 Text 1")]), Mock(outputs=[Mock(text="Batch 1 Text 2")])]
            return [Mock(outputs=[Mock(text="Batch 2 Text 1")]), Mock(outputs=[Mock(text="Batch 2 Text 2")])]

        mock_model.generate.side_effect = mock_generate_side_effect

        result = self.qwen_vl.generate(videos, generate_stage2_caption=False, batch_size=2)

        assert result == ["Batch 1 Text 1", "Batch 1 Text 2", "Batch 2 Text 1", "Batch 2 Text 2"]
        assert mock_model.generate.call_count == 2

    @patch("nemo_curator.models.qwen_vl.grouping.split_by_chunk_size")
    @patch("nemo_curator.models.qwen_vl.re.sub")
    def test_generate_with_stage2_caption(self, mock_re_sub: Mock, mock_split_by_chunk_size: Mock) -> None:
        """Test generate method with stage2 caption generation."""
        mock_model = Mock()
        self.qwen_vl.model = mock_model
        self.qwen_vl.sampling_params = Mock()
        self.qwen_vl.pattern = r"(.*)(user_prompt)(.*)"

        videos = [{"prompt": "Initial prompt", "multi_modal_data": {"video": "video1"}}]
        mock_split_by_chunk_size.return_value = [videos]

        mock_model.generate.side_effect = [
            [Mock(outputs=[Mock(text="Stage 1 caption")])],
            [Mock(outputs=[Mock(text="Stage 2 final caption")])],
        ]
        mock_re_sub.return_value = "Updated prompt with stage 2"

        result = self.qwen_vl.generate(videos, generate_stage2_caption=True, batch_size=16)

        assert result == ["Stage 2 final caption"]
        assert mock_model.generate.call_count == 2
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
        mock_model = Mock()
        self.qwen_vl.model = mock_model
        self.qwen_vl.sampling_params = Mock()

        videos = [{"prompt": "Test prompt", "multi_modal_data": {"video": "video1"}}]
        mock_split_by_chunk_size.return_value = [videos]
        mock_model.generate.side_effect = Exception("Test error")

        with pytest.raises(Exception, match="Test error"):
            self.qwen_vl.generate(videos)

        mock_logger.error.assert_called_once_with("Error generating caption for batch: Test error")

    def test_generate_empty_videos(self) -> None:
        """Test generate method with empty videos list."""
        mock_model = Mock()
        self.qwen_vl.model = mock_model
        self.qwen_vl.sampling_params = Mock()

        result = self.qwen_vl.generate([])

        assert result == []
        mock_model.generate.assert_not_called()

    def test_weight_file_path_construction(self) -> None:
        """Test that weight_file path is constructed correctly."""
        expected_path = str(pathlib.Path(self.model_dir) / _QWEN_VL_MODEL_ID)
        assert self.qwen_vl.weight_file == expected_path

        qwen_vl2 = QwenVL(model_dir="/different/path", caption_batch_size=1)
        expected_path2 = str(pathlib.Path("/different/path") / _QWEN_VL_MODEL_ID)
        assert qwen_vl2.weight_file == expected_path2

    def test_max_output_tokens_parameter(self) -> None:
        """Test that max_output_tokens parameter is properly handled."""
        custom_tokens = 1024
        qwen_vl = QwenVL(
            model_dir=self.model_dir,
            caption_batch_size=self.caption_batch_size,
            max_output_tokens=custom_tokens,
        )
        assert qwen_vl.max_output_tokens == custom_tokens

    @patch("nemo_curator.models.qwen_vl._check_vllm_supports_model")
    @patch("nemo_curator.models.qwen_vl.LLM")
    @patch("nemo_curator.models.qwen_vl.SamplingParams")
    def test_setup_sampling_params_with_custom_tokens(
        self, mock_sampling_params: Mock, mock_llm: Mock, mock_check_vllm: Mock
    ) -> None:
        """Test that SamplingParams uses the custom max_output_tokens."""
        custom_tokens = 256
        qwen_vl = QwenVL(
            model_dir=self.model_dir,
            caption_batch_size=self.caption_batch_size,
            max_output_tokens=custom_tokens,
        )
        mock_llm.return_value = Mock()

        with patch("nemo_curator.models.qwen_vl.Path") as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.glob.return_value = ["model.safetensors"]
            mock_path.return_value = mock_path_instance
            qwen_vl.setup()

        mock_sampling_params.assert_called_once_with(
            temperature=0.1,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=custom_tokens,
            stop_token_ids=[],
        )

    @patch("nemo_curator.models.qwen_vl.download_model_from_hf")
    def test_download_weights_on_node_custom_model(self, mock_download: Mock) -> None:
        """Test download_weights_on_node with a custom model name."""
        custom_model = "Qwen/Qwen2.5-VL-72B-Instruct"
        with patch("nemo_curator.models.qwen_vl.Path") as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.__truediv__ = Mock(return_value=mock_path_instance)
            mock_path_instance.exists.return_value = False
            mock_path_instance.glob.return_value = []
            mock_path_instance.mkdir = Mock()
            mock_path.return_value = mock_path_instance

            QwenVL.download_weights_on_node(model_dir="/some/dir", model_name=custom_model)

            mock_download.assert_called_once_with(model_id=custom_model, local_dir=mock_path_instance, revision=None)

    def test_download_weights_on_node_invalid_model(self) -> None:
        """Test download_weights_on_node raises for non-Qwen models."""
        with pytest.raises(ValueError, match="must be a Qwen model"):
            QwenVL.download_weights_on_node(model_dir="/some/dir", model_name="mistralai/Mistral-7B-Instruct")
