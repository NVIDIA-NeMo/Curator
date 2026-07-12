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

import json
from pathlib import Path
from unittest.mock import Mock, patch

from nemo_curator.models.qwen_lm import _QWEN_LM_VARIANTS_INFO, QwenLM, _weights_complete


class TestQwenLM:
    """Test cases for QwenLM model class."""

    def setup_method(self) -> None:
        # Ensure tests can run without vllm installed by forcing availability flag to True.
        self.vllm_available_patcher = patch("nemo_curator.models.qwen_lm.VLLM_AVAILABLE", True)
        self.vllm_available_patcher.start()

        self.model_dir = "/test/model/dir"
        self.caption_batch_size = 4
        self.fp8 = True
        self.max_output_tokens = 256
        self.qwen_lm = QwenLM(
            model_dir=self.model_dir,
            model_variant="qwen2.5",
            caption_batch_size=self.caption_batch_size,
            fp8=self.fp8,
            max_output_tokens=self.max_output_tokens,
        )

    def test_constants(self) -> None:
        assert _QWEN_LM_VARIANTS_INFO["qwen2.5"][0] == "Qwen/Qwen2.5-14B-Instruct"
        assert _QWEN_LM_VARIANTS_INFO["qwen3"][0] == "Qwen/Qwen3-14B"

    def test_initialization(self) -> None:
        assert self.qwen_lm.model_dir == self.model_dir
        assert self.qwen_lm.caption_batch_size == self.caption_batch_size
        assert self.qwen_lm.fp8 == self.fp8
        assert self.qwen_lm.max_output_tokens == self.max_output_tokens

    def test_model_id_names(self) -> None:
        model_ids = self.qwen_lm.model_id_names()
        assert isinstance(model_ids, list)
        assert len(model_ids) == 1
        assert model_ids[0] == _QWEN_LM_VARIANTS_INFO["qwen2.5"][0]

    @patch("nemo_curator.models.qwen_lm.AutoTokenizer")
    @patch("nemo_curator.models.qwen_lm.SamplingParams")
    @patch("nemo_curator.models.qwen_lm.LLM")
    @patch("nemo_curator.models.qwen_lm.Path")
    def test_setup_with_fp8(
        self, mock_path: Mock, mock_llm_class: Mock, mock_sampling_params_class: Mock, mock_tokenizer_class: Mock
    ) -> None:
        # Mock Path behavior
        mock_path_instance = Mock()
        mock_path_instance.__truediv__ = Mock(return_value="/test/model/dir/Qwen/Qwen2.5-14B-Instruct")
        mock_path.return_value = mock_path_instance

        # Mock LLM
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm

        # Mock SamplingParams
        mock_sampling_params = Mock()
        mock_sampling_params_class.return_value = mock_sampling_params

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        self.qwen_lm.setup()

        # Verify LLM initialization
        mock_llm_class.assert_called_once_with(
            model="/test/model/dir/Qwen/Qwen2.5-14B-Instruct",
            quantization="fp8",
        )

        # Verify SamplingParams initialization
        mock_sampling_params_class.assert_called_once_with(
            temperature=0.1,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=self.max_output_tokens,
            stop_token_ids=[],
        )

        # Verify tokenizer initialization
        mock_tokenizer_class.from_pretrained.assert_called_once_with("/test/model/dir/Qwen/Qwen2.5-14B-Instruct")

        # Verify attributes are set
        assert self.qwen_lm.weight_file == "/test/model/dir/Qwen/Qwen2.5-14B-Instruct"
        assert self.qwen_lm.llm == mock_llm
        assert self.qwen_lm.sampling_params == mock_sampling_params
        assert self.qwen_lm.tokenizer == mock_tokenizer

    @patch("nemo_curator.models.qwen_lm.AutoTokenizer")
    @patch("nemo_curator.models.qwen_lm.SamplingParams")
    @patch("nemo_curator.models.qwen_lm.LLM")
    @patch("nemo_curator.models.qwen_lm.Path")
    def test_setup_without_fp8(
        self, mock_path: Mock, mock_llm_class: Mock, mock_sampling_params_class: Mock, mock_tokenizer_class: Mock
    ) -> None:
        # Create QwenLM without fp8
        qwen_lm_no_fp8 = QwenLM(
            model_dir=self.model_dir,
            caption_batch_size=self.caption_batch_size,
            fp8=False,
            max_output_tokens=self.max_output_tokens,
        )

        # Mock Path behavior
        mock_path_instance = Mock()
        mock_path_instance.__truediv__ = Mock(return_value="/test/model/dir/Qwen/Qwen2.5-14B-Instruct")
        mock_path.return_value = mock_path_instance

        # Mock LLM
        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm

        # Mock SamplingParams
        mock_sampling_params = Mock()
        mock_sampling_params_class.return_value = mock_sampling_params

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        qwen_lm_no_fp8.setup()

        # Verify LLM initialization without fp8
        mock_llm_class.assert_called_once_with(
            model="/test/model/dir/Qwen/Qwen2.5-14B-Instruct",
            quantization=None,
        )

    def test_generate_single_input(self) -> None:
        # Setup mocks
        mock_llm = Mock()
        mock_tokenizer = Mock()
        mock_sampling_params = Mock()

        # Mock tokenizer behavior
        formatted_input = "formatted_prompt"
        mock_tokenizer.apply_chat_template.return_value = formatted_input

        # Mock LLM generate response
        mock_output = Mock()
        mock_output.text = "Generated response"
        mock_result = Mock()
        mock_result.outputs = [mock_output]
        mock_llm.generate.return_value = [mock_result]

        # Set up the model
        self.qwen_lm.llm = mock_llm
        self.qwen_lm.tokenizer = mock_tokenizer
        self.qwen_lm.sampling_params = mock_sampling_params

        # Test input
        test_input = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, world!"},
        ]

        result = self.qwen_lm.generate([test_input])

        # Verify tokenizer was called correctly
        mock_tokenizer.apply_chat_template.assert_called_once_with(
            [test_input], tokenize=False, add_generation_prompt=True
        )

        # Verify LLM generate was called correctly
        mock_llm.generate.assert_called_once_with(formatted_input, sampling_params=mock_sampling_params)

        # Verify result
        assert result == ["Generated response"]

    def test_generate_multiple_inputs(self) -> None:
        # Setup mocks
        mock_llm = Mock()
        mock_tokenizer = Mock()
        mock_sampling_params = Mock()

        # Mock tokenizer behavior for multiple inputs
        formatted_inputs = ["formatted_prompt_1", "formatted_prompt_2"]
        mock_tokenizer.apply_chat_template.return_value = formatted_inputs

        # Mock LLM generate response for multiple inputs
        mock_output_1 = Mock()
        mock_output_1.text = "Response 1"
        mock_result_1 = Mock()
        mock_result_1.outputs = [mock_output_1]

        mock_output_2 = Mock()
        mock_output_2.text = "Response 2"
        mock_result_2 = Mock()
        mock_result_2.outputs = [mock_output_2]

        mock_llm.generate.return_value = [mock_result_1, mock_result_2]

        # Set up the model
        self.qwen_lm.llm = mock_llm
        self.qwen_lm.tokenizer = mock_tokenizer
        self.qwen_lm.sampling_params = mock_sampling_params

        # Test inputs
        test_inputs = [[{"role": "user", "content": "First message"}], [{"role": "user", "content": "Second message"}]]

        result = self.qwen_lm.generate(test_inputs)

        # Verify tokenizer was called correctly
        mock_tokenizer.apply_chat_template.assert_called_once_with(
            test_inputs, tokenize=False, add_generation_prompt=True
        )

        # Verify LLM generate was called correctly
        mock_llm.generate.assert_called_once_with(formatted_inputs, sampling_params=mock_sampling_params)

        # Verify result
        assert result == ["Response 1", "Response 2"]

    def test_generate_empty_input(self) -> None:
        # Setup mocks
        mock_llm = Mock()
        mock_tokenizer = Mock()
        mock_sampling_params = Mock()

        # Mock tokenizer behavior for empty input
        mock_tokenizer.apply_chat_template.return_value = []
        mock_llm.generate.return_value = []

        # Set up the model
        self.qwen_lm.llm = mock_llm
        self.qwen_lm.tokenizer = mock_tokenizer
        self.qwen_lm.sampling_params = mock_sampling_params

        result = self.qwen_lm.generate([])

        # Verify tokenizer was called correctly
        mock_tokenizer.apply_chat_template.assert_called_once_with([], tokenize=False, add_generation_prompt=True)

        # Verify LLM generate was called correctly
        mock_llm.generate.assert_called_once_with([], sampling_params=mock_sampling_params)

        # Verify result
        assert result == []

    def test_generate_with_complex_conversation(self) -> None:
        """Test generate method with complex multi-turn conversation."""
        # Setup mocks
        mock_llm = Mock()
        mock_tokenizer = Mock()
        mock_sampling_params = Mock()

        # Mock tokenizer behavior
        formatted_input = "complex_formatted_prompt"
        mock_tokenizer.apply_chat_template.return_value = formatted_input

        # Mock LLM generate response
        mock_output = Mock()
        mock_output.text = "Complex response with multiple sentences. This is a detailed answer."
        mock_result = Mock()
        mock_result.outputs = [mock_output]
        mock_llm.generate.return_value = [mock_result]

        # Set up the model
        self.qwen_lm.llm = mock_llm
        self.qwen_lm.tokenizer = mock_tokenizer
        self.qwen_lm.sampling_params = mock_sampling_params

        # Test input with multi-turn conversation
        test_input = [
            {"role": "system", "content": "You are a video caption enhancer."},
            {"role": "user", "content": "A person walks down the street."},
            {"role": "assistant", "content": "A person is walking down a busy city street."},
            {"role": "user", "content": "Add more details about the environment."},
        ]

        result = self.qwen_lm.generate([test_input])

        # Verify result
        assert result == ["Complex response with multiple sentences. This is a detailed answer."]

        # Verify tokenizer was called with correct parameters
        mock_tokenizer.apply_chat_template.assert_called_once_with(
            [test_input], tokenize=False, add_generation_prompt=True
        )

    def test_weight_file_path_construction(self) -> None:
        with patch("nemo_curator.models.qwen_lm.Path") as mock_path:
            mock_path_instance = Mock()
            expected_path = "/test/model/dir/Qwen/Qwen2.5-14B-Instruct"
            mock_path_instance.__truediv__ = Mock(return_value=expected_path)
            mock_path.return_value = mock_path_instance

            with (
                patch("nemo_curator.models.qwen_lm.LLM"),
                patch("nemo_curator.models.qwen_lm.SamplingParams"),
                patch("nemo_curator.models.qwen_lm.AutoTokenizer"),
            ):
                self.qwen_lm.setup()

                # Verify Path was called correctly
                mock_path.assert_called_once_with(self.model_dir)
                mock_path_instance.__truediv__.assert_called_once_with(_QWEN_LM_VARIANTS_INFO["qwen2.5"][0])
                assert self.qwen_lm.weight_file == expected_path

    def test_sampling_params_configuration(self) -> None:
        with (
            patch("nemo_curator.models.qwen_lm.LLM"),
            patch("nemo_curator.models.qwen_lm.AutoTokenizer"),
            patch("nemo_curator.models.qwen_lm.Path") as mock_path,
            patch("nemo_curator.models.qwen_lm.SamplingParams") as mock_sampling_params_class,
        ):
            mock_path_instance = Mock()
            mock_path_instance.__truediv__ = Mock(return_value="test_path")
            mock_path.return_value = mock_path_instance

            mock_sampling_params = Mock()
            mock_sampling_params_class.return_value = mock_sampling_params

            self.qwen_lm.setup()

            # Verify SamplingParams was initialized with correct parameters
            mock_sampling_params_class.assert_called_once_with(
                temperature=0.1,
                top_p=0.001,
                repetition_penalty=1.05,
                max_tokens=self.max_output_tokens,
                stop_token_ids=[],
            )

    def teardown_method(self) -> None:
        self.vllm_available_patcher.stop()


class TestWeightsComplete:
    """Test cases for _weights_complete helper function."""

    def test_single_safetensors_file_exists(self, tmp_path: Path) -> None:
        (tmp_path / "model.safetensors").touch()
        assert _weights_complete(tmp_path) is True

    def test_no_files_returns_false(self, tmp_path: Path) -> None:
        assert _weights_complete(tmp_path) is False

    def test_index_with_all_shards_present(self, tmp_path: Path) -> None:
        shard1 = "model-00001-of-00002.safetensors"
        shard2 = "model-00002-of-00002.safetensors"
        (tmp_path / shard1).touch()
        (tmp_path / shard2).touch()
        index_data = {"weight_map": {"w1": shard1, "w2": shard2}}
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index_data))
        assert _weights_complete(tmp_path) is True

    def test_index_with_missing_shard_returns_false(self, tmp_path: Path) -> None:
        shard1 = "model-00001-of-00002.safetensors"
        shard2 = "model-00002-of-00002.safetensors"
        (tmp_path / shard1).touch()
        index_data = {"weight_map": {"w1": shard1, "w2": shard2}}
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index_data))
        assert _weights_complete(tmp_path) is False

    def test_index_with_invalid_json_returns_false(self, tmp_path: Path) -> None:
        (tmp_path / "model.safetensors.index.json").write_text("not valid json {{{")
        assert _weights_complete(tmp_path) is False

    def test_index_with_empty_weight_map_returns_false(self, tmp_path: Path) -> None:
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
        assert _weights_complete(tmp_path) is False

    def test_index_with_oserror_returns_false(self, tmp_path: Path) -> None:
        index_file = tmp_path / "model.safetensors.index.json"
        index_file.write_text("{}")
        with patch("builtins.open", side_effect=OSError("permission denied")):
            assert _weights_complete(tmp_path) is False


class TestQwenLMDownloadWeightsOnNode:
    """Test cases for QwenLM.download_weights_on_node class method."""

    @patch("nemo_curator.models.qwen_lm._weights_complete", return_value=True)
    @patch("nemo_curator.models.qwen_lm.download_model_from_hf")
    @patch("nemo_curator.models.qwen_lm.logger")
    def test_skips_download_when_weights_complete(
        self, mock_logger: Mock, mock_download: Mock, mock_complete: Mock, tmp_path: Path
    ) -> None:
        QwenLM.download_weights_on_node(str(tmp_path), variant="qwen2.5")
        mock_download.assert_not_called()
        assert "already present" in mock_logger.info.call_args[0][0]

    @patch("nemo_curator.models.qwen_lm._weights_complete", return_value=False)
    @patch("nemo_curator.models.qwen_lm.download_model_from_hf")
    @patch("nemo_curator.models.qwen_lm.logger")
    def test_downloads_when_weights_missing(
        self, mock_logger: Mock, mock_download: Mock, mock_complete: Mock, tmp_path: Path
    ) -> None:
        QwenLM.download_weights_on_node(str(tmp_path), variant="qwen2.5")
        expected_model_id = _QWEN_LM_VARIANTS_INFO["qwen2.5"][0]
        mock_download.assert_called_once_with(
            model_id=expected_model_id,
            local_dir=tmp_path / expected_model_id,
            revision=_QWEN_LM_VARIANTS_INFO["qwen2.5"][1],
        )
        assert "downloaded" in mock_logger.info.call_args[0][0]
