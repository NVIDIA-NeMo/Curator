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

import pytest

from nemo_curator.models.qwen_lm import _QWEN_LM_MODEL_ID, QwenLM, _check_vllm_supports_model


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
            caption_batch_size=self.caption_batch_size,
            fp8=self.fp8,
            max_output_tokens=self.max_output_tokens,
        )

    def test_default_model(self) -> None:
        assert (
            QwenLM(
                model_dir=self.model_dir,
                caption_batch_size=self.caption_batch_size,
                fp8=self.fp8,
                max_output_tokens=self.max_output_tokens,
            ).model_id
            == "Qwen/Qwen3-14B"
        )

    def test_initialization(self) -> None:
        assert self.qwen_lm.model_dir == self.model_dir
        assert self.qwen_lm.caption_batch_size == self.caption_batch_size
        assert self.qwen_lm.fp8 == self.fp8
        assert self.qwen_lm.max_output_tokens == self.max_output_tokens
        assert self.qwen_lm.model_id == _QWEN_LM_MODEL_ID
        assert self.qwen_lm.model_revision is None

    def test_model_id_names(self) -> None:
        model_ids = self.qwen_lm.model_id_names()
        assert isinstance(model_ids, list)
        assert len(model_ids) == 1
        assert model_ids[0] == _QWEN_LM_MODEL_ID

    def test_custom_model_id(self) -> None:
        custom_model = "Qwen/Qwen3-8B"
        qwen_lm = QwenLM(
            model_dir=self.model_dir,
            caption_batch_size=self.caption_batch_size,
            fp8=self.fp8,
            max_output_tokens=self.max_output_tokens,
            model_id=custom_model,
        )
        assert qwen_lm.model_id == custom_model
        assert qwen_lm.model_id_names() == [custom_model]

    def test_invalid_model_id_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a Qwen model"):
            QwenLM(
                model_dir=self.model_dir,
                caption_batch_size=self.caption_batch_size,
                fp8=self.fp8,
                max_output_tokens=self.max_output_tokens,
                model_id="mistralai/Mistral-7B-Instruct",
            )

    @patch("nemo_curator.models.qwen_lm._check_vllm_supports_model")
    @patch("nemo_curator.models.qwen_lm.AutoTokenizer")
    @patch("nemo_curator.models.qwen_lm.SamplingParams")
    @patch("nemo_curator.models.qwen_lm.LLM")
    @patch("nemo_curator.models.qwen_lm.Path")
    def test_setup_with_fp8(
        self,
        mock_path: Mock,
        mock_llm_class: Mock,
        mock_sampling_params_class: Mock,
        mock_tokenizer_class: Mock,
        mock_check_vllm: Mock,
    ) -> None:
        expected_weight_path = "/test/model/dir/Qwen/Qwen3-14B"
        mock_path_instance = Mock()
        mock_path_instance.__truediv__ = Mock(return_value=expected_weight_path)
        # Simulate weights already present so auto-download is skipped
        mock_path_instance.exists.return_value = True
        mock_path_instance.glob.return_value = ["model.safetensors"]
        mock_path.return_value = mock_path_instance

        mock_llm = Mock()
        mock_llm_class.return_value = mock_llm
        mock_sampling_params = Mock()
        mock_sampling_params_class.return_value = mock_sampling_params
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        self.qwen_lm.setup()

        mock_check_vllm.assert_called_once_with(self.qwen_lm.model_id)
        mock_llm_class.assert_called_once_with(
            model=expected_weight_path,
            quantization="fp8",
            enforce_eager=False,
        )
        mock_sampling_params_class.assert_called_once_with(
            temperature=0.1,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=self.max_output_tokens,
            stop_token_ids=[],
        )
        mock_tokenizer_class.from_pretrained.assert_called_once_with(expected_weight_path)
        assert self.qwen_lm.weight_file == expected_weight_path
        assert self.qwen_lm.llm == mock_llm
        assert self.qwen_lm.sampling_params == mock_sampling_params
        assert self.qwen_lm.tokenizer == mock_tokenizer

    @patch("nemo_curator.models.qwen_lm._check_vllm_supports_model")
    @patch("nemo_curator.models.qwen_lm.AutoTokenizer")
    @patch("nemo_curator.models.qwen_lm.SamplingParams")
    @patch("nemo_curator.models.qwen_lm.LLM")
    @patch("nemo_curator.models.qwen_lm.Path")
    def test_setup_without_fp8(
        self,
        mock_path: Mock,
        mock_llm_class: Mock,
        mock_sampling_params_class: Mock,
        mock_tokenizer_class: Mock,
        mock_check_vllm: Mock,
    ) -> None:
        qwen_lm_no_fp8 = QwenLM(
            model_dir=self.model_dir,
            caption_batch_size=self.caption_batch_size,
            fp8=False,
            max_output_tokens=self.max_output_tokens,
        )

        expected_weight_path = "/test/model/dir/Qwen/Qwen3-14B"
        mock_path_instance = Mock()
        mock_path_instance.__truediv__ = Mock(return_value=expected_weight_path)
        mock_path_instance.exists.return_value = True
        mock_path_instance.glob.return_value = ["model.safetensors"]
        mock_path.return_value = mock_path_instance

        mock_llm_class.return_value = Mock()
        mock_sampling_params_class.return_value = Mock()
        mock_tokenizer_class.from_pretrained.return_value = Mock()

        qwen_lm_no_fp8.setup()

        mock_llm_class.assert_called_once_with(
            model=expected_weight_path,
            quantization=None,
            enforce_eager=False,
        )

    @patch("nemo_curator.models.qwen_lm._check_vllm_supports_model")
    @patch("nemo_curator.models.qwen_lm.download_model_from_hf")
    @patch("nemo_curator.models.qwen_lm.AutoTokenizer")
    @patch("nemo_curator.models.qwen_lm.SamplingParams")
    @patch("nemo_curator.models.qwen_lm.LLM")
    @patch("nemo_curator.models.qwen_lm.Path")
    def test_setup_auto_downloads_if_missing(  # noqa: PLR0913
        self,
        mock_path: Mock,
        mock_llm_class: Mock,
        mock_sampling_params_class: Mock,
        mock_tokenizer_class: Mock,
        mock_download: Mock,
        mock_check_vllm: Mock,
    ) -> None:
        mock_path_instance = Mock()
        mock_path_instance.__truediv__ = Mock(return_value=mock_path_instance)
        mock_path_instance.__str__ = Mock(return_value="/test/model/dir/Qwen/Qwen3-14B")
        mock_path_instance.exists.return_value = False
        mock_path_instance.glob.return_value = []
        mock_path_instance.mkdir = Mock()
        mock_path.return_value = mock_path_instance

        mock_llm_class.return_value = Mock()
        mock_sampling_params_class.return_value = Mock()
        mock_tokenizer_class.from_pretrained.return_value = Mock()

        self.qwen_lm.setup()

        mock_download.assert_called_once_with(
            model_id=self.qwen_lm.model_id, local_dir=mock_path_instance, revision=None
        )

    @patch("nemo_curator.models.qwen_lm._check_vllm_supports_model")
    @patch("nemo_curator.models.qwen_lm.download_model_from_hf")
    @patch("nemo_curator.models.qwen_lm.AutoTokenizer")
    @patch("nemo_curator.models.qwen_lm.SamplingParams")
    @patch("nemo_curator.models.qwen_lm.LLM")
    @patch("nemo_curator.models.qwen_lm.Path")
    def test_setup_skips_download_if_weights_present(  # noqa: PLR0913
        self,
        mock_path: Mock,
        mock_llm_class: Mock,
        mock_sampling_params_class: Mock,
        mock_tokenizer_class: Mock,
        mock_download: Mock,
        mock_check_vllm: Mock,
    ) -> None:
        mock_path_instance = Mock()
        mock_path_instance.__truediv__ = Mock(return_value=mock_path_instance)
        mock_path_instance.__str__ = Mock(return_value="/test/model/dir/Qwen/Qwen3-14B")
        mock_path_instance.exists.return_value = True
        mock_path_instance.glob.return_value = ["model.safetensors"]
        mock_path.return_value = mock_path_instance

        mock_llm_class.return_value = Mock()
        mock_sampling_params_class.return_value = Mock()
        mock_tokenizer_class.from_pretrained.return_value = Mock()

        self.qwen_lm.setup()

        mock_download.assert_not_called()

    def test_setup_raises_if_vllm_unsupported(self) -> None:
        with (
            patch(
                "nemo_curator.models.qwen_lm._check_vllm_supports_model",
                side_effect=ValueError("not supported by vLLM"),
            ),
            pytest.raises(ValueError, match="not supported by vLLM"),
        ):
            self.qwen_lm.setup()

    def test_generate_single_input(self) -> None:
        mock_llm = Mock()
        mock_tokenizer = Mock()
        mock_sampling_params = Mock()

        formatted_input = "formatted_prompt"
        mock_tokenizer.apply_chat_template.return_value = formatted_input

        mock_output = Mock()
        mock_output.text = "Generated response"
        mock_result = Mock()
        mock_result.outputs = [mock_output]
        mock_llm.generate.return_value = [mock_result]

        self.qwen_lm.llm = mock_llm
        self.qwen_lm.tokenizer = mock_tokenizer
        self.qwen_lm.sampling_params = mock_sampling_params

        test_input = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, world!"},
        ]

        result = self.qwen_lm.generate([test_input])

        mock_tokenizer.apply_chat_template.assert_called_once_with(
            [test_input], tokenize=False, add_generation_prompt=True
        )
        mock_llm.generate.assert_called_once_with(formatted_input, sampling_params=mock_sampling_params)
        assert result == ["Generated response"]

    def test_generate_multiple_inputs(self) -> None:
        mock_llm = Mock()
        mock_tokenizer = Mock()
        mock_sampling_params = Mock()

        formatted_inputs = ["formatted_prompt_1", "formatted_prompt_2"]
        mock_tokenizer.apply_chat_template.return_value = formatted_inputs

        mock_output_1 = Mock()
        mock_output_1.text = "Response 1"
        mock_result_1 = Mock()
        mock_result_1.outputs = [mock_output_1]

        mock_output_2 = Mock()
        mock_output_2.text = "Response 2"
        mock_result_2 = Mock()
        mock_result_2.outputs = [mock_output_2]

        mock_llm.generate.return_value = [mock_result_1, mock_result_2]

        self.qwen_lm.llm = mock_llm
        self.qwen_lm.tokenizer = mock_tokenizer
        self.qwen_lm.sampling_params = mock_sampling_params

        test_inputs = [[{"role": "user", "content": "First message"}], [{"role": "user", "content": "Second message"}]]

        result = self.qwen_lm.generate(test_inputs)

        mock_tokenizer.apply_chat_template.assert_called_once_with(
            test_inputs, tokenize=False, add_generation_prompt=True
        )
        mock_llm.generate.assert_called_once_with(formatted_inputs, sampling_params=mock_sampling_params)
        assert result == ["Response 1", "Response 2"]

    def test_generate_empty_input(self) -> None:
        mock_llm = Mock()
        mock_tokenizer = Mock()
        mock_sampling_params = Mock()

        mock_tokenizer.apply_chat_template.return_value = []
        mock_llm.generate.return_value = []

        self.qwen_lm.llm = mock_llm
        self.qwen_lm.tokenizer = mock_tokenizer
        self.qwen_lm.sampling_params = mock_sampling_params

        result = self.qwen_lm.generate([])

        mock_tokenizer.apply_chat_template.assert_called_once_with([], tokenize=False, add_generation_prompt=True)
        mock_llm.generate.assert_called_once_with([], sampling_params=mock_sampling_params)
        assert result == []

    def test_generate_with_complex_conversation(self) -> None:
        """Test generate method with complex multi-turn conversation."""
        mock_llm = Mock()
        mock_tokenizer = Mock()
        mock_sampling_params = Mock()

        formatted_input = "complex_formatted_prompt"
        mock_tokenizer.apply_chat_template.return_value = formatted_input

        mock_output = Mock()
        mock_output.text = "Complex response with multiple sentences. This is a detailed answer."
        mock_result = Mock()
        mock_result.outputs = [mock_output]
        mock_llm.generate.return_value = [mock_result]

        self.qwen_lm.llm = mock_llm
        self.qwen_lm.tokenizer = mock_tokenizer
        self.qwen_lm.sampling_params = mock_sampling_params

        test_input = [
            {"role": "system", "content": "You are a video caption enhancer."},
            {"role": "user", "content": "A person walks down the street."},
            {"role": "assistant", "content": "A person is walking down a busy city street."},
            {"role": "user", "content": "Add more details about the environment."},
        ]

        result = self.qwen_lm.generate([test_input])

        assert result == ["Complex response with multiple sentences. This is a detailed answer."]
        mock_tokenizer.apply_chat_template.assert_called_once_with(
            [test_input], tokenize=False, add_generation_prompt=True
        )

    @patch("nemo_curator.models.qwen_lm._check_vllm_supports_model")
    def test_weight_file_path_construction(self, mock_check_vllm: Mock) -> None:
        with patch("nemo_curator.models.qwen_lm.Path") as mock_path:
            mock_path_instance = Mock()
            expected_path = "/test/model/dir/Qwen/Qwen3-14B"
            mock_path_instance.__truediv__ = Mock(return_value=expected_path)
            mock_path_instance.exists.return_value = True
            mock_path_instance.glob.return_value = ["model.safetensors"]
            mock_path.return_value = mock_path_instance

            with (
                patch("nemo_curator.models.qwen_lm.LLM"),
                patch("nemo_curator.models.qwen_lm.SamplingParams"),
                patch("nemo_curator.models.qwen_lm.AutoTokenizer"),
            ):
                self.qwen_lm.setup()

                mock_path.assert_any_call(self.model_dir)
                mock_path_instance.__truediv__.assert_called_once_with(self.qwen_lm.model_id)
                assert self.qwen_lm.weight_file == expected_path

    def test_sampling_params_configuration(self) -> None:
        with (
            patch("nemo_curator.models.qwen_lm._check_vllm_supports_model"),
            patch("nemo_curator.models.qwen_lm.LLM"),
            patch("nemo_curator.models.qwen_lm.AutoTokenizer"),
            patch("nemo_curator.models.qwen_lm.Path") as mock_path,
            patch("nemo_curator.models.qwen_lm.SamplingParams") as mock_sampling_params_class,
        ):
            mock_path_instance = Mock()
            mock_path_instance.__truediv__ = Mock(return_value="test_path")
            mock_path_instance.exists.return_value = True
            mock_path_instance.glob.return_value = ["model.safetensors"]
            mock_path.return_value = mock_path_instance

            mock_sampling_params = Mock()
            mock_sampling_params_class.return_value = mock_sampling_params

            self.qwen_lm.setup()

            mock_sampling_params_class.assert_called_once_with(
                temperature=0.1,
                top_p=0.001,
                repetition_penalty=1.05,
                max_tokens=self.max_output_tokens,
                stop_token_ids=[],
            )

    @patch("nemo_curator.models.qwen_lm.download_model_from_hf")
    def test_download_weights_on_node_custom_model_id(self, mock_download: Mock) -> None:
        custom_model = "Qwen/Qwen3-8B"
        with patch("nemo_curator.models.qwen_lm.Path") as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.__truediv__ = Mock(return_value=mock_path_instance)
            mock_path_instance.exists.return_value = False
            mock_path_instance.glob.return_value = []
            mock_path_instance.mkdir = Mock()
            mock_path.return_value = mock_path_instance

            QwenLM.download_weights_on_node(model_dir="/some/dir", model_id=custom_model)

            mock_download.assert_called_once_with(model_id=custom_model, local_dir=mock_path_instance, revision=None)

    @patch("nemo_curator.models.qwen_lm.download_model_from_hf")
    def test_download_weights_on_node_custom_revision(self, mock_download: Mock) -> None:
        with patch("nemo_curator.models.qwen_lm.Path") as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.__truediv__ = Mock(return_value=mock_path_instance)
            mock_path_instance.exists.return_value = False
            mock_path_instance.glob.return_value = []
            mock_path_instance.mkdir = Mock()
            mock_path.return_value = mock_path_instance

            QwenLM.download_weights_on_node(model_dir="/some/dir", model_revision="abc1234")

            mock_download.assert_called_once_with(
                model_id=_QWEN_LM_MODEL_ID, local_dir=mock_path_instance, revision="abc1234"
            )

    def test_download_weights_on_node_invalid_model_id(self) -> None:
        with pytest.raises(ValueError, match="must be a Qwen model"):
            QwenLM.download_weights_on_node(model_dir="/some/dir", model_id="mistralai/Mistral-7B-Instruct")

    def test_setup_raises_if_vllm_not_available(self) -> None:
        with (
            patch("nemo_curator.models.qwen_lm.VLLM_AVAILABLE", False),
            pytest.raises(ImportError, match="vllm is required"),
        ):
            self.qwen_lm.setup()

    @patch("nemo_curator.models.qwen_lm.download_model_from_hf")
    def test_download_weights_on_node_skips_if_weights_present(self, mock_download: Mock) -> None:
        with patch("nemo_curator.models.qwen_lm.Path") as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.__truediv__ = Mock(return_value=mock_path_instance)
            mock_path_instance.exists.return_value = True
            mock_path_instance.glob.return_value = ["model.safetensors"]
            mock_path.return_value = mock_path_instance

            QwenLM.download_weights_on_node(model_dir="/some/dir")

            mock_download.assert_not_called()

    @patch("nemo_curator.models.qwen_lm.download_model_from_hf")
    def test_download_weights_on_node_default_model(self, mock_download: Mock) -> None:
        with patch("nemo_curator.models.qwen_lm.Path") as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.__truediv__ = Mock(return_value=mock_path_instance)
            mock_path_instance.exists.return_value = False
            mock_path_instance.glob.return_value = []
            mock_path_instance.mkdir = Mock()
            mock_path.return_value = mock_path_instance

            QwenLM.download_weights_on_node(model_dir="/some/dir")

            mock_download.assert_called_once_with(
                model_id=_QWEN_LM_MODEL_ID, local_dir=mock_path_instance, revision=None
            )

    def teardown_method(self) -> None:
        self.vllm_available_patcher.stop()


class TestCheckVllmSupportsModel:
    """Tests for _check_vllm_supports_model helper."""

    def test_skips_when_vllm_not_available(self) -> None:
        with patch("nemo_curator.models.qwen_lm.VLLM_AVAILABLE", False):
            # Should return without raising even for a bogus model id
            _check_vllm_supports_model("NotQwen/fake-model")

    def test_skips_when_no_architectures(self) -> None:
        mock_config = Mock()
        mock_config.architectures = []
        with (
            patch("nemo_curator.models.qwen_lm.VLLM_AVAILABLE", True),
            patch("nemo_curator.models.qwen_lm.AutoConfig") as mock_auto_config,
        ):
            mock_auto_config.from_pretrained.return_value = mock_config
            _check_vllm_supports_model("Qwen/Qwen3-14B")  # no error expected

    def test_skips_when_architectures_is_none(self) -> None:
        mock_config = Mock()
        mock_config.architectures = None
        with (
            patch("nemo_curator.models.qwen_lm.VLLM_AVAILABLE", True),
            patch("nemo_curator.models.qwen_lm.AutoConfig") as mock_auto_config,
        ):
            mock_auto_config.from_pretrained.return_value = mock_config
            _check_vllm_supports_model("Qwen/Qwen3-14B")  # no error expected

    def test_passes_when_all_architectures_supported(self) -> None:
        mock_config = Mock()
        mock_config.architectures = ["Qwen2ForCausalLM"]
        with (
            patch("nemo_curator.models.qwen_lm.VLLM_AVAILABLE", True),
            patch("nemo_curator.models.qwen_lm.AutoConfig") as mock_auto_config,
        ):
            mock_auto_config.from_pretrained.return_value = mock_config
            with patch("nemo_curator.models.qwen_lm.ModelRegistry") as mock_registry:
                mock_registry.is_model_supported.return_value = True
                _check_vllm_supports_model("Qwen/Qwen3-14B")  # no error expected

    def test_raises_when_all_architectures_unsupported(self) -> None:
        mock_config = Mock()
        mock_config.architectures = ["UnknownArch"]
        with (
            patch("nemo_curator.models.qwen_lm.VLLM_AVAILABLE", True),
            patch("nemo_curator.models.qwen_lm.AutoConfig") as mock_auto_config,
        ):
            mock_auto_config.from_pretrained.return_value = mock_config
            with patch("nemo_curator.models.qwen_lm.ModelRegistry") as mock_registry:
                mock_registry.is_model_supported.return_value = False
                with pytest.raises(ValueError, match="not supported by vLLM"):
                    _check_vllm_supports_model("Qwen/Qwen3-14B")

    def test_skips_check_when_registry_import_fails(self) -> None:
        mock_config = Mock()
        mock_config.architectures = ["Qwen2ForCausalLM"]
        with (
            patch("nemo_curator.models.qwen_lm.VLLM_AVAILABLE", True),
            patch("nemo_curator.models.qwen_lm.AutoConfig") as mock_auto_config,
            patch("builtins.__import__", side_effect=ImportError("no module")),
        ):
            mock_auto_config.from_pretrained.return_value = mock_config
            _check_vllm_supports_model("Qwen/Qwen3-14B")  # no error expected
