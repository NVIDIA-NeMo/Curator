"""Unit tests for QwenLM model."""

import sys
from unittest.mock import Mock, patch

# Mock vllm before importing the module to avoid dependency issues
sys.modules["vllm"] = Mock()
sys.modules["transformers"] = Mock()

from ray_curator.models.qwen_lm import _QWEN_LM_MODEL_ID, QwenLM  # noqa: E402


class TestQwenLM:
    """Test cases for QwenLM model class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
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

    def test_constants(self) -> None:
        """Test that module constants are correctly defined."""
        assert _QWEN_LM_MODEL_ID == "Qwen/Qwen2.5-14B-Instruct"

    def test_initialization(self) -> None:
        """Test initialization with parameters."""
        assert self.qwen_lm.model_dir == self.model_dir
        assert self.qwen_lm.caption_batch_size == self.caption_batch_size
        assert self.qwen_lm.fp8 == self.fp8
        assert self.qwen_lm.max_output_tokens == self.max_output_tokens

    def test_model_id_names(self) -> None:
        """Test model_id_names method returns correct model IDs."""
        model_ids = self.qwen_lm.model_id_names()
        assert isinstance(model_ids, list)
        assert len(model_ids) == 1
        assert model_ids[0] == _QWEN_LM_MODEL_ID

    @patch("ray_curator.models.qwen_lm.AutoTokenizer")
    @patch("ray_curator.models.qwen_lm.SamplingParams")
    @patch("ray_curator.models.qwen_lm.LLM")
    @patch("ray_curator.models.qwen_lm.Path")
    def test_setup_with_fp8(
        self, mock_path: Mock, mock_llm_class: Mock, mock_sampling_params_class: Mock, mock_tokenizer_class: Mock
    ) -> None:
        """Test setup method with fp8 quantization enabled."""
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
            enforce_eager=False,
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

    @patch("ray_curator.models.qwen_lm.AutoTokenizer")
    @patch("ray_curator.models.qwen_lm.SamplingParams")
    @patch("ray_curator.models.qwen_lm.LLM")
    @patch("ray_curator.models.qwen_lm.Path")
    def test_setup_without_fp8(
        self, mock_path: Mock, mock_llm_class: Mock, mock_sampling_params_class: Mock, mock_tokenizer_class: Mock
    ) -> None:
        """Test setup method without fp8 quantization."""
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
            enforce_eager=False,
        )

    def test_generate_single_input(self) -> None:
        """Test generate method with single input."""
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
        """Test generate method with multiple inputs."""
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
        """Test generate method with empty input list."""
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
        """Test that weight file path is constructed correctly."""
        with patch("ray_curator.models.qwen_lm.Path") as mock_path:
            mock_path_instance = Mock()
            expected_path = "/test/model/dir/Qwen/Qwen2.5-14B-Instruct"
            mock_path_instance.__truediv__ = Mock(return_value=expected_path)
            mock_path.return_value = mock_path_instance

            with (
                patch("ray_curator.models.qwen_lm.LLM"),
                patch("ray_curator.models.qwen_lm.SamplingParams"),
                patch("ray_curator.models.qwen_lm.AutoTokenizer"),
            ):
                self.qwen_lm.setup()

                # Verify Path was called correctly
                mock_path.assert_called_once_with(self.model_dir)
                mock_path_instance.__truediv__.assert_called_once_with(_QWEN_LM_MODEL_ID)
                assert self.qwen_lm.weight_file == expected_path

    def test_sampling_params_configuration(self) -> None:
        """Test that sampling parameters are configured correctly."""
        with (
            patch("ray_curator.models.qwen_lm.LLM"),
            patch("ray_curator.models.qwen_lm.AutoTokenizer"),
            patch("ray_curator.models.qwen_lm.Path") as mock_path,
            patch("ray_curator.models.qwen_lm.SamplingParams") as mock_sampling_params_class,
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
