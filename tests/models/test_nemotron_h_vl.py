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

from __future__ import annotations

import pathlib
import tempfile
from unittest.mock import Mock, patch

import pytest

from nemo_curator.models.nemotron_h_vl import (
    _NEMOTRON_REVISION_INFO,
    _NEMOTRON_VARIANTS_INFO,
    NemotronHVL,
    NemotronVariant,
)


class TestNemotronHVLVariantInfo:
    """Test cases for variant info constants."""

    def test_variants_info_contains_all_variants(self) -> None:
        """Verify all expected variants are defined."""
        expected_variants = {"nemotron", "nemotron-bf16", "nemotron-fp8", "nemotron-nvfp4"}
        assert set(_NEMOTRON_VARIANTS_INFO.keys()) == expected_variants

    def test_variants_info_hf_ids(self) -> None:
        """Verify HuggingFace model IDs are correct."""
        assert _NEMOTRON_VARIANTS_INFO["nemotron"] == "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
        assert _NEMOTRON_VARIANTS_INFO["nemotron-bf16"] == "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
        assert _NEMOTRON_VARIANTS_INFO["nemotron-fp8"] == "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8"
        assert _NEMOTRON_VARIANTS_INFO["nemotron-nvfp4"] == "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD"

    def test_revision_info_contains_all_variants(self) -> None:
        """Verify revision info has entries for all variants."""
        expected_variants = {"nemotron", "nemotron-bf16", "nemotron-fp8", "nemotron-nvfp4"}
        assert set(_NEMOTRON_REVISION_INFO.keys()) == expected_variants


class TestNemotronHVL:
    def setup_method(self) -> None:
        """Set up model and mocks for each test."""
        self.vllm_patcher = patch("nemo_curator.models.nemotron_h_vl.VLLM_AVAILABLE", True)
        self.vllm_patcher.start()

        self.model_dir = "/test/model/dir"
        self.model_variant: NemotronVariant = "nemotron"
        self.caption_batch_size = 4
        self.model = NemotronHVL(
            model_dir=self.model_dir,
            model_variant=self.model_variant,
            caption_batch_size=self.caption_batch_size,
            max_output_tokens=512,
            stage2_prompt_text="Test stage2: ",
            verbose=False,
        )

    def teardown_method(self) -> None:
        """Tear down mocks after each test."""
        self.vllm_patcher.stop()

    def test_init_defaults(self) -> None:
        """Verify default initialization values."""
        model = NemotronHVL(model_dir=self.model_dir)
        assert model.model_dir == self.model_dir
        assert model.model_variant == "nemotron"
        assert model._normalized_variant == "nemotron-bf16"
        assert model.caption_batch_size == 8
        assert model.max_output_tokens == 512
        assert model.stage2_prompt == "Please refine this caption: "
        assert model.verbose is False
        # Weight file should include HF model ID path
        expected_hf_id = _NEMOTRON_VARIANTS_INFO["nemotron-bf16"]
        assert model.weight_file == str(pathlib.Path(self.model_dir) / expected_hf_id)

    def test_init_bf16_variant(self) -> None:
        """Verify BF16 variant initialization."""
        model = NemotronHVL(model_dir=self.model_dir, model_variant="nemotron-bf16")
        assert model.model_variant == "nemotron-bf16"
        assert model._normalized_variant == "nemotron-bf16"
        expected_hf_id = _NEMOTRON_VARIANTS_INFO["nemotron-bf16"]
        assert model.weight_file == str(pathlib.Path(self.model_dir) / expected_hf_id)

    def test_init_fp8_variant(self) -> None:
        """Verify FP8 variant initialization."""
        model = NemotronHVL(model_dir=self.model_dir, model_variant="nemotron-fp8")
        assert model.model_variant == "nemotron-fp8"
        assert model._normalized_variant == "nemotron-fp8"
        expected_hf_id = _NEMOTRON_VARIANTS_INFO["nemotron-fp8"]
        assert model.weight_file == str(pathlib.Path(self.model_dir) / expected_hf_id)

    def test_init_nvfp4_variant(self) -> None:
        """Verify NVFP4 variant initialization."""
        model = NemotronHVL(model_dir=self.model_dir, model_variant="nemotron-nvfp4")
        assert model.model_variant == "nemotron-nvfp4"
        assert model._normalized_variant == "nemotron-nvfp4"
        expected_hf_id = _NEMOTRON_VARIANTS_INFO["nemotron-nvfp4"]
        assert model.weight_file == str(pathlib.Path(self.model_dir) / expected_hf_id)

    def test_init_invalid_variant(self) -> None:
        """Raise ValueError for invalid variant."""
        with pytest.raises(ValueError, match="Invalid model_variant: invalid"):
            NemotronHVL(model_dir=self.model_dir, model_variant="invalid")  # type: ignore[arg-type]

    def test_model_id_names(self) -> None:
        """Return HuggingFace model ID for selected variant."""
        model = NemotronHVL(model_dir=self.model_dir, model_variant="nemotron-fp8")
        expected_hf_id = _NEMOTRON_VARIANTS_INFO["nemotron-fp8"]
        assert model.model_id_names == [expected_hf_id]

    def test_setup_no_vllm(self) -> None:
        """Raise ImportError when vLLM is unavailable."""
        with patch("nemo_curator.models.nemotron_h_vl.VLLM_AVAILABLE", False):
            model = NemotronHVL(model_dir=self.model_dir)
            with pytest.raises(ImportError, match="vllm is required for NemotronHVL"):
                model.setup()

    @patch("nemo_curator.models.nemotron_h_vl.SamplingParams")
    @patch("nemo_curator.models.nemotron_h_vl.LLM")
    def test_setup_bf16(self, mock_llm: Mock, mock_sampling_params: Mock) -> None:
        """Initialize LLM with BF16 settings (no quantization)."""
        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance
        mock_sampling_params_instance = Mock()
        mock_sampling_params.return_value = mock_sampling_params_instance

        model = NemotronHVL(model_dir=self.model_dir, model_variant="nemotron-bf16")
        model.setup()

        mock_llm.assert_called_once_with(
            model=model.weight_file,
            trust_remote_code=True,
            dtype="bfloat16",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=32768,
            limit_mm_per_prompt={"video": 1},
            quantization=None,
            video_pruning_rate=0,
        )
        mock_sampling_params.assert_called_once_with(
            temperature=0.6,
            max_tokens=model.max_output_tokens,
            top_p=0.95,
            stop=["</s>", "<|endoftext|>", "<SPECIAL_12>", "</think>"],
        )
        assert model.model is mock_llm_instance
        assert model.sampling_params is mock_sampling_params_instance

    @patch("nemo_curator.models.nemotron_h_vl.SamplingParams")
    @patch("nemo_curator.models.nemotron_h_vl.LLM")
    def test_setup_fp8(self, mock_llm: Mock, mock_sampling_params: Mock) -> None:
        """Initialize LLM with FP8 quantization settings."""
        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance
        mock_sampling_params.return_value = Mock()

        model = NemotronHVL(model_dir=self.model_dir, model_variant="nemotron-fp8")
        model.setup()

        mock_llm.assert_called_once_with(
            model=model.weight_file,
            trust_remote_code=True,
            dtype="auto",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=32768,
            limit_mm_per_prompt={"video": 1},
            quantization="modelopt",
            video_pruning_rate=0,
        )

    @patch("nemo_curator.models.nemotron_h_vl.SamplingParams")
    @patch("nemo_curator.models.nemotron_h_vl.LLM")
    def test_setup_nvfp4(self, mock_llm: Mock, mock_sampling_params: Mock) -> None:
        """Initialize LLM with NVFP4 quantization settings."""
        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance
        mock_sampling_params.return_value = Mock()

        model = NemotronHVL(model_dir=self.model_dir, model_variant="nemotron-nvfp4")
        model.setup()

        mock_llm.assert_called_once_with(
            model=model.weight_file,
            trust_remote_code=True,
            dtype="auto",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=32768,
            limit_mm_per_prompt={"video": 1},
            quantization="modelopt_fp4",
            video_pruning_rate=0,
        )

    @patch("nemo_curator.models.nemotron_h_vl.grouping.split_by_chunk_size")
    def test_generate_basic(self, mock_split_by_chunk_size: Mock) -> None:
        """Generate captions for a single batch without stage2."""
        mock_model = Mock()
        self.model.model = mock_model
        self.model.sampling_params = Mock()

        videos = [
            {"prompt": "Describe this video", "multi_modal_data": {"video": "video1"}},
            {"prompt": "What is happening?", "multi_modal_data": {"video": "video2"}},
        ]
        mock_split_by_chunk_size.return_value = [videos]

        out1, out2 = Mock(), Mock()
        out1.outputs = [Mock(text="Text 1")]
        out2.outputs = [Mock(text="Text 2")]
        mock_model.generate.return_value = [out1, out2]

        result = self.model.generate(videos, batch_size=16)

        assert result == ["Text 1", "Text 2"]
        mock_split_by_chunk_size.assert_called_once_with(videos, 16)
        assert mock_model.generate.call_count == 1
        args, kwargs = mock_model.generate.call_args
        assert args[0] == list(videos)
        assert kwargs["sampling_params"] == self.model.sampling_params
        assert kwargs["use_tqdm"] is False

    @patch("nemo_curator.models.nemotron_h_vl.grouping.split_by_chunk_size")
    def test_generate_batched(self, mock_split_by_chunk_size: Mock) -> None:
        """Generate captions across multiple batches."""
        mock_model = Mock()
        self.model.model = mock_model
        self.model.sampling_params = Mock()

        videos = [{"prompt": f"Video {i}", "multi_modal_data": {"video": f"video{i}"}} for i in range(4)]
        batch1, batch2 = videos[:2], videos[2:]
        mock_split_by_chunk_size.return_value = [batch1, batch2]

        def gen_outputs(batch_idx: int) -> list[Mock]:
            return [Mock(outputs=[Mock(text=f"B{batch_idx} T{i}")]) for i in range(2)]

        mock_model.generate.side_effect = [gen_outputs(1), gen_outputs(2)]

        result = self.model.generate(videos, batch_size=2)
        assert result == ["B1 T0", "B1 T1", "B2 T0", "B2 T1"]
        assert mock_model.generate.call_count == 2

    @patch("nemo_curator.models.nemotron_h_vl.grouping.split_by_chunk_size")
    def test_generate_stage2(self, mock_split_by_chunk_size: Mock) -> None:
        """Generate with stage2 refinement flow."""
        mock_model = Mock()
        self.model.model = mock_model
        self.model.sampling_params = Mock()

        videos = [{"prompt": "Human: <video>test<SPECIAL_11>Assistant", "multi_modal_data": {"video": "video1"}}]
        mock_split_by_chunk_size.return_value = [videos]

        out_stage1, out_stage2 = Mock(), Mock()
        out_stage1.outputs = [Mock(text="Stage 1")]
        out_stage2.outputs = [Mock(text="Stage 2")]
        mock_model.generate.side_effect = [[out_stage1], [out_stage2]]

        result = self.model.generate(videos, generate_stage2_caption=True, batch_size=16)
        assert result == ["Stage 2"]

        assert mock_model.generate.call_count == 2
        second_args, _ = mock_model.generate.call_args
        updated_inputs = second_args[0]
        assert "Test stage2: Stage 1" in updated_inputs[0]["prompt"]
        assert updated_inputs[0]["multi_modal_data"]["video"] == "video1"

    def test_generate_empty(self) -> None:
        """Return empty when no videos are provided."""
        assert self.model.generate([]) == []

    @patch("nemo_curator.models.nemotron_h_vl.logger")
    @patch("nemo_curator.models.nemotron_h_vl.grouping.split_by_chunk_size")
    def test_generate_error(self, mock_split_by_chunk_size: Mock, mock_logger: Mock) -> None:
        """Log and raise on generation errors."""
        mock_model = Mock()
        self.model.model = mock_model
        self.model.sampling_params = Mock()

        videos = [{"prompt": "Test", "multi_modal_data": {"video": "video1"}}]
        mock_split_by_chunk_size.return_value = [videos]

        mock_model.generate.side_effect = Exception("Test error")

        with pytest.raises(Exception, match="Test error"):
            self.model.generate(videos)

        mock_logger.error.assert_called_once()

    @patch("nemo_curator.models.nemotron_h_vl.download_model_from_hf")
    def test_download_weights_bf16(self, mock_download: Mock) -> None:
        """Download BF16 weights from HuggingFace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            NemotronHVL.download_weights_on_node(tmpdir, variant="nemotron-bf16")

            expected_hf_id = _NEMOTRON_VARIANTS_INFO["nemotron-bf16"]
            expected_path = pathlib.Path(tmpdir) / expected_hf_id

            mock_download.assert_called_once_with(
                model_id=expected_hf_id,
                local_dir=expected_path,
                revision=_NEMOTRON_REVISION_INFO["nemotron-bf16"],
            )

    @patch("nemo_curator.models.nemotron_h_vl.download_model_from_hf")
    def test_download_weights_fp8(self, mock_download: Mock) -> None:
        """Download FP8 weights from HuggingFace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            NemotronHVL.download_weights_on_node(tmpdir, variant="nemotron-fp8")

            expected_hf_id = _NEMOTRON_VARIANTS_INFO["nemotron-fp8"]
            expected_path = pathlib.Path(tmpdir) / expected_hf_id

            mock_download.assert_called_once_with(
                model_id=expected_hf_id,
                local_dir=expected_path,
                revision=_NEMOTRON_REVISION_INFO["nemotron-fp8"],
            )

    @patch("nemo_curator.models.nemotron_h_vl.download_model_from_hf")
    def test_download_weights_nvfp4(self, mock_download: Mock) -> None:
        """Download NVFP4 weights from HuggingFace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            NemotronHVL.download_weights_on_node(tmpdir, variant="nemotron-nvfp4")

            expected_hf_id = _NEMOTRON_VARIANTS_INFO["nemotron-nvfp4"]
            expected_path = pathlib.Path(tmpdir) / expected_hf_id

            mock_download.assert_called_once_with(
                model_id=expected_hf_id,
                local_dir=expected_path,
                revision=_NEMOTRON_REVISION_INFO["nemotron-nvfp4"],
            )

    @patch("nemo_curator.models.nemotron_h_vl.download_model_from_hf")
    def test_download_weights_default_variant(self, mock_download: Mock) -> None:
        """Download default variant (nemotron -> bf16) from HuggingFace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            NemotronHVL.download_weights_on_node(tmpdir, variant="nemotron")

            expected_hf_id = _NEMOTRON_VARIANTS_INFO["nemotron"]
            expected_path = pathlib.Path(tmpdir) / expected_hf_id

            mock_download.assert_called_once_with(
                model_id=expected_hf_id,
                local_dir=expected_path,
                revision=_NEMOTRON_REVISION_INFO["nemotron"],
            )

    def test_download_weights_already_exists(self) -> None:
        """Skip download when weights already exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the expected directory with a safetensors file
            expected_hf_id = _NEMOTRON_VARIANTS_INFO["nemotron-bf16"]
            model_path = pathlib.Path(tmpdir) / expected_hf_id
            model_path.mkdir(parents=True)
            (model_path / "model.safetensors").write_bytes(b"fake")

            with patch("nemo_curator.models.nemotron_h_vl.download_model_from_hf") as mock_download:
                NemotronHVL.download_weights_on_node(tmpdir, variant="nemotron-bf16")
                mock_download.assert_not_called()

    def test_download_weights_invalid_variant(self) -> None:
        """Raise ValueError for invalid variant."""
        with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(ValueError, match="Invalid variant: invalid"):
            NemotronHVL.download_weights_on_node(tmpdir, variant="invalid")  # type: ignore[arg-type]

    def test_refine_caption_prompt_with_video_tag(self) -> None:
        """Test _refine_caption_prompt with video tag in prompt."""
        original_prompt = "Human: <video>original question<SPECIAL_11>Assistant"
        refinement_text = "Please refine: initial caption"

        result = self.model._refine_caption_prompt(original_prompt, refinement_text)

        assert "<video>" in result
        assert "Please refine: initial caption" in result
        assert "<SPECIAL_11>Assistant" in result

    def test_refine_caption_prompt_without_video_tag(self) -> None:
        """Test _refine_caption_prompt without video tag returns refinement text."""
        original_prompt = "Human: original question"
        refinement_text = "Please refine: initial caption"

        result = self.model._refine_caption_prompt(original_prompt, refinement_text)

        assert result == refinement_text
