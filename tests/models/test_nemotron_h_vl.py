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

from nemo_curator.models.nemotron_h_vl import NemotronHVL


class TestNemotronHVL:
    def setup_method(self) -> None:
        """Set up model and mocks for each test."""
        self.vllm_patcher = patch("nemo_curator.models.nemotron_h_vl.VLLM_AVAILABLE", True)
        self.vllm_patcher.start()

        self.model_dir = "/test/model/dir"
        self.model_variant = "nemotron"
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
        assert model.caption_batch_size == 8
        assert model.max_output_tokens == 512
        assert model.stage2_prompt == "Please refine this caption: "
        assert model.verbose is False
        assert model.weight_file == self.model_dir

    def test_model_id_names_local(self) -> None:
        """Return local directory name when no HF id is set."""
        local_dir = "/models/local_ckpt"
        model = NemotronHVL(model_dir=local_dir)
        assert model.model_id_names == [pathlib.Path(local_dir).name]

    def test_model_id_names_config(self) -> None:
        """Read name from config.json when present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = pathlib.Path(tmpdir) / "config.json"
            cfg_path.write_text('{"_name_or_path": "nemotron-hnano"}')
            model = NemotronHVL(model_dir=tmpdir)
            assert model.model_id_names == ["nemotron-hnano"]

    def test_setup_no_vllm(self) -> None:
        """Raise ImportError when vLLM is unavailable."""
        with patch("nemo_curator.models.nemotron_h_vl.VLLM_AVAILABLE", False):
            model = NemotronHVL(model_dir=self.model_dir)
            with pytest.raises(ImportError, match="vllm is required for NemotronHVL"):
                model.setup()

    @patch("nemo_curator.models.nemotron_h_vl.logger")
    @patch("nemo_curator.models.nemotron_h_vl.SamplingParams")
    @patch("nemo_curator.models.nemotron_h_vl.LLM")
    def test_setup(self, mock_llm: Mock, mock_sampling_params: Mock, mock_logger: Mock) -> None:
        """Initialize LLM and SamplingParams with expected args."""
        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance
        mock_sampling_params_instance = Mock()
        mock_sampling_params.return_value = mock_sampling_params_instance

        self.model.setup()

        mock_llm.assert_called_once_with(
            model=self.model.weight_file,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=32768,
            limit_mm_per_prompt={"video": 1},
        )
        mock_sampling_params.assert_called_once_with(
            temperature=0.6,
            max_tokens=self.model.max_output_tokens,
            top_p=0.95,
            stop=["</s>", "<|endoftext|>", "<SPECIAL_12>", "</think>"],
        )
        assert self.model.model is mock_llm_instance
        assert self.model.sampling_params is mock_sampling_params_instance
        mock_logger.info.assert_any_call("Using vLLM V0 engine.")

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

    def test_download_weights_missing_path(self) -> None:
        """Raise when checkpoint path does not exist."""
        with pytest.raises(FileNotFoundError):
            NemotronHVL.download_weights_on_node("/non/existent/path")

    def test_download_weights_no_files(self) -> None:
        """Raise when checkpoint path has no model files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                NemotronHVL.download_weights_on_node(tmpdir)

    def test_download_weights_ok(self) -> None:
        """Succeed when checkpoint path has model files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pathlib.Path(tmpdir, "model.safetensors").write_bytes(b"fake")
            NemotronHVL.download_weights_on_node(tmpdir)
