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

import json
import pathlib
import tempfile
from unittest.mock import Mock, patch

import pytest

from nemo_curator.models.nemotron_3_nano_omni import (
    _HF_MODEL_ID,
    _HF_REVISION,
    Nemotron3NanoOmni,
)


class TestNemotron3NanoOmniConstants:
    def test_hf_model_id_is_set(self) -> None:
        assert _HF_MODEL_ID == "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning"

    def test_hf_revision_is_set(self) -> None:
        assert _HF_REVISION is not None
        assert len(_HF_REVISION) > 0


class TestNemotron3NanoOmniLocalMode:
    """Tests in local-checkpoint mode (_HF_MODEL_ID = None)."""

    def setup_method(self) -> None:
        self._hf_id_patcher = patch("nemo_curator.models.nemotron_3_nano_omni._HF_MODEL_ID", None)
        self._hf_id_patcher.start()
        self.model_dir = "/aot/checkpoints/nemotron_3_nano_omni"
        self.model = Nemotron3NanoOmni(
            model_dir=self.model_dir,
            caption_batch_size=4,
            max_output_tokens=512,
            stage2_prompt_text="Refine: ",
            verbose=False,
        )

    def teardown_method(self) -> None:
        self._hf_id_patcher.stop()

    def test_init_defaults(self) -> None:
        model = Nemotron3NanoOmni(model_dir=self.model_dir)
        assert model.model_dir == self.model_dir
        assert model.caption_batch_size == 8
        assert model.max_output_tokens == 512
        assert model.stage2_prompt == "Please refine this caption: "
        assert model.verbose is False

    def test_init_weight_file_is_model_dir_in_local_mode(self) -> None:
        assert self.model.weight_file == self.model_dir

    def test_model_id_names_empty_in_local_mode(self) -> None:
        assert self.model.model_id_names == []

    @patch("nemo_curator.models.nemotron_3_nano_omni.VLLM_AVAILABLE", True)
    @patch("nemo_curator.models.nemotron_3_nano_omni.multiprocessing")
    @patch("nemo_curator.models.nemotron_3_nano_omni.SamplingParams")
    @patch("nemo_curator.models.nemotron_3_nano_omni.LLM")
    def test_setup_creates_llm(self, mock_llm_cls: Mock, mock_sp_cls: Mock, mock_mp: Mock) -> None:
        mock_mp.get_start_method.return_value = "spawn"
        Nemotron3NanoOmni(model_dir=self.model_dir).setup()
        mock_llm_cls.assert_called_once_with(
            model=self.model_dir,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=131072,
            limit_mm_per_prompt={"video": 1},
            video_pruning_rate=0,
            mamba_ssm_cache_dtype="float32",
        )

    @patch("nemo_curator.models.nemotron_3_nano_omni.VLLM_AVAILABLE", True)
    @patch("nemo_curator.models.nemotron_3_nano_omni.multiprocessing")
    @patch("nemo_curator.models.nemotron_3_nano_omni.SamplingParams")
    @patch("nemo_curator.models.nemotron_3_nano_omni.LLM")
    def test_setup_sets_sampling_params(self, mock_llm_cls: Mock, mock_sp_cls: Mock, mock_mp: Mock) -> None:
        mock_mp.get_start_method.return_value = "spawn"
        Nemotron3NanoOmni(model_dir=self.model_dir).setup()
        mock_sp_cls.assert_called_once_with(
            temperature=0.6,
            max_tokens=512,
            top_p=0.95,
            stop=["<|im_end|>"],
        )

    @patch("nemo_curator.models.nemotron_3_nano_omni.VLLM_AVAILABLE", True)
    @patch("nemo_curator.models.nemotron_3_nano_omni.multiprocessing")
    @patch("nemo_curator.models.nemotron_3_nano_omni.SamplingParams")
    @patch("nemo_curator.models.nemotron_3_nano_omni.LLM")
    def test_setup_forces_spawn_when_not_spawn(self, mock_llm_cls: Mock, mock_sp_cls: Mock, mock_mp: Mock) -> None:
        mock_mp.get_start_method.return_value = "fork"
        Nemotron3NanoOmni(model_dir=self.model_dir).setup()
        mock_mp.set_start_method.assert_called_once_with("spawn", force=True)

    @patch("nemo_curator.models.nemotron_3_nano_omni.VLLM_AVAILABLE", True)
    @patch("nemo_curator.models.nemotron_3_nano_omni.multiprocessing")
    @patch("nemo_curator.models.nemotron_3_nano_omni.SamplingParams")
    @patch("nemo_curator.models.nemotron_3_nano_omni.LLM")
    def test_setup_skips_set_spawn_when_already_spawn(
        self, mock_llm_cls: Mock, mock_sp_cls: Mock, mock_mp: Mock
    ) -> None:
        mock_mp.get_start_method.return_value = "spawn"
        Nemotron3NanoOmni(model_dir=self.model_dir).setup()
        mock_mp.set_start_method.assert_not_called()

    def test_generate_empty_returns_empty(self) -> None:
        assert self.model.generate([]) == []

    def test_refine_caption_prompt_no_video_tag_returns_refinement(self) -> None:
        result = self.model._refine_caption_prompt("no video tag here", "refined text")
        assert result == "refined text"

    def test_refine_caption_prompt_with_video_tag(self) -> None:
        original = "prefix<video>suffix<|im_end|>rest"
        result = self.model._refine_caption_prompt(original, "refined")
        assert result.startswith("prefix<video>")
        assert "refined" in result
        assert "<|im_end|>" in result

    def test_generate_calls_model_generate(self) -> None:
        mock_output = Mock()
        mock_output.outputs = [Mock(text="A caption")]
        self.model.model = Mock()
        self.model.model.generate.return_value = [mock_output]
        self.model.sampling_params = Mock()

        item = {
            "prompt": "<formatted>",
            "multi_modal_data": {"video": (None, {"fps": 2.0})},
        }
        result = self.model.generate([item])
        assert result == ["A caption"]
        self.model.model.generate.assert_called_once()

    def test_generate_stage2_calls_model_generate_twice(self) -> None:
        mock_out1 = Mock()
        mock_out1.outputs = [Mock(text="Initial")]
        mock_out2 = Mock()
        mock_out2.outputs = [Mock(text="Refined")]
        self.model.model = Mock()
        self.model.model.generate.side_effect = [[mock_out1], [mock_out2]]
        self.model.sampling_params = Mock()

        item = {
            "prompt": "prefix<video>suffix<|im_end|>",
            "multi_modal_data": {"video": (None, {"fps": 1.0})},
        }
        result = self.model.generate([item], generate_stage2_caption=True)
        assert result == ["Refined"]
        assert self.model.model.generate.call_count == 2

    @patch("nemo_curator.models.nemotron_3_nano_omni.logger")
    def test_generate_error_logs_and_raises(self, mock_logger: Mock) -> None:
        self.model.model = Mock()
        self.model.model.generate.side_effect = RuntimeError("boom")
        self.model.sampling_params = Mock()

        item = {"prompt": "Q", "multi_modal_data": {"video": None}}
        with pytest.raises(RuntimeError, match="boom"):
            self.model.generate([item])
        mock_logger.error.assert_called_once()

    @patch("nemo_curator.models.nemotron_3_nano_omni.logger")
    def test_download_weights_skips_in_local_mode(self, mock_logger: Mock) -> None:
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("nemo_curator.models.nemotron_3_nano_omni.download_model_from_hf") as mock_dl,
        ):
            Nemotron3NanoOmni.download_weights_on_node(tmpdir)
            mock_dl.assert_not_called()
        mock_logger.info.assert_called_once()
        assert "local checkpoint" in mock_logger.info.call_args[0][0]


class TestNemotron3NanoOmniHFMode:
    """Tests for when _HF_MODEL_ID is set (future HF release mode)."""

    _MOCK_HF_ID = "nvidia/Nemotron-3-Nano-Omni-30B"
    _MOCK_REVISION = "abc123"

    def setup_method(self) -> None:
        self.hf_id_patcher = patch("nemo_curator.models.nemotron_3_nano_omni._HF_MODEL_ID", self._MOCK_HF_ID)
        self.hf_id_patcher.start()
        self.model_dir = "/test/models"

    def teardown_method(self) -> None:
        self.hf_id_patcher.stop()

    def test_init_weight_file_under_model_dir_in_hf_mode(self) -> None:
        model = Nemotron3NanoOmni(model_dir=self.model_dir)
        expected = str(pathlib.Path(self.model_dir) / self._MOCK_HF_ID)
        assert model.weight_file == expected

    def test_model_id_names_returns_hf_id(self) -> None:
        model = Nemotron3NanoOmni(model_dir=self.model_dir)
        assert model.model_id_names == [self._MOCK_HF_ID]

    @patch("nemo_curator.models.nemotron_3_nano_omni._HF_REVISION", _MOCK_REVISION)
    @patch("nemo_curator.models.nemotron_3_nano_omni.download_model_from_hf")
    def test_download_weights_calls_hf_download(self, mock_dl: Mock) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            expected_path = pathlib.Path(tmpdir) / self._MOCK_HF_ID
            expected_path.mkdir(parents=True)
            (expected_path / "config.json").write_text(
                '{"architectures": ["NemotronH_Nano_Omni_Reasoning_V3"], "model_type": "NemotronH_Nano_Omni_Reasoning_V3"}'
            )
            Nemotron3NanoOmni.download_weights_on_node(tmpdir)
            mock_dl.assert_called_once_with(
                model_id=self._MOCK_HF_ID,
                local_dir=expected_path,
                revision=self._MOCK_REVISION,
            )

    @patch("nemo_curator.models.nemotron_3_nano_omni.download_model_from_hf")
    def test_download_weights_skips_if_safetensors_exist(self, mock_dl: Mock) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            existing = pathlib.Path(tmpdir) / self._MOCK_HF_ID
            existing.mkdir(parents=True)
            (existing / "model.safetensors").write_bytes(b"fake")
            (existing / "config.json").write_text(
                '{"architectures": ["NemotronH_Nano_Omni_Reasoning_V3"], "model_type": "NemotronH_Nano_Omni_Reasoning_V3"}'
            )
            Nemotron3NanoOmni.download_weights_on_node(tmpdir)
            mock_dl.assert_not_called()

    @patch("nemo_curator.models.nemotron_3_nano_omni._HF_REVISION", _MOCK_REVISION)
    @patch("nemo_curator.models.nemotron_3_nano_omni.download_model_from_hf")
    def test_download_weights_patches_config_after_download(self, mock_dl: Mock) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            expected_path = pathlib.Path(tmpdir) / self._MOCK_HF_ID
            expected_path.mkdir(parents=True)
            (expected_path / "config.json").write_text(
                '{"architectures": ["NemotronH_Nano_Omni_Reasoning_V3"], "model_type": "NemotronH_Nano_Omni_Reasoning_V3"}'
            )
            Nemotron3NanoOmni.download_weights_on_node(tmpdir)
            cfg = json.loads((expected_path / "config.json").read_text())
            assert cfg["architectures"] == ["NemotronH_Nano_VL_V2"]
            assert cfg["model_type"] == "NemotronH_Nano_VL_V2"

    @patch("nemo_curator.models.nemotron_3_nano_omni.download_model_from_hf")
    def test_patch_config_also_applied_when_weights_already_exist(self, mock_dl: Mock) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            existing = pathlib.Path(tmpdir) / self._MOCK_HF_ID
            existing.mkdir(parents=True)
            (existing / "model.safetensors").write_bytes(b"fake")
            (existing / "config.json").write_text(
                '{"architectures": ["NemotronH_Nano_Omni_Reasoning_V3"], "model_type": "NemotronH_Nano_Omni_Reasoning_V3"}'
            )
            Nemotron3NanoOmni.download_weights_on_node(tmpdir)
            cfg = json.loads((existing / "config.json").read_text())
            assert cfg["architectures"] == ["NemotronH_Nano_VL_V2"]
            assert cfg["model_type"] == "NemotronH_Nano_VL_V2"
