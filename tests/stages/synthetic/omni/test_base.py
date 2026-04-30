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

"""Unit tests for nemo_curator.stages.synthetic.omni.base."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from nemo_curator.models.omni.base import InferenceConfig
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.synthetic.omni.base import (
    ModelProcessingStage,
    SkipSample,
    VLMProcessingStage,
)
from nemo_curator.tasks.image import SingleDataTask
from nemo_curator.tasks.ocr import OCRData

# ---------------------------------------------------------------------------
# Concrete subclasses (only the minimum required to instantiate abstract classes)
# ---------------------------------------------------------------------------


class _CpuVLMStage(VLMProcessingStage):
    name = "_test_cpu_vlm_stage"


class _GpuVLMStage(VLMProcessingStage):
    name = "_test_gpu_vlm_stage"
    resources = Resources(gpus=1)



class _SimpleModelStage(ModelProcessingStage):
    name = "_test_simple_model_stage"
    resources = Resources(gpus=1)

    def build_prompt(self, task: SingleDataTask) -> str:
        return "test prompt"

    def handle_response(self, task: SingleDataTask, response: str) -> SingleDataTask:
        task.data.error = response  # record for assertions
        return task

    def load_image(self, task: SingleDataTask) -> Image.Image:
        return Image.new("RGB", (4, 4))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task(*, task_id: str = "t0", is_valid: bool = True) -> SingleDataTask[OCRData]:
    data = OCRData(image_path=Path("test.jpg"), image_id="img_0", is_valid=is_valid)
    return SingleDataTask(task_id=task_id, dataset_name="test", data=data)


def _make_model_stage() -> _SimpleModelStage:
    mock_model = MagicMock()
    mock_model.is_loaded = False
    return _SimpleModelStage(model=mock_model, inference_config=InferenceConfig(), batch_size=4)


# ---------------------------------------------------------------------------
# VLMProcessingStage.__init__ — cuda_devices validation
# ---------------------------------------------------------------------------


class TestVLMProcessingStageInit:
    def test_cuda_devices_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="cuda_devices length"):
            _GpuVLMStage(cuda_devices=[0, 1])  # resources.gpus=1, but 2 devices

    def test_matching_cuda_devices_ok(self):
        stage = _GpuVLMStage(cuda_devices=[0])
        assert stage.cuda_devices == [0]

    def test_none_cuda_devices_accepted(self):
        stage = _GpuVLMStage(cuda_devices=None)
        assert stage.cuda_devices is None

    def test_cpu_stage_no_cuda_devices(self):
        stage = _CpuVLMStage()
        assert stage.cuda_devices is None


# ---------------------------------------------------------------------------
# VLMProcessingStage._get_gpu_memory_utilization
# ---------------------------------------------------------------------------


class TestGetGpuMemoryUtilization:
    def test_returns_09_for_whole_gpu_stage(self):
        stage = _GpuVLMStage()
        assert stage._get_gpu_memory_utilization() == pytest.approx(0.9)

    def test_raises_when_gpu_memory_gb_is_zero(self):
        stage = _CpuVLMStage()  # gpu_memory_gb=0.0
        with pytest.raises(ValueError, match="greater than 0"):
            stage._get_gpu_memory_utilization()



# ---------------------------------------------------------------------------
# VLMProcessingStage._get_tensor_parallel_size
# ---------------------------------------------------------------------------


class TestGetTensorParallelSize:
    def test_returns_gpu_count_for_gpu_stage(self):
        assert _GpuVLMStage()._get_tensor_parallel_size() == 1

    def test_returns_1_for_cpu_stage(self):
        assert _CpuVLMStage()._get_tensor_parallel_size() == 1


# ---------------------------------------------------------------------------
# ModelProcessingStage.process_batch
# ---------------------------------------------------------------------------


class TestModelProcessingStageProcessBatch:
    def test_empty_input_returns_empty(self):
        assert _make_model_stage().process_batch([]) == []

    def test_skips_invalid_task_without_calling_generate(self):
        stage = _make_model_stage()
        results = stage.process_batch([_make_task(is_valid=False)])
        assert len(results) == 1
        stage.model.generate.assert_not_called()

    def test_valid_task_calls_generate_and_handle_response(self):
        stage = _make_model_stage()
        stage.model.generate.return_value = ["response_text"]
        results = stage.process_batch([_make_task()])
        stage.model.generate.assert_called_once()
        assert results[0].data.error == "response_text"

    def test_multiple_tasks_batched_in_one_generate_call(self):
        stage = _make_model_stage()
        stage.model.generate.return_value = ["r0", "r1", "r2"]
        stage.process_batch([_make_task(task_id=f"t{i}") for i in range(3)])
        assert stage.model.generate.call_count == 1
        prompts = stage.model.generate.call_args[0][0]
        assert len(prompts) == 3

    def test_skip_sample_in_build_prompt_skips_without_marking_invalid(self):
        stage = _make_model_stage()
        stage.build_prompt = MagicMock(side_effect=SkipSample)
        results = stage.process_batch([_make_task()])
        assert results[0].data.is_valid is True
        stage.model.generate.assert_not_called()

    def test_exception_in_build_prompt_marks_task_invalid(self):
        stage = _make_model_stage()
        stage.build_prompt = MagicMock(side_effect=RuntimeError("bad prompt"))
        results = stage.process_batch([_make_task()])
        assert results[0].data.is_valid is False
        assert "bad prompt" in (results[0].data.error or "")

    def test_exception_in_handle_response_marks_task_invalid(self):
        stage = _make_model_stage()
        stage.model.generate.return_value = ["response"]
        stage.handle_response = MagicMock(side_effect=RuntimeError("parse failure"))
        results = stage.process_batch([_make_task()])
        assert results[0].data.is_valid is False
        assert "parse failure" in (results[0].data.error or "")

    def test_batch_generate_exception_marks_all_tasks_invalid(self):
        stage = _make_model_stage()
        stage.model.generate.side_effect = RuntimeError("GPU OOM")
        results = stage.process_batch([_make_task(task_id=f"t{i}") for i in range(3)])
        assert all(not r.data.is_valid for r in results)
        assert all("GPU OOM" in (r.data.error or "") for r in results)

    def test_non_multimodal_passes_none_images_to_generate(self):
        stage = _make_model_stage()
        stage.multimodal = False
        stage.model.generate.return_value = ["r"]
        stage.process_batch([_make_task()])
        images_arg = stage.model.generate.call_args[0][1]
        assert images_arg is None


# ---------------------------------------------------------------------------
# ModelProcessingStage.process (single-task delegation)
# ---------------------------------------------------------------------------


class TestModelProcessingStageProcess:
    def test_process_delegates_to_process_batch(self):
        stage = _make_model_stage()
        stage.model.generate.return_value = ["result"]
        task = _make_task()
        result = stage.process(task)
        stage.model.generate.assert_called_once()
        assert result is task


# ---------------------------------------------------------------------------
# ModelProcessingStage.setup / teardown
# ---------------------------------------------------------------------------


class TestModelProcessingStageSetupTeardown:
    def test_setup_loads_model_when_not_loaded(self):
        stage = _make_model_stage()
        stage.model.is_loaded = False
        stage.setup()
        stage.model.load.assert_called_once()

    def test_setup_skips_load_when_already_loaded(self):
        stage = _make_model_stage()
        stage.model.is_loaded = True
        stage.setup()
        stage.model.load.assert_not_called()

    def test_teardown_unloads_model(self):
        stage = _make_model_stage()
        stage.teardown()
        stage.model.unload.assert_called_once()
