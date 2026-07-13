# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for nemo_curator.stages.synthetic.omni.base.

The async client + its ``query_model`` are exercised in
``tests/models/omni/test_base.py``; here we mock the stage's batch-level
``_generate`` and verify the stage's batch dispatch, message assembly, and
error-handling logic.
"""

from pathlib import Path
from unittest.mock import MagicMock

from PIL import Image

from nemo_curator.stages.resources import Resources
from nemo_curator.stages.synthetic.omni.base import ModelProcessingStage, SkipSample
from nemo_curator.tasks.image import ImageSampleTask
from nemo_curator.tasks.ocr import OCRData


class _SimpleModelStage(ModelProcessingStage):
    name = "_test_simple_model_stage"
    resources = Resources(gpus=1)

    def build_prompt(self, task: ImageSampleTask) -> str:
        return "test prompt"

    def handle_response(self, task: ImageSampleTask, response: str) -> ImageSampleTask:
        task.data.error = response  # record for assertions
        return task

    def load_image(self, task: ImageSampleTask) -> Image.Image:
        return Image.new("RGB", (4, 4))


def _make_task(*, is_valid: bool = True) -> ImageSampleTask[OCRData]:
    # task_id is framework-assigned at stage boundaries; tests don't set it (see PR #2036).
    data = OCRData(image_path=Path("test.jpg"), image_id="img_0", is_valid=is_valid)
    return ImageSampleTask(dataset_name="test", data=data)


def _make_model_stage() -> _SimpleModelStage:
    stage = _SimpleModelStage(client=MagicMock(), model_name="test/model", batch_size=4)
    # _generate wraps the async client; mock it so these tests stay focused on dispatch.
    stage._generate = MagicMock()
    return stage


class TestModelProcessingStage:
    """Batch dispatch, message assembly, and error containment."""

    def test_invalid_inputs_pass_through_untouched(self) -> None:
        stage = _make_model_stage()
        results = stage.process_batch([_make_task(is_valid=False)])
        assert len(results) == 1
        stage._generate.assert_not_called()

    def test_batch_dispatches_in_one_generate_call(self) -> None:
        stage = _make_model_stage()
        stage._generate.return_value = ["r0", "r1", "r2"]
        results = stage.process_batch([_make_task() for _ in range(3)])
        assert stage._generate.call_count == 1
        messages_batch = stage._generate.call_args.args[0]
        assert len(messages_batch) == 3
        # each message carries an image part (multimodal) + a text part
        first_content = messages_batch[0][0]["content"]
        assert any(part["type"] == "image_url" for part in first_content)
        # handle_response stored each response into task.data.error
        assert [r.data.error for r in results] == ["r0", "r1", "r2"]

    def test_non_multimodal_builds_text_only_messages(self) -> None:
        stage = _make_model_stage()
        stage.multimodal = False
        stage._generate.return_value = ["r"]
        stage.process_batch([_make_task()])
        content = stage._generate.call_args.args[0][0][0]["content"]
        assert all(part["type"] != "image_url" for part in content)

    def test_skip_sample_in_build_prompt_drops_task_without_invalidating(self) -> None:
        stage = _make_model_stage()
        stage.build_prompt = MagicMock(side_effect=SkipSample)
        results = stage.process_batch([_make_task()])
        assert results[0].data.is_valid is True
        stage._generate.assert_not_called()

    def test_build_prompt_exception_marks_only_its_task_invalid(self) -> None:
        stage = _make_model_stage()
        stage._generate.return_value = ["ok"]
        tasks = [_make_task(), _make_task()]
        # Fail only on t0
        stage.build_prompt = MagicMock(side_effect=[RuntimeError("bad prompt t0"), "p1"])
        results = stage.process_batch(tasks)
        assert results[0].data.is_valid is False
        assert "bad prompt t0" in (results[0].data.error or "")
        assert results[1].data.is_valid is True

    def test_handle_response_exception_marks_only_its_task_invalid(self) -> None:
        stage = _make_model_stage()
        stage._generate.return_value = ["r0", "r1"]
        stage.handle_response = MagicMock(side_effect=[RuntimeError("parse t0"), None])
        results = stage.process_batch([_make_task(), _make_task()])
        assert results[0].data.is_valid is False
        assert "parse t0" in (results[0].data.error or "")
        assert results[1].data.is_valid is True

    def test_generate_exception_marks_entire_batch_invalid(self) -> None:
        stage = _make_model_stage()
        stage._generate.side_effect = RuntimeError("api down")
        results = stage.process_batch([_make_task() for _ in range(3)])
        assert all(not r.data.is_valid for r in results)
        assert all("api down" in (r.data.error or "") for r in results)

    def test_response_length_mismatch_fails_batch_without_partial_writes(self) -> None:
        """Regression: strict=True zip used to process the shorter sequence first,
        then the outer except clobbered already-handled tasks. The fix raises
        on length mismatch BEFORE any _handle_response_one call."""
        stage = _make_model_stage()
        stage._generate.return_value = ["r0", "r1"]  # 2 responses, 3 prompts
        handle_calls: list[int] = []
        stage._handle_response_one = MagicMock(side_effect=lambda _t, idx, _r: handle_calls.append(idx))
        results = stage.process_batch([_make_task() for _ in range(3)])
        assert handle_calls == [], "no per-task writes should happen on contract violation"
        assert all(not r.data.is_valid for r in results)
        assert any("returned 2 responses for 3 prompts" in (r.data.error or "") for r in results)
