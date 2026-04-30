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

"""Unit tests for nemo_curator.stages.synthetic.omni.ocr_nemotron_v2.

Most tests are CPU-only (model mocked out).
The GPU test is marked @pytest.mark.gpu and requires the nemotron_ocr
package and a GPU to be present.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from nemo_curator.stages.synthetic.omni.ocr_nemotron_v2 import (
    OCRNemotronV2Stage,
    _to_ocr_dense_word,
)
from nemo_curator.tasks.image import SingleDataTask
from nemo_curator.tasks.ocr import OCRData, OCRDenseWord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task(
    image_path: Path,
    *,
    task_id: str = "t0",
    is_valid: bool = True,
) -> SingleDataTask[OCRData]:
    data = OCRData(
        image_path=image_path,
        image_id="img_0",
        is_valid=is_valid,
    )
    return SingleDataTask(task_id=task_id, dataset_name="test", data=data)


def _make_rgb_jpeg(tmp_path: Path, name: str = "img.jpg") -> Path:
    p = tmp_path / name
    Image.new("RGB", (64, 64), (100, 150, 200)).save(p, format="JPEG")
    return p


def _mock_stage(model_dir: str = "/fake/model") -> OCRNemotronV2Stage:
    return OCRNemotronV2Stage(model_dir=model_dir)


# ---------------------------------------------------------------------------
# _to_ocr_dense_word
# ---------------------------------------------------------------------------


class TestToOcrDenseWord:
    def test_normal_prediction(self):
        pred = {"left": 0.1, "right": 0.5, "upper": 0.2, "lower": 0.4, "text": "HELLO"}
        word = _to_ocr_dense_word(pred)
        assert isinstance(word, OCRDenseWord)
        assert word.text_content == "HELLO"
        assert word.bbox_2d == [100, 200, 500, 400]

    def test_inverted_upper_lower_sorted(self):
        # NemotronOCR-v2: "upper" may be > "lower" (inverted naming)
        pred = {"left": 0.0, "right": 1.0, "upper": 0.8, "lower": 0.2, "text": "WORD"}
        word = _to_ocr_dense_word(pred)
        _x1, y1, _x2, y2 = word.bbox_2d
        assert y1 <= y2, "y1 must be <= y2 after sorting"
        assert y1 == 200  # min(0.8*1000, 0.2*1000) = 200
        assert y2 == 800

    def test_coordinates_scaled_to_0_1000(self):
        pred = {"left": 0.0, "right": 1.0, "upper": 0.0, "lower": 1.0, "text": "X"}
        word = _to_ocr_dense_word(pred)
        assert word.bbox_2d == [0, 0, 1000, 1000]

    def test_empty_text(self):
        pred = {"left": 0.0, "right": 0.5, "upper": 0.0, "lower": 0.5, "text": ""}
        word = _to_ocr_dense_word(pred)
        assert word.text_content == ""

    def test_word_is_valid_by_default(self):
        pred = {"left": 0.1, "right": 0.5, "upper": 0.1, "lower": 0.5, "text": "HI"}
        word = _to_ocr_dense_word(pred)
        assert word.valid is True


# ---------------------------------------------------------------------------
# OCRNemotronV2Stage constructor & xenna_stage_spec
# ---------------------------------------------------------------------------


class TestOCRNemotronV2StageInit:
    def test_model_dir_stored(self):
        stage = _mock_stage("/path/to/model")
        assert stage.model_dir == Path("/path/to/model")

    def test_model_dir_none(self):
        stage = OCRNemotronV2Stage()
        assert stage.model_dir is None

    def test_xenna_stage_spec_empty_without_num_workers(self):
        stage = _mock_stage()
        spec = stage.xenna_stage_spec()
        assert spec == {}

    def test_xenna_stage_spec_includes_num_workers(self):
        stage = OCRNemotronV2Stage(model_dir="/fake", num_workers=4)
        spec = stage.xenna_stage_spec()
        assert spec["num_workers"] == 4

    def test_merge_level_stored(self):
        stage = OCRNemotronV2Stage(model_dir="/fake", merge_level="sentence")
        assert stage.merge_level == "sentence"

    def test_resources_require_gpu(self):
        stage = _mock_stage()
        assert stage.resources.gpus >= 1


# ---------------------------------------------------------------------------
# process_batch (model mocked)
# ---------------------------------------------------------------------------


class TestProcessBatch:
    def _make_stage_with_mock_model(self, predictions: list[dict]) -> OCRNemotronV2Stage:
        stage = _mock_stage()
        mock_model = MagicMock(return_value=predictions)
        stage._model = mock_model
        return stage

    def test_skips_invalid_task(self, tmp_path: Path):
        stage = self._make_stage_with_mock_model([])
        p = _make_rgb_jpeg(tmp_path)
        task = _make_task(p, is_valid=False)
        results = stage.process_batch([task])
        assert results[0].data.ocr_dense is None
        stage._model.assert_not_called()

    def test_populates_ocr_dense(self, tmp_path: Path):
        preds = [
            {"left": 0.1, "right": 0.5, "upper": 0.1, "lower": 0.5, "text": "HELLO"},
        ]
        stage = self._make_stage_with_mock_model(preds)
        p = _make_rgb_jpeg(tmp_path)
        task = _make_task(p)
        results = stage.process_batch([task])
        dense = results[0].data.ocr_dense
        assert dense is not None
        assert len(dense) == 1
        assert dense[0].text_content == "HELLO"

    def test_empty_predictions_sets_empty_ocr_dense(self, tmp_path: Path):
        stage = self._make_stage_with_mock_model([])
        p = _make_rgb_jpeg(tmp_path)
        task = _make_task(p)
        results = stage.process_batch([task])
        assert results[0].data.ocr_dense == []

    def test_exception_in_process_one_marks_task_invalid(self, tmp_path: Path):
        stage = _mock_stage()
        stage._model = MagicMock(side_effect=RuntimeError("GPU OOM"))
        p = _make_rgb_jpeg(tmp_path)
        task = _make_task(p)
        results = stage.process_batch([task])
        assert results[0].data.is_valid is False
        assert "GPU OOM" in (results[0].data.error or "")

    def test_processes_multiple_tasks(self, tmp_path: Path):
        preds = [{"left": 0.0, "right": 1.0, "upper": 0.0, "lower": 1.0, "text": "X"}]
        stage = self._make_stage_with_mock_model(preds)
        tasks = [_make_task(_make_rgb_jpeg(tmp_path, f"img{i}.jpg"), task_id=f"t{i}") for i in range(3)]
        results = stage.process_batch(tasks)
        assert len(results) == 3
        assert all(r.data.ocr_dense is not None for r in results)

    def test_passes_merge_level_to_model(self, tmp_path: Path):
        preds: list[dict] = []
        stage = OCRNemotronV2Stage(model_dir="/fake", merge_level="sentence")
        stage._model = MagicMock(return_value=preds)
        p = _make_rgb_jpeg(tmp_path)
        task = _make_task(p)
        stage.process_batch([task])
        call_kwargs = stage._model.call_args
        assert call_kwargs.kwargs.get("merge_level") == "sentence" or (
            len(call_kwargs.args) >= 2 and call_kwargs.args[1] == "sentence"
        )


# ---------------------------------------------------------------------------
# setup (_resolve_model_dir)
# ---------------------------------------------------------------------------


class TestResolveModelDir:
    def test_returns_explicit_model_dir(self):
        stage = OCRNemotronV2Stage(model_dir="/my/model")
        result = stage._resolve_model_dir()
        assert result == "/my/model"

    def test_calls_snapshot_download_when_no_model_dir(self):
        stage = OCRNemotronV2Stage(model_dir=None)
        with patch(
            "huggingface_hub.snapshot_download",
            return_value="/cache/nvidia/nemotron-ocr-v2",
        ) as mock_dl:
            result = stage._resolve_model_dir()
        mock_dl.assert_called_once()
        assert result.endswith("v2_multilingual")


# ---------------------------------------------------------------------------
# GPU test (requires @pytest.mark.gpu + nemotron_ocr installed + GPU present)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_setup_loads_model():
    """Integration: setup() loads NemotronOCRV2 onto the GPU.

    Requires nemotron_ocr installed and a GPU.
    """
    import huggingface_hub

    snapshot = huggingface_hub.snapshot_download("nvidia/nemotron-ocr-v2")
    model_dir = Path(snapshot) / "v2_multilingual"
    stage = OCRNemotronV2Stage(model_dir=str(model_dir))
    stage.setup()
    assert stage._model is not None
    stage.teardown()
