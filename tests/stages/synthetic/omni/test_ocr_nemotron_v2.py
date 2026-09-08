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

"""Unit tests for nemo_curator.stages.synthetic.omni.ocr_nemotron_v2.

CPU tests cover prediction-to-OCRDenseItem conversion and stage dispatch
(model is mocked). The live model load lives behind @pytest.mark.gpu and
requires ``nemotron_ocr`` + a GPU.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from nemo_curator.stages.synthetic.omni.ocr_nemotron_v2 import (
    OCRNemotronV2Stage,
    _to_ocr_dense_item,
)
from nemo_curator.tasks.image import ImageSampleTask
from nemo_curator.tasks.ocr import OCRData


def _make_task(image_path: Path, *, is_valid: bool = True) -> ImageSampleTask[OCRData]:
    return ImageSampleTask(
        dataset_name="test",
        data=OCRData(image_path=image_path, image_id="img_0", is_valid=is_valid),
    )


def _make_rgb_jpeg(tmp_path: Path, name: str = "img.jpg") -> Path:
    p = tmp_path / name
    Image.new("RGB", (64, 64), (100, 150, 200)).save(p, format="JPEG")
    return p


class TestToOcrDenseItem:
    """Coordinate transform from NemotronOCR-v2 prediction dicts."""

    def test_scales_normalized_coords_to_0_1000(self) -> None:
        item = _to_ocr_dense_item({"left": 0.1, "right": 0.5, "upper": 0.2, "lower": 0.4, "text": "HELLO"})
        assert item.bbox_2d == [100, 200, 500, 400]
        assert item.text_content == "HELLO"
        assert item.valid is True

    def test_sorts_inverted_upper_lower(self) -> None:
        # NemotronOCR-v2 sometimes swaps `upper`/`lower`; output must have y1 <= y2.
        item = _to_ocr_dense_item({"left": 0.0, "right": 1.0, "upper": 0.8, "lower": 0.2, "text": "W"})
        _x1, y1, _x2, y2 = item.bbox_2d
        assert (y1, y2) == (200, 800)


class TestOCRNemotronV2Stage:
    """Stage-level: model resolution, batch dispatch, error containment."""

    def _stage_with_preds(self, predictions: list[dict]) -> OCRNemotronV2Stage:
        stage = OCRNemotronV2Stage(model_dir="/fake/model")
        stage._model = MagicMock(return_value=predictions)
        return stage

    def test_resolve_model_dir_falls_back_to_hf_snapshot(self) -> None:
        with patch(
            "huggingface_hub.snapshot_download",
            return_value="/cache/nvidia/nemotron-ocr-v2",
        ) as mock_dl:
            result = OCRNemotronV2Stage(model_dir=None)._resolve_model_dir()
        mock_dl.assert_called_once()
        assert result.endswith("v2_multilingual")

    def test_invalid_input_skips_model_call(self, tmp_path: Path) -> None:
        stage = self._stage_with_preds([])
        task = _make_task(_make_rgb_jpeg(tmp_path), is_valid=False)
        results = stage.process_batch([task])
        assert results[0].data.ocr_dense is None
        stage._model.assert_not_called()

    def test_populates_ocr_dense_from_predictions(self, tmp_path: Path) -> None:
        preds = [{"left": 0.1, "right": 0.5, "upper": 0.1, "lower": 0.5, "text": "HELLO"}]
        stage = self._stage_with_preds(preds)
        task = _make_task(_make_rgb_jpeg(tmp_path))
        results = stage.process_batch([task])
        dense = results[0].data.ocr_dense
        assert dense is not None
        assert len(dense) == 1
        assert dense[0].text_content == "HELLO"
        assert dense[0].bbox_2d == [100, 100, 500, 500]

    def test_empty_predictions_yield_empty_list(self, tmp_path: Path) -> None:
        stage = self._stage_with_preds([])
        task = _make_task(_make_rgb_jpeg(tmp_path))
        results = stage.process_batch([task])
        assert results[0].data.ocr_dense == []

    def test_model_failure_marks_only_failing_task_invalid(self, tmp_path: Path) -> None:
        stage = OCRNemotronV2Stage(model_dir="/fake/model")
        stage._model = MagicMock(
            side_effect=[
                RuntimeError("GPU OOM"),
                [{"left": 0.0, "right": 1.0, "upper": 0.0, "lower": 1.0, "text": "ok"}],
            ]
        )
        results = stage.process_batch(
            [
                _make_task(_make_rgb_jpeg(tmp_path, "a.jpg")),
                _make_task(_make_rgb_jpeg(tmp_path, "b.jpg")),
            ]
        )
        assert results[0].data.is_valid is False
        assert "GPU OOM" in (results[0].data.error or "")
        assert results[1].data.is_valid is True
        assert results[1].data.ocr_dense is not None
        assert results[1].data.ocr_dense[0].text_content == "ok"


@pytest.mark.gpu
def test_setup_loads_model_on_gpu() -> None:
    """Live integration: setup() downloads + loads NemotronOCRV2 onto the GPU."""
    pytest.importorskip("nemotron_ocr.inference.pipeline_v2")
    import huggingface_hub

    snapshot = huggingface_hub.snapshot_download("nvidia/nemotron-ocr-v2")
    stage = OCRNemotronV2Stage(model_dir=str(Path(snapshot) / "v2_multilingual"))
    stage.setup()
    assert stage._model is not None
    stage.teardown()
