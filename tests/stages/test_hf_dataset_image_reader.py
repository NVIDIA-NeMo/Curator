"""Tests for HFDatasetImageReaderStage.

Covers:
  - Hub dataset loading with limit (partial download)
  - Image deduplication (VQA-style datasets with multiple rows per image)
  - Idempotency (second run skips re-saving images)
  - Local path loading (imagefolder)
  - _to_pil() handles PIL Image, bytes dict, raw bytes, and file path string
"""

import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from nemo_curator.stages.synthetic.omni.io import HFDatasetImageReaderStage
from nemo_curator.tasks import _EmptyTask
from nemo_curator.tasks.image import ImageTaskData, SingleDataTask


def _empty_task() -> _EmptyTask:
    return _EmptyTask(task_id="test", dataset_name="test", data=None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rgb_image(width: int = 32, height: int = 32, color=(255, 0, 0)) -> Image.Image:
    img = Image.new("RGB", (width, height), color)
    return img


def _make_rgba_image() -> Image.Image:
    return Image.new("RGBA", (32, 32), (255, 0, 0, 128))


def _pil_to_bytes(img: Image.Image, fmt: str = "JPEG") -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt)
    return buf.getvalue()


def _fake_dataset(num_images: int = 5, *, duplicate_ids: bool = False):
    """Build a minimal list-of-dicts that mimics a HF Dataset."""
    rows = []
    for i in range(num_images):
        img_id = f"img_{i // 2:04d}" if duplicate_ids else f"img_{i:04d}"
        rows.append({
            "image": _make_rgb_image(color=(i * 40 % 256, 0, 0)),
            "image_id": img_id,
            "question": f"What is in image {i}?",
        })
    return rows


# ---------------------------------------------------------------------------
# _to_pil tests (unit — no I/O)
# ---------------------------------------------------------------------------

class TestToPil:
    def test_pil_image_passthrough(self):
        img = _make_rgb_image()
        assert HFDatasetImageReaderStage._to_pil(img) is img

    def test_bytes_dict_bytes_key(self):
        img = _make_rgb_image()
        result = HFDatasetImageReaderStage._to_pil({"bytes": _pil_to_bytes(img)})
        assert isinstance(result, Image.Image)
        assert result.size == (32, 32)

    def test_bytes_dict_data_key(self):
        img = _make_rgb_image()
        result = HFDatasetImageReaderStage._to_pil({"data": _pil_to_bytes(img)})
        assert isinstance(result, Image.Image)

    def test_raw_bytes(self):
        img = _make_rgb_image()
        result = HFDatasetImageReaderStage._to_pil(_pil_to_bytes(img))
        assert isinstance(result, Image.Image)

    def test_raw_bytearray(self):
        img = _make_rgb_image()
        result = HFDatasetImageReaderStage._to_pil(bytearray(_pil_to_bytes(img)))
        assert isinstance(result, Image.Image)

    def test_file_path_string(self, tmp_path):
        img = _make_rgb_image()
        p = tmp_path / "test.jpg"
        img.save(p, format="JPEG")
        result = HFDatasetImageReaderStage._to_pil(str(p))
        assert isinstance(result, Image.Image)

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Cannot convert"):
            HFDatasetImageReaderStage._to_pil(12345)


# ---------------------------------------------------------------------------
# process() tests (mock _load_dataset)
# ---------------------------------------------------------------------------

class TestProcess:
    def _make_stage(self, image_dir, **kwargs) -> HFDatasetImageReaderStage:
        return HFDatasetImageReaderStage(
            dataset_name="textvqa",
            image_dir=image_dir,
            split="train",
            id_column="image_id",
            **kwargs,
        )

    def test_creates_one_task_per_image(self, tmp_path):
        stage = self._make_stage(tmp_path)
        with patch.object(stage, "_load_dataset", return_value=_fake_dataset(5)):
            tasks = stage.process(_empty_task())

        assert len(tasks) == 5
        assert all(isinstance(t, SingleDataTask) for t in tasks)

    def test_image_files_saved_to_disk(self, tmp_path):
        stage = self._make_stage(tmp_path)
        with patch.object(stage, "_load_dataset", return_value=_fake_dataset(3)):
            tasks = stage.process(_empty_task())

        for task in tasks:
            assert task.data.image_path.exists(), f"Missing: {task.data.image_path}"
            img = Image.open(task.data.image_path)
            assert img.mode == "RGB"

    def test_image_id_set_from_column(self, tmp_path):
        stage = self._make_stage(tmp_path)
        with patch.object(stage, "_load_dataset", return_value=_fake_dataset(4)):
            tasks = stage.process(_empty_task())

        ids = [t.data.image_id for t in tasks]
        assert ids == ["img_0000", "img_0001", "img_0002", "img_0003"]

    def test_fallback_to_index_when_no_id_column(self, tmp_path):
        stage = HFDatasetImageReaderStage(
            dataset_name="textvqa",
            image_dir=tmp_path,
            split="train",
            id_column=None,
        )
        with patch.object(stage, "_load_dataset", return_value=_fake_dataset(3)):
            tasks = stage.process(_empty_task())

        ids = [t.data.image_id for t in tasks]
        assert ids == ["000000", "000001", "000002"]

    def test_deduplication_skips_repeated_image_ids(self, tmp_path):
        """10 rows but only 5 unique image_ids (duplicate_ids=True pairs them)."""
        stage = self._make_stage(tmp_path)
        with patch.object(stage, "_load_dataset", return_value=_fake_dataset(10, duplicate_ids=True)):
            tasks = stage.process(_empty_task())

        assert len(tasks) == 5
        ids = [t.data.image_id for t in tasks]
        assert len(ids) == len(set(ids)), "Duplicate image_ids in output"

    def test_idempotent_second_run_skips_saving(self, tmp_path):
        """Second call must not re-write images that already exist."""
        stage = self._make_stage(tmp_path)
        dataset = _fake_dataset(3)

        with patch.object(stage, "_load_dataset", return_value=dataset):
            tasks1 = stage.process(_empty_task())

        mtimes_before = {t.data.image_path: t.data.image_path.stat().st_mtime for t in tasks1}

        with patch.object(stage, "_load_dataset", return_value=dataset):
            tasks2 = stage.process(_empty_task())

        for task in tasks2:
            assert task.data.image_path.stat().st_mtime == mtimes_before[task.data.image_path], (
                f"Image was re-written on second run: {task.data.image_path}"
            )

    def test_rgba_converted_to_rgb(self, tmp_path):
        stage = self._make_stage(tmp_path)
        rows = [{"image": _make_rgba_image(), "image_id": "rgba_test"}]
        with patch.object(stage, "_load_dataset", return_value=rows):
            tasks = stage.process(_empty_task())

        saved = Image.open(tasks[0].data.image_path)
        assert saved.mode == "RGB"

    def test_task_type_respected(self, tmp_path):
        from nemo_curator.tasks.ocr import OCRData

        stage = HFDatasetImageReaderStage(
            dataset_name="textvqa",
            image_dir=tmp_path,
            split="train",
            id_column="image_id",
            task_type=OCRData,
        )
        with patch.object(stage, "_load_dataset", return_value=_fake_dataset(2)):
            tasks = stage.process(_empty_task())

        assert all(isinstance(t.data, OCRData) for t in tasks)


# ---------------------------------------------------------------------------
# _load_dataset routing tests (no real network calls)
# ---------------------------------------------------------------------------

class TestLoadDatasetRouting:
    def test_hub_dataset_without_limit(self, tmp_path):
        stage = HFDatasetImageReaderStage(
            dataset_name="textvqa",
            image_dir=tmp_path,
            limit=None,
        )
        with patch("nemo_curator.stages.synthetic.omni.io.HFDatasetImageReaderStage._load_dataset") as mock_load:
            mock_load.return_value = []
            stage.process(_empty_task())
            mock_load.assert_called_once()

    def test_local_save_to_disk_path(self, tmp_path):
        """A directory with dataset_info.json triggers load_from_disk."""
        (tmp_path / "dataset_info.json").write_text("{}")

        stage = HFDatasetImageReaderStage(
            dataset_name=str(tmp_path),
            image_dir=tmp_path / "out",
            limit=5,
        )

        # Simulate a DatasetDict: has "keys", contains "train", ["train"] returns a leaf dataset
        leaf_ds = MagicMock()
        leaf_ds.__len__ = MagicMock(return_value=10)
        leaf_ds.select = MagicMock(return_value=leaf_ds)

        mock_dict = MagicMock()
        mock_dict.__contains__ = MagicMock(return_value=True)   # "train" in mock_dict → True
        mock_dict.__getitem__ = MagicMock(return_value=leaf_ds)  # mock_dict["train"] → leaf_ds

        with patch("datasets.load_from_disk", return_value=mock_dict) as mock_lfd:
            stage._load_dataset()
            mock_lfd.assert_called_once_with(str(tmp_path))
            leaf_ds.select.assert_called_once_with(range(5))

    def test_local_imagefolder_path(self, tmp_path):
        """An existing directory without dataset_info.json triggers imagefolder loader."""
        stage = HFDatasetImageReaderStage(
            dataset_name=str(tmp_path),
            image_dir=tmp_path / "out",
            split="train",
            limit=10,
        )
        with patch("datasets.load_dataset") as mock_ld:
            mock_ld.return_value = []
            stage._load_dataset()
            mock_ld.assert_called_once_with(
                "imagefolder", data_dir=str(tmp_path), split="train[:10]"
            )

    def test_hub_limit_embedded_in_split(self, tmp_path):
        """Hub path with limit passes split slice notation."""
        stage = HFDatasetImageReaderStage(
            dataset_name="textvqa",
            image_dir=tmp_path,
            split="train",
            limit=50,
        )
        with patch("datasets.load_dataset") as mock_ld:
            mock_ld.return_value = []
            stage._load_dataset()
            mock_ld.assert_called_once_with("textvqa", None, split="train[:50]")

    def test_hub_config_name_forwarded(self, tmp_path):
        stage = HFDatasetImageReaderStage(
            dataset_name="some_multilingual_dataset",
            image_dir=tmp_path,
            split="train",
            config_name="en",
        )
        with patch("datasets.load_dataset") as mock_ld:
            mock_ld.return_value = []
            stage._load_dataset()
            mock_ld.assert_called_once_with("some_multilingual_dataset", "en", split="train")
