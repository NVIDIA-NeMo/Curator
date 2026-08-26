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

"""End-to-end I/O tests for JsonlSampleWriterStage + merge_output_shards.

All tests write real files to ``tmp_path`` and read them back; no mocks.
"""

import json
from pathlib import Path

from PIL import Image

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.synthetic.omni.io import (
    HFDatasetImageReaderStage,
    JsonlSampleWriterStage,
    merge_output_shards,
)
from nemo_curator.tasks.image import ImageSampleTask, ImageTaskData
from nemo_curator.tasks.ocr import OCRData


def _make_rgb_jpeg(tmp_path: Path, name: str = "img.jpg") -> Path:
    p = tmp_path / name
    Image.new("RGB", (32, 32), (100, 150, 200)).save(p, format="JPEG")
    return p


def _make_image_task(
    image_path: Path,
    *,
    image_id: str = "img_0",
    is_valid: bool = True,
) -> ImageSampleTask[ImageTaskData]:
    return ImageSampleTask(
        dataset_name="test",
        data=ImageTaskData(image_path=image_path, image_id=image_id, is_valid=is_valid),
    )


def _make_ocr_task(image_path: Path) -> ImageSampleTask[OCRData]:
    return ImageSampleTask(
        dataset_name="test",
        data=OCRData(image_path=image_path, image_id="img_0", ocr_dense=None),
    )


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def test_hf_dataset_image_reader_ray_stage_spec(tmp_path: Path) -> None:
    stage = HFDatasetImageReaderStage(dataset_name="unused", image_dir=tmp_path)

    assert stage.ray_stage_spec()["is_fanout_stage"] is True


class TestJsonlSampleWriterStage:
    """Per-worker JSONL writes, valid_only filtering, image-path relativization."""

    def _run(
        self,
        output: Path,
        tasks: list[ImageSampleTask],
        *,
        worker_id: str = "w1",
        **kwargs: object,
    ) -> tuple[JsonlSampleWriterStage, Path]:
        stage = JsonlSampleWriterStage(str(output), **kwargs)
        stage.setup(WorkerMetadata(worker_id=worker_id))
        for t in tasks:
            stage.process(t)
        stage.teardown()
        shard = output.parent / f"{output.stem}_worker{worker_id}{output.suffix or '.jsonl'}"
        return stage, shard

    def test_writes_per_worker_shard(self, tmp_path: Path) -> None:
        p = _make_rgb_jpeg(tmp_path)
        _, shard = self._run(tmp_path / "out.jsonl", [_make_image_task(p)], worker_id="w1")
        assert shard.exists()
        # Base path is not written to directly; merge_output_shards is the consolidation step.
        assert not (tmp_path / "out.jsonl").exists()

    def test_written_record_has_expected_shape(self, tmp_path: Path) -> None:
        p = _make_rgb_jpeg(tmp_path)
        _, shard = self._run(tmp_path / "out.jsonl", [_make_image_task(p, image_id="my_img")])
        records = [json.loads(line) for line in shard.read_text().splitlines() if line.strip()]
        assert len(records) == 1
        assert records[0]["image_id"] == "my_img"
        assert "is_valid" not in records[0], "is_valid is always stripped from output"

    def test_valid_only_filters_invalid_tasks(self, tmp_path: Path) -> None:
        p = _make_rgb_jpeg(tmp_path)
        stage, shard = self._run(
            tmp_path / "out.jsonl",
            [
                _make_image_task(p, is_valid=True),
                _make_image_task(p, is_valid=False),
            ],
            valid_only=True,
        )
        lines = [line for line in shard.read_text().splitlines() if line.strip()]
        assert len(lines) == 1
        assert stage.stats == {"saved": 1, "skipped": 1}

    def test_valid_only_false_keeps_invalid_records(self, tmp_path: Path) -> None:
        p = _make_rgb_jpeg(tmp_path)
        _, shard = self._run(
            tmp_path / "out.jsonl",
            [_make_image_task(p, is_valid=False)],
            valid_only=False,
        )
        lines = [line for line in shard.read_text().splitlines() if line.strip()]
        assert len(lines) == 1

    def test_image_path_relativized_to_image_parent(self, tmp_path: Path) -> None:
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        p = _make_rgb_jpeg(img_dir)
        _, shard = self._run(
            tmp_path / "out.jsonl",
            [_make_image_task(p)],
            image_parent=str(img_dir),
        )
        rec = json.loads(shard.read_text().splitlines()[0])
        assert rec["image_path"] == "img.jpg"

    def test_none_fields_are_stripped(self, tmp_path: Path) -> None:
        p = _make_rgb_jpeg(tmp_path)
        _, shard = self._run(tmp_path / "out.jsonl", [_make_ocr_task(p)])
        rec = json.loads(shard.read_text().splitlines()[0])
        assert "ocr_dense" not in rec
        assert "ocr_scoring_prompt" not in rec


class TestMergeOutputShards:
    """Concatenate per-worker shards into a single merged JSONL."""

    def _write_shard(self, directory: Path, worker_id: str, lines: list[dict]) -> Path:
        p = directory / f"out_worker{worker_id}.jsonl"
        _write_jsonl(p, lines)
        return p

    def test_concatenates_and_deletes_shards(self, tmp_path: Path) -> None:
        s0 = self._write_shard(tmp_path, "0", [{"a": 1}])
        s1 = self._write_shard(tmp_path, "1", [{"a": 2}])
        merged = merge_output_shards(tmp_path / "out.jsonl")
        assert merged == tmp_path / "out.jsonl"
        recs = [json.loads(line) for line in merged.read_text().splitlines()]
        assert recs == [{"a": 1}, {"a": 2}]
        assert not s0.exists()
        assert not s1.exists()

    def test_delete_shards_false_leaves_shards(self, tmp_path: Path) -> None:
        s0 = self._write_shard(tmp_path, "0", [{"x": 0}])
        merge_output_shards(tmp_path / "out.jsonl", delete_shards=False)
        assert s0.exists()

    def test_no_shards_is_noop(self, tmp_path: Path) -> None:
        output = tmp_path / "out.jsonl"
        result = merge_output_shards(output)
        assert result == output
        assert not output.exists()

    def test_overwrites_existing_merged_file(self, tmp_path: Path) -> None:
        output = tmp_path / "out.jsonl"
        _write_jsonl(output, [{"existing": True}])
        self._write_shard(tmp_path, "0", [{"new": True}])
        merge_output_shards(output)
        recs = [json.loads(line) for line in output.read_text().splitlines()]
        assert recs == [{"new": True}]
