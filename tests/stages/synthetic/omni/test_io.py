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

"""Unit tests for nemo_curator.stages.synthetic.omni.io.

Covers: RegularFileReader, load_image_from_task,
        ResultWriterStage, SkipProcessedStage, merge_output_shards.
"""

import json
from pathlib import Path

import pytest
from PIL import Image

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.synthetic.omni.io import (
    RegularFileReader,
    ResultWriterStage,
    SkipProcessedStage,
    load_image_from_task,
    merge_output_shards,
)
from nemo_curator.tasks.image import ImageTaskData, SingleDataTask
from nemo_curator.tasks.ocr import OCRData, OCRDenseWord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rgb_jpeg(tmp_path: Path, name: str = "img.jpg") -> Path:
    p = tmp_path / name
    Image.new("RGB", (32, 32), (100, 150, 200)).save(p, format="JPEG")
    return p


def _make_worker_metadata(worker_id: str = "1") -> WorkerMetadata:
    return WorkerMetadata(worker_id=worker_id)


def _make_image_task(
    image_path: Path,
    *,
    task_id: str = "t0",
    image_id: str = "img_0",
    is_valid: bool = True,
) -> SingleDataTask[ImageTaskData]:
    data = ImageTaskData(
        image_path=image_path,
        image_id=image_id,
        is_valid=is_valid,
    )
    return SingleDataTask(task_id=task_id, dataset_name="test", data=data)


def _make_ocr_task(
    image_path: Path,
    *,
    task_id: str = "t0",
    image_id: str = "img_0",
    is_valid: bool = True,
    words: list[OCRDenseWord] | None = None,
) -> SingleDataTask[OCRData]:
    data = OCRData(
        image_path=image_path,
        image_id=image_id,
        is_valid=is_valid,
        ocr_dense=words,
    )
    return SingleDataTask(task_id=task_id, dataset_name="test", data=data)


# ---------------------------------------------------------------------------
# RegularFileReader
# ---------------------------------------------------------------------------


class TestRegularFileReader:
    def test_can_read_existing_file(self, tmp_path: Path):
        p = _make_rgb_jpeg(tmp_path)
        reader = RegularFileReader()
        assert reader.can_read(p) is True

    def test_cannot_read_missing_file(self, tmp_path: Path):
        reader = RegularFileReader()
        assert reader.can_read(tmp_path / "nonexistent.jpg") is False

    def test_cannot_read_directory(self, tmp_path: Path):
        reader = RegularFileReader()
        assert reader.can_read(tmp_path) is False

    def test_read_bytes_returns_correct_content(self, tmp_path: Path):
        p = tmp_path / "data.bin"
        p.write_bytes(b"\x01\x02\x03")
        reader = RegularFileReader()
        assert reader.read_bytes(p) == b"\x01\x02\x03"

    def test_read_bytes_raises_for_missing_file(self, tmp_path: Path):
        reader = RegularFileReader()
        with pytest.raises(FileNotFoundError):
            reader.read_bytes(tmp_path / "missing.jpg")

    def test_open_image_returns_pil_image(self, tmp_path: Path):
        p = _make_rgb_jpeg(tmp_path)
        reader = RegularFileReader()
        img = reader.open_image(p)
        assert isinstance(img, Image.Image)
        assert img.size == (32, 32)


# ---------------------------------------------------------------------------
# load_image_from_task
# ---------------------------------------------------------------------------


class TestLoadImageFromTask:
    def test_correct_dimensions(self, tmp_path: Path):
        p = _make_rgb_jpeg(tmp_path, "64x64.jpg")
        Image.new("RGB", (64, 64)).save(p, format="JPEG")
        task = _make_image_task(p)
        img = load_image_from_task(task)
        assert img.size == (64, 64)


# ---------------------------------------------------------------------------
# ResultWriterStage
# ---------------------------------------------------------------------------


class TestResultWriterStage:
    def _setup_stage(self, stage: ResultWriterStage, worker_id: str = "w1") -> None:
        stage.setup(_make_worker_metadata(worker_id))

    def test_setup_creates_worker_file(self, tmp_path: Path):
        output = tmp_path / "out.jsonl"
        stage = ResultWriterStage(str(output))
        self._setup_stage(stage, "w1")
        stage.teardown()
        # Worker-suffixed file should exist
        assert (tmp_path / "out_workerw1.jsonl").exists()

    def test_setup_single_file_creates_exact_path(self, tmp_path: Path):
        output = tmp_path / "out.jsonl"
        stage = ResultWriterStage(str(output), single_file=True)
        self._setup_stage(stage)
        stage.teardown()
        assert output.exists()

    def test_process_writes_valid_record(self, tmp_path: Path):
        p = _make_rgb_jpeg(tmp_path)
        output = tmp_path / "results.jsonl"
        stage = ResultWriterStage(str(output), single_file=True)
        self._setup_stage(stage)
        task = _make_image_task(p, image_id="my_img")
        stage.process(task)
        stage.teardown()
        lines = output.read_text().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["image_id"] == "my_img"
        assert "is_valid" not in rec  # always stripped

    def test_process_valid_only_skips_invalid(self, tmp_path: Path):
        p = _make_rgb_jpeg(tmp_path)
        output = tmp_path / "results.jsonl"
        stage = ResultWriterStage(str(output), valid_only=True, single_file=True)
        self._setup_stage(stage)
        task = _make_image_task(p, is_valid=False)
        stage.process(task)
        stage.teardown()
        lines = [line for line in output.read_text().splitlines() if line.strip()]
        assert len(lines) == 0

    def test_process_valid_only_false_writes_invalid(self, tmp_path: Path):
        p = _make_rgb_jpeg(tmp_path)
        output = tmp_path / "results.jsonl"
        stage = ResultWriterStage(str(output), valid_only=False, single_file=True)
        self._setup_stage(stage)
        task = _make_image_task(p, is_valid=False)
        stage.process(task)
        stage.teardown()
        lines = [line for line in output.read_text().splitlines() if line.strip()]
        assert len(lines) == 1

    def test_image_path_relative_to_image_parent(self, tmp_path: Path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        p = _make_rgb_jpeg(img_dir)
        output = tmp_path / "results.jsonl"
        stage = ResultWriterStage(str(output), image_parent=str(img_dir), single_file=True)
        self._setup_stage(stage)
        task = _make_image_task(p)
        stage.process(task)
        stage.teardown()
        rec = json.loads(output.read_text().splitlines()[0])
        # Path should be relative (just "img.jpg", not the full path)
        assert rec["image_path"] == "img.jpg"

    def test_stats_counts_saved_and_skipped(self, tmp_path: Path):
        p = _make_rgb_jpeg(tmp_path)
        output = tmp_path / "stats.jsonl"
        stage = ResultWriterStage(str(output), valid_only=True, single_file=True)
        self._setup_stage(stage)
        stage.process(_make_image_task(p, task_id="t0", image_id="a", is_valid=True))
        stage.process(_make_image_task(p, task_id="t1", image_id="b", is_valid=False))
        stage.teardown()
        assert stage.stats["saved"] == 1
        assert stage.stats["skipped"] == 1

    def test_append_mode_does_not_truncate(self, tmp_path: Path):
        p = _make_rgb_jpeg(tmp_path)
        output = tmp_path / "out.jsonl"
        # First write
        s1 = ResultWriterStage(str(output), single_file=True)
        self._setup_stage(s1)
        s1.process(_make_image_task(p, task_id="t0", image_id="first"))
        s1.teardown()
        # Second write in append mode
        s2 = ResultWriterStage(str(output), single_file=True, append=True)
        self._setup_stage(s2)
        s2.process(_make_image_task(p, task_id="t1", image_id="second"))
        s2.teardown()
        lines = [line for line in output.read_text().splitlines() if line.strip()]
        assert len(lines) == 2

    def test_none_fields_excluded_from_output(self, tmp_path: Path):
        p = _make_rgb_jpeg(tmp_path)
        output = tmp_path / "none_test.jsonl"
        stage = ResultWriterStage(str(output), single_file=True)
        self._setup_stage(stage)
        task = _make_ocr_task(p)  # ocr_dense=None, scoring fields None
        stage.process(task)
        stage.teardown()
        rec = json.loads(output.read_text().splitlines()[0])
        # None fields should not appear
        assert "ocr_dense" not in rec
        assert "ocr_scoring_prompt" not in rec


# ---------------------------------------------------------------------------
# SkipProcessedStage
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


class TestSkipProcessedStage:
    def test_loads_processed_keys_from_file(self, tmp_path: Path):
        p = _make_rgb_jpeg(tmp_path)
        output = tmp_path / "done.jsonl"
        _write_jsonl(output, [{"image_path": str(p), "image_id": "0"}])

        stage = SkipProcessedStage(output_path=output)
        stage.setup(_make_worker_metadata())
        assert stage.stats["loaded"] == 1

    def test_skips_already_processed_task(self, tmp_path: Path):
        p = _make_rgb_jpeg(tmp_path)
        output = tmp_path / "done.jsonl"
        _write_jsonl(output, [{"image_path": str(p), "image_id": "0"}])

        stage = SkipProcessedStage(output_path=output)
        stage.setup(_make_worker_metadata())
        results = stage.process_batch([_make_image_task(p)])
        assert results == []
        assert stage.stats["skipped_existing"] == 1

    def test_passes_new_task(self, tmp_path: Path):
        p_done = _make_rgb_jpeg(tmp_path, "done.jpg")
        p_new = _make_rgb_jpeg(tmp_path, "new.jpg")
        output = tmp_path / "done.jsonl"
        _write_jsonl(output, [{"image_path": str(p_done), "image_id": "0"}])

        stage = SkipProcessedStage(output_path=output)
        stage.setup(_make_worker_metadata())
        results = stage.process_batch([_make_image_task(p_new)])
        assert len(results) == 1
        assert stage.stats["passed"] == 1

    def test_skips_within_run_duplicates(self, tmp_path: Path):
        p = _make_rgb_jpeg(tmp_path)
        output = tmp_path / "done.jsonl"
        _write_jsonl(output, [])

        stage = SkipProcessedStage(output_path=output)
        stage.setup(_make_worker_metadata())
        t1 = _make_image_task(p, task_id="t1")
        t2 = _make_image_task(p, task_id="t2")
        results = stage.process_batch([t1, t2])
        assert len(results) == 1  # second is deduplicated
        assert stage.stats["skipped_duplicate"] == 1

    def test_reads_worker_shards_when_merged_absent(self, tmp_path: Path):
        p = _make_rgb_jpeg(tmp_path)
        output = tmp_path / "results.jsonl"
        shard = tmp_path / "results_workerw0.jsonl"
        _write_jsonl(shard, [{"image_path": str(p), "image_id": "0"}])

        stage = SkipProcessedStage(output_path=output)
        stage.setup(_make_worker_metadata())
        assert stage.stats["loaded"] == 1

    def test_require_exists_true_raises_when_missing(self, tmp_path: Path):
        output = tmp_path / "nonexistent.jsonl"
        stage = SkipProcessedStage(output_path=output, require_exists=True)
        with pytest.raises(FileNotFoundError):
            stage.setup(_make_worker_metadata())

    def test_require_exists_false_silent_when_missing(self, tmp_path: Path):
        output = tmp_path / "nonexistent.jsonl"
        stage = SkipProcessedStage(output_path=output, require_exists=False)
        stage.setup(_make_worker_metadata())  # should not raise
        results = stage.process_batch([_make_image_task(_make_rgb_jpeg(tmp_path))])
        assert len(results) == 1  # everything passes through

    def test_bad_json_lines_are_ignored(self, tmp_path: Path):
        output = tmp_path / "bad.jsonl"
        output.write_text('{"image_path": "a.jpg"}\nBAD JSON\n{"image_path": "b.jpg"}\n')
        stage = SkipProcessedStage(output_path=output)
        stage.setup(_make_worker_metadata())
        assert stage.stats["loaded"] == 2  # only 2 valid lines parsed

    def test_image_path_relative_to_image_parent(self, tmp_path: Path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        p = _make_rgb_jpeg(img_dir)
        output = tmp_path / "done.jsonl"
        # The output file stores the relative path (as written by ResultWriterStage)
        _write_jsonl(output, [{"image_path": "img.jpg", "image_id": "0"}])

        stage = SkipProcessedStage(output_path=output, image_parent=img_dir)
        stage.setup(_make_worker_metadata())
        results = stage.process_batch([_make_image_task(p)])
        assert results == []  # correctly matched via relative key


# ---------------------------------------------------------------------------
# merge_output_shards
# ---------------------------------------------------------------------------


class TestMergeOutputShards:
    def _write_shard(self, directory: Path, stem: str, worker_id: str, lines: list[dict]) -> Path:
        p = directory / f"{stem}_worker{worker_id}.jsonl"
        _write_jsonl(p, lines)
        return p

    def test_merges_shards_into_single_file(self, tmp_path: Path):
        self._write_shard(tmp_path, "out", "0", [{"a": 1}])
        self._write_shard(tmp_path, "out", "1", [{"a": 2}])

        merged = merge_output_shards(tmp_path / "out.jsonl")
        lines = merged.read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"a": 1}
        assert json.loads(lines[1]) == {"a": 2}

    def test_deletes_shards_by_default(self, tmp_path: Path):
        s0 = self._write_shard(tmp_path, "out", "0", [{"x": 0}])
        s1 = self._write_shard(tmp_path, "out", "1", [{"x": 1}])
        merge_output_shards(tmp_path / "out.jsonl")
        assert not s0.exists()
        assert not s1.exists()

    def test_delete_shards_false_leaves_shards(self, tmp_path: Path):
        s0 = self._write_shard(tmp_path, "out", "0", [{"x": 0}])
        merge_output_shards(tmp_path / "out.jsonl", delete_shards=False)
        assert s0.exists()

    def test_returns_merged_path(self, tmp_path: Path):
        self._write_shard(tmp_path, "out", "0", [{"x": 0}])
        result = merge_output_shards(tmp_path / "out.jsonl")
        assert result == tmp_path / "out.jsonl"

    def test_no_shards_returns_output_path(self, tmp_path: Path):
        output = tmp_path / "out.jsonl"
        result = merge_output_shards(output)
        assert result == output
        assert not output.exists()  # nothing was created

    def test_appends_to_existing_merged_file(self, tmp_path: Path):
        output = tmp_path / "out.jsonl"
        _write_jsonl(output, [{"existing": True}])
        self._write_shard(tmp_path, "out", "0", [{"new": True}])
        merge_output_shards(output)
        lines = output.read_text().splitlines()
        assert len(lines) == 2

    def test_empty_shards_produce_empty_merged(self, tmp_path: Path):
        self._write_shard(tmp_path, "out", "0", [])
        merged = merge_output_shards(tmp_path / "out.jsonl")
        assert merged.read_text() == ""
