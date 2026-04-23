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

"""Unit tests for CheckpointManager and checkpoint stages.

Tests cover:
- Basic load/read/write lifecycle
- Single fan-out: 1 source → N leaves; all must complete
- Partial fan-out: 1 source → N leaves; M < N complete → not done
- Chained fan-outs: 1 source → Stage1: 10 batches → Stage2: each 3 sub-batches → 30 leaves
- _CheckpointFilterStage: skips completed, passes incomplete
- _CheckpointRecorderStage: writes shard, passes task through
- FilePartitioningStage: hash-based task IDs are stable across runs
"""

from __future__ import annotations

import json
from pathlib import Path

import ray

from nemo_curator.tasks import FileGroupTask
from nemo_curator.utils.checkpoint import CheckpointManager, _path_join

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _source_key(source_files: list[str]) -> str:
    return "|".join(sorted(source_files))


# ---------------------------------------------------------------------------
# _path_join
# ---------------------------------------------------------------------------


class TestPathJoin:
    def test_local_path(self):
        assert _path_join("/a/b", "c", "d.json") == "/a/b/c/d.json"

    def test_remote_path(self):
        result = _path_join("s3://bucket/prefix", "sub", "file.json")
        assert result == "s3://bucket/prefix/sub/file.json"

    def test_trailing_slash_stripped(self):
        assert _path_join("s3://bucket/prefix/", "file.json") == "s3://bucket/prefix/file.json"


# ---------------------------------------------------------------------------
# CheckpointManager — basic write/read cycle
# ---------------------------------------------------------------------------


class TestCheckpointManagerBasic:
    def test_exists_returns_false_for_nonexistent(self, tmp_path: Path):
        assert not CheckpointManager.exists(str(tmp_path / "nonexistent"))

    def test_exists_returns_true_after_creation(self, tmp_path: Path):
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        assert CheckpointManager.exists(str(ckpt_dir))

    def test_load_empty_directory(self, tmp_path: Path):
        mgr = CheckpointManager(str(tmp_path / "ckpt")).load()
        assert mgr._expected == {}
        assert dict(mgr._completed_counts) == {}

    def test_is_task_completed_returns_false_when_nothing_recorded(self, tmp_path: Path):
        mgr = CheckpointManager(str(tmp_path / "ckpt")).load()
        assert not mgr.is_task_completed(["file_a.tar"])

    def test_is_task_completed_returns_false_for_empty_source_files(self, tmp_path: Path):
        mgr = CheckpointManager(str(tmp_path / "ckpt")).load()
        assert not mgr.is_task_completed([])

    def test_mark_completed_then_load(self, tmp_path: Path):
        ckpt_path = str(tmp_path / "ckpt")
        mgr = CheckpointManager(ckpt_path)
        mgr.mark_completed("task_001", ["file_a.tar"])

        mgr2 = CheckpointManager(ckpt_path).load()
        assert mgr2.is_task_completed(["file_a.tar"])

    def test_mark_completed_multiple_sources(self, tmp_path: Path):
        ckpt_path = str(tmp_path / "ckpt")
        mgr = CheckpointManager(ckpt_path)
        mgr.mark_completed("task_a", ["file_a.tar"])
        mgr.mark_completed("task_b", ["file_b.tar"])

        mgr2 = CheckpointManager(ckpt_path).load()
        assert mgr2.is_task_completed(["file_a.tar"])
        assert mgr2.is_task_completed(["file_b.tar"])
        assert not mgr2.is_task_completed(["file_c.tar"])

    def test_source_key_ordering_is_stable(self, tmp_path: Path):
        """source_key uses sorted order so ["b","a"] == ["a","b"]."""
        ckpt_path = str(tmp_path / "ckpt")
        mgr = CheckpointManager(ckpt_path)
        mgr.mark_completed("task_x", ["b.tar", "a.tar"])

        mgr2 = CheckpointManager(ckpt_path).load()
        # Both orderings should match
        assert mgr2.is_task_completed(["a.tar", "b.tar"])
        assert mgr2.is_task_completed(["b.tar", "a.tar"])


# ---------------------------------------------------------------------------
# CheckpointManager — single fan-out (1 source → N leaves)
# ---------------------------------------------------------------------------


class TestCheckpointManagerSingleFanOut:
    def test_single_fanout_not_complete_until_all_leaves_recorded(self, tmp_path: Path):
        """1 source → 5 leaves; after 4 completions, not done; after 5th, done."""
        ckpt_path = str(tmp_path / "ckpt")
        source = ["a.tar"]
        n = 5

        # Write increment: source → n leaves
        mgr_write = CheckpointManager(ckpt_path)
        mgr_write.write_expected_increment(
            source_key=_source_key(source),
            triggering_task_id="file_group_abc123",
            increment=n - 1,  # 4
        )

        # Write n-1 completions
        for i in range(n - 1):
            mgr_write.mark_completed(f"leaf_task_{i}", source)

        mgr_read = CheckpointManager(ckpt_path).load()
        # expected = 1 + (n-1) = n = 5; completed = 4 → not done
        assert not mgr_read.is_task_completed(source)

        # Write the last completion
        mgr_write.mark_completed(f"leaf_task_{n - 1}", source)
        mgr_read2 = CheckpointManager(ckpt_path).load()
        assert mgr_read2.is_task_completed(source)

    def test_no_increment_written_means_expected_is_1(self, tmp_path: Path):
        """Without secondary fan-out, 1 completion is enough."""
        ckpt_path = str(tmp_path / "ckpt")
        source = ["b.tar"]
        mgr = CheckpointManager(ckpt_path)
        mgr.mark_completed("single_leaf", source)

        mgr2 = CheckpointManager(ckpt_path).load()
        assert mgr2.is_task_completed(source)

    def test_partial_fanout_not_complete(self, tmp_path: Path):
        """7 out of 10 completed — should NOT be marked done."""
        ckpt_path = str(tmp_path / "ckpt")
        source = ["c.tar"]
        n = 10

        mgr = CheckpointManager(ckpt_path)
        mgr.write_expected_increment(_source_key(source), "trig_task_xyz", n - 1)
        for i in range(7):
            mgr.mark_completed(f"leaf_{i}", source)

        mgr2 = CheckpointManager(ckpt_path).load()
        assert not mgr2.is_task_completed(source)


# ---------------------------------------------------------------------------
# CheckpointManager — chained fan-outs
# ---------------------------------------------------------------------------


class TestCheckpointManagerChainedFanOut:
    def test_chained_fanout_correct_expected_count(self, tmp_path: Path):
        """1 source → Stage1: 10 batches → Stage2: each 3 sub-batches → 30 leaves.

        Stage1 fan-out: 1 FileGroupTask triggers N=10 → increment written once: +9
        Stage2 fan-out: each of 10 batches triggers N=3 → increment written 10 times: +2 each

        expected = 1 + 9 + 10*2 = 30
        """
        ckpt_path = str(tmp_path / "ckpt")
        source = ["d.tar"]
        key = _source_key(source)

        mgr = CheckpointManager(ckpt_path)

        # Stage1 fan-out: 1 input → 10 batches
        mgr.write_expected_increment(key, "file_group_d", increment=9)

        # Stage2 fan-out: each of 10 batches → 3 sub-batches
        for batch_i in range(10):
            mgr.write_expected_increment(key, f"batch_{batch_i}", increment=2)

        # Complete 29 out of 30 leaves
        for leaf_i in range(29):
            mgr.mark_completed(f"sub_batch_{leaf_i}", source)

        mgr2 = CheckpointManager(ckpt_path).load()
        assert not mgr2.is_task_completed(source)  # 29 < 30

        # Complete the 30th
        mgr.mark_completed("sub_batch_29", source)
        mgr3 = CheckpointManager(ckpt_path).load()
        assert mgr3.is_task_completed(source)  # 30 >= 30

    def test_chained_fanout_partial_second_stage(self, tmp_path: Path):
        """Same chained setup but only Stage1 completes partially — not done."""
        ckpt_path = str(tmp_path / "ckpt")
        source = ["e.tar"]
        key = _source_key(source)

        mgr = CheckpointManager(ckpt_path)

        # Only the Stage1 increment (no Stage2 increments yet, only 5 Stage2 batches written)
        mgr.write_expected_increment(key, "fg_e", increment=9)  # expected so far: 10
        for batch_i in range(5):
            mgr.write_expected_increment(key, f"batch_e_{batch_i}", increment=2)
        # expected now: 1+9+5*2 = 20 (Stage2 only ran for 5 of 10 batches before crash)

        # 15 completions
        for leaf_i in range(15):
            mgr.mark_completed(f"leaf_e_{leaf_i}", source)

        mgr2 = CheckpointManager(ckpt_path).load()
        # 15 < 20 — not complete
        assert not mgr2.is_task_completed(source)


# ---------------------------------------------------------------------------
# CheckpointManager — get_completed_source_keys
# ---------------------------------------------------------------------------


class TestCheckpointManagerCompletedKeys:
    def test_completed_keys_returns_only_fully_done(self, tmp_path: Path):
        ckpt_path = str(tmp_path / "ckpt")
        source_a = ["a.tar"]
        source_b = ["b.tar"]

        mgr = CheckpointManager(ckpt_path)

        # source_a: 1 completion (no fan-out) — done
        mgr.mark_completed("task_a", source_a)

        # source_b: fan-out=3, only 2 completed — not done
        mgr.write_expected_increment(_source_key(source_b), "trig_b", increment=2)
        mgr.mark_completed("task_b1", source_b)
        mgr.mark_completed("task_b2", source_b)

        mgr2 = CheckpointManager(ckpt_path).load()
        completed_keys = mgr2.get_completed_source_keys()
        assert _source_key(source_a) in completed_keys
        assert _source_key(source_b) not in completed_keys


# ---------------------------------------------------------------------------
# _CheckpointFilterStage
# ---------------------------------------------------------------------------


class TestCheckpointFilterStage:
    def _make_task(self, task_id: str, source_files: list[str]) -> FileGroupTask:
        """Create a minimal FileGroupTask-like object for testing."""
        return FileGroupTask(
            task_id=task_id,
            dataset_name="test_ds",
            data=source_files,
            _metadata={"source_files": source_files},
        )

    def test_filter_passes_incomplete_task(self, tmp_path: Path):
        from nemo_curator.stages.checkpoint import _CheckpointFilterStage

        ckpt_path = str(tmp_path / "ckpt")
        stage = _CheckpointFilterStage(checkpoint_path=ckpt_path)
        stage.setup()

        task = self._make_task("task_a", ["file_a.tar"])
        result = stage.process(task)
        assert result == [task]

    def test_filter_skips_completed_task(self, tmp_path: Path):
        from nemo_curator.stages.checkpoint import _CheckpointFilterStage

        ckpt_path = str(tmp_path / "ckpt")
        # Pre-record completion
        mgr = CheckpointManager(ckpt_path)
        mgr.mark_completed("task_a_leaf", ["file_a.tar"])

        stage = _CheckpointFilterStage(checkpoint_path=ckpt_path)
        stage.setup()

        task = self._make_task("task_a", ["file_a.tar"])
        result = stage.process(task)
        assert result == []

    def test_filter_passes_task_with_no_source_files(self, tmp_path: Path):
        from nemo_curator.stages.checkpoint import _CheckpointFilterStage

        ckpt_path = str(tmp_path / "ckpt")
        stage = _CheckpointFilterStage(checkpoint_path=ckpt_path)
        stage.setup()

        from nemo_curator.tasks import FileGroupTask

        task = FileGroupTask(
            task_id="no_src",
            dataset_name="ds",
            data=[],
            _metadata={},  # no source_files
        )
        result = stage.process(task)
        assert result == [task]

    def test_filter_partial_fanout_not_skipped(self, tmp_path: Path):
        """If only 3 of 5 sub-tasks are complete, the source partition is NOT skipped."""
        from nemo_curator.stages.checkpoint import _CheckpointFilterStage

        ckpt_path = str(tmp_path / "ckpt")
        source = ["multi.tar"]
        key = _source_key(source)

        mgr = CheckpointManager(ckpt_path)
        mgr.write_expected_increment(key, "trig", increment=4)  # expect 5
        for i in range(3):
            mgr.mark_completed(f"leaf_{i}", source)

        stage = _CheckpointFilterStage(checkpoint_path=ckpt_path)
        stage.setup()

        task = self._make_task("source_task", source)
        result = stage.process(task)
        assert result == [task]


# ---------------------------------------------------------------------------
# _CheckpointRecorderStage
# ---------------------------------------------------------------------------


class TestCheckpointRecorderStage:
    def _make_task(self, task_id: str, source_files: list[str]) -> FileGroupTask:
        from nemo_curator.tasks import FileGroupTask

        return FileGroupTask(
            task_id=task_id,
            dataset_name="test_ds",
            data=source_files,
            _metadata={"source_files": source_files},
        )

    def test_recorder_passes_task_through(self, tmp_path: Path):
        from nemo_curator.stages.checkpoint import _CheckpointRecorderStage

        ckpt_path = str(tmp_path / "ckpt")
        stage = _CheckpointRecorderStage(checkpoint_path=ckpt_path)
        stage.setup()

        task = self._make_task("task_z", ["z.tar"])
        result = stage.process(task)
        assert result is task

    def test_recorder_writes_completed_shard(self, tmp_path: Path):
        from nemo_curator.stages.checkpoint import _CheckpointRecorderStage

        ckpt_path = str(tmp_path / "ckpt")
        stage = _CheckpointRecorderStage(checkpoint_path=ckpt_path)
        stage.setup()

        task = self._make_task("task_z", ["z.tar"])
        stage.process(task)

        # Reload and verify
        mgr = CheckpointManager(ckpt_path).load()
        assert mgr.is_task_completed(["z.tar"])

    def test_recorder_no_shard_when_no_source_files(self, tmp_path: Path):
        from nemo_curator.stages.checkpoint import _CheckpointRecorderStage

        ckpt_path = str(tmp_path / "ckpt")
        stage = _CheckpointRecorderStage(checkpoint_path=ckpt_path)
        stage.setup()

        from nemo_curator.tasks import FileGroupTask

        task = FileGroupTask(
            task_id="no_src",
            dataset_name="ds",
            data=[],
            _metadata={},
        )
        stage.process(task)

        ckpt_dir = Path(ckpt_path)
        assert not ckpt_dir.exists() or len(list(ckpt_dir.glob("*.json"))) == 0

    def test_recorder_multiple_tasks_writes_multiple_shards(self, tmp_path: Path):
        from nemo_curator.stages.checkpoint import _CheckpointRecorderStage

        ckpt_path = str(tmp_path / "ckpt")
        stage = _CheckpointRecorderStage(checkpoint_path=ckpt_path)
        stage.setup()

        source = ["shared.tar"]
        for i in range(5):
            task = self._make_task(f"leaf_{i}", source)
            stage.process(task)

        # One file per source (not per task)
        ckpt_dir = Path(ckpt_path)
        shards = list(ckpt_dir.glob("*.json"))
        assert len(shards) == 1

        data = json.loads(shards[0].read_text())
        assert data["source_key"] == _source_key(source)
        assert len(data["completed"]) == 5

    def test_recorder_no_collision_across_sources(self, tmp_path: Path):
        """Regression test: tasks with the SAME task_id from DIFFERENT source partitions
        must NOT overwrite each other's checkpoint files.

        This is the image pipeline bug: ImageReaderStage produces ``image_batch_0``,
        ``image_batch_1``, ... starting from 0 for EVERY input tar file.  Without
        source_key prefixing, tar2's ``image_batch_0.json`` overwrites tar1's, so
        tar1 ends up with 0 completions recorded and is never skipped on re-run.
        """
        from nemo_curator.stages.checkpoint import _CheckpointRecorderStage

        ckpt_path = str(tmp_path / "ckpt")
        stage = _CheckpointRecorderStage(checkpoint_path=ckpt_path)
        stage.setup()

        # Simulate: 2 tar files, each producing 2 image batches
        # Both tar files produce tasks with identical task_ids (image_batch_0, image_batch_1)
        source_a = ["tar_a.tar"]
        source_b = ["tar_b.tar"]
        for batch_id in range(2):
            stage.process(self._make_task(f"image_batch_{batch_id}", source_a))
            stage.process(self._make_task(f"image_batch_{batch_id}", source_b))

        # 2 files — one per source (no collision between sources with same task_ids)
        ckpt_dir = Path(ckpt_path)
        shards = list(ckpt_dir.glob("*.json"))
        assert len(shards) == 2, (
            f"Expected 2 source files but got {len(shards)}. "
            "Likely collision: tasks from different source partitions overwrote each other."
        )

        # Each source has exactly 2 completions
        mgr = CheckpointManager(ckpt_path).load()
        assert mgr._completed_counts[_source_key(source_a)] == 2
        assert mgr._completed_counts[_source_key(source_b)] == 2


# ---------------------------------------------------------------------------
# FilePartitioningStage — hash-based task ID stability
# ---------------------------------------------------------------------------


class TestFilePartitioningHashTaskId:
    def test_task_id_stable_across_runs(self, tmp_path: Path):
        """Same file set → same task IDs regardless of invocation order."""
        files = [str(tmp_path / f"f{i}.jsonl") for i in range(5)]
        for f in files:
            Path(f).write_text("{}\n")

        from nemo_curator.stages.file_partitioning import FilePartitioningStage
        from nemo_curator.tasks import _EmptyTask

        empty = _EmptyTask(task_id="empty", dataset_name="ds", data=None, _metadata={})

        stage1 = FilePartitioningStage(file_paths=files, files_per_partition=1)
        stage2 = FilePartitioningStage(file_paths=files, files_per_partition=1)

        result1 = stage1.process(empty)
        result2 = stage2.process(empty)

        ids1 = {t.task_id for t in result1}
        ids2 = {t.task_id for t in result2}
        assert ids1 == ids2

    def test_task_id_independent_of_other_partitions(self, tmp_path: Path):
        """Adding a new file doesn't change the task IDs of existing partitions."""
        files_a = [str(tmp_path / f"a{i}.jsonl") for i in range(3)]
        for f in files_a:
            Path(f).write_text("{}\n")
        file_new = str(tmp_path / "new.jsonl")
        Path(file_new).write_text("{}\n")

        from nemo_curator.stages.file_partitioning import FilePartitioningStage
        from nemo_curator.tasks import _EmptyTask

        empty = _EmptyTask(task_id="empty", dataset_name="ds", data=None, _metadata={})

        stage_orig = FilePartitioningStage(file_paths=files_a, files_per_partition=1)
        stage_extended = FilePartitioningStage(file_paths=[*files_a, file_new], files_per_partition=1)

        result_orig = stage_orig.process(empty)
        result_extended = stage_extended.process(empty)

        orig_ids = {t.task_id for t in result_orig}
        extended_ids = {t.task_id for t in result_extended}

        # All original IDs must appear in the extended run
        assert orig_ids.issubset(extended_ids)
        # Extended has one more (the new file)
        assert len(extended_ids) == len(orig_ids) + 1

    def test_source_files_in_metadata(self, tmp_path: Path):
        """Every FileGroupTask must have source_files set in _metadata."""
        files = [str(tmp_path / f"f{i}.jsonl") for i in range(3)]
        for f in files:
            Path(f).write_text("{}\n")

        from nemo_curator.stages.file_partitioning import FilePartitioningStage
        from nemo_curator.tasks import _EmptyTask

        empty = _EmptyTask(task_id="empty", dataset_name="ds", data=None, _metadata={})
        stage = FilePartitioningStage(file_paths=files, files_per_partition=1)
        result = stage.process(empty)

        for task in result:
            assert "source_files" in task._metadata
            assert len(task._metadata["source_files"]) > 0

    def test_is_source_stage(self):
        from nemo_curator.stages.file_partitioning import FilePartitioningStage

        stage = FilePartitioningStage(file_paths=[])
        assert stage.is_source_stage() is True


# ---------------------------------------------------------------------------
# Bug-fix regression tests
# ---------------------------------------------------------------------------


class TestNoneReturnFromProcess:
    """Bug 1: process() returning None must not crash process_batch()."""

    def test_none_return_is_silently_dropped(self):
        from dataclasses import dataclass

        from nemo_curator.stages.base import ProcessingStage
        from nemo_curator.stages.resources import Resources
        from nemo_curator.tasks import Task

        @dataclass
        class _DummyTask(Task):
            name: str = "dummy"

            @property
            def num_items(self) -> int:
                return 1

            def validate(self) -> bool:
                return True

        @dataclass
        class _FilterAllStage(ProcessingStage):
            name: str = "filter_all"
            resources: Resources = None

            def __post_init__(self):
                if self.resources is None:
                    self.resources = Resources(cpus=0.1)

            def process(self, task):
                return None  # Filter every task

        stage = _FilterAllStage()
        tasks = [_DummyTask(task_id=f"t{i}", dataset_name="ds", data=None) for i in range(3)]
        results = stage.process_batch(tasks)
        assert results == [], f"Expected empty list, got {results}"

    def test_mixed_none_and_valid_returns(self):
        from dataclasses import dataclass

        from nemo_curator.stages.base import ProcessingStage
        from nemo_curator.stages.resources import Resources
        from nemo_curator.tasks import Task

        @dataclass
        class _DummyTask(Task):
            name: str = "dummy"

            @property
            def num_items(self) -> int:
                return 1

            def validate(self) -> bool:
                return True

        @dataclass
        class _KeepEvenStage(ProcessingStage):
            name: str = "keep_even"
            resources: Resources = None

            def __post_init__(self):
                if self.resources is None:
                    self.resources = Resources(cpus=0.1)

            def process(self, task):
                idx = int(task.task_id[1:])
                return task if idx % 2 == 0 else None

        stage = _KeepEvenStage()
        tasks = [_DummyTask(task_id=f"t{i}", dataset_name="ds", data=None) for i in range(4)]
        results = stage.process_batch(tasks)
        assert len(results) == 2
        assert all(r is not None for r in results)
        assert {r.task_id for r in results} == {"t0", "t2"}


class TestFanOutDetection:
    """Bug 2: fan-out increments must only fire when len(results) > len(tasks)."""

    def test_batch_size_gt1_no_fanout_writes_no_increment(self, tmp_path):
        """1:1 stage with batch_size=2 must NOT write any fan-out increment."""
        from dataclasses import dataclass

        from nemo_curator.backends.base import BaseStageAdapter
        from nemo_curator.stages.base import ProcessingStage
        from nemo_curator.stages.resources import Resources
        from nemo_curator.tasks import Task
        from nemo_curator.utils.checkpoint import CheckpointManager

        @dataclass
        class _DummyTask(Task):
            name: str = "dummy"

            @property
            def num_items(self) -> int:
                return 1

            def validate(self) -> bool:
                return True

        @dataclass
        class _IdentityStage(ProcessingStage):
            name: str = "identity"
            resources: Resources = None

            def __post_init__(self):
                if self.resources is None:
                    self.resources = Resources(cpus=0.1)

            def process(self, task):
                return task

        ckpt_path = str(tmp_path / "ckpt")
        stage = _IdentityStage()
        stage._checkpoint_path = ckpt_path
        stage._checkpoint_storage_options = {}

        adapter = BaseStageAdapter(stage)
        mgr = CheckpointManager(ckpt_path)
        adapter._write_checkpoint_mgr = mgr

        src_a = ["file_a.jsonl"]
        src_b = ["file_b.jsonl"]
        tasks = [
            _DummyTask(task_id="tA", dataset_name="ds", data=None, _metadata={"source_files": src_a}),
            _DummyTask(task_id="tB", dataset_name="ds", data=None, _metadata={"source_files": src_b}),
        ]
        results = adapter.process_batch(tasks)

        assert len(results) == 2
        ckpt_dir = tmp_path / "ckpt"
        assert not ckpt_dir.exists() or len(list(ckpt_dir.glob("*.json"))) == 0

    def test_true_fanout_single_input_writes_increment(self, tmp_path):
        """1→N stage with a single input task must write the correct increment."""
        from dataclasses import dataclass

        from nemo_curator.backends.base import BaseStageAdapter
        from nemo_curator.stages.base import ProcessingStage
        from nemo_curator.stages.resources import Resources
        from nemo_curator.tasks import Task
        from nemo_curator.utils.checkpoint import CheckpointManager

        @dataclass
        class _DummyTask(Task):
            name: str = "dummy"

            @property
            def num_items(self) -> int:
                return 1

            def validate(self) -> bool:
                return True

        @dataclass
        class _FanOutStage(ProcessingStage):
            name: str = "fanout"
            resources: Resources = None

            def __post_init__(self):
                if self.resources is None:
                    self.resources = Resources(cpus=0.1)

            def process(self, task):
                return [_DummyTask(task_id=f"child_{i}", dataset_name="ds", data=None) for i in range(3)]

        ckpt_path = str(tmp_path / "ckpt")
        stage = _FanOutStage()
        adapter = BaseStageAdapter(stage)
        mgr = CheckpointManager(ckpt_path)
        adapter._write_checkpoint_mgr = mgr

        task = _DummyTask(
            task_id="parent",
            dataset_name="ds",
            data=None,
            _metadata={"source_files": ["archive.tar"]},
        )
        results = adapter.process_batch([task])

        assert len(results) == 3
        ckpt_dir = tmp_path / "ckpt"
        shards = list(ckpt_dir.glob("*.json"))
        assert len(shards) == 1
        data = json.loads(shards[0].read_text())
        assert len(data["increments"]) == 1
        assert data["increments"][0]["increment"] == 2  # 3 results - 1


class TestSourceFilesPropagation:
    """Bug 3: source_files must not mix across source partitions in multi-task batches."""

    def test_single_input_propagates_to_all_outputs(self, tmp_path):
        from dataclasses import dataclass

        from nemo_curator.backends.base import BaseStageAdapter
        from nemo_curator.stages.base import ProcessingStage
        from nemo_curator.stages.resources import Resources
        from nemo_curator.tasks import Task

        @dataclass
        class _DummyTask(Task):
            name: str = "dummy"

            @property
            def num_items(self) -> int:
                return 1

            def validate(self) -> bool:
                return True

        @dataclass
        class _FanOutStage(ProcessingStage):
            name: str = "fanout"
            resources: Resources = None

            def __post_init__(self):
                if self.resources is None:
                    self.resources = Resources(cpus=0.1)

            def process(self, task):
                return [_DummyTask(task_id=f"c{i}", dataset_name="ds", data=None) for i in range(2)]

        stage = _FanOutStage()
        adapter = BaseStageAdapter(stage)
        task = _DummyTask(
            task_id="p",
            dataset_name="ds",
            data=None,
            _metadata={"source_files": ["a.tar"]},
        )
        results = adapter.process_batch([task])
        for r in results:
            assert r._metadata["source_files"] == ["a.tar"]

    def test_mixed_partition_batch_skips_propagation(self, tmp_path):
        """Multi-task batch with different source_files must not merge partition identities."""
        from dataclasses import dataclass

        from nemo_curator.backends.base import BaseStageAdapter
        from nemo_curator.stages.base import ProcessingStage
        from nemo_curator.stages.resources import Resources
        from nemo_curator.tasks import Task

        @dataclass
        class _DummyTask(Task):
            name: str = "dummy"

            @property
            def num_items(self) -> int:
                return 1

            def validate(self) -> bool:
                return True

        @dataclass
        class _IdentityStage(ProcessingStage):
            name: str = "identity"
            resources: Resources = None

            def __post_init__(self):
                if self.resources is None:
                    self.resources = Resources(cpus=0.1)

            def process(self, task):
                return task

        stage = _IdentityStage()
        adapter = BaseStageAdapter(stage)
        task_a = _DummyTask(task_id="a", dataset_name="ds", data=None, _metadata={"source_files": ["a.jsonl"]})
        task_b = _DummyTask(task_id="b", dataset_name="ds", data=None, _metadata={"source_files": ["b.jsonl"]})
        results = adapter.process_batch([task_a, task_b])

        # Each result should retain its own source_files, not get the merged union
        result_map = {r.task_id: r for r in results}
        assert result_map["a"]._metadata.get("source_files") == ["a.jsonl"]
        assert result_map["b"]._metadata.get("source_files") == ["b.jsonl"]

    def test_fanin_stage_gets_union(self, tmp_path):
        """Fan-in (N→1): output should get the union of all inputs' source_files."""
        from dataclasses import dataclass

        from nemo_curator.backends.base import BaseStageAdapter
        from nemo_curator.stages.base import ProcessingStage
        from nemo_curator.stages.resources import Resources
        from nemo_curator.tasks import Task

        @dataclass
        class _DummyTask(Task):
            name: str = "dummy"

            @property
            def num_items(self) -> int:
                return 1

            def validate(self) -> bool:
                return True

        @dataclass
        class _FanInStage(ProcessingStage):
            name: str = "fanin"
            resources: Resources = None

            def __post_init__(self):
                if self.resources is None:
                    self.resources = Resources(cpus=0.1)

            def process(self, task):  # required by abstract base
                return task

            def process_batch(self, tasks):
                return [_DummyTask(task_id="merged", dataset_name="ds", data=None)]

        stage = _FanInStage()
        adapter = BaseStageAdapter(stage)
        tasks = [
            _DummyTask(task_id=f"t{i}", dataset_name="ds", data=None, _metadata={"source_files": [f"f{i}.parquet"]})
            for i in range(3)
        ]
        results = adapter.process_batch(tasks)
        assert len(results) == 1
        assert sorted(results[0]._metadata["source_files"]) == ["f0.parquet", "f1.parquet", "f2.parquet"]


class TestFilteredTaskCheckpointing:
    """Bug 4: tasks filtered by a stage must count toward partition completion."""

    def test_mark_filtered_then_load(self, tmp_path):
        from nemo_curator.utils.checkpoint import CheckpointManager

        mgr = CheckpointManager(str(tmp_path / "ckpt"))
        mgr.mark_filtered("task_1", ["a.tar"])
        mgr2 = CheckpointManager(str(tmp_path / "ckpt")).load()
        assert mgr2.is_task_completed(["a.tar"])

    def test_filtered_counts_toward_fanout_expected(self, tmp_path):
        """Fan-out expected=3, 2 complete + 1 filtered → partition done."""
        from nemo_curator.utils.checkpoint import CheckpointManager

        ckpt = str(tmp_path / "ckpt")
        mgr = CheckpointManager(ckpt)
        mgr.write_expected_increment(source_key="a.tar", triggering_task_id="parent", increment=2)
        mgr.mark_completed("child_0", ["a.tar"])
        mgr.mark_completed("child_1", ["a.tar"])
        mgr.mark_filtered("child_2", ["a.tar"])

        mgr2 = CheckpointManager(ckpt).load()
        assert mgr2.is_task_completed(["a.tar"])

    def test_partial_filtered_not_complete(self, tmp_path):
        """Fan-out expected=3, only 1 complete + 1 filtered → NOT done."""
        from nemo_curator.utils.checkpoint import CheckpointManager

        ckpt = str(tmp_path / "ckpt")
        mgr = CheckpointManager(ckpt)
        mgr.write_expected_increment(source_key="a.tar", triggering_task_id="parent", increment=2)
        mgr.mark_completed("child_0", ["a.tar"])
        mgr.mark_filtered("child_1", ["a.tar"])
        # child_2 is neither completed nor filtered

        mgr2 = CheckpointManager(ckpt).load()
        assert not mgr2.is_task_completed(["a.tar"])

    def test_adapter_records_filtered_task(self, tmp_path):
        """BaseStageAdapter must write a filtered shard when a task produces 0 outputs."""
        from dataclasses import dataclass

        from nemo_curator.backends.base import BaseStageAdapter
        from nemo_curator.stages.base import ProcessingStage
        from nemo_curator.stages.resources import Resources
        from nemo_curator.tasks import Task
        from nemo_curator.utils.checkpoint import CheckpointManager

        @dataclass
        class _DummyTask(Task):
            name: str = "dummy"

            @property
            def num_items(self) -> int:
                return 1

            def validate(self) -> bool:
                return True

        @dataclass
        class _DropAllStage(ProcessingStage):
            name: str = "drop_all"
            resources: Resources = None

            def __post_init__(self):
                if self.resources is None:
                    self.resources = Resources(cpus=0.1)

            def process(self, task):
                return []

        ckpt_path = str(tmp_path / "ckpt")
        stage = _DropAllStage()
        adapter = BaseStageAdapter(stage)
        mgr = CheckpointManager(ckpt_path)
        adapter._write_checkpoint_mgr = mgr

        task = _DummyTask(
            task_id="t1",
            dataset_name="ds",
            data=None,
            _metadata={"source_files": ["f.tar"]},
        )
        results = adapter.process_batch([task])
        assert results == []

        ckpt_dir = tmp_path / "ckpt"
        shards = list(ckpt_dir.glob("*.json"))
        assert len(shards) == 1
        data = json.loads(shards[0].read_text())
        assert data["source_key"] == "f.tar"
        assert "t1" in data["filtered"]

    def test_get_completed_source_keys_includes_filtered_partitions(self, tmp_path):
        from nemo_curator.utils.checkpoint import CheckpointManager

        ckpt = str(tmp_path / "ckpt")
        mgr = CheckpointManager(ckpt)
        mgr.mark_filtered("dropped", ["only_filtered.tar"])

        mgr2 = CheckpointManager(ckpt).load()
        keys = mgr2.get_completed_source_keys()
        assert "only_filtered.tar" in keys


# ---------------------------------------------------------------------------
# Ray actor concurrency
# ---------------------------------------------------------------------------


@ray.remote
def _remote_write_increment(actor, source_key: str, batch_id: int) -> None:
    import ray

    ray.get(actor.write_expected_increment.remote(source_key, f"batch_{batch_id}", 1))


@ray.remote
def _remote_mark_filtered(actor, task_id: str, source_files: list) -> None:
    import ray

    ray.get(actor.mark_filtered.remote(task_id, source_files))


class TestCheckpointActorConcurrency:
    """Verify the Ray actor serializes concurrent writes with no lost updates."""

    def test_concurrent_increments_are_all_recorded(self, tmp_path):
        """20 parallel write_expected_increment calls must all land in the JSON file."""
        import ray

        from nemo_curator.utils.checkpoint import CheckpointManager, get_or_create_checkpoint_actor

        ray.init(ignore_reinit_error=True)
        try:
            ckpt_path = str(tmp_path / "ckpt")
            actor = get_or_create_checkpoint_actor(ckpt_path)
            source_key = "concurrent.tar"
            n = 20

            ray.get([_remote_write_increment.remote(actor, source_key, i) for i in range(n)])

            mgr = CheckpointManager(ckpt_path).load()
            # expected = 1 + n x 1 = 21; all n increments must be recorded
            assert mgr._expected.get(source_key, 1) == 1 + n
        finally:
            ray.shutdown()

    def test_concurrent_mark_filtered_are_all_recorded(self, tmp_path):
        """20 parallel mark_filtered calls must all be written without clobbering each other."""
        import ray

        from nemo_curator.utils.checkpoint import CheckpointManager, get_or_create_checkpoint_actor

        ray.init(ignore_reinit_error=True)
        try:
            ckpt_path = str(tmp_path / "ckpt")
            actor = get_or_create_checkpoint_actor(ckpt_path)
            source = ["shared.tar"]
            source_key = _source_key(source)
            n = 20

            # Write increment so the completion threshold is high enough
            ray.get(actor.write_expected_increment.remote(source_key, "parent", n - 1))
            # Concurrently mark n tasks as filtered
            ray.get([_remote_mark_filtered.remote(actor, f"task_{i}", source) for i in range(n)])

            mgr = CheckpointManager(ckpt_path).load()
            assert mgr._filtered_counts[source_key] == n
            assert mgr.is_task_completed(source)
        finally:
            ray.shutdown()
