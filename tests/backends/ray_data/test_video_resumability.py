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

"""End-to-end resumability tests for a video pipeline.

Pipeline shape (mirrors test_text_resumability.py for the video domain):

    FilePartitioningStage           # REAL source fan-out (1 _EmptyTask -> N FileGroupTasks)
      -> VideoReaderStage           # REAL 1->1 (FileGroupTask -> VideoTask, ffprobe metadata)
      -> _FanOutVideoStage          # TEST helper fan-out A
      -> _SecondFanOutVideoStage    # TEST helper fan-out B
      -> _ActorPassThroughVideoStage  # forces actor-style so checkpoint actor wires up

Tiny synthetic .mp4 fixtures are generated at runtime via ``ffmpeg -f lavfi``;
the test is skipped if ffmpeg or ffprobe is missing from the system PATH.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest

if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
    pytest.skip("ffmpeg/ffprobe not available", allow_module_level=True)

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.video.io.video_reader import VideoReaderStage
from nemo_curator.tasks import _EmptyTask
from nemo_curator.tasks.video import VideoTask
from nemo_curator.utils.checkpoint import CheckpointManager

if TYPE_CHECKING:
    from pathlib import Path


def _write_mp4_fixtures(data_dir: Path, num_files: int) -> None:
    """Write deterministic 1-second 64x64 color-source .mp4 fixtures.

    Uses ``ffmpeg -f lavfi -i color=...`` with libx264 (yuv420p) so ffprobe can
    read all required metadata fields. Each file's color is derived from its
    index, so SHA256-based ``resumability_key``s are stable across runs.
    """
    for f_idx in range(num_files):
        path = data_dir / f"video_{f_idx:03d}.mp4"
        r = (37 * f_idx) % 256
        g = (71 * f_idx + 13) % 256
        b = (113 * f_idx + 7) % 256
        color_hex = f"0x{r:02x}{g:02x}{b:02x}"
        # mpeg4 is broadly available in stripped ffmpeg builds; libx264 is not.
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color_hex}:s=64x64:d=1:r=10",
            "-c:v",
            "mpeg4",
            "-t",
            "1",
            str(path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)  # noqa: S603


class _FanOutVideoStage(ProcessingStage[VideoTask, VideoTask]):
    name = "fanout_video_a"
    resources = Resources(cpus=0.5)

    def __init__(self, factor: int = 3):
        self._factor = factor

    def process(self, task: VideoTask) -> list[VideoTask]:
        # Share the heavy Video object reference; downstream test stages don't mutate it.
        return [
            VideoTask(
                task_id=f"{task.task_id}_fanA{i}",
                dataset_name=task.dataset_name,
                data=task.data,
                _metadata=dict(task._metadata),
            )
            for i in range(self._factor)
        ]


class _SecondFanOutVideoStage(ProcessingStage[VideoTask, VideoTask]):
    name = "fanout_video_b"
    resources = Resources(cpus=0.5)

    def __init__(self, factor: int = 2):
        self._factor = factor

    def process(self, task: VideoTask) -> list[VideoTask]:
        return [
            VideoTask(
                task_id=f"{task.task_id}_fanB{i}",
                dataset_name=task.dataset_name,
                data=task.data,
                _metadata=dict(task._metadata),
            )
            for i in range(self._factor)
        ]


class _ActorPassThroughVideoStage(ProcessingStage[VideoTask, VideoTask]):
    """Pass-through with an overridden ``setup`` so ``is_actor_stage()`` is True.

    Ray-Data task-style stages skip ``BaseStageAdapter.setup``, which leaves
    ``_checkpoint_actor`` unset and disables ``_drop_completed_inputs`` on the
    last user stage. Forcing actor-style execution wires the actor up so the
    leaf-skip path actually runs in the second test.
    """

    name = "actor_passthrough_video"
    resources = Resources(cpus=0.5)

    def setup(self, _worker_metadata: object | None = None) -> None:
        return

    def process(self, task: VideoTask) -> VideoTask:
        return task


class _MarkFanoutLeafVideoStage(ProcessingStage[VideoTask, VideoTask]):
    """Mark the i_B=0 leaves of the second fan-out complete in the checkpoint DB.

    Inserted between the second fan-out and the last user stage to simulate
    partial leaf completion. The next stage's adapter (``_is_last_user_stage =
    True``) consults the DB via ``_drop_completed_inputs`` and skips the marked
    leaves, so only the unmarked ones reach the recorder.
    """

    name = "mark_fanout_leaf_video"
    resources = Resources(cpus=0.5)

    def __init__(self, checkpoint_path: str, target_suffix: str = "_fanB0"):
        self._checkpoint_path = checkpoint_path
        self._target_suffix = target_suffix
        self._actor = None

    def setup(self, _worker_metadata: object | None = None) -> None:
        from nemo_curator.utils.checkpoint import get_or_create_checkpoint_actor

        self._actor = get_or_create_checkpoint_actor(self._checkpoint_path)

    def process(self, task: VideoTask) -> VideoTask:
        if task.task_id.endswith(self._target_suffix):
            from nemo_curator.utils.checkpoint import _checkpoint_get

            key = task._metadata["resumability_key"]
            cur = task._metadata["resumability_task_key"]
            # This stage is 1->1, so _propagate_resumability_metadata will append
            # "::0" to task_key before the next stage sees it. Mark the post-
            # propagation leaf key so _drop_completed_inputs at the last user
            # stage finds and drops the task.
            future_task_key = f"{cur}::0"
            _checkpoint_get(self._actor.mark_completed.remote(future_task_key, key))
        return task


def _strip_recorder_suffix(task_key: str) -> str:
    """The auto-injected _CheckpointRecorderStage is itself 1->1, so its adapter
    propagates a trailing ``::0`` onto each output's ``resumability_task_key``.
    Strip it to recover the key the recorder actually wrote to the DB.
    """
    assert task_key.endswith("::0"), f"unexpected task_key shape: {task_key!r}"
    return task_key[: -len("::0")]


def test_video_pipeline_resumability_with_multiple_fanouts(
    tmp_path: Path,
    shared_ray_client: None,  # noqa: ARG001
) -> None:
    """Run a multi-fan-out video pipeline twice; second run must skip all files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    ckpt_dir = tmp_path / "ckpt"

    num_files = 3
    _write_mp4_fixtures(data_dir, num_files=num_files)

    # ---- Compute the deterministic source-task keys -----------------------
    probe = FilePartitioningStage(
        file_paths=str(data_dir),
        files_per_partition=1,
        file_extensions=[".mp4"],
    )
    probe_tasks = probe.process(_EmptyTask(task_id="probe", dataset_name="video_dataset", data=None))
    expected_source_keys = {t._metadata["resumability_key"] for t in probe_tasks}
    assert len(expected_source_keys) == num_files

    factor_a = 3
    factor_b = 2

    def build_pipeline() -> Pipeline:
        p = Pipeline(name="video_resumability_test")
        p.add_stage(
            FilePartitioningStage(
                file_paths=str(data_dir),
                files_per_partition=1,
                file_extensions=[".mp4"],
            )
        )
        p.add_stage(VideoReaderStage(input_path=str(data_dir), verbose=False))
        p.add_stage(_FanOutVideoStage(factor=factor_a))
        p.add_stage(_SecondFanOutVideoStage(factor=factor_b))
        p.add_stage(_ActorPassThroughVideoStage())
        return p

    # ---- First run: process every video end-to-end -----------------------
    out1 = build_pipeline().run(RayDataExecutor(), checkpoint_path=str(ckpt_dir))
    assert out1 is not None
    expected_leaves = num_files * factor_a * factor_b
    assert len(out1) == expected_leaves

    leaf_pairs: list[tuple[str, str]] = [
        (t._metadata["resumability_key"], _strip_recorder_suffix(t._metadata["resumability_task_key"])) for t in out1
    ]
    observed_source_keys = {key for key, _ in leaf_pairs}

    mgr = CheckpointManager(str(ckpt_dir))
    try:
        assert observed_source_keys == expected_source_keys
        for key in expected_source_keys:
            assert mgr.is_task_completed(key), f"source key {key!r} not fully completed in checkpoint DB"

        leaf_flags = mgr.are_leaves_completed(leaf_pairs)
        missing = [pair for pair, ok in zip(leaf_pairs, leaf_flags, strict=True) if not ok]
        assert not missing, f"leaf keys missing from checkpoint DB: {missing}"
        assert len(set(leaf_pairs)) == expected_leaves, "expected unique leaf task_keys per fan-out output"
    finally:
        mgr.close()

    # ---- Second run: every video must be skipped -------------------------
    out2 = build_pipeline().run(RayDataExecutor(), checkpoint_path=str(ckpt_dir))
    assert out2 is not None
    assert len(out2) == 0, f"expected resume to skip all videos, got {len(out2)} output tasks"

    mgr2 = CheckpointManager(str(ckpt_dir))
    try:
        for key in expected_source_keys:
            assert mgr2.is_task_completed(key)
        assert all(mgr2.are_leaves_completed(leaf_pairs))
    finally:
        mgr2.close()


def test_video_pipeline_leaf_resumability_drops_completed_leaves(
    tmp_path: Path,
    shared_ray_client: None,  # noqa: ARG001
) -> None:
    """Leaves marked complete mid-pipeline are skipped by the last user stage.

    Pipeline:
        FilePartitioningStage -> VideoReaderStage
          -> _FanOutVideoStage(factor_a)
          -> _SecondFanOutVideoStage(factor_b)
          -> _MarkFanoutLeafVideoStage(target_suffix="_fanB0")
          -> _ActorPassThroughVideoStage

    The marker pre-records the i_B=0 fan-out leaf of every (video, fanA_i) pair.
    Both fan-outs fire and re-register expected; the marker records the leaves;
    the last user stage's adapter drops them via ``_drop_completed_inputs``;
    only ``factor_a * (factor_b - 1)`` leaves per video reach the recorder.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    ckpt_dir = tmp_path / "ckpt"

    num_files = 2
    _write_mp4_fixtures(data_dir, num_files=num_files)

    factor_a = 3
    factor_b = 2

    pipeline = Pipeline(name="video_leaf_resumability_test")
    pipeline.add_stage(
        FilePartitioningStage(
            file_paths=str(data_dir),
            files_per_partition=1,
            file_extensions=[".mp4"],
        )
    )
    pipeline.add_stage(VideoReaderStage(input_path=str(data_dir), verbose=False))
    pipeline.add_stage(_FanOutVideoStage(factor=factor_a))
    pipeline.add_stage(_SecondFanOutVideoStage(factor=factor_b))
    pipeline.add_stage(_MarkFanoutLeafVideoStage(str(ckpt_dir), target_suffix="_fanB0"))
    pipeline.add_stage(_ActorPassThroughVideoStage())

    out = pipeline.run(RayDataExecutor(), checkpoint_path=str(ckpt_dir))

    expected_count = num_files * factor_a * (factor_b - 1)
    assert out is not None
    assert len(out) == expected_count, f"expected {expected_count} surviving tasks, got {len(out)}"

    surviving_task_ids = {t.task_id for t in out}
    assert not any(tid.endswith("_fanB0") for tid in surviving_task_ids), (
        f"_fanB0 leaves should have been dropped at the last user stage, but found: {surviving_task_ids}"
    )

    # Per video, the marker records (factor_a) leaves and the recorder records
    # the remaining (factor_a * (factor_b - 1)) — total = factor_a * factor_b,
    # matching the expected count set by the two synchronous add_expected calls.
    source_keys = {t._metadata["resumability_key"] for t in out}
    assert len(source_keys) == num_files

    mgr = CheckpointManager(str(ckpt_dir))
    try:
        for key in source_keys:
            assert mgr.is_task_completed(key), f"video {key!r} not fully complete after run"
    finally:
        mgr.close()
