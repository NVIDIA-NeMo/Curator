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

"""Tests for SEDPostprocessingStage: framewise probabilities to labelled events."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from nemo_curator.stages.audio.postprocessing import (
    SEDPostprocessingStage as PublicSEDPostprocessingStage,
)
from nemo_curator.stages.audio.postprocessing.sed_postprocessing import SEDPostprocessingStage
from nemo_curator.stages.audio.postprocessing.sed_utils import SUPERCLASS_GROUPS
from nemo_curator.tasks import AudioTask

_FPS = 50.0
_CLASSES = 527
_SPEECH_IDX = SUPERCLASS_GROUPS["speech"][0]
_MUSIC_IDX = SUPERCLASS_GROUPS["music"][0]


def _framewise(*, active: dict[int, tuple[int, int]], frames: int = 100) -> np.ndarray:
    """A (T, C) matrix that is confident for each class over its [start, stop) span."""
    matrix = np.full((frames, _CLASSES), 0.01, dtype=np.float32)
    for class_idx, (start, stop) in active.items():
        matrix[start:stop, class_idx] = 0.95
    return matrix


def _task(framewise: np.ndarray, **extra: object) -> AudioTask:
    return AudioTask(
        data={"_sed_framewise": framewise, "sed_fps": _FPS, "sed_valid_frames": framewise.shape[0], **extra}
    )


def _stage(**kwargs: object) -> SEDPostprocessingStage:
    return SEDPostprocessingStage(min_duration_sec=0.0, **kwargs)


# ----------------------------------------------------------------------
# Per-class labelling (the default)
# ----------------------------------------------------------------------


def test_a_confident_span_becomes_one_labelled_event() -> None:
    task = _stage().process(_task(_framewise(active={_SPEECH_IDX: (20, 60)})))
    (event,) = task.data["sed_events"]
    assert event["start_time"] == pytest.approx(0.4)
    assert event["end_time"] == pytest.approx(1.2)


def test_each_event_carries_a_class_label_and_its_superclass() -> None:
    task = _stage().process(_task(_framewise(active={_SPEECH_IDX: (20, 60)})))
    (event,) = task.data["sed_events"]
    assert event["superclass"] == "speech"
    assert isinstance(event["label"], str)
    assert event["label"]


def test_silence_produces_no_events() -> None:
    task = _stage().process(_task(np.full((100, _CLASSES), 0.01, dtype=np.float32)))
    assert task.data["sed_events"] == []


def test_two_classes_active_at_once_produce_two_events() -> None:
    task = _stage().process(_task(_framewise(active={_SPEECH_IDX: (10, 40), _MUSIC_IDX: (50, 80)})))
    superclasses = {event["superclass"] for event in task.data["sed_events"]}
    assert superclasses == {"speech", "music"}


def test_events_are_sorted_by_start_time() -> None:
    task = _stage().process(_task(_framewise(active={_MUSIC_IDX: (60, 90), _SPEECH_IDX: (10, 40)})))
    starts = [event["start_time"] for event in task.data["sed_events"]]
    assert starts == sorted(starts)


# ----------------------------------------------------------------------
# Superclass mode
# ----------------------------------------------------------------------


def test_superclass_mode_labels_events_by_group_only() -> None:
    task = _stage(emit_superclasses=True).process(_task(_framewise(active={_SPEECH_IDX: (20, 60)})))
    (event,) = task.data["sed_events"]
    assert event["label"] == "speech"
    assert "superclass" not in event


def test_superclass_mode_collapses_two_classes_of_one_group_into_a_single_event() -> None:
    """Per-class mode would emit two overlapping events; noisy-or merges them."""
    speech_a, speech_b = SUPERCLASS_GROUPS["speech"][:2]
    framewise = _framewise(active={speech_a: (20, 60), speech_b: (20, 60)})

    per_class = _stage().process(_task(framewise.copy())).data["sed_events"]
    grouped = _stage(emit_superclasses=True).process(_task(framewise.copy())).data["sed_events"]

    assert len([e for e in per_class if e["superclass"] == "speech"]) == 2
    assert len([e for e in grouped if e["label"] == "speech"]) == 1


# ----------------------------------------------------------------------
# Detection knobs
# ----------------------------------------------------------------------


def test_a_higher_threshold_suppresses_a_weak_detection() -> None:
    framewise = np.full((100, _CLASSES), 0.01, dtype=np.float32)
    framewise[20:60, _SPEECH_IDX] = 0.45
    assert _stage(threshold=0.4).process(_task(framewise.copy())).data["sed_events"]
    assert _stage(threshold=0.8).process(_task(framewise.copy())).data["sed_events"] == []


def test_min_duration_drops_a_brief_detection() -> None:
    # 5 frames at 50 fps = 0.1 s.
    framewise = _framewise(active={_SPEECH_IDX: (20, 25)})
    stage = SEDPostprocessingStage(min_duration_sec=0.3)
    assert stage.process(_task(framewise)).data["sed_events"] == []


def test_merge_gap_fuses_a_briefly_interrupted_detection() -> None:
    framewise = _framewise(active={_SPEECH_IDX: (10, 30)})
    framewise[35:60, _SPEECH_IDX] = 0.95

    split = _stage().process(_task(framewise.copy())).data["sed_events"]
    fused = _stage(merge_gap_sec=0.2).process(_task(framewise.copy())).data["sed_events"]

    assert len(split) == 2
    assert len(fused) == 1


# ----------------------------------------------------------------------
# Input sources and task plumbing
# ----------------------------------------------------------------------


def test_valid_frames_truncates_batch_padding_before_detection() -> None:
    """Frames past the real audio are model output over zero-padding, not signal."""
    framewise = _framewise(active={_SPEECH_IDX: (60, 100)})
    task = AudioTask(data={"_sed_framewise": framewise, "sed_fps": _FPS, "sed_valid_frames": 50})
    assert _stage().process(task).data["sed_events"] == []


def test_the_framewise_array_is_dropped_after_use() -> None:
    """It is the largest thing on the task and nothing downstream needs it."""
    task = _stage().process(_task(_framewise(active={_SPEECH_IDX: (20, 60)})))
    assert "_sed_framewise" not in task.data


def test_events_are_read_from_an_npz_sidecar_when_no_array_is_in_memory(tmp_path: Path) -> None:
    npz_path = tmp_path / "framewise.npz"
    np.savez_compressed(
        npz_path,
        framewise=_framewise(active={_SPEECH_IDX: (20, 60)}),
        fps=np.float32(_FPS),
        valid_frames=np.int32(100),
    )
    task = AudioTask(data={"npz_filepath": str(npz_path)})
    assert _stage().process(task).data["sed_events"]


def test_an_in_memory_array_is_preferred_over_the_sidecar(tmp_path: Path) -> None:
    npz_path = tmp_path / "framewise.npz"
    np.savez_compressed(
        npz_path,
        framewise=np.full((100, _CLASSES), 0.01, dtype=np.float32),
        fps=np.float32(_FPS),
        valid_frames=np.int32(100),
    )
    task = _task(_framewise(active={_SPEECH_IDX: (20, 60)}), npz_filepath=str(npz_path))
    assert _stage().process(task).data["sed_events"]


def test_a_task_with_no_framewise_data_yields_no_events() -> None:
    assert _stage().process(AudioTask(data={})).data["sed_events"] == []


def test_a_missing_sidecar_path_yields_no_events(tmp_path: Path) -> None:
    task = AudioTask(data={"npz_filepath": str(tmp_path / "missing.npz")})
    assert _stage().process(task).data["sed_events"] == []


def test_fp16_framewise_output_is_accepted() -> None:
    """The inference stage stores float16 by default."""
    framewise = _framewise(active={_SPEECH_IDX: (20, 60)}).astype(np.float16)
    assert _stage().process(_task(framewise)).data["sed_events"]


def test_process_batch_labels_each_task_independently() -> None:
    tasks = [
        _task(_framewise(active={_SPEECH_IDX: (20, 60)})),
        _task(np.full((100, _CLASSES), 0.01, dtype=np.float32)),
    ]
    results = _stage().process_batch(tasks)
    assert results[0].data["sed_events"]
    assert results[1].data["sed_events"] == []


def test_stage_declares_its_exact_io_contract() -> None:
    stage = SEDPostprocessingStage(events_key="my_events")
    assert stage.inputs() == (["data"], [])
    assert stage.outputs() == (["data"], ["my_events"])


def test_events_are_written_under_a_configurable_key() -> None:
    task = SEDPostprocessingStage(min_duration_sec=0.0, events_key="my_events").process(
        _task(_framewise(active={_SPEECH_IDX: (20, 60)}))
    )
    assert task.data["my_events"]


def test_stage_is_cpu_only() -> None:
    assert SEDPostprocessingStage().resources.gpus == 0


def test_postprocessing_package_exports_the_stage() -> None:
    assert PublicSEDPostprocessingStage is SEDPostprocessingStage


def test_importing_the_stage_needs_no_torch_dependency() -> None:
    """Check in a clean interpreter because other audio tests import torch."""
    probe = (
        "import sys;"
        "from nemo_curator.stages.audio.postprocessing import SEDPostprocessingStage;"
        "print('torch' in sys.modules, 'torchlibrosa' in sys.modules)"
    )
    result = subprocess.run(  # noqa: S603 - fixed probe run under the test interpreter
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False False"
