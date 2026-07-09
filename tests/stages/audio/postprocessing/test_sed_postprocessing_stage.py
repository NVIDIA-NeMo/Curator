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

"""Stage-level tests for SEDPostprocessingStage (real stage, synthetic framewise).

Complements ``test_sed_postprocessing.py`` (which unit-tests the numpy utility
functions in isolation) by exercising the actual ``SEDPostprocessingStage.process``
code path: in-memory framewise -> per-superclass event labels, subcategory mode,
and the framewise handoff key being dropped after use.
"""

from __future__ import annotations

import numpy as np

from nemo_curator.stages.audio.postprocessing.sed_postprocessing import SEDPostprocessingStage
from nemo_curator.tasks import AudioTask

CLASSES = 527
FPS = 50.0


def _task_with_speech_blob(start: int = 100, end: int = 200, total: int = 500) -> AudioTask:
    """AudioTask carrying framewise probs with a speech blob in [start, end)."""
    fw = np.full((total, CLASSES), 0.01, dtype=np.float32)
    fw[start:end, :7] = 0.9  # speech classes 0-6 high
    return AudioTask(
        task_id="t1",
        dataset_name="test",
        data={"_sed_framewise": fw, "sed_fps": FPS, "sed_valid_frames": total},
    )


class TestSEDPostprocessingStage:
    def test_detects_speech_event(self) -> None:
        stage = SEDPostprocessingStage(threshold=0.5, min_duration_sec=0.0)
        out = stage.process(_task_with_speech_blob())

        events = out.data["sed_events"]
        speech = [e for e in events if e["label"] == "speech"]
        assert len(speech) == 1
        assert abs(speech[0]["start_time"] - 2.0) < 0.05  # frame 100 / 50 fps
        assert abs(speech[0]["end_time"] - 4.0) < 0.05  # frame 200 / 50 fps
        assert speech[0]["mean_confidence"] > 0.8

    def test_framewise_key_is_dropped(self) -> None:
        stage = SEDPostprocessingStage(threshold=0.5, min_duration_sec=0.0)
        out = stage.process(_task_with_speech_blob())
        assert "_sed_framewise" not in out.data

    def test_events_sorted_by_start_time(self) -> None:
        stage = SEDPostprocessingStage(threshold=0.5, min_duration_sec=0.0)
        out = stage.process(_task_with_speech_blob())
        starts = [e["start_time"] for e in out.data["sed_events"]]
        assert starts == sorted(starts)

    def test_silence_yields_no_events(self) -> None:
        stage = SEDPostprocessingStage(threshold=0.5, min_duration_sec=0.0)
        fw = np.full((300, CLASSES), 0.01, dtype=np.float32)
        task = AudioTask(
            task_id="t1",
            dataset_name="test",
            data={"_sed_framewise": fw, "sed_fps": FPS, "sed_valid_frames": 300},
        )
        out = stage.process(task)
        assert out.data["sed_events"] == []

    def test_subcategory_mode_labels_parent_superclass(self) -> None:
        stage = SEDPostprocessingStage(threshold=0.5, min_duration_sec=0.0, emit_subcategories=True)
        out = stage.process(_task_with_speech_blob())

        events = out.data["sed_events"]
        assert events, "expected at least one subcategory event"
        speech_children = [e for e in events if e.get("superclass") == "speech"]
        assert speech_children
        assert all("label" in e and "superclass" in e for e in events)

    def test_missing_framewise_returns_empty(self) -> None:
        stage = SEDPostprocessingStage()
        task = AudioTask(task_id="t1", dataset_name="test", data={"audio_filepath": "a.wav"})
        out = stage.process(task)
        assert out.data["sed_events"] == []

    def test_reads_from_npz_fallback(self, tmp_path: object) -> None:
        # No in-memory _sed_framewise: the stage must load the sidecar NPZ instead.
        fw = np.full((500, CLASSES), 0.01, dtype=np.float32)
        fw[100:200, :7] = 0.9
        npz_path = str(tmp_path / "fw.npz")
        np.savez_compressed(
            npz_path,
            framewise=fw.astype(np.float16),
            fps=np.float32(FPS),
            valid_frames=np.int32(500),
        )
        stage = SEDPostprocessingStage(threshold=0.5, min_duration_sec=0.0)
        task = AudioTask(task_id="t1", dataset_name="test", data={"npz_filepath": npz_path})

        out = stage.process(task)

        speech = [e for e in out.data["sed_events"] if e["label"] == "speech"]
        assert len(speech) == 1
        assert abs(speech[0]["end_time"] - 4.0) < 0.05
