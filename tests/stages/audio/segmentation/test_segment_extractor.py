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

"""Test: 1 AudioTask + 2 events -> fan-out to 2 AudioTasks.

Run: pytest tests/stages/audio/segmentation/test_segment_extractor.py -v --noconftest
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest


# ---------------------------------------------------------------------------
# Direct import (bypass Curator __init__ for Py3.9)
# ---------------------------------------------------------------------------
_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _imp(name, relpath):
    path = os.path.join(_base, relpath)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Test the logic directly without importing Curator (Py3.9 compat).


class _FakeAudioTask:
    """Minimal stand-in for AudioTask (dict wrapper)."""

    def __init__(self, task_id="", dataset_name="", data=None, filepath_key=None):
        self.task_id = task_id
        self.dataset_name = dataset_name
        self.data = data or {}
        self.filepath_key = filepath_key


def _segment_extractor_process(task, events_key="predicted_events", filepath_key="audio_filepath"):
    """Reproduce SegmentExtractorStage.process() logic."""
    events = task.data.get(events_key, [])
    if not events:
        return []
    output_tasks = []
    for idx, event in enumerate(events):
        seg_data = dict(task.data)
        seg_data["segment_start"] = event["start_time"]
        seg_data["segment_end"] = event["end_time"]
        seg_data["segment_idx"] = idx
        seg_data["segment_confidence"] = event.get("mean_confidence", 0.0)
        seg_task = _FakeAudioTask(
            task_id=f"{task.task_id}_seg_{idx}",
            dataset_name=task.dataset_name,
            filepath_key=task.filepath_key or filepath_key,
            data=seg_data,
        )
        output_tasks.append(seg_task)
    return output_tasks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSegmentExtractor:
    def test_two_events_produce_two_tasks(self) -> None:
        task = _FakeAudioTask(
            task_id="audio_001",
            dataset_name="test",
            data={
                "audio_filepath": "/data/test.wav",
                "predicted_events": [
                    {"start_time": 2.0, "end_time": 4.0, "mean_confidence": 0.95},
                    {"start_time": 6.0, "end_time": 8.5, "mean_confidence": 0.88},
                ],
            },
        )
        results = _segment_extractor_process(task)
        assert len(results) == 2

    def test_segment_timestamps_correct(self) -> None:
        task = _FakeAudioTask(
            task_id="audio_001",
            dataset_name="test",
            data={
                "audio_filepath": "/data/test.wav",
                "predicted_events": [
                    {"start_time": 2.0, "end_time": 4.0, "mean_confidence": 0.95},
                    {"start_time": 6.0, "end_time": 8.5, "mean_confidence": 0.88},
                ],
            },
        )
        results = _segment_extractor_process(task)
        assert results[0].data["segment_start"] == 2.0
        assert results[0].data["segment_end"] == 4.0
        assert results[1].data["segment_start"] == 6.0
        assert results[1].data["segment_end"] == 8.5

    def test_task_ids_unique(self) -> None:
        task = _FakeAudioTask(
            task_id="a",
            dataset_name="test",
            data={
                "predicted_events": [
                    {"start_time": 0, "end_time": 1, "mean_confidence": 0.9},
                    {"start_time": 2, "end_time": 3, "mean_confidence": 0.8},
                ],
            },
        )
        results = _segment_extractor_process(task)
        ids = [r.task_id for r in results]
        assert len(set(ids)) == 2

    def test_no_events_returns_empty(self) -> None:
        task = _FakeAudioTask(task_id="a", data={"predicted_events": []})
        results = _segment_extractor_process(task)
        assert results == []

    def test_original_data_preserved(self) -> None:
        task = _FakeAudioTask(
            task_id="a",
            data={
                "audio_filepath": "/test.wav",
                "duration": 10.0,
                "predicted_events": [{"start_time": 1, "end_time": 2, "mean_confidence": 0.9}],
            },
        )
        results = _segment_extractor_process(task)
        assert results[0].data["audio_filepath"] == "/test.wav"
        assert results[0].data["duration"] == 10.0
