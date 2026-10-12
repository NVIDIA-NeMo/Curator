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

import types
from pathlib import Path

import numpy as np
import pytest

from nemo_curator.stages.audio.tts.sed import (
    AUDIOSET_LABELS,
    SPEECH_CLASSES,
    SegmentSEDStage,
    SpeechEventConfig,
    annotate_framewise,
    classify_events,
    detect_speech_events,
    empty_annotation,
    extract_features_for_segment,
    get_class_indices,
    seconds_to_frames,
    weighted_summary,
)
from nemo_curator.tasks import AudioTask
from tests.stages.audio.tts.test_audio_io import make_tar_segments

FPS = 50.0
SPEECH = AUDIOSET_LABELS.index("Speech")
MUSIC = AUDIOSET_LABELS.index("Music")


def framewise_with_speech(n_frames: int, blocks: list[tuple[int, int]], level: float = 0.9) -> np.ndarray:
    framewise = np.full((n_frames, len(AUDIOSET_LABELS)), 0.01, dtype=np.float16)
    for start, end in blocks:
        framewise[start:end, SPEECH] = level
    return framewise


class FakeClassifier:
    """sklearn-Pipeline-like: clean-speech probability grows with event duration."""

    def __init__(self, feature_columns: list[str]) -> None:
        self.named_steps = {"clf": types.SimpleNamespace(classes_=np.array([0, 1, 2, 3]))}
        self.duration_col = feature_columns.index("duration")
        self.seen: list[np.ndarray] = []

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        self.seen.append(x.copy())
        p_clean = min(1.0, float(x[0, self.duration_col]) / 2.0)
        rest = (1.0 - p_clean) / 3.0
        return np.array([[p_clean, rest, rest, rest]])


def fake_bundle() -> dict:
    columns = ["cm_not_extracted", "duration", "n_frames", "speech_nor_mean", "music_max_max"]
    return {"model": FakeClassifier(columns), "feature_columns": columns, "multiclass": True, "clean_class_id": 0}


def test_constants_layout() -> None:
    assert len(AUDIOSET_LABELS) == 527
    assert AUDIOSET_LABELS[:2] == ("Speech", "Male speech, man speaking")
    assert AUDIOSET_LABELS[7] == "Speech synthesizer"
    assert get_class_indices(AUDIOSET_LABELS, SPEECH_CLASSES) == [0, 1, 2, 3, 4, 5, 7]
    assert seconds_to_frames(0.2, FPS) == 11


def test_detects_and_merges_speech_events() -> None:
    # blocks 10-60 and 64-100 (gap 4 frames <= 5 -> merged); block 150-200 separate
    framewise = framewise_with_speech(250, [(10, 60), (64, 100), (150, 200)])
    events = detect_speech_events(framewise, FPS, AUDIOSET_LABELS)
    assert [(e["start_time"], e["end_time"]) for e in events] == [(0.2, 2.0), (3.0, 4.0)]
    assert all(e["label"] == "Speech" for e in events)
    # noisy-OR over the 7 speech classes: 1 - (1 - 0.9) * 0.99**6
    assert events[1]["max_confidence"] == pytest.approx(1 - 0.1 * 0.99**6, abs=1e-3)


def test_short_blips_removed_by_median_filter() -> None:
    framewise = framewise_with_speech(100, [(40, 44)])  # 4 frames < half of the 11-frame window
    assert detect_speech_events(framewise, FPS, AUDIOSET_LABELS) == []


def test_weighted_summary_uses_whole_segment_duration() -> None:
    events = [
        {"start_time": 0.0, "end_time": 2.0, "mean_confidence": 0.8, "classifier_prob": 0.5},
        {"start_time": 3.0, "end_time": 5.0, "mean_confidence": 0.4, "classifier_prob": 1.0},  # clipped at 4 s
    ]
    summary = weighted_summary(events, 4.0)
    assert summary["weighted_speech_rate"] == pytest.approx(0.75)
    assert summary["weighted_model_confidence"] == pytest.approx(0.5 * 0.8 + 0.25 * 0.4)
    assert summary["weighted_classifier_confidence"] == pytest.approx(0.5 * 0.5 + 0.25 * 1.0)
    assert weighted_summary(events, 0.0) == {
        "weighted_speech_rate": 0.0,
        "weighted_model_confidence": 0.0,
        "weighted_classifier_confidence": 0.0,
    }


def test_features_cover_groups_and_key_classes() -> None:
    framewise = framewise_with_speech(100, [(0, 100)])
    framewise[:, MUSIC] = 0.5
    labels = list(AUDIOSET_LABELS)
    features = extract_features_for_segment(
        framewise,
        FPS,
        labels,
        0.0,
        1.0,
        {label: i for i, label in enumerate(labels)},
        {"speech": get_class_indices(labels, SPEECH_CLASSES), "music": [MUSIC], "empty": []},
    )
    assert features["n_frames"] == 50
    assert features["speech_max_max"] == pytest.approx(0.9, abs=1e-3)
    assert features["music_max_frac_gt_0.3"] == 1.0
    assert np.isnan(features["empty_max_mean"])
    assert "class_Hubbub_speech_noise_speech_babble_mean" in features
    assert extract_features_for_segment(framewise, FPS, labels, 5.0, 6.0, {}, {}) == {}


def test_classifier_filters_and_zero_fills_missing_features() -> None:
    bundle = fake_bundle()
    events = [
        {"start_time": 0.0, "end_time": 0.2, "mean_confidence": 0.9},  # p=0.1 < threshold -> dropped
        {"start_time": 0.5, "end_time": 1.5, "mean_confidence": 0.9},  # p=0.5 kept
    ]
    kept = classify_events(events, framewise_with_speech(100, [(0, 100)]), AUDIOSET_LABELS, FPS, bundle)
    assert [e["classifier_prob"] for e in kept] == [pytest.approx(0.5)]
    assert events[0]["classifier_prob"] == pytest.approx(0.1)
    assert bundle["model"].seen[0][0, 0] == 0.0  # feature not produced by the extractor -> 0.0


def test_annotate_framewise_truncates_padding_and_scores() -> None:
    framewise = framewise_with_speech(64, [(0, 64)])  # speech everywhere incl. padded tail
    annotation = annotate_framewise(framewise, FPS, 0.5, valid_frames=25, bundle=fake_bundle())
    assert annotation["error"] is None
    assert [(e["start_time"], e["end_time"]) for e in annotation["predicted_events"]] == [(0.0, 0.5)]
    assert annotation["weighted_speech_rate"] == pytest.approx(1.0)
    assert annotation["weighted_classifier_confidence"] == pytest.approx(0.25)


class FakeCnn14:
    """Returns speech on the first half of the frames; records input lengths."""

    def __init__(self) -> None:
        self.lengths: list[int] = []

    def to(self, _device: object) -> "FakeCnn14":
        return self

    def eval(self) -> None:
        pass

    def __call__(self, x: object, _mixup: object) -> dict:
        import torch

        self.lengths.append(int(x.shape[1]))
        frames = int(x.shape[1]) // 320 + 1
        out = torch.full((1, frames, 527), 0.01)
        out[0, : frames // 2, SPEECH] = 0.95
        return {"framewise_output": out}


def make_stage(**kwargs: object) -> SegmentSEDStage:
    stage = SegmentSEDStage(classifier_path="unused.joblib", device="cpu", **kwargs)
    stage._model = FakeCnn14()
    stage._bundle = fake_bundle()
    stage.setup()
    return stage


def test_stage_annotates_segments_and_reports_audio_errors(tmp_path: Path) -> None:
    segments = make_tar_segments(tmp_path, [("a.wav", 2.0, 22050), ("b.wav", 0.3, 16000), ("c.wav", 1.0, 16000)])
    segments[2]["GranaryHifi"]["AudioMetadata"].pop("tar_member")
    task = AudioTask(data={"orig_sample_id": "vid", "segments": segments})
    stage = make_stage(framewise_key="sed_framewise")
    out = stage.process(task)
    a, b, c = (s["sed"] for s in out.data["segments"])
    assert a["error"] is None
    assert a["predicted_events"][0]["start_time"] == 0.0
    assert a["weighted_speech_rate"] == pytest.approx(0.5, abs=0.05)
    assert stage._model.lengths == [32000, 10240]  # 22.05 kHz resampled to 16 kHz; 0.3 s padded to 10240
    assert out.data["segments"][1]["sed_framewise"].shape[0] == 15  # ceil(4800 / 320) valid frames
    assert b["error"] is None
    assert c == empty_annotation("missing_tar_member")


def test_stage_requires_classifier_path_unless_disabled() -> None:
    with pytest.raises(ValueError, match="classifier_path"):
        SegmentSEDStage()
    stage = SegmentSEDStage(enable_classifier=False)
    assert stage.outputs() == ([], ["sed"])
    assert stage.fps == 50.0


def test_custom_detection_config_is_applied() -> None:
    framewise = framewise_with_speech(200, [(10, 60), (68, 100)])  # 8-frame gap survives the median filter
    assert len(detect_speech_events(framewise, FPS, AUDIOSET_LABELS)) == 2  # default merge gap 0.1 s = 5 frames
    config = SpeechEventConfig(merge_max_gap=0.2)
    assert len(detect_speech_events(framewise, FPS, AUDIOSET_LABELS, config)) == 1
