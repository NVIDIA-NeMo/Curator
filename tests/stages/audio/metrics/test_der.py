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

from pathlib import Path

import pytest

from nemo_curator.stages.audio._agent._conformance import assert_agent_ready
from nemo_curator.stages.audio.metrics import DiarizationErrorRateStage
from nemo_curator.tasks import AudioTask

# Speaker A talks 0-10 s, speaker B 10-20 s.
_CHA = "@Begin\n*A:\thello there \x150_10000\x15\n*B:\thi \x1510000_20000\x15\n@End\n"


def _task(tmp_path: Path, segments: list[dict]) -> AudioTask:
    (tmp_path / "en_4065.cha").write_text(_CHA)
    return AudioTask(data={"audio_filepath": "/audio/en_4065.wav", "diar_segments": segments})


def test_perfect_prediction_with_other_labels_scores_zero(tmp_path: Path) -> None:
    pred = [{"start": 0.0, "end": 10.0, "speaker": "spk1"}, {"start": 10.0, "end": 20.0, "speaker": "spk0"}]
    stage = DiarizationErrorRateStage(cha_dir=str(tmp_path), collar=0.0)

    metrics = stage.process(_task(tmp_path, pred)).data["der_metrics"]

    assert metrics["der"] == pytest.approx(0.0, abs=0.2)
    assert (metrics["gt_speakers"], metrics["pred_speakers"]) == (2, 2)


def test_single_speaker_prediction_scores_unmapped_half(tmp_path: Path) -> None:
    stage = DiarizationErrorRateStage(cha_dir=str(tmp_path), collar=0.0)

    metrics = stage.process(_task(tmp_path, [{"start": 0.0, "end": 20.0, "speaker": "s"}])).data["der_metrics"]

    # The frame scorer counts the wrongly labelled half as both confusion and false alarm.
    assert (metrics["miss"], metrics["fa"], metrics["conf"], metrics["correct"]) == pytest.approx(
        (0.0, 50.0, 50.0, 50.0), abs=0.2
    )


def test_missing_reference_writes_none(tmp_path: Path) -> None:
    task = AudioTask(
        data={"audio_filepath": "/audio/other.wav", "diar_segments": [{"start": 0, "end": 1, "speaker": "s"}]}
    )

    assert DiarizationErrorRateStage(cha_dir=str(tmp_path)).process(task).data["der_metrics"] is None


def test_agent_ready_conformance(tmp_path: Path) -> None:
    task = _task(tmp_path, [{"start": 0.0, "end": 20.0, "speaker": "s"}])

    assert_agent_ready(
        DiarizationErrorRateStage(cha_dir=str(tmp_path)),
        fixture_factory=lambda: task,
        available_keys={"audio_filepath", "diar_segments"},
    )
