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

import pytest

from nemo_curator.stages.audio.metrics.get_wer import (
    GetPairwiseWerStage,
    get_cer,
    get_charrate,
    get_wer,
    get_wordrate,
)
from nemo_curator.tasks import AudioTask


def test_get_wer_basic() -> None:
    assert get_wer("a b c", "a x c") == 33.33


def test_get_cer_basic() -> None:
    assert get_cer("abc", "axc") == 33.33


def test_rates() -> None:
    assert get_charrate("abcd", 2.0) == 2.0
    assert get_wordrate("a b c d", 2.0) == 2.0


def test_pairwise_wer_validate_input_valid() -> None:
    stage = GetPairwiseWerStage()
    assert stage.validate_input(AudioTask(data={"text": "a b c", "pred_text": "a x c"})) is True


def test_pairwise_wer_validate_input_missing_text() -> None:
    stage = GetPairwiseWerStage()
    assert stage.validate_input(AudioTask(data={"pred_text": "a x c"})) is False


def test_pairwise_wer_validate_input_missing_pred_text() -> None:
    stage = GetPairwiseWerStage()
    assert stage.validate_input(AudioTask(data={"text": "a b c"})) is False


def test_pairwise_wer_process_batch_raises_on_missing_text() -> None:
    stage = GetPairwiseWerStage()
    with pytest.raises(ValueError, match="failed validation"):
        stage.process_batch([AudioTask(data={"pred_text": "a x c"})])


def test_pairwise_wer_process_batch_raises_on_missing_pred_text() -> None:
    stage = GetPairwiseWerStage()
    with pytest.raises(ValueError, match="failed validation"):
        stage.process_batch([AudioTask(data={"text": "a b c"})])


def test_pairwise_wer_stage() -> None:
    stage = GetPairwiseWerStage()
    entry = AudioTask(data={"text": "a b c", "pred_text": "a x c"})
    result = stage.process(entry)
    assert isinstance(result, AudioTask)
    assert result.data["wer"] == 33.33
