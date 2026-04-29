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

from dataclasses import dataclass

import editdistance

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


def get_wer(text: str, pred_text: str) -> float:
    text_words = text.split()
    pred_text_words = pred_text.split()
    if not text_words:
        return 0.0 if not pred_text_words else 100.0
    word_dist = editdistance.eval(text_words, pred_text_words)
    return round(word_dist / len(text_words) * 100.0, 2)


def get_cer(text: str, pred_text: str) -> float:
    if not text:
        return 0.0 if not pred_text else 100.0
    char_dist = editdistance.eval(text, pred_text)
    return round(char_dist / len(text) * 100.0, 2)


def get_charrate(text: str, duration: float) -> float:
    if duration == 0.0:
        return 0.0
    return round(len(text) / duration, 2)


def get_wordrate(text: str, duration: float) -> float:
    if duration == 0.0:
        return 0.0
    return round(len(text.split()) / duration, 2)


@dataclass
class GetPairwiseWerStage(ProcessingStage[AudioTask, AudioTask]):
    """Count pairwise word-error-rate (WER) * 100% for each pair of text and pred_text.

    WER is measured between ``data[self.text_key]`` and ``data[self.pred_text_key]``.

    Args:
        text_key: Key for the utterance transcript. Defaults to "text".
        pred_text_key: Key for the ASR predictions. Defaults to "pred_text".
        wer_key: Key to store the computed WER. Defaults to "wer".
    """

    name: str = "GetPairwiseWerStage"
    text_key: str = "text"
    pred_text_key: str = "pred_text"
    wer_key: str = "wer"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.text_key, self.pred_text_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.text_key, self.pred_text_key, self.wer_key]

    def process(self, task: AudioTask) -> AudioTask:
        task.data[self.wer_key] = get_wer(task.data[self.text_key], task.data[self.pred_text_key])
        return task
