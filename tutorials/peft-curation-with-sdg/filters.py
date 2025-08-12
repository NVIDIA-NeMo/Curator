# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_curator.filters import DocumentFilter


class FilterLowScores(DocumentFilter):
    """
    Discards documents with scores (human-assigned, or reward model assiegned) below a threshold.
    """

    def __init__(self, score_threshold: int):
        super().__init__()
        self._score_threshold = score_threshold

    def score_document(self, text: str) -> bool:
        return text >= self._score_threshold

    def keep_document(self, score: bool) -> bool:
        return score
