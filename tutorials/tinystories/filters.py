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


class IncompleteStoryFilter(DocumentFilter):
    """
    If the document doesn't end with a terminating punctuation mark, then discard.
    """

    def __init__(self):
        super().__init__()
        # Accepted story terminators.
        self._story_terminators = {".", "!", "?", '"', "”"}

    def score_document(self, text: str) -> bool:
        """
        Determines if a document's score is valid based on the last character of the text.

        Args:
            text (str): The document text.

        Returns:
            bool: True if the document's score is valid, False otherwise.
        """
        return text.strip()[-1] in self._story_terminators

    def keep_document(self, score: bool) -> bool:
        return score
