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


class FilterEmailsWithLongBody(DocumentFilter):
    """
    If the email is too long, discard.
    """

    def __init__(self, max_length: int = 5000):
        super().__init__()
        self.max_length = max_length

    def score_document(self, text: str) -> bool:
        return len(text) <= self.max_length

    def keep_document(self, score) -> bool:
        return score


class FilterEmptyEmails(DocumentFilter):
    """
    Detects empty emails (either empty body, or labeled as empty). Returns `True` for empty emails.
    """

    def score_document(self, text: str) -> bool:
        return (
            not isinstance(text, str)  # The text is not a string
            or len(text.strip()) == 0  # The text is empty
            or "Empty message" in text  # The email is labeled as empty
        )

    def keep_document(self, score) -> bool:
        return score
