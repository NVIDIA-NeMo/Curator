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

from nemo_curator.stages.function_decorators import processing_stage
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.text.filters import DocumentFilter
from nemo_curator.tasks import DocumentBatch


# Filter by thinking ON or OFF
class ThinkingOnFilter(DocumentFilter):
    def __init__(self):
        super().__init__()

    def score_document(self, text: str) -> bool:
        return "on" in text.lower()

    def keep_document(self, score: bool) -> bool:
        return score


# Skip if not used for Nano training
class NanoFilter(DocumentFilter):
    def __init__(self):
        super().__init__()

    def score_document(self, text: str) -> bool:
        return "nano" in text.lower()

    def keep_document(self, score: bool) -> bool:
        return score


# Filter out samples with empty think tags
class EmptyThinkTagsFilter(DocumentFilter):
    def __init__(self):
        super().__init__()

    def score_document(self, text: str) -> bool:
        return not ("<think>\n\n</think>" in text or "<think>\n</think>" in text or "<think></think>" in text)

    def keep_document(self, score: bool) -> bool:
        return score


# Skip if malformed
# We use the @processing_stage decorator here instead of the DocumentFilter class
# because the MalformedFilter uses 2 text fields instead of 1
@processing_stage(name="MalformedFilter", resources=Resources(cpus=1.0), batch_size=1)
def malformed_filter(task: DocumentBatch) -> DocumentBatch:
    task.data["is_malformed"] = task.data["input"].str.contains(r"\\boxed", na=False) & ~task.data[
        "output"
    ].str.contains(r"\\boxed", na=False)
    task.data = task.data[~task.data["is_malformed"]]
    task.data = task.data.drop(columns=["is_malformed"], axis=1)
    return task


# Doesn't contain think close tag
class MissingThinkCloseTagFilter(DocumentFilter):
    def __init__(self):
        super().__init__()

    def score_document(self, text: str) -> bool:
        return not ("<think>" in text and "</think>" not in text)

    def keep_document(self, score: bool) -> bool:
        return score


# Reasoning off and contains think open tag
class ContainsThinkOpenTagFilter(DocumentFilter):
    def __init__(self):
        super().__init__()

    def score_document(self, text: str) -> bool:
        return not ("<think>" in text or "</think>" in text)

    def keep_document(self, score: bool) -> bool:
        return score


# Reasoning on and doesn't contain think open tag
class MissingThinkOpenTagFilter(DocumentFilter):
    def __init__(self):
        super().__init__()

    def score_document(self, text: str) -> bool:
        return not ("<think>" not in text or "</think>" not in text)

    def keep_document(self, score: bool) -> bool:
        return score
