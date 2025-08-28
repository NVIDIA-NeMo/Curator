# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_curator.stages.text.modifiers.doc_modifier import DocumentModifier


class LineRemover(DocumentModifier):
    """
    Removes lines from a document if the content of the line matches a given string.
    """

    def __init__(self, patterns: list[str]):
        """
        Args:
            patterns (List[str]): The patterns to check
        """
        super().__init__()
        self._patterns = patterns

    def modify_document(self, text: str) -> str:
        lines = text.split("\n")
        new_lines = [line for line in lines if line not in self._patterns]
        return "\n".join(new_lines)
