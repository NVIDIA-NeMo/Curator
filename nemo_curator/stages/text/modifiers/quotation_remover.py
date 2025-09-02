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


class QuotationRemover(DocumentModifier):
    """
    Removes quotations from a document following a few rules:
    - If the document is less than 2 characters, it is returned unchanged.
    - If the document starts and ends with a quotation mark and there are
        no newlines in the document, the quotation marks are removed.
    - If the document starts and ends with a quotation mark and there are
        newlines in the document, the quotation marks are removed only if
        the first line does not end with a quotation mark.
    """

    def __init__(self):
        super().__init__()

    def modify_document(self, text: str) -> str:
        if len(text.strip()) > 2 and text[0] == '"' and text[-1] == '"':  # noqa: SIM102, PLR2004
            if "\n" not in text.strip() or text.split("\n")[0][-1] != '"':
                text = text[1:-1]
        return text
