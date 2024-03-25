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

from nemo_curator.modifiers.doc_modifier import DocumentModifier
from nemo_curator.utils.constants import policy_substrings
from nemo_curator.utils.text_utils import (
    get_paragraphs,
    is_paragraph_indices_in_top_or_bottom_only,
)


class BoilerPlateStringModifier(DocumentModifier):
    """
    If the sentence contains any of the boilerplate strings then discard.
    This includes things like "terms of use", "privacy policy", etc.
    Source: Adapted significantly from Google C4 processing.
    """

    def __init__(
        self,
        remove_if_at_top_or_bottom=True,
    ):
        super().__init__()
        self._remove_if_at_top_or_bottom = remove_if_at_top_or_bottom
        self._top_or_bottom_only = False
        self._boilerplate_paragraph_indices = []
        self._name = "boilerplate_string_ratio"

    def modify_document(self, text):
        # Initialize variables
        self._boilerplate_paragraph_indices = []

        # Return an empty string when the document should be removed entirely
        empty_string = ""

        # Get the paragraphs
        paragraphs = self._paragraphs
        if paragraphs is None:
            paragraphs = get_paragraphs(text)

        # Check each paragraph
        for idx, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip().lower()
            if "lorem ipsum" in paragraph:
                return empty_string
            if any(p in paragraph for p in policy_substrings):
                if not self._remove_if_at_top_or_bottom:
                    return empty_string
                else:
                    self._boilerplate_paragraph_indices.append(idx)

        # Keep the document if we did not find any boilerplate
        if len(self._boilerplate_paragraph_indices) == 0:
            return text

        # Mark if boilerplate is only at top or bottom
        self._top_or_bottom_only = is_paragraph_indices_in_top_or_bottom_only(
            self._boilerplate_paragraph_indices,
            len(paragraphs),
        )

        if self._top_or_bottom_only:
            # In case paragraphs is None, recompute it
            if self._paragraphs is None:
                self._paragraphs = get_paragraphs(text)
            modified_doc = "\n\n".join(
                [
                    p
                    for idx, p in enumerate(self._paragraphs)
                    if idx not in self._boilerplate_paragraph_indices
                ]
            )
            # Set the paragraphs back to None as the document has been
            # changed
        else:
            modified_doc = text

        self._paragraphs = None
        return modified_doc
