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


class Slicer(DocumentModifier):
    """
    Slices a document based on indices or strings.
    """

    def __init__(
        self,
        left: int | str | None = 0,
        right: int | str | None = None,
        include_left: bool = True,
        include_right: bool = True,
        strip: bool = True,
    ):
        """
        Args:
            left (Union[int, str], optional): If the provided value is an int, slice the string from this index (inclusive).
                If the provided value is a str, slice the string from the first occurence of this substring.
            right (Union[int, str], optional): If the provided value is an int, slice the string to this index (exclusive).
                If the provided value is a str, slice the string to the last occurence of this substring. If None,
                right is set to the length of the string.
            include_left (bool): Only used if `left` is a string. If True, the value of `left` is included in the
                slicing result. Defaults to False.
            include_right (bool): Only used if `right` is a string. If True, the value of `right` is included in the
                slicing result. Defaults to False.
            strip (bool): If True, strip the resulting string.
        """
        super().__init__()
        self._left = left
        self._right = right
        self._include_left = include_left
        self._include_right = include_right
        self._strip = strip

    def modify_document(self, text: str) -> str:
        # Determine start index based on left type
        if isinstance(self._left, int):
            left_index = self._left
        elif isinstance(self._left, str):
            left_index_found = text.find(self._left)
            if left_index_found == -1:
                return ""
            left_index = left_index_found if self._include_left else left_index_found + len(self._left)
        else:
            left_index = 0  # default if neither int nor str

        # Determine end index based on right type
        if isinstance(self._right, int):
            right_index = self._right
        elif isinstance(self._right, str):
            right_index_found = text.rfind(self._right)
            if right_index_found == -1:
                return ""
            right_index = right_index_found + len(self._right) if self._include_right else right_index_found
        else:
            right_index = len(text)  # default if neither int nor str

        result = text[left_index:right_index]
        if self._strip:
            result = result.strip()
        return result
