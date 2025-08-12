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

from nemo_curator.modifiers import DocumentModifier


class QuotationUnifier(DocumentModifier):
    """
    A simple modifier that unifies the quotation marks in the documents.
    """

    def modify_document(self, text: str) -> str:
        """
        Modifies the given text by replacing left and right single quotes with normal single quotes,
        and replacing left and right double quotes with normal double quotes.

        Args:
            text (str): The text to be modified.

        Returns:
            str: The modified text.
        """
        text = text.replace("‘", "'").replace("’", "'")  # noqa: RUF001
        return text.replace("“", '"').replace("”", '"')
