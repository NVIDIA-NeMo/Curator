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

# The system prompt template to be inserted into the documents.
SYS_PROMPT_TEMPLATE = """[INST] <<SYS>> You are reviewing the contents of an email. Based on the content, please categorize this email into one of the following categories:
1. 'Company Business/Strategy.'
2. 'Purely Personal.'
3. 'Personal but in a professional context.'
4. 'Logistic Arrangements.'
5. 'Employment arrangements.'
6. 'Document editing/checking/collaboration.'
Please provide only one category (e.g., 'Purely Personal.'). <</SYS>>

Content::
%s

What should this email be categorized as?
[/INST]
Answer:: """


class AddSystemPrompt(DocumentModifier):
    """
    A simple modifier that adds system prompts to each document.
    """

    def modify_document(self, text: str) -> str:
        """
        Inserts system prompts into the document.

        Args:
            text (str): The text to be modified.

        Returns:
            str: The modified text.
        """
        return SYS_PROMPT_TEMPLATE % text


class AddPeriod(DocumentModifier):
    """
    A simple modifier that adds a period to the end of each email category.
    """

    def modify_document(self, text: str) -> str:
        """
        Adds a period to the end of each email category.

        Args:
            text (str): The text to be modified.

        Returns:
            str: The modified text.
        """
        return text + "."
