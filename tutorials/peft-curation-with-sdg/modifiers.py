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

import re
import warnings

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

from nemo_curator.modifiers import DocumentModifier

# Ignore the specific BeautifulSoup warning
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


class CleanHTML(DocumentModifier):
    """
    A simple modifier that removes HTML tags from the document.
    """

    def modify_document(self, text: str) -> str:
        """
        Removes HTML tags from the document.

        Args:
            text (str): The text to be modified.

        Returns:
            str: The modified text.
        """
        text = BeautifulSoup(text, "lxml").get_text()
        # Remove extra whitespaces and newlines.
        return re.sub(r"\s+", " ", text).strip()
