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

import usaddress
from presidio_analyzer import LocalRecognizer, RecognizerResult


class AddressRecognizer(LocalRecognizer):
    """
    Address Detector based on usaddress library

    Typical US address format:
    * The recipient's first and last name
    * Street number and name (address line 1)
    * Apartment or unit and its number (address line 2)
    * City, state and zip code (include all of this on one line with a comma between city and state, but not zip code)
    * Country
    """

    def load(self) -> None:
        pass

    def analyze(self, text: str, entities, nlp_artifacts) -> list[RecognizerResult]:  # noqa: ANN001, ARG002
        output = usaddress.parse(text)
        curr_pos = 0
        results = []

        for token, _type in output:
            token_pos_start = text.find(token, curr_pos)
            token_pos_end = token_pos_start + len(token)
            curr_pos = token_pos_end + 1
            if _type != "Recipient" and not _type.startswith("USPS"):
                result = RecognizerResult(
                    entity_type="ADDRESS",
                    start=token_pos_start,
                    end=token_pos_end,
                    score=1,
                )
                results.append(result)
        return results
