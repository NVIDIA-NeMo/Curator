# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterator

from spacy.attrs import ORTH
from spacy.lang.en import English


nlp = English()
nlp.add_pipe("sentencizer")

# abbreviations confuse sentencizer
abbrs = (
    "ref. fig. sec. thm. obs. eq. mr. mrs. ms. dr. prep. prof. vs. etc. e.g. i.e. cf. al. "
    "Ref. Fig. Sec. Thm. Obs. Eq. Mr. Mrs. Ms. Dr. Prep. Prof."
)
for abbr in abbrs.split():
    nlp.tokenizer.add_special_case(abbr, [{ORTH: abbr}])

# prevent [E088] Text of length exceeds maximum
nlp.max_length = 10_000_000


def split_sentences(text: str) -> Iterator[slice]:
    """Split English text/paragraph into sentences, excluding whitespace."""
    doc = nlp(text)
    for sent in doc.sents:
        while sent.start < len(doc) and doc[sent.start].is_space:
            sent.start_char += len(doc[sent.start].text)
            sent.start += 1
        trim_ws = False
        while sent.end > 0 and doc[sent.end - 1].is_space:
            sent.end_char -= len(doc[sent.end - 1].text)
            sent.end -= 1
            trim_ws = True
        if trim_ws:
            # spacy handles trailing whitespace by default but if we changed
            # sentence end, need to take care of it ourselves
            sent.end_char -= len(doc[sent.end - 1].whitespace_)
        if sent.start_char < sent.end_char:
            yield slice(sent.start_char, sent.end_char)
