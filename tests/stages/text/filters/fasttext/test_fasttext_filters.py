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

from unittest.mock import Mock

import numpy as np
import pytest

from nemo_curator.stages.text.filters.fasttext import FastTextLangId


@pytest.mark.parametrize(
    ("label", "expected_language"),
    [
        ("__label__en", "en"),
        ("__label__eng_Latn", "eng_Latn"),
    ],
)
def test_score_document_preserves_complete_fasttext_label(label: str, expected_language: str) -> None:
    lang_id = FastTextLangId(model_path="model.bin")
    lang_id._fasttext_langid_model = Mock()
    lang_id._fasttext_langid_model.predict.return_value = ([[label]], [np.array([0.9])])

    assert lang_id.score_document("Hello, world!") == str([0.9, expected_language])


@pytest.mark.parametrize(
    ("language_filter", "prediction", "expected"),
    [
        ("en", "en", True),
        ("EN", "en", True),
        ("eng", "eng_Latn", True),
        ("eng_Latn", "eng_Latn", True),
        ("ENG_LATN", "eng_Latn", True),
        ("eng_Cyrl", "eng_Latn", False),
        ("deu", "eng_Latn", False),
    ],
)
def test_keep_document_filters_language_or_language_script(
    language_filter: str, prediction: str, expected: bool
) -> None:
    lang_id = FastTextLangId(model_path="model.bin", lang=language_filter)

    assert lang_id.keep_document(str([0.9, prediction])) is expected


def test_keep_document_applies_score_cutoff_with_glotlid_label() -> None:
    lang_id = FastTextLangId(model_path="model.bin", min_langid_score=0.8, lang="eng")

    assert not lang_id.keep_document(str([0.7, "eng_Latn"]))
