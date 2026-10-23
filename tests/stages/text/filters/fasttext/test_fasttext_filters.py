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
import pandas as pd
import pytest

from nemo_curator.stages.text.filters import DocumentFilter, ScoreFilter
from nemo_curator.stages.text.filters.fasttext import FastTextLangId
from nemo_curator.tasks import DocumentBatch


class FakeQualityFilter(DocumentFilter):
    """Emulate ``FastTextQualityFilter`` without loading a model."""

    def __init__(self, alpha: float = 3, seed: int = 42):
        super().__init__()
        self._alpha = alpha
        self._seed = np.random.seed(seed)  # noqa: NPY002

    def load_model(self) -> None:
        pass

    def score_document(self, text: str) -> float:
        scores = {"a": 0.00, "b": 0.25, "c": 0.50, "d": 0.75}
        try:
            return scores[text]
        except KeyError:
            msg = f"Unexpected text: {text}"
            raise ValueError(msg) from None

    def keep_document(self, score: float) -> bool:
        return np.random.pareto(self._alpha) > 1 - score  # noqa: NPY002


class FakeLangId(DocumentFilter):
    """Emulate ``FastTextLangId`` without loading a model."""

    def __init__(self, min_langid_score: float = 0.3):
        super().__init__()
        self._cutoff = min_langid_score

    def load_model(self) -> None:
        pass

    def score_document(self, text: str) -> str:
        scores = {
            "a": [0.5, "EN"],
            "b": [0.7, "HI"],
            "c": [0.2, "PT"],
            "d": [0.5, "EN"],
        }
        try:
            return str(scores[text])
        except KeyError:
            msg = f"Unexpected text: {text}"
            raise ValueError(msg) from None

    def keep_document(self, score: float | str) -> bool:
        if isinstance(score, str):
            score = eval(score)  # noqa: S307

        return score[0] >= self._cutoff


def list_to_dataset(documents: list[str]) -> DocumentBatch:
    return DocumentBatch(data=pd.DataFrame({"text": documents}), dataset_name="test_1")


def assert_datasets_equal(expected: DocumentBatch, actual: DocumentBatch) -> None:
    pd.testing.assert_frame_equal(
        expected.to_pandas().reset_index(drop=True),
        actual.to_pandas().reset_index(drop=True),
    )
    assert actual.dataset_name == expected.dataset_name


@pytest.mark.parametrize(
    ("label", "expected_language"),
    [
        ("__label__en", "en"),
        ("__label__EN", "EN"),
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
        ("en", "EN", True),
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


def test_fake_quality_filter_pipeline() -> None:
    dataset = list_to_dataset(["a", "b", "c", "d"])

    filtered_data = ScoreFilter(FakeQualityFilter()).process(dataset)

    expected_data = list_to_dataset(["b", "c", "d"])
    assert_datasets_equal(expected_data, filtered_data)


def test_fake_langid_filter_pipeline() -> None:
    dataset = list_to_dataset(["a", "b", "c", "d"])

    filtered_data = ScoreFilter(FakeLangId()).process(dataset)

    expected_data = list_to_dataset(["a", "b", "d"])
    assert_datasets_equal(expected_data, filtered_data)
