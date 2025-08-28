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

import os

import fasttext
import numpy as np

from nemo_curator.stages.text.filters.doc_filter import DocumentFilter


class FastTextQualityFilter(DocumentFilter):
    def __init__(self, model_path: str | None = None, label: str = "__label__hq", alpha: float = 3, seed: int = 42):
        if model_path is None:
            msg = "Must provide a valid path to a FastText model to compute document scores with this filter"
            raise ValueError(msg)
        self._model_path = model_path
        self._label = label
        self._alpha = alpha
        self._seed = np.random.seed(seed)  # noqa: NPY002
        self._name = "fasttext_quality_filter"

    def model_check_or_download(self) -> None:
        if not os.path.exists(self._model_path):
            msg = f"Model file {self._model_path} not found"
            raise FileNotFoundError(msg)

    def load_model(self) -> None:
        self._fasttext_quality_filter_model = fasttext.load_model(self._model_path)

    def score_document(self, text: str) -> float:
        # See setup() function in modules/filter.py
        model = self._fasttext_quality_filter_model

        text = text.replace("\n", " ").replace("__label__", " ")
        pred = model.predict(text)
        document_score = pred[1][0]
        if pred[0][0] != self._label:
            document_score = 1 - document_score

        return document_score

    def keep_document(self, score: float) -> bool:
        return np.random.pareto(self._alpha) > 1 - score  # noqa: NPY002


class FastTextLangId(DocumentFilter):
    def __init__(self, model_path: str | None = None, min_langid_score: float = 0.3):
        if model_path is None:
            msg = "Must provide a valid path to a FastText model to identify languages with this filter"
            raise ValueError(msg)
        self._model_path = model_path
        self._lang_code = None
        self._cutoff = min_langid_score
        self._name = "lang_id"

    def model_check_or_download(self) -> None:
        if not os.path.exists(self._model_path):
            msg = f"Model file {self._model_path} not found"
            raise FileNotFoundError(msg)

    def load_model(self) -> None:
        self._fasttext_langid_model = fasttext.load_model(self._model_path)

    def score_document(self, text: str) -> list[float | str]:
        # See setup() function in modules/filter.py
        model = self._fasttext_langid_model

        pp = text.strip().replace("\n", " ")
        label, score = model.predict(pp, k=1)
        score = score[0]
        lang_code = label[0][-2:].upper()

        # Need to convert it to a string to allow backend conversions
        return str([score, lang_code])

    def keep_document(self, score: float | str) -> bool:
        if isinstance(score, str):
            score = eval(score)  # noqa: S307

        return score[0] >= self._cutoff
