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

import dask
import fasttext
import numpy as np
import pandas as pd

from nemo_curator.filters.doc_filter import DocumentFilter
from nemo_curator.utils.decorators import batched
from nemo_curator.utils.distributed_utils import NoWorkerError, load_object_on_worker


class FastTextQualityFilter(DocumentFilter):

    def __init__(self, model_path=None, label="__label__hq", alpha=3, seed=42):
        if model_path is None:
            raise ValueError(
                "Must provide a valid path to a FastText model "
                "to compute document scores with this filter"
            )
        self._model_path = model_path
        self._label = label
        self._alpha = alpha
        self._seed = np.random.seed(seed)
        self._name = "fasttext_quality_filter"

    @batched
    def score_document(self, df: pd.Series):
        model_attr = f"{self._name}_{self._model_path}"
        try:
            model = load_object_on_worker(model_attr, self._load_model, {})
        except NoWorkerError:
            return pd.Series(np.ones(len(df)), dtype=float)

        def _score_document(text):
            text = text.replace("\n", " ").replace("__label__", " ")
            pred = model.predict(text)
            document_score = pred[1][0]
            if pred[0][0] != self._label:
                document_score = 1 - document_score

            return document_score

        return df.apply(_score_document)

    @batched
    def keep_document(self, df: pd.Series):
        return np.random.pareto(self._alpha, size=len(df)) > 1 - df

    def _load_model(self):
        return fasttext.load_model(self._model_path)


class FastTextLangId(DocumentFilter):

    def __init__(self, model_path=None, min_langid_score=0.3):
        if model_path is None:
            raise ValueError(
                "Must provide a valid path to a FastText model "
                "to identify languages with this filter"
            )
        self._model_path = model_path
        self._lang_code = None
        self._cutoff = min_langid_score
        self._name = "lang_id"

    @batched
    def score_document(self, df: pd.Series):
        model_attr = f"{self._name}_{self._model_path}"
        try:
            model = load_object_on_worker(model_attr, self._load_model, {})
        except NoWorkerError:
            return pd.Series([[1.0, "N/A"] for _ in range(len(df))])

        def _score_document(text):
            pp = text.strip().replace("\n", " ")
            label, score = model.predict(pp, k=1)
            score = score[0]
            lang_code = label[0][-2:].upper()

            return [score, lang_code]

        return df.apply(_score_document)

    def keep_document(self, score):
        return score[0] >= self._cutoff

    def _load_model(self):
        return fasttext.load_model(self._model_path)
