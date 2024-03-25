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

import importlib
from abc import ABC, abstractmethod


class DocumentFilter(ABC):

    def __init__(self):
        super().__init__()
        self._name = self.__class__.__name__
        self._sentences = None
        self._paragraphs = None
        self._ngrams = None

    @abstractmethod
    def score_document(self, text):
        pass

    @abstractmethod
    def keep_document(self, scores):
        pass

    @property
    def name(self):
        return self._name

    @property
    def sentences(self):
        return self._sentences

    @sentences.setter
    def sentences(self, sentences):
        self._sentences = sentences

    @property
    def paragraphs(self):
        return self._paragraphs

    @paragraphs.setter
    def paragraphs(self, paragraphs):
        self._paragraphs = paragraphs

    @property
    def ngrams(self):
        return self._ngrams

    @ngrams.setter
    def ngrams(self, ngrams):
        self._ngrams = ngrams


def import_filter(filter_path):
    module_path, filter_name = filter_path.rsplit(".", 1)
    filter_module = importlib.import_module(module_path)
    filter_class = getattr(filter_module, filter_name)
    if not issubclass(filter_class, DocumentFilter):
        raise ValueError(
            f"Input filter {filter_class.__name__} must be derived "
            "from DocumentFilter defined in nemo_curator.filters.doc_filter"
        )
    return filter_class
