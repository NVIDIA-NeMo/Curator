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

from nemo_curator.utils.text_utils import get_words


class DownstreamTask(ABC):

    def __init__(self):
        super().__init__()
        self._task_name = None
        self._ngrams = {}

    @abstractmethod
    def generate_ngrams(self):
        pass

    @property
    def ngrams(self):
        return self._ngrams

    def _update_ngrams(self, text, min_ngram_size=8, max_ngram_size=13):
        words, positions = get_words(text)
        if len(words) < min_ngram_size:
            return

        if len(words) < max_ngram_size:
            seq = " ".join(words)
            if seq not in self._ngrams:
                self._ngrams[seq] = 0

        for i in range(len(words) - max_ngram_size + 1):
            seq = " ".join(words[i : i + max_ngram_size])
            if seq not in self._ngrams:
                self._ngrams[seq] = 0


def import_task(task_path):
    module_path, task_name = task_path.rsplit(".", 1)
    task_module = importlib.import_module(module_path)
    task_class = getattr(task_module, task_name)
    if not issubclass(task_class, DownstreamTask):
        raise ValueError(
            f"Input iterator {task_class.__name__} "
            "must be derived from DownstreamTask"
            "defined in nemo_curator.tasks.downstream_task"
        )
    return task_class
