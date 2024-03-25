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

from collections import defaultdict

import dask.dataframe as dd
import pandas as pd
import pytest

import nemo_curator
from nemo_curator.datasets import DocumentDataset
from nemo_curator.tasks import DownstreamTask


class SimpleTask(DownstreamTask):
    def __init__(self, min_ngram_size=8, max_ngram_size=13):
        super().__init__()
        self._task_name = "simple"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = [
            "This is a simple example task document with enough words.",
            "Two plus two equals four. Two times two equals four as well. However, two divided by two is one.",
            "Cars are (sometimes) red. Bicycles can (occasionally) be blue. Trees often have green leaves.",
            "This is a short one.",
            "This is a short one.",
        ]

    def generate_ngrams(self):
        for line in self._dataset:
            self._update_ngrams(line, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class TinyTask(DownstreamTask):
    def __init__(self, min_ngram_size=0, max_ngram_size=4):
        super().__init__()
        self._task_name = "tiny"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = ["Super tiny task with one document."]

    def generate_ngrams(self):
        for line in self._dataset:
            self._update_ngrams(line, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


def list_to_dataset(documents, col_name="text", npartitions=2):
    data = {col_name: documents}
    pdf = pd.DataFrame(data)

    return DocumentDataset(dd.from_pandas(pdf, npartitions=npartitions))


@pytest.fixture
def contaminated_dataset():
    return list_to_dataset(
        [
            "This document is fine",
            "Before contamination. This is a simple example task document with enough words. After contamination.",
            "This document is not good. Two plus two equals four. Two times two equals four as well. However, one minus one is zero.",
            "Long contamination. Birds are (sometimes) red. Bicycles can (occasionally) be blue. Trees often have green leaves. After contamination.",
            "Small contamination in a very short document 1. SupeR tiNy task with one document!",
            "Small contamination in a very short document 2. Super tiNy task with one document?",
        ],
        col_name="text",
    )


class TestPrepareTaskData:
    def test_single_task(self):
        decontaminator = nemo_curator.TaskDecontamination(SimpleTask())
        actual_count = decontaminator.prepare_task_ngram_count().compute()
        expected_ngrams = [
            "this is a simple example task document with enough words",
            "two plus two equals four two times two equals four as well however",
            "plus two equals four two times two equals four as well however two",
            "two equals four two times two equals four as well however two divided",
            "equals four two times two equals four as well however two divided by",
            "four two times two equals four as well however two divided by two",
            "two times two equals four as well however two divided by two is",
            "times two equals four as well however two divided by two is one",
            "cars are sometimes red bicycles can occasionally be blue trees often have green",
            "are sometimes red bicycles can occasionally be blue trees often have green leaves",
        ]
        expected_count = {ngram: 0 for ngram in expected_ngrams}

        assert len(expected_count) == len(
            actual_count
        ), f"Expected #{len(expected_count)}, got #{len(actual_count)}"
        assert (
            expected_count == actual_count
        ), f"Expected: {expected_count}, got: {actual_count}"

    def test_multiple_tasks(self):
        decontaminator = nemo_curator.TaskDecontamination([SimpleTask(), TinyTask()])
        actual_count = decontaminator.prepare_task_ngram_count().compute()
        expected_ngrams = [
            "this is a simple example task document with enough words",
            "two plus two equals four two times two equals four as well however",
            "plus two equals four two times two equals four as well however two",
            "two equals four two times two equals four as well however two divided",
            "equals four two times two equals four as well however two divided by",
            "four two times two equals four as well however two divided by two",
            "two times two equals four as well however two divided by two is",
            "times two equals four as well however two divided by two is one",
            "cars are sometimes red bicycles can occasionally be blue trees often have green",
            "are sometimes red bicycles can occasionally be blue trees often have green leaves",
            "super tiny task with",
            "tiny task with one",
            "task with one document",
        ]
        expected_count = {ngram: 0 for ngram in expected_ngrams}

        assert len(expected_count) == len(
            actual_count
        ), f"Expected #{len(expected_count)}, got #{len(actual_count)}"
        assert (
            expected_count == actual_count
        ), f"Expected: {expected_count}, got: {actual_count}"


class TestFindMatchingNgrams:
    def test_single_task(self, contaminated_dataset):
        decontaminator = nemo_curator.TaskDecontamination(SimpleTask())
        task_ngrams = decontaminator.prepare_task_ngram_count()
        actual_result = decontaminator.find_matching_ngrams(
            task_ngrams, contaminated_dataset
        ).compute()
        actual_ngrams, actual_freq = (
            actual_result["matched-ngrams"],
            actual_result["ngrams-freq"],
        )

        expected_ngrams = defaultdict(int)
        expected_ngrams["this is a simple example task document with enough words"] = 1
        expected_ngrams[
            "two plus two equals four two times two equals four as well however"
        ] = 1
        expected_ngrams[
            "are sometimes red bicycles can occasionally be blue trees often have green leaves"
        ] = 1

        expected_freq = [(10, 1), (13, 9)]

        assert (
            expected_freq == actual_freq
        ), f"Expected #{expected_freq}, got #{actual_freq}"
        assert (
            expected_ngrams == actual_ngrams
        ), f"Expected: {expected_ngrams}, got: {actual_ngrams}"

    def test_multiple_tasks(self, contaminated_dataset):
        decontaminator = nemo_curator.TaskDecontamination([SimpleTask(), TinyTask()])
        task_ngrams = decontaminator.prepare_task_ngram_count()
        actual_result = decontaminator.find_matching_ngrams(
            task_ngrams, contaminated_dataset
        ).compute()
        actual_ngrams, actual_freq = (
            actual_result["matched-ngrams"],
            actual_result["ngrams-freq"],
        )

        expected_ngrams = defaultdict(int)
        expected_ngrams["this is a simple example task document with enough words"] = 1
        expected_ngrams[
            "two plus two equals four two times two equals four as well however"
        ] = 1
        expected_ngrams[
            "are sometimes red bicycles can occasionally be blue trees often have green leaves"
        ] = 1
        expected_ngrams["super tiny task with"] = 2

        expected_freq = [(4, 3), (10, 1), (13, 9)]

        assert (
            expected_freq == actual_freq
        ), f"Expected #{expected_freq}, got #{actual_freq}"
        assert (
            expected_ngrams == actual_ngrams
        ), f"Expected: {expected_ngrams}, got: {actual_ngrams}"


class TestRemoveMatchingNgrams:
    def test_single_task(self, contaminated_dataset):
        decontaminator = nemo_curator.TaskDecontamination(
            SimpleTask(), min_document_length=1, remove_char_each_side=1
        )
        task_ngrams = decontaminator.prepare_task_ngram_count()
        ngram_data = decontaminator.find_matching_ngrams(
            task_ngrams, contaminated_dataset
        )
        matched_ngrams, ngram_freq = (
            ngram_data["matched-ngrams"],
            ngram_data["ngrams-freq"],
        )
        filtered_dataset = decontaminator.remove_matching_ngrams(
            matched_ngrams, ngram_freq, contaminated_dataset
        )

        actual_data = sorted(filtered_dataset.df.compute()["text"].to_list())
        expected_data = [
            "This document is fine",
            "Before contamination.",
            " After contamination.",
            "This document is not good.",
            "Long contamination.",
            " After contamination.",
            "Small contamination in a very short document 1. SupeR tiNy task with one document!",
            "Small contamination in a very short document 2. Super tiNy task with one document?",
        ]
        expected_data.sort()
        assert (
            expected_data == actual_data
        ), f"Expected #{expected_data}, got #{actual_data}"

    def test_multiple_tasks(self, contaminated_dataset):
        decontaminator = nemo_curator.TaskDecontamination(
            [SimpleTask(), TinyTask()], min_document_length=1, remove_char_each_side=1
        )
        task_ngrams = decontaminator.prepare_task_ngram_count()
        ngram_data = decontaminator.find_matching_ngrams(
            task_ngrams, contaminated_dataset
        )
        matched_ngrams, ngram_freq = (
            ngram_data["matched-ngrams"],
            ngram_data["ngrams-freq"],
        )
        filtered_dataset = decontaminator.remove_matching_ngrams(
            matched_ngrams, ngram_freq, contaminated_dataset
        )

        actual_data = sorted(filtered_dataset.df.compute()["text"].to_list())
        expected_data = [
            "This document is fine",
            "Before contamination.",
            " After contamination.",
            "This document is not good.",
            "Long contamination.",
            " After contamination.",
            "Small contamination in a very short document 1.",
            "Small contamination in a very short document 2.",
        ]
        expected_data.sort()
        assert (
            expected_data == actual_data
        ), f"Expected #{expected_data}, got #{actual_data}"


class TestFullPipeline:
    def test_single_task(self, contaminated_dataset):
        decontaminator = nemo_curator.TaskDecontamination(
            SimpleTask(), min_document_length=1, remove_char_each_side=1
        )
        filtered_dataset = decontaminator(contaminated_dataset)

        actual_data = sorted(filtered_dataset.df.compute()["text"].to_list())
        expected_data = [
            "This document is fine",
            "Before contamination.",
            " After contamination.",
            "This document is not good.",
            "Long contamination.",
            " After contamination.",
            "Small contamination in a very short document 1. SupeR tiNy task with one document!",
            "Small contamination in a very short document 2. Super tiNy task with one document?",
        ]
        expected_data.sort()
        assert (
            expected_data == actual_data
        ), f"Expected #{expected_data}, got #{actual_data}"

    def test_multiple_tasks(self, contaminated_dataset):
        decontaminator = nemo_curator.TaskDecontamination(
            [SimpleTask(), TinyTask()], min_document_length=1, remove_char_each_side=1
        )
        filtered_dataset = decontaminator(contaminated_dataset)

        actual_data = sorted(filtered_dataset.df.compute()["text"].to_list())
        expected_data = [
            "This document is fine",
            "Before contamination.",
            " After contamination.",
            "This document is not good.",
            "Long contamination.",
            " After contamination.",
            "Small contamination in a very short document 1.",
            "Small contamination in a very short document 2.",
        ]
        expected_data.sort()
        assert (
            expected_data == actual_data
        ), f"Expected #{expected_data}, got #{actual_data}"
