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

import json

from datasets import load_dataset

from nemo_curator.tasks.downstream_task import DownstreamTask
from nemo_curator.utils.file_utils import get_all_files_paths_under


class Race(DownstreamTask):

    def __init__(self, min_ngram_size=8, max_ngram_size=13):
        super().__init__()
        self._task_name = "race"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(self._task_name, "all", split="test")

    def generate_ngrams(self):
        for line in self._dataset:
            text = line["question"]
            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class Squad(DownstreamTask):

    def __init__(self, min_ngram_size=8, max_ngram_size=13):
        super().__init__()
        self._task_name = "squad"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("squad_v2", split="validation")

    def generate_ngrams(self):
        for line in self._dataset:
            text = line["question"]
            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class ArcEasy(DownstreamTask):

    def __init__(self, min_ngram_size=8, max_ngram_size=13):
        super().__init__()
        self._task_name = "arceasy"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("ai2_arc", "ARC-Easy", split="test")

    def generate_ngrams(self):
        for line in self._dataset:
            text = line["question"]
            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class ArcChallenge(DownstreamTask):

    def __init__(self, min_ngram_size=8, max_ngram_size=13):
        super().__init__()
        self._task_name = "arcchallenge"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("ai2_arc", "ARC-Challenge", split="test")

    def generate_ngrams(self):
        for line in self._dataset:
            text = line["question"]
            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class OpenBookQA(DownstreamTask):

    def __init__(self, min_ngram_size=8, max_ngram_size=13):
        super().__init__()
        self._task_name = "openbookqa"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("openbookqa", "main", split="test")

    def generate_ngrams(self):
        for line in self._dataset:
            text = line["question_stem"]
            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class BoolQ(DownstreamTask):

    def __init__(self, min_ngram_size=8, max_ngram_size=13):
        super().__init__()
        self._task_name = "boolq"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("super_glue", "boolq", split="validation")

    def generate_ngrams(self):
        for line in self._dataset:
            text = line["question"]
            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class Copa(DownstreamTask):

    def __init__(self, min_ngram_size=8, max_ngram_size=13):
        super().__init__()
        self._task_name = "copa"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("super_glue", "copa", split="validation")

    def generate_ngrams(self):
        for line in self._dataset:
            text = line["premise"]
            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class RTE(DownstreamTask):

    def __init__(self, min_ngram_size=8, max_ngram_size=13):
        super().__init__()
        self._task_name = "rte"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("glue", "rte", split="validation")

    def generate_ngrams(self):
        for line in self._dataset:
            text = line["sentence1"] + "\n" + line["sentence2"]
            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class MultiRC(DownstreamTask):

    def __init__(self, min_ngram_size=8, max_ngram_size=13):
        super().__init__()
        self._task_name = "multirc"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("super_glue", "multirc", split="validation")

    def generate_ngrams(self):
        for line in self._dataset:
            text = line["question"]
            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class WSC(DownstreamTask):

    def __init__(self, min_ngram_size=8, max_ngram_size=13):
        super().__init__()
        self._task_name = "wsc"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("super_glue", "multirc", split="validation")

    def generate_ngrams(self):
        for line in self._dataset:
            text = line["question"]
            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class CB(DownstreamTask):

    def __init__(self, min_ngram_size=8, max_ngram_size=13):
        super().__init__()
        self._task_name = "cb"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("super_glue", "cb", split="validation")

    def generate_ngrams(self):
        for line in self._dataset:
            text = line["premise"]
            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class ANLI(DownstreamTask):

    def __init__(self, min_ngram_size=8, max_ngram_size=13):
        super().__init__()
        self._task_name = "anli"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("anli")
        self._keys = ["test_r1", "test_r2", "test_r3"]

    def generate_ngrams(self):
        for key in self._keys:
            data = self._dataset[key]
            for line in data:
                try:
                    text = line["premise"]
                    self._update_ngrams(
                        text, self._min_ngram_size, self._max_ngram_size
                    )
                    text = line["hypothesis"]
                    self._update_ngrams(
                        text, self._min_ngram_size, self._max_ngram_size
                    )
                except Exception as e:
                    print("Error:", e)

        return self.ngrams


class Record(DownstreamTask):

    def __init__(self, min_ngram_size=8, max_ngram_size=13):
        super().__init__()
        self._task_name = "record"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("super_glue", "record", split="validation")

    def generate_ngrams(self):
        for line in self._dataset:
            text = line["query"]
            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class COQA(DownstreamTask):

    def __init__(self, file_path, min_ngram_size=8, max_ngram_size=13):
        super().__init__()
        self._task_name = "coqa"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        if file_path is None:
            raise Exception("Must provide a path to the coqa.json file")
        self._dataset = json.load(open(file_path))["data"]

    def generate_ngrams(self):
        for line in self._dataset:
            all_questions = line["questions"]
            for question in all_questions:
                self._update_ngrams(
                    question["input_text"],
                    self._min_ngram_size,
                    self._max_ngram_size,
                )
            story = line["story"]
            self._update_ngrams(story, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class TriviaQA(DownstreamTask):

    def __init__(self, min_ngram_size=8, max_ngram_size=13):
        super().__init__()
        self._task_name = "trivia_qa"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("trivia_qa", "unfiltered", split="test")

    def generate_ngrams(self):
        for line in self._dataset:
            text = line["question"]
            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class Quac(DownstreamTask):

    def __init__(self, min_ngram_size=8, max_ngram_size=13):
        super().__init__()
        self._task_name = "quac"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("quac", split="validation")

    def generate_ngrams(self):
        for line in self._dataset:
            all_questions = line["questions"]
            for question in all_questions:
                self._update_ngrams(
                    question,
                    self._min_ngram_size,
                    self._max_ngram_size,
                )

        return self.ngrams


class WebQA(DownstreamTask):

    def __init__(self, min_ngram_size=8, max_ngram_size=13):
        super().__init__()
        self._task_name = "webqa"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("web_questions", split="test")

    def generate_ngrams(self):
        for line in self._dataset:
            text = line["question"]
            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class Drop(DownstreamTask):

    def __init__(self, min_ngram_size=8, max_ngram_size=13):
        super().__init__()
        self._task_name = "drop"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset("drop", split="validation")

    def generate_ngrams(self):
        for line in self._dataset:
            text = line["question"]
            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class WiC(DownstreamTask):

    def __init__(self, min_ngram_size=8, max_ngram_size=13):
        super().__init__()
        self._task_name = "wic"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            path="super_glue",
            name="wic",
            split="validation",
        )

    def generate_ngrams(self):
        for line in self._dataset:
            text = line["sentence1"] + "\n" + line["sentence2"]
            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class MMLU(DownstreamTask):

    def __init__(self, path=None, min_ngram_size=8, max_ngram_size=13):
        super().__init__()
        self._task_name = "mmlu"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._path = path
        if self._path is None:
            raise Exception(
                "Must provide path that contain " "MMLU task data in JSONL format"
            )

    def generate_ngrams(self):
        for ifile in get_all_files_paths_under(self._path):
            for iline in open(ifile, "rb"):
                document = json.loads(iline)
                text = document["text"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class BigBenchHard(DownstreamTask):

    def __init__(self, path=None, min_ngram_size=8, max_ngram_size=13):
        super().__init__()
        self._task_name = "bigbench_hard"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._path = path
        if self._path is None:
            raise Exception(
                "Must provide path that contain "
                "BigBenchHard task data in JSONL format"
            )

    def generate_ngrams(self):
        for ifile in get_all_files_paths_under(self._path):
            for iline in open(ifile, "rb"):
                document = json.loads(iline)
                text = document["text"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class BigBenchLight(DownstreamTask):

    def __init__(self, path=None, min_ngram_size=8, max_ngram_size=13):
        super().__init__()
        self._task_name = "bigbench_light"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._path = path
        if self._path is None:
            raise Exception(
                "Must provide path that contain "
                "BigBenchLight task data in JSONL format"
            )

    def generate_ngrams(self):
        for ifile in get_all_files_paths_under(self._path):
            for iline in open(ifile, "rb"):
                document = json.loads(iline)
                text = document["text"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class Multilingual(DownstreamTask):

    def __init__(self, path=None, min_ngram_size=8, max_ngram_size=13):
        super().__init__()
        self._task_name = "multilingual"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._path = path
        if self._path is None:
            raise Exception(
                "Must provide path to " "multilingual task data in JSONL format"
            )

    def generate_ngrams(self):
        for ifile in get_all_files_paths_under(self._path):
            for iline in open(ifile, "rb"):
                document = json.loads(iline)
                text = document["text"]
                self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class PIQA(DownstreamTask):

    def __init__(self, min_ngram_size=8, max_ngram_size=13):
        super().__init__()
        self._task_name = "piqa"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(self._task_name, split="test")

    def generate_ngrams(self):
        for line in self._dataset:
            text = line["goal"]
            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class Winogrande(DownstreamTask):

    def __init__(self, min_ngram_size=8, max_ngram_size=13):
        super().__init__()
        self._task_name = "winogrande"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._dataset = load_dataset(
            path="winogrande",
            name="winogrande_xl",
            split="validation",
        )

    def generate_ngrams(self):
        for line in self._dataset:
            text = line["sentence"]
            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class Lambada(DownstreamTask):

    def __init__(self, file_path, min_ngram_size=8, max_ngram_size=13):
        super().__init__()
        self._task_name = "lambada"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._file_path = file_path

    def generate_ngrams(self):
        with open(self._file_path, "r") as f:
            for line in f:
                try:
                    myjson = json.loads(line)
                    text = myjson["text"]
                    self._update_ngrams(
                        text, self._min_ngram_size, self._max_ngram_size
                    )
                except Exception as e:
                    print(f"Error {e}")

        return self.ngrams


class NumDasc(DownstreamTask):

    def __init__(self, n, file_path, min_ngram_size=8, max_ngram_size=13):
        super().__init__()
        self._n = n
        self._task_name = "{n}dasc"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._file_path = file_path

    def generate_ngrams(self):
        with open(self._file_path, "r") as f:
            for line in f:
                try:
                    myjson = json.loads(line)
                    text = myjson["context"] + myjson["completion"]
                    self._update_ngrams(
                        text, self._min_ngram_size, self._max_ngram_size
                    )
                except Exception as e:
                    print(f"Error {e}")

        return self.ngrams


class StoryCloze(DownstreamTask):

    def __init__(self, file_path, min_ngram_size=8, max_ngram_size=13):
        super().__init__()
        self._task_name = "story_cloze"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._file_path = file_path

    def generate_ngrams(self):
        with open(self._file_path, "r") as f:
            for line in f:
                try:
                    myjson = json.loads(line)
                    text = " ".join(
                        [
                            myjson["InputSentence1"],
                            myjson["InputSentence2"],
                            myjson["InputSentence3"],
                            myjson["InputSentence4"],
                        ]
                    )
                    self._update_ngrams(
                        text, self._min_ngram_size, self._max_ngram_size
                    )
                except Exception as e:
                    print(f"Error {e}")

        return self.ngrams
