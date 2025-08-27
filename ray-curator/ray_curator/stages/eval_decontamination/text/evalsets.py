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

import json

from datasets import load_dataset

from ray_curator.utils.file_utils import get_all_files_paths_under

from .evalset_base import EvaluationSetBase


class RaceEvalSet(EvaluationSetBase):
    def __init__(self, min_ngram_size: int = 8, max_ngram_size: int = 13):
        self._task_name = "race"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        super().__init__()

    def _load_dataset(self) -> None:
        self._dataset = load_dataset(self._task_name, "all", split="test")

    def generate_ngrams(self) -> dict[str, int]:
        for line in self._dataset:
            text = line["question"]
            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class SquadEvalSet(EvaluationSetBase):
    def __init__(self, min_ngram_size: int = 8, max_ngram_size: int = 13):
        self._task_name = "squad"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        super().__init__()

    def _load_dataset(self) -> None:
        self._dataset = load_dataset("squad_v2", split="validation")

    def generate_ngrams(self) -> dict[str, int]:
        for line in self._dataset:
            text = line["question"]
            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class ArcEasyEvalSet(EvaluationSetBase):
    def __init__(self, min_ngram_size: int = 8, max_ngram_size: int = 13):
        self._task_name = "arceasy"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        super().__init__()

    def _load_dataset(self) -> None:
        self._dataset = load_dataset("ai2_arc", "ARC-Easy", split="test")

    def generate_ngrams(self) -> dict[str, int]:
        for line in self._dataset:
            text = line["question"]
            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class ArcChallengeEvalSet(EvaluationSetBase):
    def __init__(self, min_ngram_size: int = 8, max_ngram_size: int = 13):
        self._task_name = "arcchallenge"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        super().__init__()

    def _load_dataset(self) -> None:
        self._dataset = load_dataset("ai2_arc", "ARC-Challenge", split="test")

    def generate_ngrams(self) -> dict[str, int]:
        for line in self._dataset:
            text = line["question"]
            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class OpenBookQAEvalSet(EvaluationSetBase):
    def __init__(self, min_ngram_size: int = 8, max_ngram_size: int = 13):
        self._task_name = "openbookqa"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        super().__init__()

    def _load_dataset(self) -> None:
        self._dataset = load_dataset("openbookqa", "main", split="test")

    def generate_ngrams(self) -> dict[str, int]:
        for line in self._dataset:
            text = line["question_stem"]
            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class BoolQEvalSet(EvaluationSetBase):
    def __init__(self, min_ngram_size: int = 8, max_ngram_size: int = 13):
        self._task_name = "boolq"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        super().__init__()

    def _load_dataset(self) -> None:
        self._dataset = load_dataset("super_glue", "boolq", split="validation")

    def generate_ngrams(self) -> dict[str, int]:
        for line in self._dataset:
            text = line["question"]
            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class CopaEvalSet(EvaluationSetBase):
    def __init__(self, min_ngram_size: int = 8, max_ngram_size: int = 13):
        self._task_name = "copa"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        super().__init__()

    def _load_dataset(self) -> None:
        self._dataset = load_dataset("super_glue", "copa", split="validation")

    def generate_ngrams(self) -> dict[str, int]:
        for line in self._dataset:
            text = line["premise"]
            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class RTEEvalSet(EvaluationSetBase):
    def __init__(self, min_ngram_size: int = 8, max_ngram_size: int = 13):
        self._task_name = "rte"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        super().__init__()

    def _load_dataset(self) -> None:
        self._dataset = load_dataset("glue", "rte", split="validation")

    def generate_ngrams(self) -> dict[str, int]:
        for line in self._dataset:
            text = line["sentence1"] + "\n" + line["sentence2"]
            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class MultiRCEvalSet(EvaluationSetBase):
    def __init__(self, min_ngram_size: int = 8, max_ngram_size: int = 13):
        self._task_name = "multirc"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        super().__init__()

    def _load_dataset(self) -> None:
        self._dataset = load_dataset("super_glue", "multirc", split="validation")

    def generate_ngrams(self) -> dict[str, int]:
        for line in self._dataset:
            text = line["question"]
            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class WSCEvalSet(EvaluationSetBase):
    def __init__(self, min_ngram_size: int = 8, max_ngram_size: int = 13):
        self._task_name = "wsc"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        super().__init__()

    def _load_dataset(self) -> None:
        self._dataset = load_dataset("super_glue", "wsc", split="validation")

    def generate_ngrams(self) -> dict[str, int]:
        for line in self._dataset:
            text = line["text"]
            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class CBEvalSet(EvaluationSetBase):
    def __init__(self, min_ngram_size: int = 8, max_ngram_size: int = 13):
        self._task_name = "cb"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        super().__init__()

    def _load_dataset(self) -> None:
        self._dataset = load_dataset("super_glue", "cb", split="validation")

    def generate_ngrams(self) -> dict[str, int]:
        for line in self._dataset:
            text = line["premise"]
            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class ANLIEvalSet(EvaluationSetBase):
    def __init__(self, min_ngram_size: int = 8, max_ngram_size: int = 13):
        self._task_name = "anli"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._keys = ["test_r1", "test_r2", "test_r3"]
        super().__init__()

    def _load_dataset(self) -> None:
        self._dataset = load_dataset("anli")

    def generate_ngrams(self) -> dict[str, int]:
        for key in self._keys:
            data = self._dataset[key]
            for line in data:
                try:
                    text = line["premise"]
                    self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                    text = line["hypothesis"]
                    self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                except Exception as e:  # noqa: BLE001, PERF203
                    print("Error:", e)

        return self.ngrams


class RecordEvalSet(EvaluationSetBase):
    def __init__(self, min_ngram_size: int = 8, max_ngram_size: int = 13):
        self._task_name = "record"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        super().__init__()

    def _load_dataset(self) -> None:
        self._dataset = load_dataset("super_glue", "record", split="validation")

    def generate_ngrams(self) -> dict[str, int]:
        for line in self._dataset:
            text = line["query"]
            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class COQAEvalSet(EvaluationSetBase):
    def __init__(self, file_path: str | None = None, min_ngram_size: int = 8, max_ngram_size: int = 13):
        self._task_name = "coqa"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        if file_path is None:
            msg = "Must provide a path to the coqa.json file"
            raise ValueError(msg)
        self._file_path = file_path
        super().__init__()

    def _load_dataset(self) -> None:
        with open(self._file_path) as f:
            self._dataset = json.load(f)["data"]

    def generate_ngrams(self) -> dict[str, int]:
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


class TriviaQAEvalSet(EvaluationSetBase):
    def __init__(self, min_ngram_size: int = 8, max_ngram_size: int = 13):
        self._task_name = "trivia_qa"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        super().__init__()

    def _load_dataset(self) -> None:
        self._dataset = load_dataset("trivia_qa", "unfiltered", split="test")

    def generate_ngrams(self) -> dict[str, int]:
        for line in self._dataset:
            text = line["question"]
            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class QuacEvalSet(EvaluationSetBase):
    def __init__(self, min_ngram_size: int = 8, max_ngram_size: int = 13):
        self._task_name = "quac"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        super().__init__()

    def _load_dataset(self) -> None:
        self._dataset = load_dataset("quac", split="validation")

    def generate_ngrams(self) -> dict[str, int]:
        for line in self._dataset:
            all_questions = line["questions"]
            for question in all_questions:
                self._update_ngrams(
                    question,
                    self._min_ngram_size,
                    self._max_ngram_size,
                )

        return self.ngrams


class WebQAEvalSet(EvaluationSetBase):
    def __init__(self, min_ngram_size: int = 8, max_ngram_size: int = 13):
        self._task_name = "webqa"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        super().__init__()

    def _load_dataset(self) -> None:
        self._dataset = load_dataset("web_questions", split="test")

    def generate_ngrams(self) -> dict[str, int]:
        for line in self._dataset:
            text = line["question"]
            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class DropEvalSet(EvaluationSetBase):
    def __init__(self, min_ngram_size: int = 8, max_ngram_size: int = 13):
        self._task_name = "drop"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        super().__init__()

    def _load_dataset(self) -> None:
        self._dataset = load_dataset("drop", split="validation")

    def generate_ngrams(self) -> dict[str, int]:
        for line in self._dataset:
            text = line["question"]
            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class WiCEvalSet(EvaluationSetBase):
    def __init__(self, min_ngram_size: int = 8, max_ngram_size: int = 13):
        self._task_name = "wic"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        super().__init__()

    def _load_dataset(self) -> None:
        self._dataset = load_dataset(
            path="super_glue",
            name="wic",
            split="validation",
        )

    def generate_ngrams(self) -> dict[str, int]:
        for line in self._dataset:
            text = line["sentence1"] + "\n" + line["sentence2"]
            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class MMLUEvalSet(EvaluationSetBase):
    def __init__(self, path: str, min_ngram_size: int = 8, max_ngram_size: int = 13):
        self._task_name = "mmlu"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._path = path
        super().__init__()

    def _load_dataset(self) -> None:
        # For file-based datasets, we don't load everything into memory
        # We just verify the path exists and is accessible
        self._dataset = self._path

    def generate_ngrams(self) -> dict[str, int]:
        for ifile in get_all_files_paths_under(self._path):
            with open(ifile, "rb") as f:
                for iline in f:
                    document = json.loads(iline)
                    text = document["text"]
                    self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class BigBenchHardEvalSet(EvaluationSetBase):
    def __init__(self, path: str, min_ngram_size: int = 8, max_ngram_size: int = 13):
        self._task_name = "bigbench_hard"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._path = path
        super().__init__()

    def _load_dataset(self) -> None:
        # For file-based datasets, we don't load everything into memory
        # We just verify the path exists and is accessible
        self._dataset = self._path

    def generate_ngrams(self) -> dict[str, int]:
        for ifile in get_all_files_paths_under(self._path):
            with open(ifile, "rb") as f:
                for iline in f:
                    document = json.loads(iline)
                    text = document["text"]
                    self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class BigBenchLightEvalSet(EvaluationSetBase):
    def __init__(self, path: str, min_ngram_size: int = 8, max_ngram_size: int = 13):
        self._task_name = "bigbench_light"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._path = path
        super().__init__()

    def _load_dataset(self) -> None:
        # For file-based datasets, we don't load everything into memory
        # We just verify the path exists and is accessible
        self._dataset = self._path

    def generate_ngrams(self) -> dict[str, int]:
        for ifile in get_all_files_paths_under(self._path):
            with open(ifile, "rb") as f:
                for iline in f:
                    document = json.loads(iline)
                    text = document["text"]
                    self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class MultilingualEvalSet(EvaluationSetBase):
    def __init__(self, path: str, min_ngram_size: int = 8, max_ngram_size: int = 13):
        self._task_name = "multilingual"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._path = path
        super().__init__()

    def _load_dataset(self) -> None:
        # For file-based datasets, we don't load everything into memory
        # We just verify the path exists and is accessible
        self._dataset = self._path

    def generate_ngrams(self) -> dict[str, int]:
        for ifile in get_all_files_paths_under(self._path):
            with open(ifile, "rb") as f:
                for iline in f:
                    document = json.loads(iline)
                    text = document["text"]
                    self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class PIQAEvalSet(EvaluationSetBase):
    def __init__(self, min_ngram_size: int = 8, max_ngram_size: int = 13):
        self._task_name = "piqa"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        super().__init__()

    def _load_dataset(self) -> None:
        self._dataset = load_dataset(self._task_name, split="test")

    def generate_ngrams(self) -> dict[str, int]:
        for line in self._dataset:
            text = line["goal"]
            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class WinograndeEvalSet(EvaluationSetBase):
    def __init__(self, min_ngram_size: int = 8, max_ngram_size: int = 13):
        self._task_name = "winogrande"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        super().__init__()

    def _load_dataset(self) -> None:
        self._dataset = load_dataset(
            path="winogrande",
            name="winogrande_xl",
            split="validation",
        )

    def generate_ngrams(self) -> dict[str, int]:
        for line in self._dataset:
            text = line["sentence"]
            self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)

        return self.ngrams


class LambadaEvalSet(EvaluationSetBase):
    def __init__(self, file_path: str, min_ngram_size: int = 8, max_ngram_size: int = 13):
        self._task_name = "lambada"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._file_path = file_path
        super().__init__()

    def _load_dataset(self) -> None:
        # For file-based datasets, we don't load everything into memory
        # We just store the file path for processing
        self._dataset = self._file_path

    def generate_ngrams(self) -> dict[str, int]:
        with open(self._file_path) as f:
            for line in f:
                try:
                    myjson = json.loads(line)
                    text = myjson["text"]
                    self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                except Exception as e:  # noqa: BLE001, PERF203
                    print(f"Error {e}")

        return self.ngrams


class NumDascEvalSet(EvaluationSetBase):
    def __init__(self, n: int, file_path: str, min_ngram_size: int = 8, max_ngram_size: int = 13):
        self._n = n
        self._task_name = f"{n}dasc"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._file_path = file_path
        super().__init__()

    def _load_dataset(self) -> None:
        # For file-based datasets, we don't load everything into memory
        # We just store the file path for processing
        self._dataset = self._file_path

    def generate_ngrams(self) -> dict[str, int]:
        with open(self._file_path) as f:
            for line in f:
                try:
                    myjson = json.loads(line)
                    text = myjson["context"] + myjson["completion"]
                    self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                except Exception as e:  # noqa: BLE001, PERF203
                    print(f"Error {e}")

        return self.ngrams


class StoryClozeEvalSet(EvaluationSetBase):
    def __init__(self, file_path: str, min_ngram_size: int = 8, max_ngram_size: int = 13):
        self._task_name = "story_cloze"
        self._min_ngram_size = min_ngram_size
        self._max_ngram_size = max_ngram_size
        self._file_path = file_path
        super().__init__()

    def _load_dataset(self) -> None:
        # For file-based datasets, we don't load everything into memory
        # We just store the file path for processing
        self._dataset = self._file_path

    def generate_ngrams(self) -> dict[str, int]:
        with open(self._file_path) as f:
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
                    self._update_ngrams(text, self._min_ngram_size, self._max_ngram_size)
                except Exception as e:  # noqa: BLE001, PERF203
                    print(f"Error {e}")

        return self.ngrams
