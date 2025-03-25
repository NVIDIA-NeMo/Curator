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
import os
import tempfile
from unittest import mock

import pytest
from datasets import Dataset

import nemo_curator
from nemo_curator.tasks.metrics import (
    ANLI,
    CB,
    COQA,
    MMLU,
    PIQA,
    RTE,
    WSC,
    ArcChallenge,
    ArcEasy,
    BigBenchHard,
    BigBenchLight,
    BoolQ,
    Copa,
    Drop,
    Lambada,
    Multilingual,
    MultiRC,
    NumDasc,
    OpenBookQA,
    Quac,
    Race,
    Record,
    Squad,
    StoryCloze,
    TriviaQA,
    WebQA,
    WiC,
    Winogrande,
)


# Mock dataset for testing
@pytest.fixture
def mock_dataset():
    return [
        {"question": "This is a test question for metrics testing"},
        {"question": "Here is another question with more words for ngram generation"},
        {"question": "Very short"},
        {
            "question": "This contains enough words to create multiple different ngrams for testing purposes"
        },
    ]


@pytest.fixture
def mock_question_stem_dataset():
    return [
        {"question_stem": "This is a test question stem for metrics testing"},
        {
            "question_stem": "Here is another question stem with more words for ngram generation"
        },
        {"question_stem": "Very short stem"},
        {
            "question_stem": "This question stem contains enough words to create multiple different ngrams for testing purposes"
        },
    ]


@pytest.fixture
def mock_premises_dataset():
    return [
        {"premise": "This is a test premise for metrics testing"},
        {"premise": "Here is another premise with more words for ngram generation"},
        {"premise": "Very short premise"},
        {
            "premise": "This premise contains enough words to create multiple different ngrams for testing purposes"
        },
    ]


@pytest.fixture
def mock_sentences_dataset():
    return [
        {
            "sentence1": "This is the first test sentence",
            "sentence2": "This is the second test sentence",
        },
        {
            "sentence1": "Here is another long first sentence with words",
            "sentence2": "And a matching second sentence",
        },
        {"sentence1": "Short one", "sentence2": "Another short"},
        {"sentence1": "This sentence contains", "sentence2": "enough words for ngrams"},
    ]


@pytest.fixture
def mock_coqa_data():
    return {
        "data": [
            {
                "questions": [
                    {"input_text": "This is a test question for COQA?"},
                    {
                        "input_text": "Another question with more words for testing COQA?"
                    },
                ],
                "story": "This is a test story for COQA that has enough words for ngram generation.",
            },
            {
                "questions": [{"input_text": "Short question?"}],
                "story": "Another story with sufficient length for ngram creation.",
            },
        ]
    }


@pytest.fixture
def mock_jsonl_data():
    return [
        '{"text": "This is a test document for MMLU with enough words to create ngrams."}\n',
        '{"text": "Here is another line for BigBench testing with sufficient words."}\n',
        '{"text": "More text for testing multilingual capabilities with proper length."}\n',
        '{"text": "Short."}\n',
    ]


@pytest.fixture
def mock_lambada_data():
    return [
        '{"text": "This is a test document for Lambada with enough words to create ngrams."}\n',
        '{"text": "Here is another line for Lambada testing with sufficient words."}\n',
        '{"text": "Short."}\n',
    ]


@pytest.fixture
def mock_numdasc_data():
    return [
        '{"context": "This is the context", "completion": "with the completion part for testing."}\n',
        '{"context": "Another context section", "completion": "with its completion that should have sufficient words for ngrams."}\n',
        '{"context": "Short", "completion": "one."}\n',
    ]


@pytest.fixture
def mock_storycloze_data():
    return [
        '{"InputSentence1": "First sentence of story.", "InputSentence2": "Second part continues.", "InputSentence3": "Third section adds detail.", "InputSentence4": "Fourth concludes test story."}\n',
        '{"InputSentence1": "Story begins here.", "InputSentence2": "It continues with details.", "InputSentence3": "Adding more narrative.", "InputSentence4": "Concluding the test story."}\n',
    ]


@pytest.fixture
def mock_query_dataset():
    return [
        {"query": "This is a test query for Record testing"},
        {"query": "Here is another query with more words for ngram generation"},
        {"query": "Very short query"},
        {
            "query": "This query contains enough words to create multiple different ngrams for testing purposes"
        },
    ]


class TestMetricsTasks:

    @mock.patch("nemo_curator.tasks.metrics.load_dataset")
    def test_race_init_and_generate_ngrams(self, mock_load_dataset, mock_dataset):
        # Setup mock
        mock_load_dataset.return_value = Dataset.from_list(mock_dataset)

        # Test initialization
        race_task = Race(min_ngram_size=5, max_ngram_size=10)
        assert race_task._task_name == "race"
        assert race_task._min_ngram_size == 5
        assert race_task._max_ngram_size == 10

        # Test generate_ngrams
        ngrams = race_task.generate_ngrams()
        assert len(ngrams) > 0

        # Verify ngram content
        # The exact count depends on the mock dataset but we can check if expected ngrams are present
        expected_ngram = "this is a test question for metrics"
        assert any(expected_ngram in ngram for ngram in ngrams.keys())

    @mock.patch("nemo_curator.tasks.metrics.load_dataset")
    def test_squad_init_and_generate_ngrams(self, mock_load_dataset, mock_dataset):
        # Setup mock
        mock_load_dataset.return_value = Dataset.from_list(mock_dataset)

        # Test initialization
        squad_task = Squad(min_ngram_size=5, max_ngram_size=10)
        assert squad_task._task_name == "squad"
        assert squad_task._min_ngram_size == 5
        assert squad_task._max_ngram_size == 10

        # Test generate_ngrams
        ngrams = squad_task.generate_ngrams()
        assert len(ngrams) > 0

    @mock.patch("nemo_curator.tasks.metrics.load_dataset")
    def test_arceasy_init_and_generate_ngrams(self, mock_load_dataset, mock_dataset):
        mock_load_dataset.return_value = Dataset.from_list(mock_dataset)

        arceasy_task = ArcEasy(min_ngram_size=5, max_ngram_size=10)
        assert arceasy_task._task_name == "arceasy"
        assert arceasy_task._min_ngram_size == 5
        assert arceasy_task._max_ngram_size == 10

        ngrams = arceasy_task.generate_ngrams()
        assert len(ngrams) > 0

    @mock.patch("nemo_curator.tasks.metrics.load_dataset")
    def test_arcchallenge_init_and_generate_ngrams(
        self, mock_load_dataset, mock_dataset
    ):
        mock_load_dataset.return_value = Dataset.from_list(mock_dataset)

        arcchallenge_task = ArcChallenge(min_ngram_size=5, max_ngram_size=10)
        assert arcchallenge_task._task_name == "arcchallenge"
        assert arcchallenge_task._min_ngram_size == 5
        assert arcchallenge_task._max_ngram_size == 10

        ngrams = arcchallenge_task.generate_ngrams()
        assert len(ngrams) > 0

    @mock.patch("nemo_curator.tasks.metrics.load_dataset")
    def test_openbookqa_init_and_generate_ngrams(
        self, mock_load_dataset, mock_question_stem_dataset
    ):
        mock_load_dataset.return_value = Dataset.from_list(mock_question_stem_dataset)

        openbookqa_task = OpenBookQA(min_ngram_size=5, max_ngram_size=10)
        assert openbookqa_task._task_name == "openbookqa"
        assert openbookqa_task._min_ngram_size == 5
        assert openbookqa_task._max_ngram_size == 10

        ngrams = openbookqa_task.generate_ngrams()
        assert len(ngrams) > 0

    @mock.patch("nemo_curator.tasks.metrics.load_dataset")
    def test_boolq_init_and_generate_ngrams(self, mock_load_dataset, mock_dataset):
        mock_load_dataset.return_value = Dataset.from_list(mock_dataset)

        boolq_task = BoolQ(min_ngram_size=5, max_ngram_size=10)
        assert boolq_task._task_name == "boolq"
        assert boolq_task._min_ngram_size == 5
        assert boolq_task._max_ngram_size == 10

        ngrams = boolq_task.generate_ngrams()
        assert len(ngrams) > 0

    @mock.patch("nemo_curator.tasks.metrics.load_dataset")
    def test_copa_init_and_generate_ngrams(
        self, mock_load_dataset, mock_premises_dataset
    ):
        mock_load_dataset.return_value = Dataset.from_list(mock_premises_dataset)

        copa_task = Copa(min_ngram_size=5, max_ngram_size=10)
        assert copa_task._task_name == "copa"
        assert copa_task._min_ngram_size == 5
        assert copa_task._max_ngram_size == 10

        ngrams = copa_task.generate_ngrams()
        assert len(ngrams) > 0

    @mock.patch("nemo_curator.tasks.metrics.load_dataset")
    def test_rte_init_and_generate_ngrams(
        self, mock_load_dataset, mock_sentences_dataset
    ):
        mock_load_dataset.return_value = Dataset.from_list(mock_sentences_dataset)

        rte_task = RTE(min_ngram_size=5, max_ngram_size=10)
        assert rte_task._task_name == "rte"
        assert rte_task._min_ngram_size == 5
        assert rte_task._max_ngram_size == 10

        ngrams = rte_task.generate_ngrams()
        assert len(ngrams) > 0

    @mock.patch("nemo_curator.tasks.metrics.load_dataset")
    def test_multirc_init_and_generate_ngrams(self, mock_load_dataset, mock_dataset):
        mock_load_dataset.return_value = Dataset.from_list(mock_dataset)

        multirc_task = MultiRC(min_ngram_size=5, max_ngram_size=10)
        assert multirc_task._task_name == "multirc"
        assert multirc_task._min_ngram_size == 5
        assert multirc_task._max_ngram_size == 10

        ngrams = multirc_task.generate_ngrams()
        assert len(ngrams) > 0

    @mock.patch("nemo_curator.tasks.metrics.load_dataset")
    def test_wsc_init_and_generate_ngrams(self, mock_load_dataset, mock_dataset):
        mock_load_dataset.return_value = Dataset.from_list(mock_dataset)

        wsc_task = WSC(min_ngram_size=5, max_ngram_size=10)
        assert wsc_task._task_name == "wsc"
        assert wsc_task._min_ngram_size == 5
        assert wsc_task._max_ngram_size == 10

        ngrams = wsc_task.generate_ngrams()
        assert len(ngrams) > 0

    @mock.patch("nemo_curator.tasks.metrics.load_dataset")
    def test_cb_init_and_generate_ngrams(
        self, mock_load_dataset, mock_premises_dataset
    ):
        mock_load_dataset.return_value = Dataset.from_list(mock_premises_dataset)

        cb_task = CB(min_ngram_size=5, max_ngram_size=10)
        assert cb_task._task_name == "cb"
        assert cb_task._min_ngram_size == 5
        assert cb_task._max_ngram_size == 10

        ngrams = cb_task.generate_ngrams()
        assert len(ngrams) > 0

    @mock.patch("nemo_curator.tasks.metrics.load_dataset")
    def test_anli_init_and_generate_ngrams(
        self, mock_load_dataset, mock_premises_dataset
    ):
        mock_dataset_dict = {
            "test_r1": Dataset.from_list(mock_premises_dataset),
            "test_r2": Dataset.from_list(mock_premises_dataset),
            "test_r3": Dataset.from_list(mock_premises_dataset),
        }
        mock_load_dataset.return_value = mock_dataset_dict

        anli_task = ANLI(min_ngram_size=5, max_ngram_size=10)
        assert anli_task._task_name == "anli"
        assert anli_task._min_ngram_size == 5
        assert anli_task._max_ngram_size == 10

        ngrams = anli_task.generate_ngrams()
        assert len(ngrams) > 0

    @mock.patch("nemo_curator.tasks.metrics.load_dataset")
    def test_record_init_and_generate_ngrams(
        self, mock_load_dataset, mock_query_dataset
    ):
        mock_load_dataset.return_value = Dataset.from_list(mock_query_dataset)

        record_task = Record(min_ngram_size=5, max_ngram_size=10)
        assert record_task._task_name == "record"
        assert record_task._min_ngram_size == 5
        assert record_task._max_ngram_size == 10

        ngrams = record_task.generate_ngrams()
        assert len(ngrams) > 0

    def test_coqa_init_and_generate_ngrams(self, mock_coqa_data, tmp_path):
        # Create a temporary file with mock COQA data
        coqa_file = tmp_path / "coqa.json"
        with open(coqa_file, "w") as f:
            json.dump(mock_coqa_data, f)

        coqa_task = COQA(file_path=str(coqa_file), min_ngram_size=5, max_ngram_size=10)
        assert coqa_task._task_name == "coqa"
        assert coqa_task._min_ngram_size == 5
        assert coqa_task._max_ngram_size == 10

        ngrams = coqa_task.generate_ngrams()
        assert len(ngrams) > 0

    def test_coqa_raises_exception_without_file_path(self):
        with pytest.raises(
            Exception, match="Must provide a path to the coqa.json file"
        ):
            COQA(file_path=None)

    @mock.patch("nemo_curator.tasks.metrics.load_dataset")
    def test_triviaqa_init_and_generate_ngrams(self, mock_load_dataset, mock_dataset):
        mock_load_dataset.return_value = Dataset.from_list(mock_dataset)

        triviaqa_task = TriviaQA(min_ngram_size=5, max_ngram_size=10)
        assert triviaqa_task._task_name == "trivia_qa"
        assert triviaqa_task._min_ngram_size == 5
        assert triviaqa_task._max_ngram_size == 10

        ngrams = triviaqa_task.generate_ngrams()
        assert len(ngrams) > 0

    @mock.patch("nemo_curator.tasks.metrics.load_dataset")
    def test_quac_init_and_generate_ngrams(self, mock_load_dataset):
        # For Quac, setup a specific dataset structure
        mock_dataset_list = [
            {
                "questions": [
                    "Question one for quac?",
                    "Question two with more words for quac testing?",
                ]
            },
            {
                "questions": [
                    "Another quac question with sufficient words for ngram generation?"
                ]
            },
        ]
        mock_load_dataset.return_value = Dataset.from_list(mock_dataset_list)

        quac_task = Quac(min_ngram_size=5, max_ngram_size=10)
        assert quac_task._task_name == "quac"
        assert quac_task._min_ngram_size == 5
        assert quac_task._max_ngram_size == 10

        ngrams = quac_task.generate_ngrams()
        assert len(ngrams) > 0

    @mock.patch("nemo_curator.tasks.metrics.load_dataset")
    def test_webqa_init_and_generate_ngrams(self, mock_load_dataset, mock_dataset):
        mock_load_dataset.return_value = Dataset.from_list(mock_dataset)

        webqa_task = WebQA(min_ngram_size=5, max_ngram_size=10)
        assert webqa_task._task_name == "webqa"
        assert webqa_task._min_ngram_size == 5
        assert webqa_task._max_ngram_size == 10

        ngrams = webqa_task.generate_ngrams()
        assert len(ngrams) > 0

    @mock.patch("nemo_curator.tasks.metrics.load_dataset")
    def test_drop_init_and_generate_ngrams(self, mock_load_dataset, mock_dataset):
        mock_load_dataset.return_value = Dataset.from_list(mock_dataset)

        drop_task = Drop(min_ngram_size=5, max_ngram_size=10)
        assert drop_task._task_name == "drop"
        assert drop_task._min_ngram_size == 5
        assert drop_task._max_ngram_size == 10

        ngrams = drop_task.generate_ngrams()
        assert len(ngrams) > 0

    @mock.patch("nemo_curator.tasks.metrics.load_dataset")
    def test_wic_init_and_generate_ngrams(
        self, mock_load_dataset, mock_sentences_dataset
    ):
        mock_load_dataset.return_value = Dataset.from_list(mock_sentences_dataset)

        wic_task = WiC(min_ngram_size=5, max_ngram_size=10)
        assert wic_task._task_name == "wic"
        assert wic_task._min_ngram_size == 5
        assert wic_task._max_ngram_size == 10

        ngrams = wic_task.generate_ngrams()
        assert len(ngrams) > 0

    @mock.patch("nemo_curator.utils.file_utils.get_all_files_paths_under")
    def test_mmlu_init_and_generate_ngrams(
        self, mock_get_paths, mock_jsonl_data, tmp_path
    ):
        # Create a temporary file with mock JSONL data
        mmlu_dir = tmp_path / "mmlu"
        mmlu_dir.mkdir()
        mmlu_file = mmlu_dir / "data.jsonl"
        with open(mmlu_file, "w") as f:
            f.writelines(mock_jsonl_data)

        mock_get_paths.return_value = [str(mmlu_file)]

        mmlu_task = MMLU(path=str(mmlu_dir), min_ngram_size=5, max_ngram_size=10)
        assert mmlu_task._task_name == "mmlu"
        assert mmlu_task._min_ngram_size == 5
        assert mmlu_task._max_ngram_size == 10

        ngrams = mmlu_task.generate_ngrams()
        assert len(ngrams) > 0

    def test_mmlu_raises_exception_without_path(self):
        with pytest.raises(Exception, match="Must provide path that contain"):
            MMLU(path=None)

    @mock.patch("nemo_curator.utils.file_utils.get_all_files_paths_under")
    def test_bigbenchhard_init_and_generate_ngrams(
        self, mock_get_paths, mock_jsonl_data, tmp_path
    ):
        bbh_dir = tmp_path / "bbh"
        bbh_dir.mkdir()
        bbh_file = bbh_dir / "data.jsonl"
        with open(bbh_file, "w") as f:
            f.writelines(mock_jsonl_data)

        mock_get_paths.return_value = [str(bbh_file)]

        bbh_task = BigBenchHard(path=str(bbh_dir), min_ngram_size=5, max_ngram_size=10)
        assert bbh_task._task_name == "bigbench_hard"
        assert bbh_task._min_ngram_size == 5
        assert bbh_task._max_ngram_size == 10

        ngrams = bbh_task.generate_ngrams()
        assert len(ngrams) > 0

    def test_bigbenchhard_raises_exception_without_path(self):
        with pytest.raises(Exception, match="Must provide path that contain"):
            BigBenchHard(path=None)

    @mock.patch("nemo_curator.utils.file_utils.get_all_files_paths_under")
    def test_bigbenchlight_init_and_generate_ngrams(
        self, mock_get_paths, mock_jsonl_data, tmp_path
    ):
        bbl_dir = tmp_path / "bbl"
        bbl_dir.mkdir()
        bbl_file = bbl_dir / "data.jsonl"
        with open(bbl_file, "w") as f:
            f.writelines(mock_jsonl_data)

        mock_get_paths.return_value = [str(bbl_file)]

        bbl_task = BigBenchLight(path=str(bbl_dir), min_ngram_size=5, max_ngram_size=10)
        assert bbl_task._task_name == "bigbench_light"
        assert bbl_task._min_ngram_size == 5
        assert bbl_task._max_ngram_size == 10

        ngrams = bbl_task.generate_ngrams()
        assert len(ngrams) > 0

    def test_bigbenchlight_raises_exception_without_path(self):
        with pytest.raises(Exception, match="Must provide path that contain"):
            BigBenchLight(path=None)

    @mock.patch("nemo_curator.utils.file_utils.get_all_files_paths_under")
    def test_multilingual_init_and_generate_ngrams(
        self, mock_get_paths, mock_jsonl_data, tmp_path
    ):
        ml_dir = tmp_path / "multilingual"
        ml_dir.mkdir()
        ml_file = ml_dir / "data.jsonl"
        with open(ml_file, "w") as f:
            f.writelines(mock_jsonl_data)

        mock_get_paths.return_value = [str(ml_file)]

        ml_task = Multilingual(path=str(ml_dir), min_ngram_size=5, max_ngram_size=10)
        assert ml_task._task_name == "multilingual"
        assert ml_task._min_ngram_size == 5
        assert ml_task._max_ngram_size == 10

        ngrams = ml_task.generate_ngrams()
        assert len(ngrams) > 0

    def test_multilingual_raises_exception_without_path(self):
        with pytest.raises(Exception, match="Must provide path to"):
            Multilingual(path=None)

    @mock.patch("nemo_curator.tasks.metrics.load_dataset")
    def test_piqa_init_and_generate_ngrams(self, mock_load_dataset):
        # For PIQA, setup a specific dataset structure
        mock_piqa_dataset = [
            {
                "goal": "This is a goal for PIQA with enough words to test ngram generation."
            },
            {"goal": "Another goal with sufficient length for PIQA testing."},
            {"goal": "Short goal."},
        ]
        mock_load_dataset.return_value = Dataset.from_list(mock_piqa_dataset)

        piqa_task = PIQA(min_ngram_size=5, max_ngram_size=10)
        assert piqa_task._task_name == "piqa"
        assert piqa_task._min_ngram_size == 5
        assert piqa_task._max_ngram_size == 10

        ngrams = piqa_task.generate_ngrams()
        assert len(ngrams) > 0

    @mock.patch("nemo_curator.tasks.metrics.load_dataset")
    def test_winogrande_init_and_generate_ngrams(self, mock_load_dataset):
        # For Winogrande, setup a specific dataset structure
        mock_winogrande_dataset = [
            {
                "sentence": "This is a Winogrande sentence with enough words to test ngram generation."
            },
            {
                "sentence": "Another sentence with sufficient length for Winogrande testing."
            },
            {"sentence": "Short sentence."},
        ]
        mock_load_dataset.return_value = Dataset.from_list(mock_winogrande_dataset)

        winogrande_task = Winogrande(min_ngram_size=5, max_ngram_size=10)
        assert winogrande_task._task_name == "winogrande"
        assert winogrande_task._min_ngram_size == 5
        assert winogrande_task._max_ngram_size == 10

        ngrams = winogrande_task.generate_ngrams()
        assert len(ngrams) > 0

    def test_lambada_init_and_generate_ngrams(self, mock_lambada_data, tmp_path):
        # Create a temporary file with mock Lambada data
        lambada_file = tmp_path / "lambada.jsonl"
        with open(lambada_file, "w") as f:
            f.writelines(mock_lambada_data)

        lambada_task = Lambada(
            file_path=str(lambada_file), min_ngram_size=5, max_ngram_size=10
        )
        assert lambada_task._task_name == "lambada"
        assert lambada_task._min_ngram_size == 5
        assert lambada_task._max_ngram_size == 10

        ngrams = lambada_task.generate_ngrams()
        assert len(ngrams) > 0

    def test_numdadc_init_and_generate_ngrams(self, mock_numdasc_data, tmp_path):
        # Create a temporary file with mock NumDasc data
        numdasc_file = tmp_path / "numdasc.jsonl"
        with open(numdasc_file, "w") as f:
            f.writelines(mock_numdasc_data)

        numdasc_task = NumDasc(
            n=2, file_path=str(numdasc_file), min_ngram_size=5, max_ngram_size=10
        )
        assert numdasc_task._task_name == "{n}dasc"
        assert numdasc_task._min_ngram_size == 5
        assert numdasc_task._max_ngram_size == 10
        assert numdasc_task._n == 2

        ngrams = numdasc_task.generate_ngrams()
        assert len(ngrams) > 0

    def test_storycloze_init_and_generate_ngrams(self, mock_storycloze_data, tmp_path):
        # Create a temporary file with mock StoryCloze data
        storycloze_file = tmp_path / "storycloze.jsonl"
        with open(storycloze_file, "w") as f:
            f.writelines(mock_storycloze_data)

        storycloze_task = StoryCloze(
            file_path=str(storycloze_file), min_ngram_size=5, max_ngram_size=10
        )
        assert storycloze_task._task_name == "story_cloze"
        assert storycloze_task._min_ngram_size == 5
        assert storycloze_task._max_ngram_size == 10

        ngrams = storycloze_task.generate_ngrams()
        assert len(ngrams) > 0

    # Tests for exception handling in Lambada, NumDasc, and StoryCloze

    def test_lambada_exception_handling(self, tmp_path):
        """Test that Lambada properly handles invalid JSON and prints error messages."""
        # Create a file with invalid JSON data
        invalid_file = tmp_path / "invalid_lambada.jsonl"
        with open(invalid_file, "w") as f:
            f.write('{"text": "Valid JSON line"}\n')
            f.write("Invalid JSON line that will cause an exception\n")
            f.write('{"incomplete": "Missing text field"}\n')

        lambada_task = Lambada(
            file_path=str(invalid_file), min_ngram_size=5, max_ngram_size=10
        )

        # Mock stdout to capture printed error messages
        with mock.patch("builtins.print") as mock_print:
            # This should not raise an exception
            ngrams = lambada_task.generate_ngrams()

            # Verify error was printed
            mock_print.assert_called()
            # At least one call should have "Error" in it
            assert any("Error" in str(call) for call in mock_print.call_args_list)

    def test_numdasc_exception_handling(self, tmp_path):
        """Test that NumDasc properly handles invalid JSON and prints error messages."""
        # Create a file with invalid JSON data
        invalid_file = tmp_path / "invalid_numdasc.jsonl"
        with open(invalid_file, "w") as f:
            f.write('{"context": "Valid", "completion": "JSON line"}\n')
            f.write("Invalid JSON line\n")
            f.write('{"context": "Missing completion field"}\n')

        numdasc_task = NumDasc(
            n=2, file_path=str(invalid_file), min_ngram_size=5, max_ngram_size=10
        )

        # Mock stdout to capture printed error messages
        with mock.patch("builtins.print") as mock_print:
            # This should not raise an exception
            ngrams = numdasc_task.generate_ngrams()

            # Verify error was printed
            mock_print.assert_called()
            # At least one call should have "Error" in it
            assert any("Error" in str(call) for call in mock_print.call_args_list)

    def test_storycloze_exception_handling(self, tmp_path):
        """Test that StoryCloze properly handles invalid JSON and prints error messages."""
        # Create a file with invalid JSON data
        invalid_file = tmp_path / "invalid_storycloze.jsonl"
        with open(invalid_file, "w") as f:
            f.write(
                '{"InputSentence1": "Valid", "InputSentence2": "JSON", "InputSentence3": "line", "InputSentence4": "here"}\n'
            )
            f.write("Invalid JSON line\n")
            f.write('{"InputSentence1": "Missing other required fields"}\n')

        storycloze_task = StoryCloze(
            file_path=str(invalid_file), min_ngram_size=5, max_ngram_size=10
        )

        # Mock stdout to capture printed error messages
        with mock.patch("builtins.print") as mock_print:
            # This should not raise an exception
            ngrams = storycloze_task.generate_ngrams()

            # Verify error was printed
            mock_print.assert_called()
            # At least one call should have "Error" in it
            assert any("Error" in str(call) for call in mock_print.call_args_list)

    # Edge cases and error handling tests

    def test_task_with_short_texts(self, tmp_path):
        """Test that tasks handle texts shorter than min_ngram_size properly."""
        # Create a file with very short texts
        short_texts_file = tmp_path / "short.jsonl"
        with open(short_texts_file, "w") as f:
            f.write('{"text": "Short."}\n')
            f.write('{"text": "Too short."}\n')

        # Mock the file paths function
        with mock.patch(
            "nemo_curator.utils.file_utils.get_all_files_paths_under"
        ) as mock_get_paths:
            mock_get_paths.return_value = [str(short_texts_file)]

            # Test with a high min_ngram_size
            task = MMLU(path=str(tmp_path), min_ngram_size=20, max_ngram_size=25)
            ngrams = task.generate_ngrams()

            # Should not generate any ngrams from short texts
            assert len(ngrams) == 0

    @mock.patch("nemo_curator.tasks.metrics.load_dataset")
    def test_error_handling_in_anli(self, mock_load_dataset, mock_premises_dataset):
        """Test that ANLI handles errors during processing."""
        # Create a dataset with bad entries that will cause errors
        bad_dataset = mock_premises_dataset.copy()
        # Add an entry without required fields
        bad_dataset.append({"not_premise": "This will cause an error"})

        mock_dataset_dict = {
            "test_r1": Dataset.from_list(bad_dataset),
            "test_r2": Dataset.from_list(mock_premises_dataset),
            "test_r3": Dataset.from_list(mock_premises_dataset),
        }
        mock_load_dataset.return_value = mock_dataset_dict

        # Should not raise an exception despite the bad data
        anli_task = ANLI(min_ngram_size=5, max_ngram_size=10)
        ngrams = anli_task.generate_ngrams()

        # Should still generate ngrams from the valid data
        assert len(ngrams) > 0
