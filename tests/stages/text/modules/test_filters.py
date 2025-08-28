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

import numpy as np
import pandas as pd
import pytest

from nemo_curator.stages.text.filters import (
    AlphaFilter,
    BoilerPlateStringFilter,
    BulletsFilter,
    CommonEnglishWordsFilter,
    DocumentFilter,
    EllipsisFilter,
    GeneralCommentToCodeFilter,
    HistogramFilter,
    HTMLBoilerplateFilter,
    LongWordFilter,
    MeanWordLengthFilter,
    NonAlphaNumericFilter,
    NumberOfLinesOfCodeFilter,
    NumbersFilter,
    ParenthesesFilter,
    PerExtensionFilter,
    PornographicUrlsFilter,
    PunctuationFilter,
    PythonCommentToCodeFilter,
    RepeatedLinesByCharFilter,
    RepeatedLinesFilter,
    RepeatedParagraphsByCharFilter,
    RepeatedParagraphsFilter,
    RepeatingDuplicateNGramsFilter,
    RepeatingTopNGramsFilter,
    SubstringFilter,
    SymbolsToWordsFilter,
    TokenCountFilter,
    UrlsFilter,
    WhiteSpaceFilter,
    WordCountFilter,
    WordsWithoutAlphabetsFilter,
    XMLHeaderFilter,
)
from nemo_curator.stages.text.modules import Filter, Score, ScoreFilter
from nemo_curator.tasks import DocumentBatch


class LetterCountFilter(DocumentFilter):
    """
    Keeps documents that have at least some number of a given letter
    """

    def __init__(self, letter: str = "a", min_count: int = 5) -> None:
        super().__init__()
        self.letter = letter
        self.min_count = min_count
        self._name = "letter_count"

    def score_document(self, text: str) -> int:
        return text.count(self.letter)

    def keep_document(self, score: int) -> bool:
        return score >= self.min_count


# A simple dummy tokenizer for our tests.
class DummyTokenizer:
    def encode(self, text: str) -> list[str]:
        # Simply splits the text on whitespace.
        return text.split()


def all_equal(left_dataset: DocumentBatch, right_dataset: DocumentBatch) -> bool:
    df_left = left_dataset.to_pandas().reset_index(drop=True)
    df_right = right_dataset.to_pandas().reset_index(drop=True)

    if not df_left.equals(df_right):
        print(f"DataFrames do not match: {df_left} != {df_right}")
        return False
    if left_dataset.task_id != right_dataset.task_id:
        print(f"Task IDs do not match: {left_dataset.task_id} != {right_dataset.task_id}")
        return False
    if left_dataset.dataset_name != right_dataset.dataset_name:
        print(f"Dataset names do not match: {left_dataset.dataset_name} != {right_dataset.dataset_name}")
        return False

    return True


def list_to_dataset(documents: list[str], col_name: str = "text") -> DocumentBatch:
    data = {col_name: documents}
    pdf = pd.DataFrame(data)

    return DocumentBatch(
        data=pdf,
        task_id="batch_1",
        dataset_name="test_1",
    )


@pytest.fixture
def letter_count_data() -> DocumentBatch:
    return DocumentBatch(
        data=pd.DataFrame({"documents": ["Two aa", "a a Three a", "Five aaa aa", "aaaSeven aaaa"]}),
        task_id="batch_1",
        dataset_name="test_1",
    )


class TestFilterModule:
    def test_score_filter(self, letter_count_data: DocumentBatch) -> None:
        letter_filter = LetterCountFilter()
        filter_step = ScoreFilter(letter_filter, text_field="documents")

        filtered_data = filter_step.process(letter_count_data)

        expected_data = DocumentBatch(
            data=pd.DataFrame({"documents": ["Five aaa aa", "aaaSeven aaaa"]}),
            task_id="batch_1_letter_count",
            dataset_name="test_1",
        )

        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    def test_score_document(self, letter_count_data: DocumentBatch) -> None:
        letter_filter = LetterCountFilter()
        score_field = "a_count"
        score_step = Score(
            letter_filter.score_document,
            text_field="documents",
            score_field=score_field,
        )

        scored_data = score_step.process(letter_count_data)

        expected_scores = pd.Series([2, 3, 5, 7])
        scores = scored_data.data[score_field]
        assert all(expected_scores == scores), f"Expected {expected_scores} but got {scores}"

    def test_score(self, letter_count_data: DocumentBatch) -> None:
        letter_filter = LetterCountFilter()
        score_field = "a_count"
        score_step = Score(
            letter_filter,
            text_field="documents",
            score_field=score_field,
        )

        scored_data = score_step.process(letter_count_data)

        expected_scores = pd.Series([2, 3, 5, 7])
        scores = scored_data.data[score_field]
        assert all(expected_scores == scores), f"Expected {expected_scores} but got {scores}"

    def test_retain_score_filter(self, letter_count_data: DocumentBatch) -> None:
        letter_filter = LetterCountFilter()
        score_field = "count_a"
        filter_step = ScoreFilter(letter_filter, text_field="documents", score_field=score_field)

        filtered_data = filter_step.process(letter_count_data)

        expected_data = DocumentBatch(
            data=pd.DataFrame({"documents": ["Five aaa aa", "aaaSeven aaaa"]}),
            task_id="batch_1_letter_count",
            dataset_name="test_1",
        )
        expected_data.data[score_field] = pd.Series([5, 7])
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    def test_filter_document(self, letter_count_data: DocumentBatch) -> None:
        letter_filter = LetterCountFilter()
        score_field = "a_count"
        score_step = Score(
            letter_filter.score_document,
            text_field="documents",
            score_field=score_field,
        )

        scored_data = score_step.process(letter_count_data)

        filter_step = Filter(letter_filter.keep_document, score_field)

        filtered_data = filter_step.process(scored_data)

        expected_data = DocumentBatch(
            data=pd.DataFrame({"documents": ["Five aaa aa", "aaaSeven aaaa"]}),
            task_id="batch_1_score_fn_filter_fn",
            dataset_name="test_1",
        )
        expected_data.data[score_field] = pd.Series([5, 7])
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    def test_filter(self, letter_count_data: DocumentBatch) -> None:
        letter_filter = LetterCountFilter()
        score_field = "a_count"
        score_step = Score(
            letter_filter,
            text_field="documents",
            score_field=score_field,
        )

        scored_data = score_step.process(letter_count_data)

        filter_step = Filter(letter_filter, score_field)

        filtered_data = filter_step.process(scored_data)

        expected_data = DocumentBatch(
            data=pd.DataFrame({"documents": ["Five aaa aa", "aaaSeven aaaa"]}),
            task_id="batch_1_letter_count_letter_count",
            dataset_name="test_1",
        )
        expected_data.data[score_field] = pd.Series([5, 7])
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    def test_invert(self, letter_count_data: DocumentBatch) -> None:
        letter_filter = LetterCountFilter()
        filter_step = ScoreFilter(letter_filter, text_field="documents", invert=True)

        filtered_data = filter_step.process(letter_count_data)

        expected_data = DocumentBatch(
            data=pd.DataFrame({"documents": ["Two aa", "a a Three a"]}),
            task_id="batch_1_letter_count",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    @pytest.mark.parametrize("score_field", [None, "a_count", ["a_count"], ["a_count", "e_count"]])
    def test_score_filter_chain(self, letter_count_data: DocumentBatch, score_field: list[str] | None) -> None:
        if score_field in ["a_count", ["a_count"]]:
            with pytest.raises(ValueError):  # noqa: PT011
                ScoreFilter(
                    [LetterCountFilter(letter="a"), LetterCountFilter(letter="e", min_count=2)],
                    text_field="documents",
                    score_field=score_field,
                )
            return

        filters = ScoreFilter(
            [LetterCountFilter(letter="a"), LetterCountFilter(letter="e", min_count=2)],
            text_field="documents",
            score_field=score_field,
        )

        filtered_data = filters.process(letter_count_data)

        if score_field is None:
            expected_df = pd.DataFrame({"documents": ["aaaSeven aaaa"]})
        else:
            expected_df = pd.DataFrame({"documents": ["aaaSeven aaaa"], "a_count": [7], "e_count": [2]})

        expected_data = DocumentBatch(
            data=expected_df,
            task_id="batch_1_score_filter_chain_of_letter_count_letter_count",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    @pytest.mark.parametrize("score_field", [None, "a_count", ["a_count"], ["a_count", "e_count"]])
    def test_score_chain(self, letter_count_data: DocumentBatch, score_field: list[str] | None) -> None:
        if score_field in [None, "a_count", ["a_count"]]:
            with pytest.raises(ValueError):  # noqa: PT011
                Score(
                    [LetterCountFilter(letter="a"), LetterCountFilter(letter="e", min_count=2)],
                    text_field="documents",
                    score_field=score_field,
                )
            return

        filters = Score(
            [LetterCountFilter(letter="a"), LetterCountFilter(letter="e", min_count=2)],
            text_field="documents",
            score_field=score_field,
        )

        filtered_data = filters.process(letter_count_data)

        expected_df = pd.DataFrame(
            {
                "documents": ["Two aa", "a a Three a", "Five aaa aa", "aaaSeven aaaa"],
                "a_count": [2, 3, 5, 7],
                "e_count": [0, 2, 1, 2],
            }
        )

        expected_data = DocumentBatch(
            data=expected_df,
            task_id="batch_1_score_chain_of_letter_count_letter_count",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    @pytest.mark.parametrize("filter_field", [None, "a_count", ["a_count"], ["a_count", "e_count"]])
    def test_filter_chain(self, filter_field: list[str] | None) -> None:
        letter_count_data = DocumentBatch(
            data=pd.DataFrame(
                {
                    "documents": ["Two aa", "a a Three a", "Five aaa aa", "aaaSeven aaaa"],
                    "a_count": [2, 3, 5, 7],
                    "e_count": [0, 2, 1, 2],
                }
            ),
            task_id="batch_1",
            dataset_name="test_1",
        )

        if filter_field in [None, "a_count", ["a_count"]]:
            with pytest.raises(ValueError):  # noqa: PT011
                Filter(
                    [LetterCountFilter(letter="a"), LetterCountFilter(letter="e", min_count=2)],
                    filter_field=filter_field,
                )
            return

        filters = Filter(
            [LetterCountFilter(letter="a"), LetterCountFilter(letter="e", min_count=2)],
            filter_field=filter_field,
        )

        filtered_data = filters.process(letter_count_data)

        expected_df = pd.DataFrame(
            {
                "documents": ["aaaSeven aaaa"],
                "a_count": [7],
                "e_count": [2],
            }
        )

        expected_data = DocumentBatch(
            data=expected_df,
            task_id="batch_1_filter_chain_of_letter_count_letter_count",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    def test_score_filter_all_rows(self, letter_count_data: DocumentBatch) -> None:
        # Processes out all rows
        filters = ScoreFilter(
            LetterCountFilter(letter="a", min_count=8),
            text_field="documents",
        )
        intermediate_data = filters.process(letter_count_data)

        # Applies a filter on an empty batch
        filters = ScoreFilter(
            LetterCountFilter(letter="e", min_count=2),
            text_field="documents",
        )
        filtered_data = filters.process(intermediate_data)

        # Empty DataFrame
        expected_df = pd.DataFrame()

        expected_data = DocumentBatch(
            data=expected_df,
            task_id="batch_1_letter_count_letter_count",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    def test_score_filter_all_rows_chain(self, letter_count_data: DocumentBatch) -> None:
        # Processes out all rows on the first filter
        filters = ScoreFilter(
            [LetterCountFilter(letter="a", min_count=8), LetterCountFilter(letter="e", min_count=2)],
            text_field="documents",
        )

        filtered_data = filters.process(letter_count_data)

        # Empty DataFrame
        expected_df = pd.DataFrame()

        expected_data = DocumentBatch(
            data=expected_df,
            task_id="batch_1_score_filter_chain_of_letter_count_letter_count",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"


class TestHeuristicFilters:
    def test_nonalpha(self) -> None:
        dataset = list_to_dataset(["", "This is a test case.", "%$^%$^%$&^$()))))", "$aaa"])
        filters = ScoreFilter(NonAlphaNumericFilter())

        filtered_data = filters.process(dataset)

        expected_data = DocumentBatch(
            data=pd.DataFrame({"text": ["This is a test case.", "$aaa"]}),
            task_id="batch_1_alpha_numeric",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    def test_symbolswords(self) -> None:
        dataset = list_to_dataset(
            [
                "mixed bag ... #",
                "full of words",
                "... # ... # #",
                "barely ok 3 4 5 6 7 8 9 #",
            ]
        )
        filters = ScoreFilter(SymbolsToWordsFilter())

        filtered_data = filters.process(dataset)

        expected_data = DocumentBatch(
            data=pd.DataFrame({"text": ["full of words", "barely ok 3 4 5 6 7 8 9 #"]}),
            task_id="batch_1_symbol_to_word",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    def test_numbers(self) -> None:
        dataset = list_to_dataset(["purely letters", "34134543", "$!@$@!$!@", "abcdefghi1"])
        filters = ScoreFilter(NumbersFilter(max_number_to_text_ratio=0.1))

        filtered_data = filters.process(dataset)

        expected_data = DocumentBatch(
            data=pd.DataFrame({"text": ["purely letters", "$!@$@!$!@", "abcdefghi1"]}),
            task_id="batch_1_numbers_ratio",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    def test_urls(self) -> None:
        dataset = list_to_dataset(
            [
                "https://www.nvidia.com/en-us/",
                "no urls here!",
                "$!@$@!$!@",
                "bunch of other words with url afdsjafidsaofjbwreowihfdsafbdashuoiotauhiofdafdsafd fdasfdafdsafdsafdsafdsafdsafdsa https://www.nvidia.com/en-us/ something else after the url etc more and more",
                "words with url https://www.nvidia.com/en-us/",
            ]
        )
        filters = ScoreFilter(UrlsFilter())

        filtered_data = filters.process(dataset)

        expected_data = DocumentBatch(
            data=pd.DataFrame(
                {
                    "text": [
                        "no urls here!",
                        "$!@$@!$!@",
                        "bunch of other words with url afdsjafidsaofjbwreowihfdsafbdashuoiotauhiofdafdsafd fdasfdafdsafdsafdsafdsafdsafdsa https://www.nvidia.com/en-us/ something else after the url etc more and more",
                    ]
                }
            ),
            task_id="batch_1_urls_ratio",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    def test_bullets(self) -> None:
        dataset = list_to_dataset(
            [
                "• not good",
                "good",
                "50 \n ⦾ 50",
                "⁌ this \n⁌ should \n⁌barely \n⁌pass \n⁌5 \n⁌6 \n⁌7 \n⁌8 \n⁌9 \n done!",
            ]
        )
        filters = ScoreFilter(BulletsFilter())

        filtered_data = filters.process(dataset)

        expected_data = DocumentBatch(
            data=pd.DataFrame(
                {
                    "text": [
                        "good",
                        "50 \n ⦾ 50",
                        "⁌ this \n⁌ should \n⁌barely \n⁌pass \n⁌5 \n⁌6 \n⁌7 \n⁌8 \n⁌9 \n done!",
                    ]
                }
            ),
            task_id="batch_1_bullet_ratio",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    def test_whitespace(self) -> None:
        dataset = list_to_dataset(["\t\n\r", "good", "50%\n\n\n", "123\b"])
        filters = ScoreFilter(WhiteSpaceFilter())

        filtered_data = filters.process(dataset)

        expected_data = DocumentBatch(
            data=pd.DataFrame({"text": ["good", "123\b"]}),
            task_id="batch_1_white_space",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    def test_parentheses(self) -> None:
        dataset = list_to_dataset(["()", "(not good)", "this is completely absolutely fine", "123456789("])
        filters = ScoreFilter(ParenthesesFilter())

        filtered_data = filters.process(dataset)

        expected_data = DocumentBatch(
            data=pd.DataFrame({"text": ["this is completely absolutely fine", "123456789("]}),
            task_id="batch_1_parentheses_ratio",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    def test_longword(self) -> None:
        dataset = list_to_dataset(["tiny", "large"])
        filters = ScoreFilter(LongWordFilter(max_word_length=4))

        filtered_data = filters.process(dataset)

        expected_data = DocumentBatch(
            data=pd.DataFrame({"text": ["tiny"]}),
            task_id="batch_1_max_word_length",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    def test_wordcount(self) -> None:
        dataset = list_to_dataset(["", "one", "two words", "$#@$ %$@$#@ !#@!", "one two three four five"])
        filters = ScoreFilter(WordCountFilter(min_words=2, max_words=4))

        filtered_data = filters.process(dataset)

        expected_data = DocumentBatch(
            data=pd.DataFrame({"text": ["two words", "$#@$ %$@$#@ !#@!"]}),
            task_id="batch_1_word_count",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    def test_wordcount_zh(self) -> None:
        dataset = list_to_dataset(["", "你好。", "我喜欢学习中文。"])
        filters = ScoreFilter(WordCountFilter(min_words=2, max_words=5, lang="zh"))

        filtered_data = filters.process(dataset)

        expected_data = DocumentBatch(
            data=pd.DataFrame({"text": ["你好。", "我喜欢学习中文。"]}),
            task_id="batch_1_word_count",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    @pytest.mark.skip(reason="Skipping MeCab runtime error for now")
    def test_wordcount_ja(self) -> None:
        dataset = list_to_dataset(["", "猫が寝ます。", "私は日本語のテキストを分割します。"])
        filters = ScoreFilter(WordCountFilter(min_words=5, max_words=11, lang="ja"))

        filtered_data = filters.process(dataset)

        expected_data = DocumentBatch(
            data=pd.DataFrame({"text": ["猫が寝ます。", "私は日本語のテキストを分割します。"]}),
            task_id="batch_1_word_count_ja",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    def test_boilerplate(self) -> None:
        dataset = list_to_dataset(
            [
                "nothing\t here",
                "1\n\n2\n\n3\n\n4\n\n5\n\n6\n\nterms of use\n\n privacy policy\n\n cookie policy\n\nuses cookies",
                "too much \n\n privacy & cookies policy",
            ]
        )
        filters = ScoreFilter(BoilerPlateStringFilter())

        filtered_data = filters.process(dataset)

        expected_data = DocumentBatch(
            data=pd.DataFrame(
                {
                    "text": [
                        "nothing\t here",
                        "1\n\n2\n\n3\n\n4\n\n5\n\n6\n\nterms of use\n\n privacy policy\n\n cookie policy\n\nuses cookies",
                    ]
                }
            ),
            task_id="batch_1_boilerplate_string_ratio",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    def test_meanwordlength(self) -> None:
        dataset = list_to_dataset(
            [
                "a",
                "aa",
                "superlongword short",
                "evenly balanced",
                "waytoolongforasingleword",
            ]
        )
        filters = ScoreFilter(MeanWordLengthFilter())

        filtered_data = filters.process(dataset)

        expected_data = DocumentBatch(
            data=pd.DataFrame({"text": ["superlongword short", "evenly balanced"]}),
            task_id="batch_1_mean_word_length",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    def test_repeatedlines(self) -> None:
        dataset = list_to_dataset(["totally unique", "half.\nhalf."])
        filters = ScoreFilter(RepeatedLinesFilter())

        filtered_data = filters.process(dataset)

        expected_data = DocumentBatch(
            data=pd.DataFrame({"text": ["totally unique"]}),
            task_id="batch_1_repeated_lines",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    def test_repeatedparagraphs(self) -> None:
        dataset = list_to_dataset(["totally unique", "half.\n\nhalf."])
        filters = ScoreFilter(RepeatedParagraphsFilter())

        filtered_data = filters.process(dataset)

        expected_data = DocumentBatch(
            data=pd.DataFrame({"text": ["totally unique"]}),
            task_id="batch_1_repeated_paragraphs",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    def test_repeatedlineschar(self) -> None:
        dataset = list_to_dataset(
            [
                "totally unique",
                "a.\na.\nvery very very short duplicate.",
                "half.\nhalf.",
                "super very incredibly huge long duplicate.\nsuper very incredibly huge long duplicate.\na.\nb.\nc.",
            ]
        )
        filters = ScoreFilter(RepeatedLinesByCharFilter())

        filtered_data = filters.process(dataset)

        expected_data = DocumentBatch(
            data=pd.DataFrame({"text": ["totally unique", "a.\na.\nvery very very short duplicate."]}),
            task_id="batch_1_repeated_lines_char",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    def test_repeatedparagraphschar(self) -> None:
        dataset = list_to_dataset(
            [
                "totally unique",
                "a.\n\n  a.\n\n  very very very short duplicate.",
                "half.\n\nhalf.",
                "super very incredibly huge long duplicate.\n\nsuper very incredibly huge long duplicate.\n\n  a.\n\n  b.\n\n  c.",
            ]
        )
        filters = ScoreFilter(RepeatedParagraphsByCharFilter())

        filtered_data = filters.process(dataset)

        expected_data = DocumentBatch(
            data=pd.DataFrame({"text": ["totally unique", "a.\n\n  a.\n\n  very very very short duplicate."]}),
            task_id="batch_1_repeated_paragraphs_char",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    def test_repeatingtopngrams(self) -> None:
        dataset = list_to_dataset(
            [
                "this is a totally fine sentence with no repeat ngrams so we are ok",
                "a b . a b",
                "a a a a a a",
                "totally fine small dupe a b a b",
            ]
        )
        filters = ScoreFilter(RepeatingTopNGramsFilter())

        filtered_data = filters.process(dataset)

        expected_data = DocumentBatch(
            data=pd.DataFrame(
                {
                    "text": [
                        "this is a totally fine sentence with no repeat ngrams so we are ok",
                        "totally fine small dupe a b a b",
                    ]
                }
            ),
            task_id="batch_1_repeating_top_2grams",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    def test_repeatingduplicatengrams(self) -> None:
        dataset = list_to_dataset(["a a b b a a b b", "totally fine", "a a a a this should be fine as well"])
        filters = ScoreFilter(RepeatingDuplicateNGramsFilter())

        filtered_data = filters.process(dataset)

        expected_data = DocumentBatch(
            data=pd.DataFrame({"text": ["totally fine", "a a a a this should be fine as well"]}),
            task_id="batch_1_repeating_dup_2gram",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    def test_punctuation(self) -> None:
        dataset = list_to_dataset(["not good", "good.", "just\n barely\n fine\n ok\n yep."])
        filters = ScoreFilter(PunctuationFilter(max_num_sentences_without_endmark_ratio=0.8))

        filtered_data = filters.process(dataset)

        expected_data = DocumentBatch(
            data=pd.DataFrame({"text": ["good.", "just\n barely\n fine\n ok\n yep."]}),
            task_id="batch_1_punctuation",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    def test_ellipsis(self) -> None:
        dataset = list_to_dataset(["not good...", "good.", "just...\n barely...\n fine...\n ok...\n yep."])
        filters = ScoreFilter(EllipsisFilter(max_num_lines_ending_with_ellipsis_ratio=0.8))

        filtered_data = filters.process(dataset)

        expected_data = DocumentBatch(
            data=pd.DataFrame({"text": ["good.", "just...\n barely...\n fine...\n ok...\n yep."]}),
            task_id="batch_1_ellipsis",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    def test_commonenglishwords(self) -> None:
        dataset = list_to_dataset(["uncommon", "the and", "the and and of to"])
        filters = ScoreFilter(CommonEnglishWordsFilter())

        filtered_data = filters.process(dataset)

        expected_data = DocumentBatch(
            data=pd.DataFrame({"text": ["the and", "the and and of to"]}),
            task_id="batch_1_common_english_words",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    def test_wordswithoutalphabets(self) -> None:
        dataset = list_to_dataset(["totally fine", "good good good good !", "@"])
        filters = ScoreFilter(WordsWithoutAlphabetsFilter())

        filtered_data = filters.process(dataset)

        expected_data = DocumentBatch(
            data=pd.DataFrame({"text": ["totally fine", "good good good good !"]}),
            task_id="batch_1_words_without_alphabets",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    def test_pornographicurls(self) -> None:
        dataset = list_to_dataset(
            [
                "no url",
                "fine url https://www.nvidia.com/en-us/",
                "bad url https://www.pornhub.com/",
            ]
        )
        filters = ScoreFilter(PornographicUrlsFilter())

        filtered_data = filters.process(dataset)

        expected_data = DocumentBatch(
            data=pd.DataFrame({"text": ["no url", "fine url https://www.nvidia.com/en-us/"]}),
            task_id="batch_1_PornographicUrlsFilter",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    def test_histogram(self) -> None:
        dataset = list_to_dataset(
            [
                "This is a perfectly fine English document.",
                "But if you insist that this is written in Chinese,",
                "it's likely that something is fishy.",
                "另一方面，这是一个好的中文文档，",  # noqa: RUF001
                "但你一定要说这是英文文档，",  # noqa: RUF001
                "那很可能有些地方出了差错。",
            ]
        )
        filter1 = ScoreFilter(HistogramFilter(lang="en"))
        filter2 = ScoreFilter(HistogramFilter(lang="zh"))

        filtered_data1 = filter1.process(dataset)
        filtered_data2 = filter2.process(dataset)

        expected_data1 = DocumentBatch(
            data=pd.DataFrame(
                {
                    "text": [
                        "This is a perfectly fine English document.",
                        "But if you insist that this is written in Chinese,",
                        "it's likely that something is fishy.",
                    ]
                }
            ),
            task_id="batch_1_histogram",
            dataset_name="test_1",
        )
        expected_data2 = DocumentBatch(
            data=pd.DataFrame(
                {
                    "text": [
                        "另一方面，这是一个好的中文文档，",  # noqa: RUF001
                        "但你一定要说这是英文文档，",  # noqa: RUF001
                        "那很可能有些地方出了差错。",
                    ]
                }
            ),
            task_id="batch_1_histogram",
            dataset_name="test_1",
        )
        assert all_equal(expected_data1, filtered_data1), f"Expected {expected_data1} but got {filtered_data1}"
        assert all_equal(expected_data2, filtered_data2), f"Expected {expected_data2} but got {filtered_data2}"


class TestTokenCountFilter:
    def test_score_document(self) -> None:
        tokenizer = DummyTokenizer()
        token_filter = TokenCountFilter(tokenizer, min_tokens=2, max_tokens=3)
        text = "another test case"  # Should yield 3 tokens.
        score = token_filter.score_document(text)
        assert score == 3

    def test_keep_document(self) -> None:
        tokenizer = DummyTokenizer()
        token_filter = TokenCountFilter(tokenizer, min_tokens=2, max_tokens=3)
        # Check that a score of 1 (too few) and 4 (too many) are rejected,
        # while scores of 2 and 3 are accepted.
        assert token_filter.keep_document(2)
        assert token_filter.keep_document(3)
        assert not token_filter.keep_document(1)
        assert not token_filter.keep_document(4)

    def test_filter_dataset(self) -> None:
        # Create a dataset of documents with different word counts.
        docs = [
            "hello",  # 1 token
            "hello world",  # 2 tokens
            "this is a test",  # 4 tokens
            "another test case",  # 3 tokens
        ]
        dataset = list_to_dataset(docs, col_name="text")

        tokenizer = DummyTokenizer()
        token_filter = TokenCountFilter(tokenizer, min_tokens=2, max_tokens=3)
        filter_step = ScoreFilter(token_filter, text_field="text")

        filtered_dataset = filter_step.process(dataset)

        # We expect to keep only the documents with exactly 2 or 3 tokens.
        expected_dataset = DocumentBatch(
            data=pd.DataFrame({"text": ["hello world", "another test case"]}),
            task_id="batch_1_token_count",
            dataset_name="test_1",
        )
        assert all_equal(expected_dataset, filtered_dataset)

    def test_filter_dataset_default(self) -> None:
        # Create a dataset of documents with different word counts.
        docs = [
            "hello",  # 1 token
            "hello world",  # 2 tokens
            "this is a test",  # 4 tokens
            "another test case",  # 3 tokens
        ]
        dataset = list_to_dataset(docs, col_name="text")

        tokenizer = DummyTokenizer()
        # Using default settings: min_tokens=0 and max_tokens=inf, so all documents pass.
        token_filter = TokenCountFilter(tokenizer)
        filter_step = ScoreFilter(token_filter, text_field="text")

        filtered_dataset = filter_step.process(dataset)

        # We expect to keep all documents.
        expected_dataset = DocumentBatch(
            data=pd.DataFrame({"text": docs}),
            task_id="batch_1_token_count",
            dataset_name="test_1",
        )
        assert all_equal(expected_dataset, filtered_dataset)


class TestSubstringFilter:
    def test_invalid_position(self) -> None:
        # Creating a SubstringFilter with an invalid position should raise a ValueError.
        with pytest.raises(ValueError):  # noqa: PT011
            SubstringFilter("foo", "middle")

    def test_prefix_mode(self) -> None:
        filter_prefix = SubstringFilter("Hello", "prefix")
        # Positive example: text starts with "Hello".
        text = "Hello world"
        score = filter_prefix.score_document(text)
        assert score == 1
        assert filter_prefix.keep_document(score)
        # Negative example: text does not start with "Hello".
        text2 = "world Hello"
        score2 = filter_prefix.score_document(text2)
        assert score2 == 0
        assert not filter_prefix.keep_document(score2)

    def test_suffix_mode(self) -> None:
        filter_suffix = SubstringFilter("end", "suffix")
        # Positive example: text ends with "end".
        text = "This is the end"
        score = filter_suffix.score_document(text)
        assert score == 1
        assert filter_suffix.keep_document(score)
        # Negative example: text does not end with "end".
        text2 = "The end is near"
        score2 = filter_suffix.score_document(text2)
        assert score2 == 0
        assert not filter_suffix.keep_document(score2)

    def test_any_mode(self) -> None:
        filter_any = SubstringFilter("test", "any")
        # Positive example: text contains "test".
        text = "this is a test string"
        score = filter_any.score_document(text)
        assert score == 1
        assert filter_any.keep_document(score)
        # Negative example: text does not contain "test".
        text2 = "this is a string"
        score2 = filter_any.score_document(text2)
        assert score2 == 0
        assert not filter_any.keep_document(score2)

    def test_filter_dataset_prefix(self) -> None:
        docs = ["Hello world", "world Hello", "Hello everyone", "Not matching"]
        dataset = list_to_dataset(docs, col_name="text")
        filter_prefix = SubstringFilter("Hello", "prefix")
        filter_step = ScoreFilter(filter_prefix, text_field="text")

        filtered_dataset = filter_step.process(dataset)

        # Expect only those records where the text starts with "Hello".
        expected_dataset = DocumentBatch(
            data=pd.DataFrame({"text": ["Hello world", "Hello everyone"]}),
            task_id="batch_1_SubstringFilter",
            dataset_name="test_1",
        )

        assert all_equal(expected_dataset, filtered_dataset)

    def test_filter_dataset_suffix(self) -> None:
        docs = [
            "This is the end",  # ends with "end"
            "end of story",  # does not end with "end"
            "ending is good",  # does not end with "end"
            "Not matching end",  # ends with "end"
            "The end",  # ends with "end"
        ]
        dataset = list_to_dataset(docs, col_name="text")
        filter_suffix = SubstringFilter("end", "suffix")
        filter_step = ScoreFilter(filter_suffix, text_field="text")

        filtered_dataset = filter_step.process(dataset)

        # Expect only those records that end with "end".
        expected_dataset = DocumentBatch(
            data=pd.DataFrame({"text": ["This is the end", "Not matching end", "The end"]}),
            task_id="batch_1_SubstringFilter",
            dataset_name="test_1",
        )

        assert all_equal(expected_dataset, filtered_dataset)

    def test_filter_dataset_any(self) -> None:
        docs = ["test case", "This is a testcase", "no match here", "another test"]
        dataset = list_to_dataset(docs, col_name="text")
        filter_any = SubstringFilter("test", "any")
        filter_step = ScoreFilter(filter_any, text_field="text")

        filtered_dataset = filter_step.process(dataset)

        # Expect documents that contain "test" anywhere.
        expected_dataset = DocumentBatch(
            data=pd.DataFrame({"text": ["test case", "This is a testcase", "another test"]}),
            task_id="batch_1_SubstringFilter",
            dataset_name="test_1",
        )

        assert all_equal(expected_dataset, filtered_dataset)


class TestCodeFilters:
    def test_python_comment_to_code(self) -> None:
        doc_1 = "# Good code\nprint('hello world')"
        doc_2 = "print('bad code')"
        doc_3 = "# Too many\n# comments!"
        doc_4 = "'''Good comment'''\nprint('hello world')"
        dataset = list_to_dataset([doc_1, doc_2, doc_3, doc_4])
        filters = ScoreFilter(PythonCommentToCodeFilter())
        filtered_data = filters.process(dataset)

        expected_data = DocumentBatch(
            data=pd.DataFrame({"text": [doc_1, doc_4]}),
            task_id="batch_1_python_comment_ratio",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    def test_general_commment_to_code(self) -> None:
        doc_1 = '// Good code\nprintf("hello world\\n")'
        doc_2 = 'printf("bad code\\n")'
        doc_3 = "// Way far too many\n// comments!"
        doc_4 = '/*\nGood comment\n*/\nprintf("hello world\\n")'
        dataset = list_to_dataset([doc_1, doc_2, doc_3, doc_4])
        filters = ScoreFilter(GeneralCommentToCodeFilter("text/x-c++"))

        filtered_data = filters.process(dataset)

        expected_data = DocumentBatch(
            data=pd.DataFrame({"text": [doc_1, doc_4]}),
            task_id="batch_1_comment_ratio",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    def test_number_lines_code(self) -> None:
        doc_1 = """print("too short")"""
        doc_2 = """print("just")
        print("right")"""
        doc_3 = """print("way")
        print("too")
        print("long")
        print("!")"""
        dataset = list_to_dataset([doc_1, doc_2, doc_3])
        filters = ScoreFilter(NumberOfLinesOfCodeFilter(min_lines=2, max_lines=3))

        filtered_data = filters.process(dataset)

        expected_data = DocumentBatch(
            data=pd.DataFrame({"text": [doc_2]}),
            task_id="batch_1_num_lines",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    def test_xml_header(self) -> None:
        dataset = list_to_dataset(["no header", "<?xml version=1.0>", "slightly offset <?xml version="])
        filters = ScoreFilter(XMLHeaderFilter())

        filtered_data = filters.process(dataset)

        expected_data = DocumentBatch(
            data=pd.DataFrame({"text": ["no header"]}),
            task_id="batch_1_xml_header",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    def test_alpha(self) -> None:
        dataset = list_to_dataset(["full of alphabet", "<>?$#@!", "mixed <>"])
        filters = ScoreFilter(AlphaFilter())

        filtered_data = filters.process(dataset)

        expected_data = DocumentBatch(
            data=pd.DataFrame({"text": ["full of alphabet", "mixed <>"]}),
            task_id="batch_1_alpha_filter",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    def test_html_boilerplate(self) -> None:
        good_doc = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Sample Webpage</title>
        </head>
        <body>
            <h1>Welcome to my sample webpage</h1>
            <p>This is a very fun paragraph on my sample webpage.</p>
        </body>
        </html>
        """
        boilerplate_heavy_doc = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Boilerplate Webpage</title>
        </head>
        <body>
            <h1><span>Welcome</span> <span>to</span> <span>my</span> <span>boilerplate</span> <span>webpage</span></h1>
            <div>
                <div>
                    <div><p>hi</p></div>
                </div>
                <div>
                    <div><p>hi</p></div>
                </div>
            </div>
        </body>
        </html>
        """
        small_doc = """
            <!DOCTYPE html>
            <html><body>hello world</body></html>
        """
        dataset = list_to_dataset([good_doc, boilerplate_heavy_doc, small_doc])
        filters = ScoreFilter(HTMLBoilerplateFilter())

        filtered_data = filters.process(dataset)

        expected_data = DocumentBatch(
            data=pd.DataFrame({"text": [good_doc]}),
            task_id="batch_1_html_boilerplate",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    @pytest.fixture
    def per_extension_filter(self) -> PerExtensionFilter:
        metadata_file = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "..",
                "..",
                "nemo_curator",
                "utils",
                "code_meta.csv",
            )
        )

        return PerExtensionFilter("c++", "cpp", metadata_file=metadata_file)

    def test_per_extension_filter(self, per_extension_filter: PerExtensionFilter) -> None:
        good_cpp = """
        #include <iostream>
        using namespace std;
        int main() {
            cout << "Hello World!" << endl;
            return 0;
        };
        """
        dataset = list_to_dataset([good_cpp])
        filters = ScoreFilter(per_extension_filter)

        filtered_data = filters.process(dataset)

        expected_data = DocumentBatch(
            data=pd.DataFrame({"text": [good_cpp]}),
            task_id="batch_1_per_extension_filter",
            dataset_name="test_1",
        )

        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    @pytest.mark.parametrize(
        "content,expected",  # noqa: PT006
        [
            ("", (0, 0.0)),
            ("\n", (0, 0.0)),
            ("abc\n", (3, 1.5)),
            ("Lorem ipsum \ndolor sit amet,", (15, 13.5)),
        ],
    )
    def test_line_statistics(
        self, per_extension_filter: PerExtensionFilter, content: str, expected: tuple[int, float]
    ) -> None:
        line_statistics = per_extension_filter._line_statistics(content)
        assert line_statistics == expected, f"Expected {expected} but got {line_statistics}"


class FakeQualityFilter(DocumentFilter):
    """
    Emulates FastTextQualityFilter without a model
    """

    def __init__(self, alpha: float = 3, seed: int = 42):
        super().__init__()
        self._alpha = alpha
        self._seed = np.random.seed(seed)  # noqa: NPY002

    def score_document(self, text: str) -> float:
        if text == "a":
            return 0.00
        elif text == "b":
            return 0.25
        elif text == "c":
            return 0.50
        elif text == "d":
            return 0.75
        else:
            msg = f"Unexpected text: {text}"
            raise ValueError(msg)

    def keep_document(self, score: float) -> bool:
        return np.random.pareto(self._alpha) > 1 - score  # noqa: NPY002


class FakeLangId(DocumentFilter):
    """
    Emulates FastTextLangId without a model
    """

    def __init__(self, min_langid_score: float = 0.3):
        super().__init__()
        self._cutoff = min_langid_score

    def score_document(self, text: str) -> str:
        if text in ["a", "d"]:
            return str([0.5, "EN"])
        if text == "b":
            return str([0.7, "HI"])
        if text == "c":
            return str([0.2, "PT"])
        else:
            msg = f"Unexpected text: {text}"
            raise ValueError(msg)

    def keep_document(self, score: float | str) -> bool:
        if isinstance(score, str):
            score = eval(score)  # noqa: S307

        return score[0] >= self._cutoff


class TestClassifierFilters:
    def test_fake_quality_filter(self) -> None:
        dataset = list_to_dataset(["a", "b", "c", "d"])
        filters = ScoreFilter(FakeQualityFilter())

        filtered_data = filters.process(dataset)

        expected_data = DocumentBatch(
            data=pd.DataFrame({"text": ["b", "c", "d"]}),
            task_id="batch_1_FakeQualityFilter",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    def test_fake_langid_filter(self) -> None:
        dataset = list_to_dataset(["a", "b", "c", "d"])
        filters = ScoreFilter(FakeLangId())

        filtered_data = filters.process(dataset)

        expected_data = DocumentBatch(
            data=pd.DataFrame({"text": ["a", "b", "d"]}),
            task_id="batch_1_FakeLangId",
            dataset_name="test_1",
        )
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"
