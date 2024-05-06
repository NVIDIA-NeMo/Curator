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

import os

import dask
import numpy as np
import pandas as pd
import pytest
from dask import dataframe as dd

from nemo_curator.datasets import DocumentDataset
from nemo_curator.filters import (
    AlphaFilter,
    BoilerPlateStringFilter,
    BulletsFilter,
    CommonEnglishWordsFilter,
    DocumentFilter,
    EllipsisFilter,
    GeneralCommentToCodeFilter,
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
    SymbolsToWordsFilter,
    TokenizerFertilityFilter,
    UrlsFilter,
    WhiteSpaceFilter,
    WordCountFilter,
    WordsWithoutAlphabetsFilter,
    XMLHeaderFilter,
)
from nemo_curator.modules import Filter, Score, ScoreFilter, Sequential
from nemo_curator.utils.decorators import batched


class LetterCountFilter(DocumentFilter):
    """
    Keeps documents that have at least some number of a given letter
    """

    def __init__(self, letter="a", min_count=5):
        super().__init__()
        self.letter = letter
        self.min_count = min_count

    def score_document(self, text):
        return text.count(self.letter)

    def keep_document(self, score):
        return score >= self.min_count


class BatchedLengthFilter(DocumentFilter):
    """
    Keeps documents of a given length
    """

    def __init__(self, min_length=5, max_length=10):
        super().__init__()
        self.min_length = min_length
        self.max_length = max_length

    @batched
    def score_document(self, df):
        return df.str.len()

    @batched
    def keep_document(self, scores):
        min_threshold = self.min_length <= scores
        max_threshold = scores <= self.max_length
        return min_threshold & max_threshold


def all_equal(left_dataset, right_dataset):
    return all(left_dataset.df.compute() == right_dataset.df.compute())


def list_to_dataset(documents, col_name="text", npartitions=2):
    data = {col_name: documents}
    pdf = pd.DataFrame(data)

    return DocumentDataset(dd.from_pandas(pdf, npartitions=npartitions))


@pytest.fixture
def letter_count_data():
    return list_to_dataset(
        ["Two aa", "a a Three a", "Five aaa aa", "aaaSeven aaaa"], col_name="documents"
    )


class TestFilterModule:
    def test_score_filter(self, letter_count_data):
        letter_filter = LetterCountFilter()
        filter_step = ScoreFilter(letter_filter, text_field="documents")
        filtered_data = filter_step(letter_count_data)

        expected_indices = [2, 3]
        expected_data = DocumentDataset(letter_count_data.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"

    def test_score(self, letter_count_data):
        letter_filter = LetterCountFilter()
        score_field = "a_count"
        score_step = Score(
            letter_filter.score_document,
            text_field="documents",
            score_field=score_field,
        )
        scored_data = score_step(letter_count_data)

        expected_scores = pd.Series([2, 3, 5, 7])
        scores = scored_data.df[score_field]
        assert all(
            expected_scores == scores.compute()
        ), f"Expected {expected_scores} but got {scores}"

    def test_retain_score_filter(self, letter_count_data):
        letter_filter = LetterCountFilter()
        score_field = "count_a"
        filter_step = ScoreFilter(
            letter_filter, text_field="documents", score_field=score_field
        )
        filtered_data = filter_step(letter_count_data)

        expected_indices = [2, 3]
        # Compute before loc due to https://github.com/dask/dask-expr/issues/1036
        expected_data = letter_count_data.df.compute().loc[expected_indices]
        expected_data = DocumentDataset(dd.from_pandas(expected_data, 2))
        expected_data.df[score_field] = pd.Series([5, 7], index=expected_data.df.index)
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"

    def test_filter(self, letter_count_data):
        letter_filter = LetterCountFilter()
        score_field = "a_count"
        score_step = Score(
            letter_filter.score_document,
            text_field="documents",
            score_field=score_field,
        )
        scored_data = score_step(letter_count_data)
        filter_step = Filter(letter_filter.keep_document, score_field)
        filtered_data = filter_step(scored_data)

        expected_indices = [2, 3]
        # Compute before loc due to https://github.com/dask/dask-expr/issues/1036
        expected_data = letter_count_data.df.compute().loc[expected_indices]
        expected_data = dd.from_pandas(expected_data, 2)
        expected_data[score_field] = pd.Series([5, 7], index=expected_data.index)
        expected_data = DocumentDataset(expected_data)
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"

    def test_invert(self, letter_count_data):
        letter_filter = LetterCountFilter()
        filter_step = ScoreFilter(letter_filter, text_field="documents", invert=True)
        filtered_data = filter_step(letter_count_data)

        expected_indices = [0, 1]
        expected_data = DocumentDataset(letter_count_data.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"

    def test_sequential_filter(self, letter_count_data):
        filters = Sequential(
            [
                ScoreFilter(LetterCountFilter(), text_field="documents"),
                ScoreFilter(LetterCountFilter(min_count=6), text_field="documents"),
            ]
        )
        filtered_data = filters(letter_count_data)

        expected_indices = [3]
        expected_data = DocumentDataset(letter_count_data.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"

    def test_batch_score_filter(self, letter_count_data):
        length_filter = BatchedLengthFilter(min_length=8, max_length=11)
        filter_step = ScoreFilter(length_filter, text_field="documents")
        filtered_data = filter_step(letter_count_data)

        expected_indices = [1, 2]
        expected_data = DocumentDataset(letter_count_data.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"

    def test_batch_score(self, letter_count_data):
        length_filter = BatchedLengthFilter(min_length=8, max_length=11)
        score_field = "lengths"
        score_step = Score(
            length_filter.score_document,
            text_field="documents",
            score_field=score_field,
        )
        scored_data = score_step(letter_count_data)

        expected_scores = pd.Series([6, 11, 11, 13])
        scores = scored_data.df[score_field]
        assert all(
            expected_scores == scores.compute()
        ), f"Expected {expected_scores} but got {scores}"

    def test_batch_filter(self, letter_count_data):
        length_filter = BatchedLengthFilter(min_length=8, max_length=11)
        score_field = "lengths"
        score_step = Score(
            length_filter.score_document,
            text_field="documents",
            score_field=score_field,
        )
        scored_data = score_step(letter_count_data)
        filter_step = Filter(length_filter.keep_document, score_field)
        filtered_data = filter_step(scored_data)

        expected_indices = [1, 2]
        expected_data = letter_count_data.df.loc[expected_indices]
        expected_data[score_field] = pd.Series([11, 11], index=expected_data.index)
        expected_data = DocumentDataset(expected_data)
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"

    def test_score_filter_type(self, letter_count_data):
        letter_filter = LetterCountFilter()
        filter_step = ScoreFilter(letter_filter, text_field="documents", score_type=int)
        filtered_data = filter_step(letter_count_data)

        expected_indices = [2, 3]
        expected_data = DocumentDataset(letter_count_data.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"

    def test_score_type(self, letter_count_data):
        letter_filter = LetterCountFilter()
        score_field = "a_count"
        score_step = Score(
            letter_filter.score_document,
            text_field="documents",
            score_field=score_field,
            score_type=int,
        )
        scored_data = score_step(letter_count_data)

        expected_scores = pd.Series([2, 3, 5, 7])
        scores = scored_data.df[score_field]
        assert all(
            expected_scores == scores.compute()
        ), f"Expected {expected_scores} but got {scores}"

    def test_chain_filter(self, letter_count_data):
        letter_count_filter = LetterCountFilter(min_count=4)
        length_filter = BatchedLengthFilter(min_length=8, max_length=11)
        filters = Sequential(
            [
                ScoreFilter(letter_count_filter, text_field="documents"),
                ScoreFilter(length_filter, text_field="documents"),
            ]
        )
        filtered_data = filters(letter_count_data)

        expected_indices = [2]
        expected_data = DocumentDataset(letter_count_data.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"


class TestHeuristicFilters:
    def test_nonalpha(self):
        dataset = list_to_dataset(
            ["", "This is a test case.", "%$^%$^%$&^$()))))", "$aaa"]
        )
        filters = ScoreFilter(NonAlphaNumericFilter())
        filtered_data = filters(dataset)

        expected_indices = [1, 3]
        expected_data = DocumentDataset(dataset.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"

    def test_symbolswords(self):
        dataset = list_to_dataset(
            [
                "mixed bag ... #",
                "full of words",
                "... # ... # #",
                "barely ok 3 4 5 6 7 8 9 #",
            ]
        )
        filters = ScoreFilter(SymbolsToWordsFilter())
        filtered_data = filters(dataset)

        expected_indices = [1, 3]
        expected_data = DocumentDataset(dataset.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"

    def test_numbers(self):
        dataset = list_to_dataset(
            ["purely letters", "34134543", "$!@$@!$!@", "abcdefghi1"]
        )
        filters = ScoreFilter(NumbersFilter(max_number_to_text_ratio=0.1))
        filtered_data = filters(dataset)

        expected_indices = [0, 2, 3]
        expected_data = DocumentDataset(dataset.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"

    def test_urls(self):
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
        filtered_data = filters(dataset)

        expected_indices = [1, 2, 3]
        expected_data = DocumentDataset(dataset.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"

    def test_bullets(self):
        dataset = list_to_dataset(
            [
                "• not good",
                "good",
                "50 \n ⦾ 50",
                "⁌ this \n⁌ should \n⁌barely \n⁌pass \n⁌5 \n⁌6 \n⁌7 \n⁌8 \n⁌9 \n done!",
            ]
        )
        filters = ScoreFilter(BulletsFilter())
        filtered_data = filters(dataset)

        expected_indices = [1, 2, 3]
        expected_data = DocumentDataset(dataset.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"

    def test_whitespace(self):
        dataset = list_to_dataset(["\t\n\r", "good", "50%\n\n\n", "123\b"])
        filters = ScoreFilter(WhiteSpaceFilter())
        filtered_data = filters(dataset)

        expected_indices = [1, 3]
        expected_data = DocumentDataset(dataset.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"

    def test_parentheses(self):
        dataset = list_to_dataset(
            ["()", "(not good)", "this is completely absolutely fine", "123456789("]
        )
        filters = ScoreFilter(ParenthesesFilter())
        filtered_data = filters(dataset)

        expected_indices = [2, 3]
        expected_data = DocumentDataset(dataset.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"

    def test_longword(self):
        dataset = list_to_dataset(["tiny", "large"])
        filters = ScoreFilter(LongWordFilter(max_word_length=4))
        filtered_data = filters(dataset)

        expected_indices = [0]
        expected_data = DocumentDataset(dataset.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"

    def test_wordcount(self):
        dataset = list_to_dataset(
            ["", "one", "two words", "$#@$ %$@$#@ !#@!", "one two three four five"]
        )
        filters = ScoreFilter(WordCountFilter(min_words=2, max_words=4))
        filtered_data = filters(dataset)

        expected_indices = [2, 3]
        expected_data = DocumentDataset(dataset.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"

    def test_boilerplate(self):
        dataset = list_to_dataset(
            [
                "nothing\t here",
                "1\n\n2\n\n3\n\n4\n\n5\n\n6\n\nterms of use\n\n privacy policy\n\n cookie policy\n\nuses cookies",
                "too much \n\n privacy & cookies policy",
            ]
        )
        filters = ScoreFilter(BoilerPlateStringFilter())
        filtered_data = filters(dataset)

        expected_indices = [0, 1]
        expected_data = DocumentDataset(dataset.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"

    def test_meanwordlength(self):
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
        filtered_data = filters(dataset)

        expected_indices = [2, 3]
        expected_data = DocumentDataset(dataset.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"

    def test_repeatedlines(self):
        dataset = list_to_dataset(["totally unique", "half.\nhalf."])
        filters = ScoreFilter(RepeatedLinesFilter())
        filtered_data = filters(dataset)

        expected_indices = [0]
        expected_data = DocumentDataset(dataset.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"

    def test_repeatedparagraphs(self):
        dataset = list_to_dataset(["totally unique", "half.\n\nhalf."])
        filters = ScoreFilter(RepeatedParagraphsFilter())
        filtered_data = filters(dataset)

        expected_indices = [0]
        expected_data = DocumentDataset(dataset.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"

    def test_repeatedlineschar(self):
        dataset = list_to_dataset(
            [
                "totally unique",
                "a.\na.\nvery very very short duplicate.",
                "half.\nhalf.",
                "super very incredibly huge long duplicate.\nsuper very incredibly huge long duplicate.\na.\nb.\nc.",
            ]
        )
        filters = ScoreFilter(RepeatedLinesByCharFilter())
        filtered_data = filters(dataset)

        expected_indices = [0, 1]
        expected_data = DocumentDataset(dataset.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"

    def test_repeatedparagraphschar(self):
        dataset = list_to_dataset(
            [
                "totally unique",
                "a.\n\n  a.\n\n  very very very short duplicate.",
                "half.\n\nhalf.",
                "super very incredibly huge long duplicate.\n\nsuper very incredibly huge long duplicate.\n\n  a.\n\n  b.\n\n  c.",
            ]
        )
        filters = ScoreFilter(RepeatedParagraphsByCharFilter())
        filtered_data = filters(dataset)

        expected_indices = [0, 1]
        expected_data = DocumentDataset(dataset.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"

    def test_repeatingtopngrams(self):
        dataset = list_to_dataset(
            [
                "this is a totally fine sentence with no repeat ngrams so we are ok",
                "a b . a b",
                "a a a a a a",
                "totally fine small dupe a b a b",
            ]
        )
        filters = ScoreFilter(RepeatingTopNGramsFilter())
        filtered_data = filters(dataset)

        expected_indices = [0, 3]
        expected_data = DocumentDataset(dataset.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"

    def test_repeatingduplicatengrams(self):
        dataset = list_to_dataset(
            ["a a b b a a b b", "totally fine", "a a a a this should be fine as well"]
        )
        filters = ScoreFilter(RepeatingDuplicateNGramsFilter())
        filtered_data = filters(dataset)

        expected_indices = [1, 2]
        expected_data = DocumentDataset(dataset.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"

    def test_punctuation(self):
        dataset = list_to_dataset(
            ["not good", "good.", "just\n barely\n fine\n ok\n yep."]
        )
        filters = ScoreFilter(
            PunctuationFilter(max_num_sentences_without_endmark_ratio=0.8)
        )
        filtered_data = filters(dataset)

        expected_indices = [1, 2]
        expected_data = DocumentDataset(dataset.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"

    def test_ellipsis(self):
        dataset = list_to_dataset(
            ["not good...", "good.", "just...\n barely...\n fine...\n ok...\n yep."]
        )
        filters = ScoreFilter(
            EllipsisFilter(max_num_lines_ending_with_ellipsis_ratio=0.8)
        )
        filtered_data = filters(dataset)

        expected_indices = [1, 2]
        expected_data = DocumentDataset(dataset.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"

    def test_commonenglishwords(self):
        dataset = list_to_dataset(["uncommon", "the and", "the and and of to"])
        filters = ScoreFilter(CommonEnglishWordsFilter())
        filtered_data = filters(dataset)

        expected_indices = [1, 2]
        expected_data = DocumentDataset(dataset.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"

    def test_wordswithoutalphabets(self):
        dataset = list_to_dataset(["totally fine", "good good good good !", "@"])
        filters = ScoreFilter(WordsWithoutAlphabetsFilter())
        filtered_data = filters(dataset)

        expected_indices = [0, 1]
        expected_data = DocumentDataset(dataset.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"

    def test_pornographicurls(self):
        dataset = list_to_dataset(
            [
                "no url",
                "fine url https://www.nvidia.com/en-us/",
                "bad url https://www.pornhub.com/",
            ]
        )
        filters = ScoreFilter(PornographicUrlsFilter())
        filtered_data = filters(dataset)

        expected_indices = [0, 1]
        expected_data = DocumentDataset(dataset.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"


class TestCodeFilters:
    def test_python_comment_to_code(self):
        doc_1 = "# Good code\nprint('hello world')"
        doc_2 = "print('bad code')"
        doc_3 = "# Too many\n# comments!"
        doc_4 = "'''Good comment'''\nprint('hello world')"
        dataset = list_to_dataset([doc_1, doc_2, doc_3, doc_4])
        filters = ScoreFilter(PythonCommentToCodeFilter())
        filtered_data = filters(dataset)

        expected_indices = [0, 3]
        expected_data = DocumentDataset(dataset.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"

    def test_general_commment_to_code(self):
        doc_1 = '// Good code\nprintf("hello world\\n")'
        doc_2 = 'printf("bad code\\n")'
        doc_3 = "// Way far too many\n// comments!"
        doc_4 = '/*\nGood comment\n*/\nprintf("hello world\\n")'
        dataset = list_to_dataset([doc_1, doc_2, doc_3, doc_4])
        filters = ScoreFilter(GeneralCommentToCodeFilter("text/x-c++"))
        filtered_data = filters(dataset)

        expected_indices = [0, 3]
        expected_data = DocumentDataset(dataset.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"

    def test_number_lines_code(self):
        doc_1 = """print("too short")"""
        doc_2 = """print("just")
        print("right")"""
        doc_3 = """print("way")
        print("too")
        print("long")
        print("!")"""
        dataset = list_to_dataset([doc_1, doc_2, doc_3])
        filters = ScoreFilter(NumberOfLinesOfCodeFilter(min_lines=2, max_lines=3))
        filtered_data = filters(dataset)

        expected_indices = [1]
        expected_data = DocumentDataset(dataset.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"

    def test_xml_header(self):
        dataset = list_to_dataset(
            ["no header", "<?xml version=1.0>", "slightly offset <?xml version="]
        )
        filters = ScoreFilter(XMLHeaderFilter())
        filtered_data = filters(dataset)

        expected_indices = [0]
        expected_data = DocumentDataset(dataset.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"

    def test_alpha(self):
        dataset = list_to_dataset(["full of alphabet", "<>?$#@!", "mixed <>"])
        filters = ScoreFilter(AlphaFilter())
        filtered_data = filters(dataset)

        expected_indices = [0, 2]
        expected_data = DocumentDataset(dataset.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"

    def test_html_boilerplate(self):
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
        filtered_data = filters(dataset)

        expected_indices = [0]
        expected_data = DocumentDataset(dataset.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"

    def test_per_extension_filter(self):
        good_cpp = """
        #include <iostream>

        using namespace std;

        int main() {
            cout << "Hello World!" << endl;
            return 0;
        };
        """
        dataset = list_to_dataset([good_cpp])
        metadata_file = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "nemo_curator",
                "utils",
                "code_meta.csv",
            )
        )
        filters = ScoreFilter(
            PerExtensionFilter("c++", "cpp", metadata_file=metadata_file)
        )
        filtered_data = filters(dataset)

        expected_indices = [0]
        expected_data = DocumentDataset(dataset.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"


class FakeQualityFilter(DocumentFilter):
    """
    Emulates FastTextQualityFilter without a model
    """

    def __init__(self, alpha=3, seed=42):
        super().__init__()
        self._alpha = alpha
        self._seed = np.random.seed(seed)

    @batched
    def score_document(self, df):
        return pd.Series(np.arange(len(df)) / len(df))

    @batched
    def keep_document(self, df):
        return np.random.pareto(self._alpha, size=len(df)) > 1 - df


class FakeLangId(DocumentFilter):
    """
    Emulates FastTextLangId without a model
    """

    def __init__(self, min_langid_score=0.3, convert_string=False):
        super().__init__()
        self._cutoff = min_langid_score

        # Dask will automatically convert the list score type
        # to a string without this option.
        # See https://github.com/NVIDIA/NeMo-Curator/issues/33
        dask.config.set({"dataframe.convert-string": convert_string})

    @batched
    def score_document(self, df):
        scores = [[0.5, "EN"], [0.7, "HI"], [0.2, "PT"]]
        scores = scores * len(df)
        scores = scores[: len(df)]
        return pd.Series(scores)

    def keep_document(self, score):
        return score[0] >= self._cutoff


class TestClassifierFilters:
    def test_fake_quality_filter(self):
        dataset = list_to_dataset(["a", "b", "c", "d"], npartitions=1)
        filters = ScoreFilter(FakeQualityFilter())
        filtered_data = filters(dataset)

        expected_indices = [1, 2, 3]
        expected_data = DocumentDataset(dataset.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"

    def test_fake_langid_filter(self):
        dataset = list_to_dataset(["a", "b", "c", "d"], npartitions=1)
        filters = ScoreFilter(FakeLangId())
        filtered_data = filters(dataset)

        expected_indices = [0, 1, 3]
        expected_data = DocumentDataset(dataset.df.loc[expected_indices])
        assert all_equal(
            expected_data, filtered_data
        ), f"Expected {expected_data} but got {filtered_data}"
