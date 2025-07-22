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

import pandas as pd
import pytest
from dask import dataframe as dd

from nemo_curator.datasets import DocumentDataset
from nemo_curator.datasets.parallel_dataset import ParallelDataset
from nemo_curator.filters import DocumentFilter, LengthRatioFilter
from nemo_curator.filters.models.qe_models import COMET_IMPORT_MSG, PYMARIAN_IMPORT_MSG
from nemo_curator.modules import ParallelScoreFilter
from nemo_curator.utils.import_utils import is_unavailable, safe_import

comet = safe_import("comet", msg=COMET_IMPORT_MSG)
pymarian = safe_import("pymarian", msg=PYMARIAN_IMPORT_MSG)


class LetterCountFilter(DocumentFilter):
    """
    Keeps documents that have at least some number of a given letter
    """

    def __init__(self, letter: str = "a", min_count: int = 5) -> None:
        super().__init__()
        self.letter = letter
        self.min_count = min_count

    def score_document(self, text: str) -> int:
        return text.count(self.letter)

    def keep_document(self, score: int) -> bool:
        return score >= self.min_count


def all_equal(left_dataset: DocumentDataset, right_dataset: DocumentDataset) -> bool:
    return all(left_dataset.df.compute() == right_dataset.df.compute())


def list_to_dataset(documents: list[str], col_name: str = "text", npartitions: int = 2) -> DocumentDataset:
    data = {col_name: documents}
    pdf = pd.DataFrame(data)

    return DocumentDataset(dd.from_pandas(pdf, npartitions=npartitions))


def two_lists_to_parallel_dataset(  # noqa: PLR0913
    src_documents: list[str],
    tgt_documents: list[str],
    src_lang: str,
    tgt_lang: str,
    src_col_name: str = "src",
    tgt_col_name: str = "tgt",
    npartitions: int = 2,
) -> ParallelDataset:
    src_langs = [src_lang] * len(src_documents)
    tgt_langs = [tgt_lang] * len(src_documents)
    data = {
        src_col_name: src_documents,
        "src_lang": src_langs,
        tgt_col_name: tgt_documents,
        "tgt_lang": tgt_langs,
    }
    pdf = pd.DataFrame(data)

    return ParallelDataset(dd.from_pandas(pdf, npartitions=npartitions))


@pytest.fixture
def letter_count_data() -> DocumentDataset:
    return list_to_dataset(["Two aa", "a a Three a", "Five aaa aa", "aaaSeven aaaa"], col_name="documents")


@pytest.fixture
def parallel_letter_count_data() -> ParallelDataset:
    return two_lists_to_parallel_dataset(
        ["Einsa", "Zwei aaa", "a Drei a", "Fünf aaa a", "aaaSieben aaaa"],
        ["aOne", "Two aa", "a a Three a", "Five aaa aa", "aaaSeven aaaa"],
        src_lang="de",
        tgt_lang="en",
        src_col_name="src",
        tgt_col_name="tgt",
    )


@pytest.fixture
def length_ratio_data() -> ParallelDataset:
    return two_lists_to_parallel_dataset(
        ["Test", "test", "Test Test ", "Test Test"],
        ["Prueba", "prueba prueba prueba", "Prueba Prueba", "Prueba Prueba Prueba "],
        src_lang="en",
        tgt_lang="es",
    )


class TestFilterModule:
    def test_parallel_score_filter(self, parallel_letter_count_data: ParallelDataset) -> None:
        src_letter_count_filter = LetterCountFilter(min_count=2)
        tgt_letter_count_filter = LetterCountFilter(min_count=3)
        filter_step = ParallelScoreFilter(src_letter_count_filter, tgt_letter_count_filter)
        filtered_data = filter_step(parallel_letter_count_data)

        expected_indices = [2, 3, 4]
        expected_data = ParallelDataset(parallel_letter_count_data.df.loc[expected_indices])
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"

    def test_joint_score_filter(self, length_ratio_data: ParallelDataset) -> None:
        filter_ = LengthRatioFilter(
            max_ratio=1.5,
            src_lang="en",
            tgt_lang="de",
            score_field="ratio",
            score_type=float,
        )
        filtered_data = filter_(length_ratio_data)

        expected_indices = [0, 2]
        expected_data = ParallelDataset(length_ratio_data.df.loc[expected_indices])
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"


class TestClassifierFilters:
    @pytest.mark.skipif(is_unavailable(comet), reason="Test depends on COMET but it's not installed.")
    def test_comet_qe_filter(self) -> None:
        dataset = two_lists_to_parallel_dataset(
            [
                "This sentence will be translated on the Chinese side.",
                "This sentence will have something irrelevant on the Chinese side.",
            ],
            [
                "这句话在中文一侧会被翻译。",
                "至尊戒，驭众戒；至尊戒，寻众戒；魔戒至尊引众戒，禁锢众戒黑暗中。",  # noqa: RUF001
            ],
            "en",
            "zh",
        )

        from nemo_curator.filters import QualityEstimationFilter
        from nemo_curator.utils.distributed_utils import get_client

        client = get_client(n_workers=1)
        filter_ = QualityEstimationFilter(
            "comet-qe",
            cutoff=-0.25,
            mode="bidi",
            score_type=float,
            metadata_fields=["src_lang", "tgt_lang"],
        )
        filtered_data = filter_(dataset)

        expected_indices = [0]
        expected_data = ParallelDataset(dataset.df.loc[expected_indices])
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"
        client.close()

    @pytest.mark.skipif(
        is_unavailable(pymarian),
        reason="Test depends on PyMarian but it's not installed.",
    )
    def test_cometoid_qe_filter(self) -> None:
        dataset = two_lists_to_parallel_dataset(
            [
                "This sentence will be translated on the Chinese side.",
                "This sentence will have something irrelevant on the Chinese side.",
            ],
            [
                "这句话在中文一侧会被翻译。",
                "至尊戒，驭众戒；至尊戒，寻众戒；魔戒至尊引众戒，禁锢众戒黑暗中。",  # noqa: RUF001
            ],
            "en",
            "zh",
        )

        from nemo_curator.filters import QualityEstimationFilter
        from nemo_curator.utils.distributed_utils import get_client

        client = get_client(n_workers=1)
        filter_ = QualityEstimationFilter(
            "cometoid-wmt23",
            cutoff=0.75,
            mode="bidi",
            score_type=float,
            metadata_fields=["src_lang", "tgt_lang"],
        )  # enable GPU by gpu=True
        filtered_data = filter_(dataset)

        expected_indices = [0]
        expected_data = ParallelDataset(dataset.df.loc[expected_indices])
        assert all_equal(expected_data, filtered_data), f"Expected {expected_data} but got {filtered_data}"
        client.close()
