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

from dask.array import logical_and

from nemo_curator.datasets.parallel_dataset import ParallelDataset
from nemo_curator.filters import DocumentFilter
from nemo_curator.modules.base import BaseModule


class ParallelScoreFilter(BaseModule):
    def __init__(  # noqa: PLR0913
        self,
        src_filter_obj: DocumentFilter,
        tgt_filter_obj: DocumentFilter,
        src_field: str = "src",
        tgt_field: str = "tgt",
        src_score: str | None = None,
        tgt_score: str | None = None,
        score_type: str | None = None,
        invert: bool = False,
    ):
        """A filter object wrapper class for applying *monolingual* filter objects on bitext.
        If either side of the bitext is discarded, the whole bitext pair is discarded.
        If you want to apply a *bitext* filter that takes both the source and target as input, checkout `BitextFilter` class.

        Note that the goal of this wrapper class is to group the same/similar filters on bitext thus making the logic clearer,
        which is why we force the `score_type` and `invert` to be the same among source/target filters.
        If you need the extra flexibility, you should fall back to applying two filters one after the other.

        Args:
            src_filter_obj (_type_): The score function that takes in a document string and outputs a score for the source document.
            tgt_filter_obj (_type_): The score function that takes in a document string and outputs a score for the target document.
            src_field (str, optional): The field the source documents will be read from. Defaults to "src".
            tgt_field (str, optional): The field the target documents will be read from. Defaults to "tgt".
            src_score (str, optional): The field to which the source scores will be written. If None, scores will be immediately discarded after use. Defaults to None.
            tgt_score (str, optional): The field to which the target scores will be written. If None, scores will be immediately discarded after use. Defaults to None.
            score_type (Optional[str]): The datatype of the score that will be made for each document. Defaults to None.
            invert (bool, optional): If True, will keep all documents that are normally discarded. Defaults to False.
        """
        super().__init__(input_backend=src_filter_obj.backend)
        self.source_score_filter = ScoreFilter(src_filter_obj, src_field, src_score, score_type, invert)  # noqa: F821
        self.target_score_filter = ScoreFilter(tgt_filter_obj, tgt_field, tgt_score, score_type, invert)  # noqa: F821

    def call(self, dataset: ParallelDataset) -> ParallelDataset:
        src_bool_mask = self.source_score_filter.compute_filter_mask(dataset)
        tgt_bool_mask = self.target_score_filter.compute_filter_mask(dataset)

        # remove lines together if one of them is filtered
        bool_mask = logical_and(src_bool_mask, tgt_bool_mask)

        return ParallelDataset(dataset.df[bool_mask])
