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

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

from dask.typing import no_default

from nemo_curator.datasets.parallel_dataset import ParallelDataset
from nemo_curator.utils.module_utils import is_batched


class BitextFilter(ABC):
    """A base class for bitext filter objects (such as length ratio, QE filter) on bitext.
    Different from `ParallelScoreFilter`, these filters require looking at both source AND target side of the bitext to compute a score.

    This is roughly equivalent to a `ScoreFilter` wrapping over a `DocumentFilter` object.
    But aside from operating on `ParallelDataset` instead of `DocumentDataset`, it comes with some other differences:

    - It discarded the ScoreFilter/DocumentFilter hierarchy. So filter classes can directly be used instead of being wrapped by ScoreFilter.
    - Unlike an DocumentFilter object, it allows passing extra metadata information into the scoring function.
    """

    def __init__(
        self,
        src_field: str = "src",
        tgt_field: str = "tgt",
        metadata_fields: Union[List[str], str] = [],
        metadata_field_name_mapping: Dict[str, str] = {},
        score_field: Optional[str] = None,
        score_type: Union[type, str] = None,
        invert=False,
    ):
        """Args:
            src_field (str, optional): The field the source documents will be read from. Defaults to "src".
            tgt_field (str, optional): The field the target documents will be read from. Defaults to "tgt".
            metadata_fields (Union[List[str], str], optional): Name of the metadata fields in case fields other than source and target documents need to be accessed. Defaults to [].
            metadata_field_name_mapping (Dict[str, str], optional): Mapping of field names in the data to argument names in `_score_bitext` function, in case they are different.
                For example, if a field is called "src" in the data but should be passed to an argument called "source" in `_score_bitext` function,
                you should add an entry `{"src": "source"}`. Identity map is assumed if a mapping is not specified for a field name. Default to {}.
            score_field (Optional[str], optional): The field to which the scores will be written. If None, scores will be immediately discarded after use. Defaults to None.
            score_type (Union[type, str], optional): The datatype of the score that will be made for each document. Defaults to None.
            invert (bool, optional): If True, will keep all documents that are normally discarded. Defaults to False.

        Raises:
            ValueError: If length of source and target fields are different.
        """

        self.src_field = src_field
        self.tgt_field = tgt_field
        self.metadata_fields = metadata_fields
        self.metadata_field_name_mapping = metadata_field_name_mapping
        self.score_field = score_field
        self.score_type = score_type
        self.invert = invert

    def __call__(
        self,
        dataset: ParallelDataset,
    ) -> ParallelDataset:
        """Scores and filters all records in the dataset

        Args:
            dataset (ParallelDataset): The dataset to apply the module to.

        Returns:
            ParallelDataset:  A dataset with the score and filter applied
        """
        # Set the metadata for the function calls if provided
        if self.score_type:
            meta = (None, self.score_type)
        else:
            meta = no_default

        # support multiple fields if supplied
        fields = []
        fields.append(self.src_field)
        fields.append(self.tgt_field)
        fields.extend(self.metadata_fields)

        if is_batched(self.score_bitext):
            scores = dataset.df[fields].map_partitions(
                self._score_bitext_wrapper,
                metadata_field_name_mapping=self.metadata_field_name_mapping,
                meta=meta,
            )
        else:
            scores = dataset.df[fields].apply(
                self._score_bitext_wrapper,
                metadata_field_name_mapping=self.metadata_field_name_mapping,
                axis=1,
                meta=meta,
            )

        if self.score_field is not None:
            dataset.df[self.score_field] = scores

        if is_batched(self.keep_bitext):
            bool_mask = scores.map_partitions(self.keep_bitext, meta=(None, bool))
        else:
            bool_mask = scores.apply(self.keep_bitext, meta=(None, bool))
        if self.invert:
            bool_mask = ~bool_mask

        return ParallelDataset(dataset.df[bool_mask])

    def _score_bitext_wrapper(
        self,
        df,
        metadata_field_name_mapping: Dict[str, str] = {},
    ):
        """In the batch mode, pass fields in a data frame to arguments of a function, according to a name mapping.

        Args:
            df (DataFrame): data frame to perform the mapping on.
            metadata_field_name_mapping (Dict[str, str]): see `__call__` function for details.
        """
        kwargs = {}
        kwargs["src"] = df[self.src_field]
        kwargs["tgt"] = df[self.tgt_field]
        for field_name in self.metadata_fields:
            arg_name = metadata_field_name_mapping.get(field_name, field_name)
            kwargs[arg_name] = df[field_name]

        return self.score_bitext(**kwargs)

    @abstractmethod
    def score_bitext(self, src, tgt, **kwargs):
        """Scoring function for the bitext."""
        pass

    @abstractmethod
    def keep_bitext(self, **kwargs):
        pass
