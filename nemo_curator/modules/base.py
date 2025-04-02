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

import logging
import warnings
from abc import ABC, abstractmethod
from typing import Literal, Optional, Union

import dask.dataframe as dd

from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.gpu_utils import is_cudf_type


class BaseModule(ABC):
    """
    Base class for all NeMo Curator modules.

    Handles validating that data lives on the correct device for each module
    """

    SUPPORTED_BACKENDS = ["pandas", "cudf", "any"]

    def __init__(
        self,
        input_backend: Literal["pandas", "cudf", "any"],
        name: Optional[str] = None,
    ) -> None:
        """
        Constructs a Module

        Args:
            input_backend (Literal["pandas", "cudf", "any"]): The backend the input dataframe must be on for the module to work
            name (str, Optional): The name of the module. If None, defaults to self.__class__.__name__
        """
        super().__init__()
        self.name = name or self.__class__.__name__

        if input_backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"{input_backend} not one of the supported backends {self.SUPPORTED_BACKENDS}"
            )
        self.input_backend = input_backend

    @abstractmethod
    def call(self, dataset: DocumentDataset):
        """
        Performs an arbitrary operation on a dataset

        Args:
            dataset (DocumentDataset): The dataset to operate on
        """
        raise NotImplementedError("call method must be implemented by subclasses")

    def _validate_correct_backend(self, ddf: dd.DataFrame):
        if self.input_backend == "any":
            return

        backend = "cudf" if is_cudf_type(ddf) else "pandas"
        if backend != self.input_backend:
            raise ValueError(
                f"Module {self.name} requires dataset to have backend {self.input_backend} but got backend {backend}."
                "Try using nemo_curator.ToBackend to swap dataframe backends before running this module."
            )

    def __call__(self, dataset: DocumentDataset):
        """
        Validates the dataset is on the right backend, and performs an arbitrary operation on it

        Args:
            dataset (DocumentDataset): The dataset to operate on
        """
        self._validate_correct_backend(dataset.df)

        return self.call(dataset)


class BaseDeduplicationModule(BaseModule):
    """
    Base class for all NeMo Curator deduplication modules.
    """

    def __init__(
        self,
        id_field: str,
        text_field: str,
        perform_removal: bool = False,
        logger: Union[logging.LoggerAdapter, str] = "./",
        profile_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        input_backend: Literal["pandas", "cudf", "any"] = "any",
        **kwargs,
    ):
        super().__init__(input_backend=input_backend, **kwargs)
        self.id_field = id_field
        self.text_field = text_field
        self.perform_removal = perform_removal
        self.logger = logger
        self.profile_dir = profile_dir
        self.cache_dir = cache_dir

        if self.perform_removal and cache_dir is None:
            warnings.warn("cache_dir is recommended to remove duplicates.")

        if cache_dir is None and profile_dir is not None:
            warnings.warn(
                "cache_dir for intermediate outputs is required to generate profiles"
            )

        if not self.perform_removal:
            warnings.warn(
                "In future NeMo Curator releases, the default value for perform_removal will be True."
            )

    @abstractmethod
    def identify_duplicates(self, dataset: DocumentDataset) -> DocumentDataset:
        """
        Identifies duplicates in a dataset

        Args:
            dataset (DocumentDataset): The dataset to identify duplicates in
        """
        raise NotImplementedError(
            "identify_duplicates method must be implemented by subclasses"
        )

    @abstractmethod
    def remove(
        self, dataset: DocumentDataset, duplicates_to_remove: DocumentDataset
    ) -> DocumentDataset:
        """
        Removes duplicates from a dataset

        Args:
            dataset (DocumentDataset): The dataset to remove duplicates from
        """
        raise NotImplementedError("remove method must be implemented by subclasses")

    def call(self, dataset: DocumentDataset) -> DocumentDataset:
        """
        Execute the deduplication process.

        Args:
            dataset (DocumentDataset): Input dataset for deduplication.
        Returns:
            DocumentDataset: Deduplicated dataset if perform_removal is False, otherwise the dataset with duplicates removed.
        """
        duplicates = self.identify_duplicates(dataset)

        if self.perform_removal:
            return self.remove(dataset, duplicates)

        return duplicates
