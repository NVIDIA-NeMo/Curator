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

from abc import ABC, abstractmethod
from typing import Literal, Optional

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
