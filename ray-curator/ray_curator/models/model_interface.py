# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Model interface to help run one or more models inside a pipeline stage."""

import abc


class ModelInterface(abc.ABC):
    """Abstract base class that defines an interface for machine learning models.

    Specifically focused on their weight handling and environmental setup.

    This interface allows our pipeline code to download weights locally and setup models in a uniform
    way. It does not place any restrictions on how inference is run.
    """

    @property
    def weights_names(self) -> list[str]:
        """Returns a list of weight names associated with the model.

        In cosmos-curator, each set of weights has a name associated with it.
        This is often the huggingspace name for those weights (e.g. Salesforce/instructblip-vicuna-13b).
        but doesn't need to be. We use these names to push/pull weights to/from S3.

        Returns:
            A list of strings.

        """
        if self.id_file_mapping() is None:
            return []
        return list(self.id_file_mapping().keys())

    @abc.abstractmethod
    def id_file_mapping(self) -> dict[str, list[str] | None]:
        """Return a mapping of model id to a list of file names associated with the model ids.

        In cosmos-curator, each set of weights has a name associated with it.
        This is often the huggingspace name for those weights (e.g. Salesforce/instructblip-vicuna-13b).
        but doesn't need to be. We use these names to push/pull weights to/from S3. Each model id must be unique.

        Returns:
            A dictionary of model id to a list of file names.
            If the list of files is None, that means all files should be downloaded.

        """

    @property
    @abc.abstractmethod
    def conda_env_name(self) -> str:
        """Returns the name of the conda environment that this model must be run from.

        Returns:
            A string representing the conda environment name.

        """

    @abc.abstractmethod
    def setup(self) -> None:
        """Set up the model for use, such as loading weights and building computation graphs."""
