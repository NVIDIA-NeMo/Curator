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

import abc


class ModelInterface(abc.ABC):
    """Abstract base class that defines an interface for machine learning models.

    Specifically focused on their weight handling and environmental setup.

    This interface allows our pipeline code to download weights locally and setup models in a uniform
    way. It does not place any restrictions on how inference is run.
    """

    @property
    @abc.abstractmethod
    def model_id_names(self) -> list[str]:
        """Returns a list of model IDs associated with the model.

        In cosmos-curate, each model has an ID associated with it.
        This is often the huggingspace name for that model (e.g. Salesforce/instructblip-vicuna-13b).

        Returns:
            A list of strings.

        """

    @abc.abstractmethod
    def setup(self) -> None:
        """Set up the model for use, such as loading weights and building computation graphs."""
