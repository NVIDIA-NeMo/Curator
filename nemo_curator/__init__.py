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

import dask

from .modules import *
from .package_info import (
    __contact_emails__,
    __contact_names__,
    __description__,
    __download_url__,
    __homepage__,
    __keywords__,
    __license__,
    __package_name__,
    __repository_url__,
    __shortversion__,
    __version__,
)
from .services import (
    AsyncLLMClient,
    AsyncOpenAIClient,
    LLMClient,
    NemoDeployClient,
    OpenAIClient,
)
from .utils.distributed_utils import get_client, get_network_interfaces

# Dask will automatically convert the list score type
# to a string without this option.
# See https://github.com/NVIDIA/NeMo-Curator/issues/33
# This also happens when reading and writing to files
dask.config.set({"dataframe.convert-string": False})

__all__ = [
    "__contact_emails__",
    "__contact_names__",
    "__description__",
    "__download_url__",
    "__homepage__",
    "__keywords__",
    "__license__",
    "__package_name__",
    "__repository_url__",
    "__shortversion__",
    "__version__",
]
