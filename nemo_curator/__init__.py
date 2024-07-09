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

import sys

import dask

# Disable query planning if possible
# https://github.com/NVIDIA/NeMo-Curator/issues/73
if dask.config.get("dataframe.query-planning") is True or "dask_expr" in sys.modules:
    raise NotImplementedError(
        """
        NeMo Curator does not support query planning yet.
        Please disable query planning before importing
        `dask.dataframe` or `dask_cudf`. This can be done via:
        `export DASK_DATAFRAME__QUERY_PLANNING=False`, or
        importing `dask.dataframe/dask_cudf` after importing
        `nemo_curator`.
        """
    )
else:
    dask.config.set({"dataframe.query-planning": False})


from .modules import *
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
