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
from packaging.version import parse as parseVersion

_dask_version = parseVersion(dask.__version__)

# TODO: remove when dask min version gets bumped
DASK_SHUFFLE_METHOD_ARG = _dask_version > parseVersion("2024.1.0")
DASK_P2P_ERROR = _dask_version < parseVersion("2023.10.0")
DASK_SHUFFLE_CAST_DTYPE = _dask_version > parseVersion("2023.12.0")
