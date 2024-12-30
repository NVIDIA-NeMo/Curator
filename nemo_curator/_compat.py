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
from packaging.version import parse as parse_version

try:
    _dask_version = parse_version(dask.__version__)
except TypeError:
    # When mocking with autodoc the dask version is not there
    _dask_version = parse_version("2024.06.0")


try:
    import dask_cudf

    _dask_cudf_version = parse_version(dask_cudf.__version__)
except (ImportError, TypeError):
    # When mocking with autodoc the dask version is not there
    _dask_cudf_version = parse_version("2024.06.0")

try:
    import cudf

    CURRENT_CUDF_VERSION = parse_version(cudf.__version__)
except (ImportError, TypeError):
    CURRENT_CUDF_VERSION = parse_version("24.10.0")

# TODO remove this once 25.02 becomes the base version of cudf in nemo-curator

# minhash in < 24.12 used to have a minhash(txt) api which was deprecated in favor of
# minhash(a, b) in 25.02 (in 24.12, minhash_permuted(a,b) was introduced)
MINHASH_DEPRECATED_API = (
    CURRENT_CUDF_VERSION.base_version < parse_version("24.12").base_version
)
MINHASH_PERMUTED_AVAILABLE = (CURRENT_CUDF_VERSION.major == 24) & (
    CURRENT_CUDF_VERSION.minor == 12
)

# TODO: remove when dask min version gets bumped
DASK_SHUFFLE_METHOD_ARG = _dask_version > parse_version("2024.1.0")
DASK_P2P_ERROR = _dask_version < parse_version("2023.10.0")
DASK_SHUFFLE_CAST_DTYPE = _dask_version > parse_version("2023.12.0")
DASK_CUDF_PARQUET_READ_INCONSISTENT_SCHEMA = _dask_version > parse_version("2024.12")

# Query-planning check (and cache)
_DASK_QUERY_PLANNING_ENABLED = None


def query_planning_enabled():
    global _DASK_QUERY_PLANNING_ENABLED

    if _DASK_QUERY_PLANNING_ENABLED is None:
        if _dask_version > parse_version("2024.6.0"):
            import dask.dataframe as dd

            _DASK_QUERY_PLANNING_ENABLED = dd.DASK_EXPR_ENABLED
        else:
            _DASK_QUERY_PLANNING_ENABLED = "dask_expr" in sys.modules
    return _DASK_QUERY_PLANNING_ENABLED
