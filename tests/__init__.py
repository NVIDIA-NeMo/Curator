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

# Disable query planning before any tests are loaded
# https://github.com/NVIDIA/NeMo-Curator/issues/73
if dask.config.get("dataframe.query-planning") is True:
    raise NotImplementedError(
        "NeMo Curator does not support query planning yet. "
        "Please disable query planning before importing "
        "`nemo_curator`, `dask.dataframe` or `dask_cudf`."
    )
else:
    dask.config.set({"dataframe.query-planning": False})
