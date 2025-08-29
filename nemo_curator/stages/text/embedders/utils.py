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

from typing import Any

import cudf
import cupy as cp
import pylibcudf as plc


def create_list_series_from_1d_or_2d_ar(ar: Any, index: cudf.Index) -> cudf.Series:  # noqa: ANN401
    """Create a cudf list series from 2d arrays.
    This code comes from https://github.com/rapidsai/crossfit/blob/76f74d0d927cf76313a3960d7dd5575d1dff2f06/crossfit/backend/cudf/series.py#L20-L32

    Args:
        ar (cp.ndarray): any object that can be converted to a cupy array (cupy, numpy, torch, etc.)
        index (cudf.Index): index of the the dataframe to be returned

    Returns:
        cudf.Series: cudf series with the index respected
    """
    arr = cp.asarray(ar)
    if len(arr.shape) == 1:
        arr = arr.reshape(-1, 1)
    if not isinstance(index, (cudf.RangeIndex, cudf.Index, cudf.MultiIndex)):
        index = cudf.Index(index)
    return cudf.Series.from_pylibcudf(
        plc.Column.from_cuda_array_interface(arr),
        metadata={"index": index},
    )
