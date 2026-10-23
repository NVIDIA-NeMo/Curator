# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import cudf


def normalize_text(
    text_series: cudf.Series,
    *,
    lowercase: bool = True,
    normalize_space: bool = True,
) -> cudf.Series:
    """Normalize text in a cuDF Series.

    Parameters
    ----------
    text_series : cudf.Series
        Series containing the text to normalize.
    lowercase : bool, default=True
        Whether to lowercase the text.
    normalize_space : bool, default=True
        Whether to collapse whitespace runs and trim leading and trailing whitespace.

    Returns
    -------
    cudf.Series
        The normalized text series.

    Raises
    ------
    TypeError
        If ``text_series`` is not a cuDF Series.
    """
    if not isinstance(text_series, cudf.Series):
        msg = "Expected text_series of type cudf.Series"
        raise TypeError(msg)

    normalized_text = text_series
    if lowercase:
        normalized_text = normalized_text.str.lower()
    if normalize_space:
        normalized_text = normalized_text.str.normalize_spaces()

    return normalized_text
