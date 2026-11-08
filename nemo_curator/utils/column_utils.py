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

import pandas as pd
import pyarrow as pa


def pyarrow_string_to_pandas_dtype(arrow_type: pa.DataType, *, na_value: object = pd.NA) -> pd.StringDtype | None:
    """Map Arrow strings to a pandas dtype with functional string operations."""
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        try:
            return pd.StringDtype(storage="pyarrow", na_value=na_value)
        except TypeError:
            # pandas 2.1 does not accept na_value or Arrow large_string arrays.
            storage = "python" if pa.types.is_large_string(arrow_type) else "pyarrow"
            return pd.StringDtype(storage=storage)
    return None


def pyarrow_to_pandas_dtype(arrow_type: pa.DataType) -> pd.StringDtype | pd.ArrowDtype:
    """Map Arrow types to pandas dtypes with functional string operations."""
    string_dtype = pyarrow_string_to_pandas_dtype(arrow_type)
    return string_dtype if string_dtype is not None else pd.ArrowDtype(arrow_type)


def resolve_filename_column(add_filename_column: bool | str) -> str | None:
    """Resolve the filename column name based on the input parameter.

    Args:
        add_filename_column: Can be:
            - True: Use default filename column name
            - False: No filename column
            - str: Use the provided string as column name

    Returns:
        str | None: The filename column name or None if not needed
    """
    if add_filename_column is True:
        return "file_name"
    elif add_filename_column is False:
        return None
    elif isinstance(add_filename_column, str):
        return add_filename_column
    else:
        msg = f"Invalid value for add_filename_column: {add_filename_column}"
        raise ValueError(msg)
