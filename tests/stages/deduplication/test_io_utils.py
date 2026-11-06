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

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.mark.parametrize("filesystem_kind", ["none", "fsspec", "pyarrow"])
@pytest.mark.gpu
def test_read_parquet_uses_pyarrow_for_page_indexed_delta_byte_array(tmp_path: Path, filesystem_kind: str) -> None:
    pytest.importorskip("cudf")

    import fsspec
    import pyarrow.fs as pafs

    from nemo_curator.stages.deduplication.io_utils import DeduplicationIO

    path = tmp_path / "indexed.parquet"
    pq.write_table(
        pa.table({"text": ["hello", "parquet"]}),
        path,
        use_dictionary=False,
        column_encoding="DELTA_BYTE_ARRAY",
        write_page_index=True,
    )

    if filesystem_kind == "fsspec":
        filesystem = fsspec.filesystem("file")
    elif filesystem_kind == "pyarrow":
        filesystem = pafs.LocalFileSystem()
    else:
        filesystem = None
    read_kwargs = {"filesystem": filesystem} if filesystem is not None else {}
    result = DeduplicationIO(id_generator=None).read_parquet(str(path), columns=["text"], **read_kwargs)

    assert result["text"].to_arrow().to_pylist() == ["hello", "parquet"]
