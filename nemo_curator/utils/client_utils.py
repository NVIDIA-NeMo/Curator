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

import fsspec
from fsspec.core import url_to_fs


class FSPath:
    """Wrapper that combines filesystem and path for convenient file operations."""

    def __init__(self, fs: fsspec.AbstractFileSystem, path: str):
        self._fs = fs
        self._path = path

    def open(self, mode: str = "rb", **kwargs) -> fsspec.spec.AbstractBufferedFile:
        return self._fs.open(self._path, mode, **kwargs)

    def __str__(self):
        return self._path

    def __repr__(self):
        return f"FSPath({self._path})"

    def as_posix(self) -> str:
        # Get the filesystem protocol and add appropriate prefix
        protocol = getattr(self._fs, "protocol", None)
        if protocol and protocol != "file":
            # For non-local filesystems, add the protocol prefix
            if isinstance(protocol, (list, tuple)):
                protocol = protocol[0]  # Take first protocol if multiple
            return f"{protocol}://{self._path}"
        return self._path

    def get_bytes_cat_ranges(
        self,
        *,
        part_size: int = 10 * 1024**2,  # 10 MiB
    ) -> bytes:
        """
        Read object into memory using fsspec's cat_ranges.
        Modified from https://github.com/rapidsai/cudf/blob/ba64909422016ba389ab06ed01d7578336c19e8e/python/dask_cudf/dask_cudf/io/json.py#L26-L34
        """
        size = self._fs.size(self._path)
        if not size:
            return b""

        starts = list(range(0, size, part_size))
        ends = [min(s + part_size, size) for s in starts]

        # Raise on any failed range
        blocks = self._fs.cat_ranges(
            [self._path] * len(starts),
            starts,
            ends,
            on_error="raise",
        )

        out = bytearray(size)
        for s, b in zip(starts, blocks, strict=False):
            out[s : s + len(b)] = b
        return bytes(out)


def is_remote_url(url: str) -> bool:
    fs, _ = url_to_fs(url)
    proto = fs.protocol[0] if isinstance(fs.protocol, (list, tuple)) else fs.protocol
    return proto not in (None, "file")
