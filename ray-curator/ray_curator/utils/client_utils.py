import fsspec


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
        part_size: int = 32 * 1024**2,  # 32 MiB
    ) -> bytes:
        """
        Read object into memory using fsspec's cat_ranges (no threads).
        Modified from https://github.com/rapidsai/cudf/blob/ba64909422016ba389ab06ed01d7578336c19e8e/python/dask_cudf/dask_cudf/io/json.py#L26-L34
        """
        size = self._fs.size(self._path)
        if not size:
            return b""

        starts = list(range(0, size, part_size))
        ends   = [min(s + part_size, size) for s in starts]

        # Raise on any failed range
        blocks = self._fs.cat_ranges(
            [self._path] * len(starts),
            starts,
            ends,
            on_error="raise",
        )

        out = bytearray(size)
        for s, b in zip(starts, blocks, strict=False):
            out[s:s + len(b)] = b
        return bytes(out)
