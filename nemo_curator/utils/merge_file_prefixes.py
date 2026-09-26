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

"""
Simplified version of the tools/merge_datasets.py script from the Megatron-LM library
(https://github.com/NVIDIA/Megatron-LM/blob/main/tools/merge_datasets.py).
"""

import argparse
import os
import struct
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from types import TracebackType

import numpy as np

_INDEX_HEADER = b"MMIDIDX\x00\x00"


def extract_index_contents(idx_path: str) -> tuple[np.ndarray, np.ndarray, type[np.number]]:
    """Extract the index contents from the index file

    Args:
        idx_path (str): The path to the index file

    Returns:
        Tuple[np.ndarray, np.ndarray, Type[np.number]]: The sequence lengths, document indices and dtype
                of the index file
    """
    with open(idx_path, "rb") as stream:
        header = stream.read(9)
        assert header == _INDEX_HEADER, f"bad header, cannot read: {idx_path}"  # noqa: S101

        version = struct.unpack("<Q", stream.read(8))[0]
        assert version == 1, f"bad version, cannot read: {idx_path}"  # noqa: S101

        code = struct.unpack("<B", stream.read(1))[0]
        dtype = np.int32 if code == 4 else np.uint16  # noqa: PLR2004

        sequence_count = struct.unpack("<Q", stream.read(8))[0]
        document_count = struct.unpack("<Q", stream.read(8))[0]

        offset = stream.tell()

    bin_buffer_mmap = np.memmap(idx_path, mode="r", order="C")
    bin_buffer = memoryview(bin_buffer_mmap)

    sequence_lengths = np.frombuffer(bin_buffer, dtype=np.int32, count=sequence_count, offset=offset)

    sequence_pointers = np.frombuffer(
        bin_buffer,
        dtype=np.int64,
        count=sequence_count,
        offset=offset + sequence_lengths.nbytes,
    )
    document_indices = np.frombuffer(
        bin_buffer,
        dtype=np.int64,
        count=document_count,
        offset=offset + sequence_lengths.nbytes + sequence_pointers.nbytes,
    )

    return sequence_lengths, document_indices, dtype


def _kernel_copy(src_fd: int, dst_fd: int, dst_offset: int, size: int) -> None:
    """Copy ``size`` bytes from ``src_fd[0:size]`` into ``dst_fd[dst_offset:dst_offset+size]``
    using the fastest available primitive.

    Prefers ``os.copy_file_range`` (Linux 4.5+): explicit src/dst offsets, GIL released,
    in-kernel copy that becomes a reflink on filesystems that support it (XFS, Btrfs).
    Falls back to ``os.sendfile`` (writes at dst's *current* file position — callers using
    sendfile in parallel must give each worker its own dst fd) and finally to a chunked
    read/write loop.
    """
    if size == 0:
        return

    if hasattr(os, "copy_file_range"):
        try:
            sent = 0
            while sent < size:
                n = os.copy_file_range(
                    src_fd,
                    dst_fd,
                    size - sent,
                    offset_src=sent,
                    offset_dst=dst_offset + sent,
                )
                if n == 0:
                    break
                sent += n
            if sent == size:
                return
            # Short copy (some backends return 0 prematurely) - fall through
        except OSError:
            pass

    if hasattr(os, "sendfile"):
        try:
            os.lseek(dst_fd, dst_offset, os.SEEK_SET)
            sent = 0
            while sent < size:
                n = os.sendfile(dst_fd, src_fd, sent, size - sent)
                if n == 0:
                    break
                sent += n
            if sent == size:
                return
            # Short copy - fall through to the chunked read/write path
        except OSError:
            pass

    os.lseek(src_fd, 0, os.SEEK_SET)
    os.lseek(dst_fd, dst_offset, os.SEEK_SET)
    chunk_size = 64 * 1024 * 1024
    remaining = size
    while remaining > 0:
        buf = os.read(src_fd, min(chunk_size, remaining))
        if not buf:
            break
        os.write(dst_fd, buf)
        remaining -= len(buf)
    if remaining > 0:
        msg = f"premature EOF: copied {size - remaining} of {size} bytes"
        raise OSError(msg)


def _copy_bin_at_offset(src_path: str, dst_path: str, dst_offset: int, size: int) -> None:
    """Copy a full input ``.bin`` file into ``[dst_offset, dst_offset+size)`` of the output.

    Each call opens its own src and dst fds, so this is safe to invoke concurrently from
    threads — no shared file-position state.
    """
    with open(src_path, "rb") as src, open(dst_path, "r+b") as dst:
        _kernel_copy(src.fileno(), dst.fileno(), dst_offset, size)


class _IndexWriter:
    """Simplified version of the _IndexWriter class from the Megatron-LM library.

    Object class to write the index (.idx) file

    Args:
        idx_path (str): The path to the index file

        dtype (Type[np.number]): The dtype of the index file
    """

    def __init__(self, idx_path: str, dtype: type[np.number]) -> None:
        self.idx_path = idx_path
        self.dtype = dtype

    def __enter__(self) -> "_IndexWriter":
        """Enter the context introduced by the 'with' keyword

        Returns:
            _IndexWriter: The instance
        """
        self.idx_writer = open(self.idx_path, "wb")
        # fixed, vestigial practice
        self.idx_writer.write(_INDEX_HEADER)
        # fixed, vestigial practice
        self.idx_writer.write(struct.pack("<Q", 1))
        # the numeric code for the dtype
        self.idx_writer.write(struct.pack("<B", 4 if self.dtype == np.int32 else 8))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        """Exit the context introduced by the 'with' keyword

        Args:
            exc_type (Optional[Type[BaseException]]): Exception type

            exc_val (Optional[BaseException]): Exception value

            exc_tb (Optional[TracebackType]): Exception traceback object

        Returns:
            Optional[bool]: Whether to silence the exception
        """
        self.idx_writer.close()
        return None

    def write(
        self,
        sequence_lengths: Iterable[int | np.integer],
        document_indices: Iterable[int | np.integer],
    ) -> None:
        """Write the index (.idx) file

        Args:
            sequence_lengths (List[int]): The length of each sequence

            document_indices (List[int]): The sequence indices demarcating the end of each document
        """
        sequence_pointers = self._sequence_pointers(sequence_lengths)

        # the number of sequences in the dataset
        sequence_count = len(sequence_lengths)
        self.idx_writer.write(struct.pack("<Q", sequence_count))

        # the number of documents in the dataset
        document_count = len(document_indices)
        self.idx_writer.write(struct.pack("<Q", document_count))

        # the number of tokens per sequence
        self.idx_writer.write(np.ascontiguousarray(sequence_lengths, dtype=np.int32).tobytes(order="C"))

        # the byte offsets for all sequences
        self.idx_writer.write(sequence_pointers.tobytes(order="C"))

        # the sequence indices marking the end of each document
        self.idx_writer.write(np.ascontiguousarray(document_indices, dtype=np.int64).tobytes(order="C"))

    def _sequence_pointers(self, sequence_lengths: Iterable[int | np.integer]) -> np.ndarray:
        """Build the sequence pointers per the sequence lengths and dtype size

        Args:
            sequence_lengths (List[int]): The length of each sequence

        Returns:
            np.ndarray: int64 array of byte pointers to the start of each sequence in the bin file
        """
        itemsize = 4 if self.dtype == np.int32 else 2
        n = len(sequence_lengths)
        pointers = np.empty(n, dtype=np.int64)
        if n == 0:
            return pointers
        pointers[0] = 0
        if n > 1:
            np.cumsum(
                np.asarray(sequence_lengths[:-1], dtype=np.int64) * np.int64(itemsize),
                out=pointers[1:],
            )
        return pointers


class IndexedDatasetBuilder:
    """Simplified version of the IndexedDatasetBuilder class from the Megatron-LM library.

    Builder class for the IndexedDataset class

    Args:
        bin_path (str): The path to the data (.bin) file

        dtype (Type[np.number], optional): The dtype of the index file. Defaults to np.int32.

    """

    def __init__(self, bin_path: str, dtype: type[np.number]) -> None:
        self.data_file = open(bin_path, "wb")  # noqa: SIM115
        self.dtype = dtype

        # Accumulate per-input arrays and concatenate once at finalize. Extending a Python
        # list with numpy scalars and re-boxing them at the end was a major hot spot.
        self._seq_chunks: list[np.ndarray] = []
        self._doc_chunks: list[np.ndarray] = [np.array([0], dtype=np.int64)]
        self._cumulative_seq = 0

    def add_index(self, path_prefix: str) -> None:
        """Add an entire IndexedDataset to the dataset

        Args:
            path_prefix (str): The index (.idx) and data (.bin) prefix
        """
        sequence_lengths, document_indices, dtype = extract_index_contents(path_prefix + ".idx")
        assert dtype == self.dtype  # noqa: S101

        offset = self._cumulative_seq
        self._seq_chunks.append(sequence_lengths)
        self._doc_chunks.append(np.int64(offset) + document_indices[1:].astype(np.int64, copy=False))
        self._cumulative_seq += len(sequence_lengths)

        src_path = path_prefix + ".bin"
        size = os.path.getsize(src_path)
        if size:
            self.data_file.flush()
            dst_offset = self.data_file.tell()
            with open(src_path, "rb") as src:
                _kernel_copy(src.fileno(), self.data_file.fileno(), dst_offset, size)
            # Sync Python's logical position with the kernel-level file size so any
            # subsequent buffered ops see the right offset.
            self.data_file.seek(dst_offset + size)

    def finalize(self, idx_path: str) -> None:
        """Clean up and write the index (.idx) file

        Args:
            idx_path (str): The path to the index file
        """
        self.data_file.close()
        sequence_lengths = (
            np.concatenate(self._seq_chunks) if self._seq_chunks else np.empty(0, dtype=np.int32)
        )
        document_indices = np.concatenate(self._doc_chunks)
        with _IndexWriter(idx_path, self.dtype) as writer:
            writer.write(sequence_lengths, document_indices)


def _discover_prefixes(input_dir: str) -> list[str]:
    prefixes: set[str] = set()
    for basename in os.listdir(input_dir):
        prefix, ext = os.path.splitext(basename)

        if ext not in {".bin", ".idx"}:
            continue

        if prefix in prefixes:
            continue

        if not os.path.isfile(os.path.join(input_dir, basename)):
            continue

        ext_pair = ".bin" if ext == ".idx" else ".idx"
        assert os.path.isfile(os.path.join(input_dir, prefix + ext_pair)), (  # noqa: S101
            f"ERROR: {ext_pair} file not provided for {os.path.join(input_dir, prefix)}"
        )

        prefixes.add(prefix)

    if not prefixes:
        msg = f"ERROR: No valid file prefix pairs found in {input_dir}"
        raise ValueError(msg)

    return sorted(prefixes)


def merge_file_prefixes(input_dir: str, output_prefix: str, workers: int = 1) -> None:
    """Merge all .bin/.idx prefix pairs in ``input_dir`` into a single pair at ``output_prefix``.

    Args:
        input_dir: Directory containing the .bin/.idx pairs to merge.
        output_prefix: Output path prefix; produces ``<output_prefix>.bin`` and ``<output_prefix>.idx``.
        workers: Threads for parallel index reads and bin copies. Default 1 is serial. Bin copies
            use kernel zero-copy (``copy_file_range``), so ``workers > 1`` parallelizes those across
            threads — useful on parallel/multi-stream storage. A single local SSD typically saturates
            at 2-4 workers.
    """
    prefixes = _discover_prefixes(input_dir)

    if workers <= 1:
        builder: IndexedDatasetBuilder | None = None
        for prefix in prefixes:
            if builder is None:
                _, _, dtype = extract_index_contents(os.path.join(input_dir, prefix + ".idx"))
                builder = IndexedDatasetBuilder(output_prefix + ".bin", dtype=dtype)
            builder.add_index(os.path.join(input_dir, prefix))
        builder.finalize(output_prefix + ".idx")
        return

    paths = [os.path.join(input_dir, p) for p in prefixes]

    # Phase 1: read all idx files concurrently. Each thread independently mmaps its file,
    # so no shared state.
    with ThreadPoolExecutor(max_workers=workers) as ex:
        idx_results = list(ex.map(lambda p: extract_index_contents(p + ".idx"), paths))

    dtype = idx_results[0][2]
    for _, _, d in idx_results[1:]:
        assert d == dtype, f"dtype mismatch across input files: {d} vs {dtype}"  # noqa: S101

    seq_chunks = [r[0] for r in idx_results]
    doc_chunks: list[np.ndarray] = [np.array([0], dtype=np.int64)]
    cumulative_seq = 0
    for seq, doc, _ in idx_results:
        doc_chunks.append(np.int64(cumulative_seq) + doc[1:].astype(np.int64, copy=False))
        cumulative_seq += len(seq)
    sequence_lengths = np.concatenate(seq_chunks) if seq_chunks else np.empty(0, dtype=np.int32)
    document_indices = np.concatenate(doc_chunks)

    # Phase 2: pre-size the output bin and copy each input slice into its known offset
    # in parallel.
    bin_paths = [p + ".bin" for p in paths]
    bin_sizes = [os.path.getsize(p) for p in bin_paths]
    offsets: list[int] = []
    running = 0
    for s in bin_sizes:
        offsets.append(running)
        running += s
    total_size = running

    out_bin = output_prefix + ".bin"
    with open(out_bin, "wb") as f:
        if total_size:
            os.ftruncate(f.fileno(), total_size)

    if total_size:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(
                ex.map(
                    lambda args: _copy_bin_at_offset(args[0], out_bin, args[1], args[2]),
                    zip(bin_paths, offsets, bin_sizes, strict=True),
                )
            )

    # Phase 3: write the merged idx.
    with _IndexWriter(output_prefix + ".idx", dtype=dtype) as writer:
        writer.write(sequence_lengths, document_indices)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title="input data")
    group.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to directory containing all document files to merge",
    )

    group = parser.add_argument_group(title="output data")
    group.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Path to merged output file prefix",
    )

    group = parser.add_argument_group(title="parallelism")
    group.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Threads for parallel index read and bin copy. Default 1 (serial). On a single "
            "local SSD, 2-4 typically saturates I/O; on parallel/multi-stream storage "
            "(Lustre, multi-mount NFS), more pays off."
        ),
    )

    args = parser.parse_args()

    assert os.path.isdir(args.input_dir), f"ERROR: {args.input_dir} is not a directory or does not exist"  # noqa: S101

    output_dir = os.path.dirname(args.output_prefix) or "."
    assert os.path.isdir(output_dir), (  # noqa: S101
        f"ERROR: {output_dir} is not a directory or does not exist"
    )

    return args


if __name__ == "__main__":
    args = get_args()
    merge_file_prefixes(args.input_dir, args.output_prefix, workers=args.workers)
