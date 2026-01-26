"""
Simplified version of the tools/merge_datasets.py script from the Megatron-LM library.
"""

import argparse
import gc
import os
import shutil
import struct
from collections.abc import Iterable
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
        self.idx_writer.write(np.array(sequence_lengths, dtype=np.int32).tobytes(order="C"))

        # the byte offsets for all sequences
        self.idx_writer.write(np.array(sequence_pointers, dtype=np.int64).tobytes(order="C"))

        # the sequence indices marking the end of each document
        self.idx_writer.write(np.array(document_indices, dtype=np.int64).tobytes(order="C"))

    def _sequence_pointers(self, sequence_lengths: Iterable[int | np.integer]) -> list[int]:
        """Build the sequence pointers per the sequence lengths and dtype size

        Args:
            sequence_lengths (List[int]): The length of each sequence

        Returns:
            List[int]: The pointer to the beginning of each sequence
        """
        itemsize = np.int64(4 if self.dtype == np.int32 else 2)
        curr_ptr = np.int64(0)
        list_ptr = []
        for length in sequence_lengths:
            list_ptr.append(curr_ptr.item())
            curr_ptr += length * itemsize
        return list_ptr


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

        self.sequence_lengths = []
        self.document_indices = [0]

    def add_index(self, path_prefix: str) -> None:
        """Add an entire IndexedDataset to the dataset

        Args:
            path_prefix (str): The index (.idx) and data (.bin) prefix
        """
        # Concatenate index
        sequence_lengths, document_indices, dtype = extract_index_contents(path_prefix + ".idx")
        assert dtype == self.dtype  # noqa: S101

        offset = len(self.sequence_lengths)
        self.sequence_lengths.extend(sequence_lengths)
        self.document_indices.extend((offset + document_indices)[1:])

        # Free up memory to make space for new indices
        del sequence_lengths, document_indices
        gc.collect()

        # Concatenate data
        with open(path_prefix + ".bin", "rb") as f:
            shutil.copyfileobj(f, self.data_file)

    def finalize(self, idx_path: str) -> None:
        """Clean up and write the index (.idx) file

        Args:
            idx_path (str): The path to the index file
        """
        self.data_file.close()
        with _IndexWriter(idx_path, self.dtype) as writer:
            writer.write(self.sequence_lengths, self.document_indices)


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

    args = parser.parse_args()

    assert os.path.isdir(args.input_dir), f"ERROR: {args.input_dir} is not a directory or does not exist"  # noqa: S101

    assert os.path.isdir(os.path.dirname(args.output_prefix)), (  # noqa: S101
        f"ERROR: {os.path.dirname(args.output_prefix)} is not a directory or does not exist"
    )

    return args


def main(input_dir: str, output_prefix: str) -> None:
    prefixes = set()
    for basename in os.listdir(input_dir):
        prefix, ext = os.path.splitext(basename)

        if prefix in prefixes:
            continue

        if not os.path.isfile(os.path.join(input_dir, basename)):
            continue

        ext_pair = ".bin" if ext == ".idx" else ".idx"
        assert os.path.isfile(os.path.join(input_dir, prefix) + ext_pair), (  # noqa: S101
            f"ERROR: {ext_pair} file not provided for {os.path.join(input_dir, prefix)}"
        )

        prefixes.add(prefix)

    builder = None
    for prefix in sorted(prefixes):
        if builder is None:
            _, _, dtype = extract_index_contents(os.path.join(input_dir, prefix + ".idx"))
            builder = IndexedDatasetBuilder(output_prefix + ".bin", dtype=dtype)

        builder.add_index(os.path.join(input_dir, prefix))

    builder.finalize(output_prefix + ".idx")


if __name__ == "__main__":
    args = get_args()
    main(args.input_dir, args.output_prefix)
