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

from collections.abc import Generator

from nemo_curator.utils.grouping import pairwise, split_by_chunk_size, split_into_n_chunks


class TestSplitByChunkSize:
    """Test cases for split_by_chunk_size function."""

    def test_basic_functionality(self):
        """Test basic chunking with default parameters."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        chunks = list(split_by_chunk_size(data, 3))
        expected = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assert chunks == expected

    def test_incomplete_chunk_kept_by_default(self):
        """Test that incomplete chunks are kept by default."""
        data = [1, 2, 3, 4, 5]
        chunks = list(split_by_chunk_size(data, 3))
        expected = [[1, 2, 3], [4, 5]]
        assert chunks == expected

    def test_drop_incomplete_chunk_true(self):
        """Test dropping incomplete chunks when drop_incomplete_chunk=True."""
        data = [1, 2, 3, 4, 5]
        chunks = list(split_by_chunk_size(data, 3, drop_incomplete_chunk=True))
        expected = [[1, 2, 3]]
        assert chunks == expected

    def test_drop_incomplete_chunk_false_explicit(self):
        """Test keeping incomplete chunks when drop_incomplete_chunk=False."""
        data = [1, 2, 3, 4, 5]
        chunks = list(split_by_chunk_size(data, 3, drop_incomplete_chunk=False))
        expected = [[1, 2, 3], [4, 5]]
        assert chunks == expected

    def test_custom_size_function(self):
        """Test using custom size function."""
        # Each string's length counts as its size
        data = ["a", "bb", "ccc", "dddd", "e"]
        chunks = list(split_by_chunk_size(data, 5, custom_size_func=len))
        expected = [["a", "bb", "ccc"], ["dddd", "e"]]
        assert chunks == expected

    def test_custom_size_function_complex(self):
        """Test custom size function with more complex logic."""
        # Using tuples where second element is the size
        data = [("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 1)]
        chunks = list(split_by_chunk_size(data, 5, custom_size_func=lambda x: x[1]))
        expected = [[("a", 1), ("b", 2), ("c", 3)], [("d", 4), ("e", 1)]]
        assert chunks == expected

    def test_chunk_size_one(self):
        """Test with chunk_size of 1."""
        data = [1, 2, 3]
        chunks = list(split_by_chunk_size(data, 1))
        expected = [[1], [2], [3]]
        assert chunks == expected

    def test_chunk_size_larger_than_data(self):
        """Test when chunk_size is larger than the data."""
        data = [1, 2, 3]
        chunks = list(split_by_chunk_size(data, 10))
        expected = [[1, 2, 3]]
        assert chunks == expected

    def test_empty_iterable(self):
        """Test with empty iterable."""
        data = []
        chunks = list(split_by_chunk_size(data, 3))
        expected = []
        assert chunks == expected

    def test_single_item(self):
        """Test with single item."""
        data = [42]
        chunks = list(split_by_chunk_size(data, 3))
        expected = [[42]]
        assert chunks == expected

    def test_generator_input(self):
        """Test with generator as input."""

        def gen() -> Generator[int, None, None]:
            yield from range(5)

        chunks = list(split_by_chunk_size(gen(), 2))
        expected = [[0, 1], [2, 3], [4]]
        assert chunks == expected

    def test_string_iterable(self):
        """Test with string as iterable."""
        data = "hello"
        chunks = list(split_by_chunk_size(data, 2))
        expected = [["h", "e"], ["l", "l"], ["o"]]
        assert chunks == expected


class TestSplitIntoNChunks:
    """Test cases for split_into_n_chunks function."""

    def test_basic_functionality(self):
        """Test basic splitting into n chunks."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        chunks = list(split_into_n_chunks(data, 3))
        expected = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assert chunks == expected

    def test_uneven_split(self):
        """Test splitting when data doesn't divide evenly."""
        data = [1, 2, 3, 4, 5, 6, 7, 8]
        chunks = list(split_into_n_chunks(data, 3))
        expected = [[1, 2, 3], [4, 5, 6], [7, 8]]
        assert chunks == expected

    def test_more_chunks_than_items(self):
        """Test when number of chunks is greater than number of items."""
        data = [1, 2, 3]
        chunks = list(split_into_n_chunks(data, 5))
        expected = [[1], [2], [3]]
        assert chunks == expected

    def test_single_chunk(self):
        """Test splitting into a single chunk."""
        data = [1, 2, 3, 4, 5]
        chunks = list(split_into_n_chunks(data, 1))
        expected = [[1, 2, 3, 4, 5]]
        assert chunks == expected

    def test_empty_iterable(self):
        """Test with empty iterable."""
        data = []
        chunks = list(split_into_n_chunks(data, 3))
        expected = []
        assert chunks == expected

    def test_single_item(self):
        """Test with single item."""
        data = [42]
        chunks = list(split_into_n_chunks(data, 3))
        expected = [[42]]
        assert chunks == expected

    def test_equal_chunks_even_division(self):
        """Test that chunks are equal when data divides evenly."""
        data = list(range(12))
        chunks = list(split_into_n_chunks(data, 4))
        expected = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        assert chunks == expected
        # Verify all chunks have the same size
        chunk_sizes = [len(chunk) for chunk in chunks]
        assert all(size == 3 for size in chunk_sizes)

    def test_remainder_distribution(self):
        """Test that remainder is distributed among first chunks."""
        data = list(range(10))  # 10 items into 3 chunks
        chunks = list(split_into_n_chunks(data, 3))
        expected = [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assert chunks == expected
        # First chunk should have one extra item
        assert len(chunks[0]) == 4
        assert len(chunks[1]) == 3
        assert len(chunks[2]) == 3

    def test_string_iterable(self):
        """Test with string as iterable."""
        data = "hello world"
        chunks = list(split_into_n_chunks(data, 3))
        expected = [["h", "e", "l", "l"], ["o", " ", "w", "o"], ["r", "l", "d"]]
        assert chunks == expected


class TestPairwise:
    """Test cases for pairwise function."""

    def test_basic_functionality(self):
        """Test basic pairwise functionality."""
        data = [1, 2, 3, 4, 5]
        pairs = list(pairwise(data))
        expected = [(1, 2), (2, 3), (3, 4), (4, 5)]
        assert pairs == expected

    def test_two_items(self):
        """Test with exactly two items."""
        data = [1, 2]
        pairs = list(pairwise(data))
        expected = [(1, 2)]
        assert pairs == expected

    def test_single_item(self):
        """Test with single item."""
        data = [1]
        pairs = list(pairwise(data))
        expected = []
        assert pairs == expected

    def test_empty_iterable(self):
        """Test with empty iterable."""
        data = []
        pairs = list(pairwise(data))
        expected = []
        assert pairs == expected

    def test_string_iterable(self):
        """Test with string as iterable."""
        data = "hello"
        pairs = list(pairwise(data))
        expected = [("h", "e"), ("e", "l"), ("l", "l"), ("l", "o")]
        assert pairs == expected

    def test_generator_input(self):
        """Test with generator as input."""

        def gen() -> Generator[int, None, None]:
            yield from range(4)

        pairs = list(pairwise(gen()))
        expected = [(0, 1), (1, 2), (2, 3)]
        assert pairs == expected

    def test_different_types(self):
        """Test with mixed types."""
        data = [1, "a", 2.5, True]
        pairs = list(pairwise(data))
        expected = [(1, "a"), ("a", 2.5), (2.5, True)]
        assert pairs == expected

    def test_duplicate_items(self):
        """Test with duplicate items."""
        data = [1, 1, 2, 2, 3]
        pairs = list(pairwise(data))
        expected = [(1, 1), (1, 2), (2, 2), (2, 3)]
        assert pairs == expected
