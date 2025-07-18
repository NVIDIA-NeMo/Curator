"""Test suite for grouping utilities module."""

from collections.abc import Generator

from ray_curator.utils.grouping import pairwise, split_by_chunk_size, split_into_n_chunks


class TestSplitByChunkSize:
    """Test suite for split_by_chunk_size function."""

    def test_split_by_chunk_size_basic(self) -> None:
        """Test basic functionality of split_by_chunk_size."""
        iterable = [1, 2, 3, 4, 5, 6]
        result = list(split_by_chunk_size(iterable, chunk_size=2))

        assert result == [[1, 2], [3, 4], [5, 6]]

    def test_split_by_chunk_size_incomplete_chunk(self) -> None:
        """Test split_by_chunk_size with incomplete final chunk."""
        iterable = [1, 2, 3, 4, 5]
        result = list(split_by_chunk_size(iterable, chunk_size=2))

        assert result == [[1, 2], [3, 4], [5]]

    def test_split_by_chunk_size_drop_incomplete_chunk(self) -> None:
        """Test split_by_chunk_size with drop_incomplete_chunk=True."""
        iterable = [1, 2, 3, 4, 5]
        result = list(split_by_chunk_size(iterable, chunk_size=2, drop_incomplete_chunk=True))

        assert result == [[1, 2], [3, 4]]

    def test_split_by_chunk_size_empty_iterable(self) -> None:
        """Test split_by_chunk_size with empty iterable."""
        iterable = []
        result = list(split_by_chunk_size(iterable, chunk_size=2))

        assert result == []

    def test_split_by_chunk_size_single_item(self) -> None:
        """Test split_by_chunk_size with single item."""
        iterable = [42]
        result = list(split_by_chunk_size(iterable, chunk_size=2))

        assert result == [[42]]

    def test_split_by_chunk_size_chunk_size_one(self) -> None:
        """Test split_by_chunk_size with chunk_size=1."""
        iterable = [1, 2, 3]
        result = list(split_by_chunk_size(iterable, chunk_size=1))

        assert result == [[1], [2], [3]]

    def test_split_by_chunk_size_chunk_size_larger_than_iterable(self) -> None:
        """Test split_by_chunk_size with chunk_size larger than iterable length."""
        iterable = [1, 2, 3]
        result = list(split_by_chunk_size(iterable, chunk_size=10))

        assert result == [[1, 2, 3]]

    def test_split_by_chunk_size_custom_size_func(self) -> None:
        """Test split_by_chunk_size with custom size function."""
        # Using strings where size is determined by length
        iterable = ["a", "bb", "ccc", "dddd", "e"]
        result = list(split_by_chunk_size(iterable, chunk_size=5, custom_size_func=len))

        assert result == [["a", "bb", "ccc"], ["dddd", "e"]]

    def test_split_by_chunk_size_custom_size_func_exact_fit(self) -> None:
        """Test split_by_chunk_size with custom size function that fits exactly."""
        iterable = ["ab", "cd", "ef"]
        result = list(split_by_chunk_size(iterable, chunk_size=4, custom_size_func=len))

        assert result == [["ab", "cd"], ["ef"]]

    def test_split_by_chunk_size_custom_size_func_drop_incomplete(self) -> None:
        """Test split_by_chunk_size with custom size function and drop_incomplete_chunk=True."""
        iterable = ["a", "bb", "ccc", "d"]
        result = list(split_by_chunk_size(
            iterable,
            chunk_size=4,
            custom_size_func=len,
            drop_incomplete_chunk=True
        ))

        assert result == [["a", "bb", "ccc"]]

    def test_split_by_chunk_size_generator_input(self) -> None:
        """Test split_by_chunk_size with generator input."""
        def gen() -> Generator[int, None, None]:
            yield from range(5)

        result = list(split_by_chunk_size(gen(), chunk_size=2))

        assert result == [[0, 1], [2, 3], [4]]

    def test_split_by_chunk_size_string_iterable(self) -> None:
        """Test split_by_chunk_size with string iterable."""
        iterable = "hello"
        result = list(split_by_chunk_size(iterable, chunk_size=2))

        assert result == [["h", "e"], ["l", "l"], ["o"]]


class TestSplitIntoNChunks:
    """Test suite for split_into_n_chunks function."""

    def test_split_into_n_chunks_basic(self) -> None:
        """Test basic functionality of split_into_n_chunks."""
        iterable = [1, 2, 3, 4, 5, 6]
        result = list(split_into_n_chunks(iterable, num_chunks=3))

        assert result == [[1, 2], [3, 4], [5, 6]]

    def test_split_into_n_chunks_uneven_division(self) -> None:
        """Test split_into_n_chunks with uneven division."""
        iterable = [1, 2, 3, 4, 5, 6, 7]
        result = list(split_into_n_chunks(iterable, num_chunks=3))

        assert result == [[1, 2, 3], [4, 5], [6, 7]]

    def test_split_into_n_chunks_more_chunks_than_items(self) -> None:
        """Test split_into_n_chunks with more chunks than items."""
        iterable = [1, 2, 3]
        result = list(split_into_n_chunks(iterable, num_chunks=5))

        assert result == [[1], [2], [3]]

    def test_split_into_n_chunks_empty_iterable(self) -> None:
        """Test split_into_n_chunks with empty iterable."""
        iterable = []
        result = list(split_into_n_chunks(iterable, num_chunks=3))

        assert result == []

    def test_split_into_n_chunks_single_item(self) -> None:
        """Test split_into_n_chunks with single item."""
        iterable = [42]
        result = list(split_into_n_chunks(iterable, num_chunks=3))

        assert result == [[42]]

    def test_split_into_n_chunks_single_chunk(self) -> None:
        """Test split_into_n_chunks with single chunk."""
        iterable = [1, 2, 3, 4, 5]
        result = list(split_into_n_chunks(iterable, num_chunks=1))

        assert result == [[1, 2, 3, 4, 5]]

    def test_split_into_n_chunks_equal_chunks(self) -> None:
        """Test split_into_n_chunks with equal-sized chunks."""
        iterable = [1, 2, 3, 4, 5, 6, 7, 8]
        result = list(split_into_n_chunks(iterable, num_chunks=4))

        assert result == [[1, 2], [3, 4], [5, 6], [7, 8]]

    def test_split_into_n_chunks_large_uneven(self) -> None:
        """Test split_into_n_chunks with larger uneven division."""
        iterable = list(range(10))
        result = list(split_into_n_chunks(iterable, num_chunks=3))

        assert result == [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def test_split_into_n_chunks_string_iterable(self) -> None:
        """Test split_into_n_chunks with string iterable."""
        iterable = "hello"
        result = list(split_into_n_chunks(iterable, num_chunks=2))

        assert result == [["h", "e", "l"], ["l", "o"]]

    def test_split_into_n_chunks_generator_input(self) -> None:
        """Test split_into_n_chunks with generator input."""
        def gen() -> Generator[int, None, None]:
            yield from range(6)

        result = list(split_into_n_chunks(gen(), num_chunks=2))

        assert result == [[0, 1, 2], [3, 4, 5]]


class TestPairwise:
    """Test suite for pairwise function."""

    def test_pairwise_basic(self) -> None:
        """Test basic functionality of pairwise."""
        iterable = [1, 2, 3, 4, 5]
        result = list(pairwise(iterable))

        assert result == [(1, 2), (2, 3), (3, 4), (4, 5)]

    def test_pairwise_empty_iterable(self) -> None:
        """Test pairwise with empty iterable."""
        iterable = []
        result = list(pairwise(iterable))

        assert result == []

    def test_pairwise_single_item(self) -> None:
        """Test pairwise with single item."""
        iterable = [42]
        result = list(pairwise(iterable))

        assert result == []

    def test_pairwise_two_items(self) -> None:
        """Test pairwise with two items."""
        iterable = [1, 2]
        result = list(pairwise(iterable))

        assert result == [(1, 2)]

    def test_pairwise_string_iterable(self) -> None:
        """Test pairwise with string iterable."""
        iterable = "hello"
        result = list(pairwise(iterable))

        assert result == [("h", "e"), ("e", "l"), ("l", "l"), ("l", "o")]

    def test_pairwise_generator_input(self) -> None:
        """Test pairwise with generator input."""
        def gen() -> Generator[int, None, None]:
            yield from range(4)

        result = list(pairwise(gen()))

        assert result == [(0, 1), (1, 2), (2, 3)]

    def test_pairwise_different_types(self) -> None:
        """Test pairwise with different data types."""
        iterable = [1, "a", 3.14, True]
        result = list(pairwise(iterable))

        assert result == [(1, "a"), ("a", 3.14), (3.14, True)]

    def test_pairwise_repeated_elements(self) -> None:
        """Test pairwise with repeated elements."""
        iterable = [1, 1, 2, 2, 3, 3]
        result = list(pairwise(iterable))

        assert result == [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3)]
