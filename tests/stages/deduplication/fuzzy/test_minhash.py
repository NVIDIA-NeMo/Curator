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

# ruff: noqa: E402
from collections.abc import Iterable
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

cudf = pytest.importorskip("cudf", reason="MinHash tests require cudf")

from nemo_curator.stages.deduplication.fuzzy.minhash import GPUMinHash


def minhash_overlap(minhash1: np.ndarray, minhash2: np.ndarray) -> float:
    """Calculate the overlap ratio between two minhash signatures."""
    assert len(minhash1) == len(minhash2)
    overlap = sum(minhash1 == minhash2)
    return overlap / len(minhash1)


def jaccard_index(str1: str, str2: str, char_ngrams: int) -> float:
    """Calculate the true Jaccard index between two strings."""
    return cudf.Series([str1]).str.jaccard_index(cudf.Series([str2]), width=char_ngrams).values_host[0]


def generate_all_pairs(item: Iterable) -> Iterable:
    """Generate all pairs of items from an iterable."""
    return combinations(item, 2)


@pytest.fixture
def sample_data():
    """Create sample test data."""
    return pd.DataFrame(
        {
            "id": [1, 2, 300, 4, -1],
            "text": [
                "A test string",
                "A different test string",
                "A different object",
                "The quick brown fox jumps over the lazy dog",
                "The quick black cat jumps over the lazy dog",
            ],
        }
    )


@pytest.mark.gpu
class TestGPUMinHash:
    """Test suite for GPU MinHash implementation."""

    @pytest.mark.parametrize("use_64bit_hash", [False, True])
    @pytest.mark.parametrize(("seed", "char_ngrams", "num_hashes"), [(128, 3, 260)])
    def test_identical_minhash(
        self,
        sample_data: pd.DataFrame,
        use_64bit_hash: bool,
        seed: int,
        char_ngrams: int,
        num_hashes: int,
    ) -> None:
        """Test that identical MinHash configurations produce identical results."""
        # Convert to cudf DataFrame
        df = cudf.from_pandas(sample_data)

        # Create first minhasher
        minhasher1 = GPUMinHash(
            seed=seed,
            num_hashes=num_hashes,
            char_ngrams=char_ngrams,
            use_64bit_hash=use_64bit_hash,
            pool=False,
        )

        # Compute minhashes
        result1 = minhasher1.compute_minhashes(df["text"])
        sig_lengths = result1.list.len()
        assert (sig_lengths == num_hashes).all()

        # Create second minhasher with same config
        minhasher2 = GPUMinHash(
            seed=seed,
            num_hashes=num_hashes,
            char_ngrams=char_ngrams,
            use_64bit_hash=use_64bit_hash,
            pool=False,
        )

        # Compute minhashes again
        result2 = minhasher2.compute_minhashes(df["text"])

        # Results should be identical
        assert (result1.to_pandas() == result2.to_pandas()).all()

    @pytest.mark.parametrize(
        ("use_64bit_hash", "seed", "char_ngrams", "num_hashes"),
        [(False, 42, 5, 20), (True, 32768, 10, 18)],
    )
    @pytest.mark.gpu
    def test_minhash_approximation(
        self,
        sample_data: pd.DataFrame,
        use_64bit_hash: bool,
        seed: int,
        char_ngrams: int,
        num_hashes: int,
    ) -> None:
        """Test that MinHash approximates Jaccard similarity within reasonable bounds."""
        THRESHOLD = 0.15  # noqa: N806

        # Convert to cudf DataFrame
        df = cudf.from_pandas(sample_data)

        # Create minhasher
        minhasher = GPUMinHash(
            seed=seed,
            num_hashes=num_hashes,
            char_ngrams=char_ngrams,
            use_64bit_hash=use_64bit_hash,
            pool=False,
        )

        # Compute minhashes
        minhash_signatures = minhasher.compute_minhashes(df["text"]).to_pandas().values  # noqa: PD011
        strings = df["text"].to_pandas().values  # noqa: PD011

        # Compare all pairs
        for (sig1, str1), (sig2, str2) in generate_all_pairs(tuple(zip(minhash_signatures, strings, strict=False))):
            true_jaccard = jaccard_index(str1, str2, char_ngrams)
            minhash_approximation = minhash_overlap(np.array(sig1), np.array(sig2))
            assert abs(true_jaccard - minhash_approximation) < THRESHOLD

    @pytest.mark.gpu
    def test_minhash_seed_generation(self) -> None:
        """Test the seed generation method."""
        minhasher = GPUMinHash()

        # Test 32-bit seeds
        seeds_32 = minhasher.generate_seeds(n_permutations=10, seed=42, bit_width=32)
        assert len(seeds_32) == 10
        assert seeds_32.dtype == np.uint32
        assert (seeds_32 < 2**32).all()

        # Test 64-bit seeds
        seeds_64 = minhasher.generate_seeds(n_permutations=10, seed=42, bit_width=64)
        assert len(seeds_64) == 10
        assert seeds_64.dtype == np.uint64

        # Same seed should produce same results
        seeds_32_repeat = minhasher.generate_seeds(n_permutations=10, seed=42, bit_width=32)
        assert (seeds_32 == seeds_32_repeat).all()

        # Different seed should produce different results
        seeds_32_diff = minhasher.generate_seeds(n_permutations=10, seed=123, bit_width=32)
        assert not (seeds_32 == seeds_32_diff).all()

    @pytest.mark.parametrize("char_ngrams", [5, 10, 24])
    def test_different_char_ngrams(
        self,
        sample_data: pd.DataFrame,
        char_ngrams: int,
    ) -> None:
        """Test MinHash with different character n-gram sizes."""
        df = cudf.from_pandas(sample_data)

        minhasher = GPUMinHash(
            seed=42,
            num_hashes=100,
            char_ngrams=char_ngrams,
        )

        result = minhasher.compute_minhashes(df["text"])

        # Verify all signatures have correct length
        sig_lengths = result.list.len()
        assert (sig_lengths == 100).all()

        # Higher n-gram sizes should generally produce more diverse signatures
        # for similar strings
        if char_ngrams > 5:
            sigs = result.to_pandas().values  # noqa: PD011
            # Compare signatures of "A test string" and "A different test string"
            overlap = minhash_overlap(np.array(sigs[0]), np.array(sigs[1]))
            # With larger n-grams, these strings should be less similar
            assert overlap < 0.8  # Arbitrary threshold for demonstration
