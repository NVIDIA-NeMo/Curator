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
import pytest

cudf = pytest.importorskip("cudf")

from nemo_curator.stages.deduplication.semantic.ranking import RankingStrategy


@pytest.mark.gpu
class TestRankingStrategy:
    """Test cases for RankingStrategy."""

    def test_distance_based_ranking_hard(self) -> None:
        """Test distance-based ranking with hard strategy (farthest first)."""
        # Create test data
        test_data = cudf.DataFrame(
            {"id": [1, 2, 3, 4], "cosine_dist_to_cent": [0.1, 0.8, 0.3, 0.6], "other_col": ["a", "b", "c", "d"]}
        )

        # Create distance-based ranking strategy using metadata_based method
        strategy = RankingStrategy.metadata_based(
            metadata_cols=["cosine_dist_to_cent"],
            ascending=False,  # farthest first (descending)
        )

        # Apply ranking
        ranked_df = strategy.rank_cluster(test_data)

        # Should be sorted by cosine_dist_to_cent descending, then id ascending
        expected_order = [2, 4, 3, 1]  # IDs in order of descending distance
        assert ranked_df["id"].to_arrow().to_pylist() == expected_order

    def test_distance_based_ranking_easy(self) -> None:
        """Test distance-based ranking with easy strategy (closest first)."""
        test_data = cudf.DataFrame(
            {"id": [1, 2, 3, 4], "cosine_dist_to_cent": [0.1, 0.8, 0.3, 0.6], "other_col": ["a", "b", "c", "d"]}
        )

        strategy = RankingStrategy.metadata_based(
            metadata_cols=["cosine_dist_to_cent"],
            ascending=True,  # closest first (ascending)
        )

        ranked_df = strategy.rank_cluster(test_data)

        # Should be sorted by cosine_dist_to_cent ascending, then id ascending
        expected_order = [1, 3, 4, 2]  # IDs in order of ascending distance
        assert ranked_df["id"].to_arrow().to_pylist() == expected_order

    def test_distance_based_ranking_random(self) -> None:
        """Test random ranking strategy."""
        test_data = cudf.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "cosine_dist_to_cent": [0.1, 0.8, 0.3, 0.6],
            }
        )

        strategy = RankingStrategy.random(random_seed=42)

        ranked_df = strategy.rank_cluster(test_data)

        # Should have all the same IDs but in different order
        assert set(ranked_df["id"].to_arrow().to_pylist()) == {1, 2, 3, 4}
        # With fixed seed, should be deterministic
        ranked_df2 = strategy.rank_cluster(test_data)
        assert ranked_df["id"].to_arrow().to_pylist() == ranked_df2["id"].to_arrow().to_pylist()

    def test_metadata_based_ranking(self) -> None:
        """Test metadata-based ranking."""
        test_data = cudf.DataFrame({"id": [1, 2, 3, 4], "score": [0.9, 0.3, 0.7, 0.5], "priority": [2, 1, 3, 1]})

        # Sort by priority (ascending), then score (descending)
        strategy = RankingStrategy.metadata_based(
            metadata_cols=["priority", "score"],
            ascending=[True, False],  # priority ascending, score descending
        )

        ranked_df = strategy.rank_cluster(test_data)

        # Expected order: priority=1 (ids 2,4) with score desc -> id 4, id 2
        #                 priority=2 (id 1) -> id 1
        #                 priority=3 (id 3) -> id 3
        expected_order = [4, 2, 1, 3]
        assert ranked_df["id"].to_arrow().to_pylist() == expected_order

    def test_custom_ranking_strategy(self) -> None:
        """Test creating custom ranking strategy directly."""
        test_data = cudf.DataFrame({"id": [1, 2, 3], "custom_metric": [10, 5, 20]})

        # Custom strategy: sort by custom_metric descending
        strategy = RankingStrategy(metadata_cols=["custom_metric"], ascending=False)

        ranked_df = strategy.rank_cluster(test_data)

        expected_order = [3, 1, 2]  # 20, 10, 5 in descending order
        assert ranked_df["id"].to_arrow().to_pylist() == expected_order

    def test_missing_column_error(self) -> None:
        """Test error when required column is missing."""
        test_data = cudf.DataFrame({"id": [1, 2, 3], "existing_col": [1, 2, 3]})

        strategy = RankingStrategy(metadata_cols=["missing_col"], ascending=True)

        with pytest.raises(ValueError, match="Required columns.*not found"):
            strategy.rank_cluster(test_data)

    def test_invalid_strategy_error(self) -> None:
        """Test error with invalid strategy."""
        test_data = cudf.DataFrame({"id": [1, 2], "col": [1, 2]})

        strategy = RankingStrategy(metadata_cols=["col"], strategy="invalid")

        with pytest.raises(ValueError, match="Invalid strategy"):
            strategy.rank_cluster(test_data)

    def test_mismatched_ascending_length(self) -> None:
        """Test error when ascending list length doesn't match metadata_cols."""
        with pytest.raises(ValueError, match="Length of ascending.*must match"):
            RankingStrategy(
                metadata_cols=["col1", "col2"],
                ascending=[True],  # Only one bool for two columns
            )
