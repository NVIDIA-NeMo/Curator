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

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import cudf


class RankingStrategy:
    """Flexible ranking strategy that allows users to specify metadata columns and sorting order.

    This design allows for extensible ranking based on any metadata columns with
    user-specified sorting criteria.
    """

    def __init__(
        self,
        metadata_cols: list[str],
        ascending: list[bool] | bool = True,
        strategy: Literal["sort", "random"] = "sort",
        random_seed: int = 42,
    ):
        """Initialize ranking strategy.

        Args:
            metadata_cols: List of metadata column names to sort by (in priority order)
            ascending: Boolean or list of booleans indicating sort order for each column.
                      If single bool, applies to all columns.
            strategy: Ranking strategy - "sort" for sorting by metadata_cols, "random" for random
            random_seed: Seed for random strategy
        """
        self.metadata_cols = metadata_cols
        self.strategy = strategy
        self.random_seed = random_seed

        # Handle ascending parameter
        if isinstance(ascending, bool):
            self.ascending = [ascending] * len(metadata_cols)
        else:
            if len(ascending) != len(metadata_cols):
                msg = f"Length of ascending ({len(ascending)}) must match metadata_cols ({len(metadata_cols)})"
                raise ValueError(msg)
            self.ascending = ascending

    def rank_cluster(self, cluster_df: "cudf.DataFrame") -> "cudf.DataFrame":
        """Rank cluster based on the specified strategy."""
        if self.strategy == "random":
            return cluster_df.sample(frac=1, random_state=self.random_seed, ignore_index=True)
        elif self.strategy == "sort":
            # Check that all required columns exist
            missing_cols = [col for col in self.metadata_cols if col not in cluster_df.columns]
            if missing_cols:
                msg = f"Required columns {missing_cols} not found in cluster data. "
                msg += f"Available columns: {list(cluster_df.columns)}"
                raise ValueError(msg)

            # Only sort by the explicitly specified metadata columns
            return cluster_df.sort_values(by=self.metadata_cols, ascending=self.ascending, ignore_index=True)
        else:
            msg = f"Invalid strategy: {self.strategy}. Supported: 'sort', 'random'"
            raise ValueError(msg)

    @classmethod
    def metadata_based(
        cls,
        metadata_cols: list[str],
        ascending: list[bool] | bool = True,
        random_seed: int = 42,
    ) -> "RankingStrategy":
        """Create a metadata-based ranking strategy.

        Args:
            metadata_cols: List of metadata column names to sort by (in priority order)
            ascending: Boolean or list of booleans indicating sort order for each column
            random_seed: Random seed for reproducible results

        Returns:
            RankingStrategy instance configured for metadata-based ranking
        """
        return cls(metadata_cols=metadata_cols, ascending=ascending, strategy="sort", random_seed=random_seed)

    @classmethod
    def random(cls, random_seed: int = 42) -> "RankingStrategy":
        """Create a random ranking strategy.

        Args:
            random_seed: Random seed for reproducible results

        Returns:
            RankingStrategy instance configured for random ranking
        """
        return cls(metadata_cols=[], strategy="random", random_seed=random_seed)
