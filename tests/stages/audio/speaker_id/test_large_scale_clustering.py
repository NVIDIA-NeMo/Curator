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

"""Unit tests for large-scale BIRCH + AHC clustering."""

from __future__ import annotations

import numpy as np
import pytest

from nemo_curator.stages.audio.speaker_id.clustering.large_scale_clustering_and_scoring import (
    DROPPED_LABEL,
    cluster_embeddings_large_scale,
    cosine_threshold_to_birch_radius,
    filter_small_clusters,
    recommend_clustering_method,
)


class TestRecommendClusteringMethod:
    def test_small_dataset_standard(self) -> None:
        assert recommend_clustering_method(num_hours=100) == "standard"

    def test_large_hours_triggers_large_scale(self) -> None:
        assert recommend_clustering_method(num_hours=600) == "large_scale"

    def test_large_utterances_triggers_large_scale(self) -> None:
        assert recommend_clustering_method(num_utterances=200_000) == "large_scale"

    def test_both_small_is_standard(self) -> None:
        assert recommend_clustering_method(num_hours=100, num_utterances=50_000) == "standard"

    def test_neither_provided_raises(self) -> None:
        with pytest.raises(ValueError, match="Provide"):
            recommend_clustering_method()


class TestCosineThresholdToBirchRadius:
    def test_known_value(self) -> None:
        radius = cosine_threshold_to_birch_radius(0.8)
        expected = float(np.sqrt(2.0 * (1.0 - 0.8)))
        assert abs(radius - expected) < 1e-6

    def test_cosine_one_gives_zero(self) -> None:
        assert cosine_threshold_to_birch_radius(1.0) == pytest.approx(0.0, abs=1e-6)

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="cos_threshold"):
            cosine_threshold_to_birch_radius(1.5)


class TestFilterSmallClusters:
    def test_drops_small_clusters(self) -> None:
        labels = np.array([1, 1, 1, 2, 2, 2, 3])
        filtered, stats = filter_small_clusters(labels, min_cluster_size=2)

        assert stats["n_clusters_dropped"] == 1
        assert filtered[6] == DROPPED_LABEL
        assert all(filtered[i] != DROPPED_LABEL for i in range(6))

    def test_min_size_one_keeps_all(self) -> None:
        labels = np.array([1, 2, 3])
        filtered, stats = filter_small_clusters(labels, min_cluster_size=1)

        assert stats["n_clusters_dropped"] == 0
        np.testing.assert_array_equal(filtered, labels)

    def test_all_dropped(self) -> None:
        labels = np.array([1, 2, 3])
        filtered, stats = filter_small_clusters(labels, min_cluster_size=5)

        assert stats["n_utts_dropped"] == 3
        assert all(f == DROPPED_LABEL for f in filtered)


class TestClusterEmbeddingsLargeScale:
    def _make_two_clusters(self, n_per: int = 50, dim: int = 64) -> np.ndarray:
        rng = np.random.RandomState(42)
        a = rng.randn(n_per, dim).astype(np.float32) + 5.0
        b = rng.randn(n_per, dim).astype(np.float32) - 5.0
        return np.vstack([a, b])

    def test_two_clusters_found(self) -> None:
        embs = self._make_two_clusters()
        labels, confidence, stats = cluster_embeddings_large_scale(
            embs, threshold=0.3, min_cluster_size=1,
        )

        assert labels.shape == (100,)
        kept = labels[labels != DROPPED_LABEL]
        assert len(set(kept)) == 2

    def test_confidence_shape(self) -> None:
        embs = self._make_two_clusters()
        labels, confidence, _ = cluster_embeddings_large_scale(
            embs, threshold=0.3, min_cluster_size=1,
        )

        assert confidence is not None
        assert confidence.shape == (100,)
        assert confidence.min() >= 0.0
        assert confidence.max() <= 1.0

    def test_no_confidence_when_disabled(self) -> None:
        embs = self._make_two_clusters()
        _, confidence, _ = cluster_embeddings_large_scale(
            embs, threshold=0.3, min_cluster_size=1, compute_confidence=False,
        )
        assert confidence is None

    def test_empty_input(self) -> None:
        labels, confidence, stats = cluster_embeddings_large_scale(
            np.empty((0, 64), dtype=np.float32),
        )
        assert len(labels) == 0
        assert stats["n_input"] == 0

    def test_single_input(self) -> None:
        labels, confidence, stats = cluster_embeddings_large_scale(
            np.random.randn(1, 64).astype(np.float32), min_cluster_size=1,
        )
        assert len(labels) == 1
        assert stats["n_input"] == 1

    def test_min_cluster_size_filters(self) -> None:
        embs = self._make_two_clusters(n_per=50)
        labels, _, stats = cluster_embeddings_large_scale(
            embs, threshold=0.3, min_cluster_size=100,
        )
        assert (labels == DROPPED_LABEL).sum() > 0

    def test_stats_keys(self) -> None:
        embs = self._make_two_clusters(n_per=20)
        _, _, stats = cluster_embeddings_large_scale(
            embs, threshold=0.3, min_cluster_size=1,
        )
        assert "n_input" in stats
        assert "n_leaf_subclusters" in stats
        assert "filter" in stats
        assert "n_clusters_raw" in stats
