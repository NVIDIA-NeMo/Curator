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

"""Unit tests for speaker clustering and confidence scoring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nemo_curator.stages.audio.speaker_id.speaker_clustering_and_scoring import (
    SpeakerClusteringStage,
    _cosine_similarity_matrix,
    _l2_normalize,
    cluster_embeddings,
    cluster_stats,
    normalize_embeddings_for_clustering,
    speaker_confidence,
)


class TestL2Normalize:
    def test_unit_length_output(self) -> None:
        x = np.random.randn(10, 192).astype(np.float32)
        normed = _l2_normalize(x)
        norms = np.linalg.norm(normed, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_zero_vector_handled(self) -> None:
        x = np.zeros((1, 4), dtype=np.float32)
        normed = _l2_normalize(x)
        assert not np.any(np.isnan(normed))


class TestCosineSimilarityMatrix:
    def test_self_similarity_is_one(self) -> None:
        x = np.random.randn(5, 64).astype(np.float32)
        sim = _cosine_similarity_matrix(x)
        np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-5)

    def test_symmetric(self) -> None:
        x = np.random.randn(5, 64).astype(np.float32)
        sim = _cosine_similarity_matrix(x)
        np.testing.assert_allclose(sim, sim.T, atol=1e-6)


class TestNormalizeEmbeddings:
    def test_center_global_zero_mean(self) -> None:
        x = np.random.randn(20, 8).astype(np.float32)
        result = normalize_embeddings_for_clustering(x, mode="center_global")
        np.testing.assert_allclose(result.mean(axis=0), 0.0, atol=1e-5)

    def test_none_passthrough(self) -> None:
        x = np.random.randn(5, 8).astype(np.float32)
        result = normalize_embeddings_for_clustering(x, mode="none")
        np.testing.assert_allclose(result, x, atol=1e-5)

    def test_external_with_mean(self, tmp_path: Path) -> None:
        x = np.ones((3, 4), dtype=np.float32) * 5.0
        mean_path = str(tmp_path / "mean.npy")
        np.save(mean_path, np.ones(4, dtype=np.float32) * 2.0)

        result = normalize_embeddings_for_clustering(
            x, mode="external", external_mean_npy=mean_path,
        )
        np.testing.assert_allclose(result, 3.0, atol=1e-5)

    def test_external_missing_file_raises(self) -> None:
        x = np.ones((3, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="external_mean_npy"):
            normalize_embeddings_for_clustering(x, mode="external")

    def test_invalid_mode_raises(self) -> None:
        x = np.ones((3, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown"):
            normalize_embeddings_for_clustering(x, mode="bad_mode")


class TestClusterEmbeddings:
    def test_two_distinct_clusters(self) -> None:
        rng = np.random.RandomState(42)
        cluster_a = rng.randn(10, 64).astype(np.float32) + 5.0
        cluster_b = rng.randn(10, 64).astype(np.float32) - 5.0
        embeddings = np.vstack([cluster_a, cluster_b])

        labels = cluster_embeddings(embeddings, threshold=0.3)

        assert len(set(labels[:10])) == 1
        assert len(set(labels[10:])) == 1
        assert labels[0] != labels[10]

    def test_single_embedding(self) -> None:
        labels = cluster_embeddings(np.random.randn(1, 64).astype(np.float32))
        assert len(labels) == 1
        assert labels[0] == 1

    def test_empty_returns_empty(self) -> None:
        labels = cluster_embeddings(np.empty((0, 64), dtype=np.float32))
        assert len(labels) == 0


class TestClusterStats:
    def test_basic_stats(self) -> None:
        labels = np.array([1, 1, 1, 2, 2, 3])
        stats = cluster_stats(labels)
        assert stats["num_clusters"] == 3
        assert stats["largest_cluster"] == 3
        assert stats["smallest_cluster"] == 1
        assert stats["singletons"] == 1


class TestSpeakerConfidence:
    def test_well_separated_clusters_high_confidence(self) -> None:
        rng = np.random.RandomState(42)
        a = rng.randn(5, 64).astype(np.float32) + 10.0
        b = rng.randn(5, 64).astype(np.float32) - 10.0
        embeddings = np.vstack([a, b])
        labels = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

        scores = speaker_confidence(embeddings, labels)

        assert scores.shape == (10,)
        assert all(s > 0.5 for s in scores)

    def test_singleton_gets_zero(self) -> None:
        embeddings = np.random.randn(3, 64).astype(np.float32)
        labels = np.array([1, 1, 2])  # cluster 2 is singleton
        scores = speaker_confidence(embeddings, labels)
        assert scores[2] == 0.0


class TestSpeakerClusteringStage:
    def _setup_shard(
        self, tmp_path: Path, n: int = 20, dim: int = 64,
    ) -> tuple[str, str]:
        """Create a manifest + matching embedding NPZ for testing."""
        rng = np.random.RandomState(123)
        manifest_dir = tmp_path / "manifests"
        manifest_dir.mkdir()
        emb_dir = tmp_path / "embeddings"
        emb_dir.mkdir()

        entries = []
        cut_ids = []
        for i in range(n):
            afp = f"audio_{i:04d}.wav"
            entries.append({"audio_filepath": afp, "text": f"utterance {i}"})
            cut_ids.append(afp)

        manifest_path = str(manifest_dir / "manifest_0.json")
        with open(manifest_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        embeddings = rng.randn(n, dim).astype(np.float32)
        np.savez(
            str(emb_dir / "embeddings_0.npz"),
            cut_ids=np.array(cut_ids, dtype=object),
            embeddings=embeddings,
        )

        return manifest_path, str(emb_dir)

    def test_shard_level_clustering(self, tmp_path: Path) -> None:
        manifest_path, emb_dir = self._setup_shard(tmp_path)
        output_dir = str(tmp_path / "output")

        stage = SpeakerClusteringStage(
            input_manifest=manifest_path,
            embedding_dir=emb_dir,
            output_manifest_dir=output_dir,
            shard_level_clustering=True,
            show_progress=False,
        )
        stage.setup()
        stage.process(None)

        out_file = Path(output_dir) / "manifest_0.json"
        assert out_file.exists()

        with open(out_file) as f:
            lines = [json.loads(l) for l in f if l.strip()]

        assert len(lines) == 20
        assert all("speaker_label" in l for l in lines)
        assert all("confidence_score" in l for l in lines)
        assert all(l["speaker_label"] >= 1 for l in lines)

    def test_global_clustering(self, tmp_path: Path) -> None:
        manifest_path, emb_dir = self._setup_shard(tmp_path)
        output_dir = str(tmp_path / "output")

        stage = SpeakerClusteringStage(
            input_manifest=manifest_path,
            embedding_dir=emb_dir,
            output_manifest_dir=output_dir,
            shard_level_clustering=False,
            show_progress=False,
        )
        stage.setup()
        stage.process(None)

        out_file = Path(output_dir) / "manifest_0.json"
        assert out_file.exists()

    def test_missing_embedding_raises(self, tmp_path: Path) -> None:
        manifest_dir = tmp_path / "manifests"
        manifest_dir.mkdir()
        manifest_path = str(manifest_dir / "manifest_0.json")
        with open(manifest_path, "w") as f:
            f.write(json.dumps({"audio_filepath": "a.wav"}) + "\n")

        stage = SpeakerClusteringStage(
            input_manifest=manifest_path,
            embedding_dir=str(tmp_path / "empty_embs"),
            output_manifest_dir=str(tmp_path / "output"),
            show_progress=False,
        )
        stage.setup()
        with pytest.raises(FileNotFoundError):
            stage.process(None)
