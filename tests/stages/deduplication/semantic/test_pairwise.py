# modality: text

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

from contextlib import suppress
from pathlib import Path
from unittest.mock import patch

# Suppress GPU-related import errors when running pytest -m "not gpu"
with suppress(ImportError):
    import cudf
    import cupy as cp

import numpy as np
import pytest
import torch

# Suppress GPU-related import errors when running pytest -m "not gpu"
with suppress(ImportError):
    from nemo_curator.stages.deduplication.semantic.pairwise import (
        PairwiseCosineSimilarityStage,
        PairwiseStage,
        pairwise_cosine_similarity_batched,
    )
    from nemo_curator.stages.deduplication.semantic.pairwise_io import ClusterWiseFilePartitioningStage
    from nemo_curator.stages.deduplication.semantic.ranking import RankingStrategy
    from nemo_curator.stages.text.embedders.utils import create_list_series_from_1d_or_2d_ar
    from nemo_curator.tasks import FileGroupTask


@pytest.mark.gpu
class TestPairwiseCosineSimilarityBatched:
    """Test cases for pairwise_cosine_similarity_batched function."""

    def setup_method(self) -> None:
        """Setup test data similar to test_semdedup.py."""
        # Create a 6x3 array where each row is a unit vector
        # The second and last two rows are the same
        input_embeddings = torch.tensor(
            np.asarray(
                [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [1, 2, 3], [1, 2, 3]],
            ),
            dtype=torch.float32,
        )
        # Normalize the input array
        self.input_embeddings = input_embeddings / torch.norm(input_embeddings, dim=1, keepdim=True)
        self.expected_pairwise_similarity = np.array([0.0000, 0.974631, 0.998190, 0.999618, 1.0000, 1.0000])
        self.expected_indices = np.array([0, 0, 1, 2, 0, 0])

    @pytest.mark.parametrize(
        ("input_dtype", "compute_dtype", "batch_size"),
        [
            (torch.float32, "auto", 1),
            (torch.float32, "float16", 2),
            (torch.float16, "auto", 6),
            (torch.float16, "float16", 3),
        ],
    )
    def test_pairwise_cosine_similarity_batched(
        self,
        input_dtype: torch.dtype,
        compute_dtype: str,
        batch_size: int,
    ) -> None:
        """Fixed batches preserve compute precision and earlier-rank behavior."""
        max_similarity, max_indices = pairwise_cosine_similarity_batched(
            self.input_embeddings.to(input_dtype), batch_size, compute_dtype
        )
        is_fp16 = input_dtype == torch.float16 or compute_dtype == "float16"
        tolerance = 2e-3 if is_fp16 else 1e-6
        np.testing.assert_allclose(
            max_similarity.tolist(),
            self.expected_pairwise_similarity,
            rtol=tolerance,
            atol=tolerance,
        )
        np.testing.assert_array_equal(max_indices.tolist(), self.expected_indices)
        assert max_similarity.dtype == (cp.float16 if is_fp16 else cp.float32)

    def test_rejects_upcasting_fp16_embeddings(self) -> None:
        with pytest.raises(ValueError, match="float16 embeddings"):
            pairwise_cosine_similarity_batched(self.input_embeddings.to(torch.float16), compute_dtype="float32")

    @pytest.mark.parametrize("batch_size", [7, 64])
    def test_pairwise_cosine_similarity_batched_rand_array(self, batch_size: int) -> None:
        """Test fixed batching against a full-matrix reference."""
        torch.manual_seed(42)
        n, d = 64, 32
        rand_arr = torch.randn(n, d, device="cuda")
        reference = rand_arr @ rand_arr.T
        valid_neighbors = torch.ones((n, n), dtype=torch.bool, device="cuda").tril_(diagonal=-1)
        reference.masked_fill_(~valid_neighbors, -torch.inf)
        expected_similarity, expected_indices = torch.max(reference, dim=1)
        expected_similarity[0] = 0.0
        expected_indices[0] = 0

        max_similarity, max_indices = pairwise_cosine_similarity_batched(rand_arr, batch_size=batch_size)

        np.testing.assert_allclose(max_similarity.tolist(), expected_similarity.tolist(), rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(max_indices.tolist(), expected_indices.tolist())

    def test_fp16_and_fp32_agree_on_non_tie_neighbor_ids(self) -> None:
        """FP16 and FP32 select the same unique neighbors for a four-row chain.

        A points along x; B is closest to A; C is closest to B; and D is
        closest to C. The expected earlier-neighbor indices are [0, 0, 1, 2].
        """
        embeddings = torch.tensor(
            [[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.1, 0.9, 0.4], [-0.2, 0.1, 0.97]],
            device="cuda",
        )
        embeddings = embeddings / torch.linalg.vector_norm(embeddings, dim=1, keepdim=True)

        fp16_scores, fp16_indices = pairwise_cosine_similarity_batched(
            embeddings.to(torch.float16), 2, compute_dtype="float16"
        )
        fp32_scores, fp32_indices = pairwise_cosine_similarity_batched(embeddings, 2)

        assert fp16_scores.dtype == cp.float16
        assert fp32_scores.dtype == cp.float32
        np.testing.assert_array_equal(fp16_indices.tolist(), fp32_indices.tolist())
        np.testing.assert_allclose(fp16_scores.tolist(), fp32_scores.tolist(), rtol=2e-3, atol=2e-3)

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_negative_similarity_is_not_replaced_by_masked_zero(self, batch_size: int) -> None:
        """Masking future rows must not beat a valid negative earlier-row similarity."""
        embeddings = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32)

        max_similarity, max_indices = pairwise_cosine_similarity_batched(embeddings, batch_size)

        np.testing.assert_array_equal(max_indices.tolist(), [0, 0])
        np.testing.assert_allclose(max_similarity.tolist(), [0.0, -1.0])


@pytest.mark.gpu
class TestPairwiseCosineSimilarityStage:
    """Test cases for PairwiseCosineSimilarityStage."""

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"pairwise_batch_size": 0}, "positive integer"),
            ({"pairwise_batch_size": True}, "positive integer"),
            ({"compute_dtype": "bfloat16"}, "Unsupported compute_dtype"),
        ],
    )
    def test_rejects_invalid_stage_settings(self, kwargs: dict[str, object], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            PairwiseCosineSimilarityStage(
                id_field="id",
                embedding_field="embedding",
                output_path="/unused",
                ranking_strategy=RankingStrategy.random(),
                **kwargs,
            )

    def test_process_batch_sums_cluster_timings(self) -> None:
        stage = PairwiseCosineSimilarityStage(
            id_field="id",
            embedding_field="embedding",
            output_path="/unused",
            ranking_strategy=RankingStrategy.random(),
        )
        tasks = [
            FileGroupTask(dataset_name="test", data=["unused"], _metadata={"centroid_id": centroid_id})
            for centroid_id in range(2)
        ]
        durations = iter([1.0, 2.0])

        def process(task: FileGroupTask) -> FileGroupTask:
            stage._log_metric("pairwise_read_time", next(durations))
            return task

        with patch.object(stage, "process", side_effect=process):
            assert stage.process_batch(tasks) == tasks

        assert stage._consume_custom_metrics() == {"pairwise_read_time": 3.0}

    def test_single_item_cluster(self, tmp_path: Path) -> None:
        """Test processing a cluster with a single item."""
        # Create test data with single embedding
        test_data = cudf.DataFrame({"id": [1], "embedding": [[0.1, 0.2, 0.3]]})

        # Save to parquet file
        input_file = tmp_path / "single_item.parquet"
        test_data.to_parquet(input_file)

        # Create output directory
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create stage with default ranking strategy
        ranking_strategy = RankingStrategy(metadata_cols=[], strategy="random")
        stage = PairwiseCosineSimilarityStage(
            id_field="id",
            embedding_field="embedding",
            output_path=str(output_dir),
            ranking_strategy=ranking_strategy,
            read_kwargs={},
            write_kwargs={},
        )
        stage.setup()

        # Create task
        task = FileGroupTask(
            dataset_name="test",
            data=[str(input_file)],
            _metadata={"centroid_id": 0, "filetype": "parquet"},
        )

        # Process task
        result = stage.process(task)

        # Verify result
        assert isinstance(result, FileGroupTask)
        assert result._metadata["centroid_id"] == 0
        assert len(result.data) == 1

        # Check output file
        output_file = output_dir / "cluster_0.parquet"
        assert output_file.exists()

        # Read and verify output
        result_df = cudf.read_parquet(output_file)
        assert len(result_df) == 1
        assert "id" in result_df.columns
        assert "max_id" in result_df.columns
        assert "cosine_sim_score" in result_df.columns
        assert result_df["cosine_sim_score"].iloc[0] == 0.0
        assert result_df["cosine_sim_score"].dtype == np.dtype("float32")
        with patch("nemo_curator.stages.deduplication.semantic.pairwise._release_cached_memory") as release:
            stage.teardown()
        release.assert_called_once_with()

    @pytest.mark.parametrize(
        ("storage_dtype", "compute_dtype", "batch_size"),
        [
            ("float16", "auto", 2),
            ("float32", "auto", 8),
            ("float32", "float16", 2),
        ],
    )
    def test_multi_item_cluster(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        storage_dtype: str,
        compute_dtype: str,
        batch_size: int,
    ) -> None:
        """Exercise storage/compute precision, file groups, ranking, ties, and metrics together."""
        embeddings = cp.asarray(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.8, 0.6, 0.0]],
            dtype=cp.float32,
        )
        if storage_dtype == "float16":
            embeddings = embeddings.astype(cp.float16).view(cp.uint16)

        input_files = []
        for file_index, row_slice in enumerate((slice(0, 2), slice(2, 4))):
            frame = cudf.DataFrame(
                {
                    "document_key": [30, 10, 40, 20][row_slice],
                    "quality_rank": [3, 1, 4, 2][row_slice],
                }
            )
            frame["embedding"] = create_list_series_from_1d_or_2d_ar(embeddings[row_slice], index=frame.index)
            input_file = tmp_path / f"multi_item_{file_index}.parquet"
            frame.to_parquet(input_file)
            input_files.append(str(input_file))

        monkeypatch.setattr(
            "nemo_curator.stages.deduplication.semantic.pairwise.break_parquet_partition_into_groups",
            lambda _: [[path] for path in input_files],
        )
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        stage = PairwiseCosineSimilarityStage(
            id_field="document_key",
            embedding_field="embedding",
            output_path=str(output_dir),
            ranking_strategy=RankingStrategy.metadata_based(["quality_rank"], ascending=True),
            pairwise_batch_size=batch_size,
            compute_dtype=compute_dtype,
        )

        stage.process(
            FileGroupTask(
                dataset_name="test",
                data=input_files,
                _metadata={"centroid_id": 7, "filetype": "parquet"},
            )
        )

        result_df = cudf.read_parquet(output_dir / "cluster_7.parquet")
        assert result_df["id"].to_arrow().to_pylist() == [10, 20, 30, 40]
        assert result_df["max_id"].to_arrow().to_pylist() == [10, 10, 20, 10]
        assert result_df["cosine_sim_score"].dtype == np.dtype("float32")
        tolerance = 2e-3 if storage_dtype == "float16" or compute_dtype == "float16" else 1e-6
        np.testing.assert_allclose(
            result_df["cosine_sim_score"].to_numpy(),
            [0.0, 0.8, 0.6, 0.0],
            rtol=tolerance,
            atol=tolerance,
        )

        metrics = stage._consume_custom_metrics()
        assert metrics.keys() == {
            "pairwise_footer_scan_time",
            "pairwise_read_time",
            "pairwise_rank_time",
            "pairwise_conversion_time",
            "pairwise_compute_time",
            "pairwise_write_time",
        }
        assert all(metrics[name] >= 0 for name in metrics if name.endswith("_time"))

    def test_pairwise_stage_with_custom_metadata_ranking(self, tmp_path: Path) -> None:
        """Test PairwiseCosineSimilarityStage with custom metadata-based ranking."""
        # Create test data with custom metadata
        embeddings = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
        embeddings_tensor = embeddings_tensor / torch.norm(embeddings_tensor, dim=1, keepdim=True)
        embeddings_normalized = embeddings_tensor.tolist()

        test_data = cudf.DataFrame(
            {
                "id": [1, 2, 3],
                "embedding": embeddings_normalized,
                "priority": [3, 1, 2],  # Custom ranking column
                "score": [0.9, 0.5, 0.7],
            }
        )

        # Save to parquet file
        input_file = tmp_path / "custom_ranked_data.parquet"
        test_data.to_parquet(input_file)

        # Create output directory
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create custom ranking strategy - sort by priority ascending, then score descending
        ranking_strategy = RankingStrategy.metadata_based(metadata_cols=["priority", "score"], ascending=[True, False])

        # Create stage with custom ranking
        stage = PairwiseCosineSimilarityStage(
            id_field="id",
            embedding_field="embedding",
            output_path=str(output_dir),
            ranking_strategy=ranking_strategy,
            read_kwargs={},
            write_kwargs={},
        )
        stage.setup()

        # Create task
        task = FileGroupTask(
            dataset_name="test",
            data=[str(input_file)],
            _metadata={"centroid_id": 1, "filetype": "parquet"},
        )

        # Process task
        _ = stage.process(task)

        # Verify result
        output_file = output_dir / "cluster_1.parquet"
        assert output_file.exists()

        # Read and verify output - should be ranked by priority asc, then score desc
        result_df = cudf.read_parquet(output_file)
        assert len(result_df) == 3
        assert result_df["id"].to_arrow().to_pylist() == [2, 3, 1]

    def test_pairwise_stage_ranking_fails_on_missing_columns(self, tmp_path: Path) -> None:
        """Test that ranking fails when required columns are missing."""
        # Create test data without distance columns
        embeddings = [[1.0, 0.0], [0.0, 1.0]]
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
        embeddings_tensor = embeddings_tensor / torch.norm(embeddings_tensor, dim=1, keepdim=True)
        embeddings_normalized = embeddings_tensor.tolist()

        test_data = cudf.DataFrame(
            {
                "id": [1, 2],
                "embedding": embeddings_normalized,
                # No distance columns
            }
        )

        # Save to parquet file
        input_file = tmp_path / "no_distance_data.parquet"
        test_data.to_parquet(input_file)

        # Create output directory
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create distance-based ranking strategy (will fail due to missing columns)
        ranking_strategy = RankingStrategy.metadata_based(metadata_cols=["cosine_dist_to_cent"], ascending=False)

        # Create stage
        stage = PairwiseCosineSimilarityStage(
            id_field="id",
            embedding_field="embedding",
            output_path=str(output_dir),
            ranking_strategy=ranking_strategy,
            read_kwargs={},
            write_kwargs={},
        )
        stage.setup()

        # Create task
        task = FileGroupTask(
            dataset_name="test",
            data=[str(input_file)],
            _metadata={"centroid_id": 2, "filetype": "parquet"},
        )

        # Process task - should fail due to missing distance columns
        with (
            patch("nemo_curator.stages.deduplication.semantic.pairwise._release_cached_memory") as release,
            pytest.raises(KeyError, match="cosine_dist_to_cent"),
        ):
            stage.process(task)
        release.assert_called_once_with()


@pytest.mark.gpu
class TestPairwiseStage:
    """Test cases for PairwiseStage composite stage."""

    def setup_method(self) -> None:
        # We create a 6x3 array where each row is a unit vector
        # The second and last two rows are the same
        input_embeddings = torch.tensor(
            np.asarray(
                [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [1, 2, 3], [1, 2, 3]],
            ),
            dtype=torch.float32,
        )
        # Normalize the input array
        self.input_embeddings = input_embeddings / torch.norm(input_embeddings, dim=1, keepdim=True)
        self.expected_pairwise_similarity = np.array([0.0000, 0.974631, 0.998190, 0.999618, 1.0000, 1.0000])
        self.expected_indices = np.array([0, 0, 1, 2, 0, 0])

    def test_decompose(self) -> None:
        """Test that PairwiseStage correctly decomposes into constituent stages."""
        stage = PairwiseStage(
            id_field="id",
            embedding_field="embedding",
            input_path="/input/path",
            output_path="/output/path",
            pairwise_batch_size=1024,
            verbose=True,
        )

        stages = stage.decompose()

        # Should return exactly 2 stages
        assert len(stages) == 2

        # First stage should be ClusterWiseFilePartitioningStage
        assert isinstance(stages[0], ClusterWiseFilePartitioningStage)
        assert stages[0].input_path == "/input/path"

        # Second stage should be PairwiseCosineSimilarityStage
        assert isinstance(stages[1], PairwiseCosineSimilarityStage)
        assert stages[1].id_field == "id"
        assert stages[1].embedding_field == "embedding"
        assert stages[1].output_path == "/output/path"
        assert stages[1].pairwise_batch_size == 1024
        assert stages[1].verbose is True

    @pytest.mark.skip(reason="This test needs to be debugged")
    def test_stage_with_kwargs(self) -> None:
        """Test PairwiseStage with read_kwargs and write_kwargs."""
        read_kwargs = {"storage_options": {"key": "value"}}
        write_kwargs = {"storage_options": {"write_key": "write_value"}, "compression": "gzip"}

        stage = PairwiseStage(
            id_field="doc_id",
            embedding_field="embeddings",
            input_path="/test/input",
            output_path="/test/output",
            read_kwargs=read_kwargs,
            write_kwargs=write_kwargs,
        )

        stages = stage.decompose()
        assert len(stages) == 2

        # Check that ClusterWiseFilePartitioningStage gets storage_options from read_kwargs
        partitioning_stage = stages[0]
        assert partitioning_stage.storage_options == {"key": "value"}

        similarity_stage = stages[1]
        assert similarity_stage.input_storage_options == {"key": "value"}
        assert "storage_options" not in similarity_stage.read_kwargs
        assert similarity_stage.output_storage_options == {"write_key": "write_value"}
        assert "storage_options" not in similarity_stage.write_kwargs
        assert similarity_stage.write_kwargs == {"compression": "gzip"}

        """Test PairwiseStage with hard ranking (farthest first)."""
        stage = PairwiseStage(
            id_field="doc_id",
            embedding_field="embeddings",
            input_path="/test/input",
            output_path="/test/output",
            which_to_keep="hard",
            sim_metric="cosine",
        )

        stages = stage.decompose()
        similarity_stage = stages[1]
        ranking_strategy = similarity_stage.ranking_strategy

        assert ranking_strategy.metadata_cols == ["cosine_dist_to_cent", "doc_id"]
        assert ranking_strategy.ascending == [False, False]  # "hard" means descending (farthest first)

    def _setup_test_data(self, tmp_path: Path) -> tuple[Path, Path, "cudf.DataFrame", Path]:
        """Helper method to set up common test data for all ranking tests."""
        cluster_id = 0

        # Create base embeddings dataframe
        embeddings_df = cudf.DataFrame(
            {
                "embedding": self.input_embeddings.tolist(),
                "id": list(range(self.input_embeddings.shape[0])),
                "nearest_cent": [cluster_id] * self.input_embeddings.shape[0],
            }
        )

        # Compute actual distances to centroid (for distance-based ranking)
        centroid = self.input_embeddings[:1]  # Shape: (1, 3)
        embeddings_array = cp.asarray(self.input_embeddings)
        centroid_array = cp.asarray(centroid)

        # L2 distances
        l2_distances = cp.sqrt(cp.sum((embeddings_array - centroid_array) ** 2, axis=1))
        embeddings_df["l2_dist_to_cent"] = l2_distances

        # Cosine distances (1 - cosine_similarity)
        centroid_normalized = centroid_array / cp.linalg.norm(centroid_array, axis=1, keepdims=True)
        cosine_similarities = cp.sum(embeddings_array * centroid_normalized, axis=1)
        cosine_distances = 1 - cosine_similarities
        embeddings_df["cosine_dist_to_cent"] = cosine_distances

        # Add custom metadata columns for metadata-based ranking tests
        embeddings_df["custom_score"] = [0.1, 0.8, 0.3, 0.9, 0.2, 0.15]  # Custom ranking scores
        embeddings_df["priority"] = [3, 1, 2, 1, 3, 2]  # Priority levels (1=high, 3=low)

        # Save to two parquet files
        input_files = [tmp_path / f"test_cluster_{i}.parquet" for i in range(1, 3)]
        embeddings_df.iloc[:3].to_parquet(input_files[0])
        embeddings_df.iloc[3:].to_parquet(input_files[1])

        # Create output directory
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        return input_files[0], input_files[1], embeddings_df, output_dir

    def _run_pairwise_stage_test(self, tmp_path: Path, ranking_kwargs: dict) -> tuple[list, list, list]:
        """Helper method to run pairwise stage test and return results."""
        input_file_1, input_file_2, embeddings_df, output_dir = self._setup_test_data(tmp_path)
        input_files = [input_file_1, input_file_2]

        with patch(
            "nemo_curator.stages.deduplication.semantic.pairwise.break_parquet_partition_into_groups"
        ) as mock_break_into_groups:
            # Mock break_parquet_partition_into_groups to return one file per group
            mock_break_into_groups.return_value = [[str(input_files[0])], [str(input_files[1])]]

            # Create the composite PairwiseStage
            composite_stage = PairwiseStage(
                id_field="id",
                embedding_field="embedding",
                input_path=str(tmp_path),
                output_path=str(output_dir),
                **ranking_kwargs,
            )

            # Decompose and verify stage structure
            stages = composite_stage.decompose()
            assert len(stages) == 2
            assert isinstance(stages[0], ClusterWiseFilePartitioningStage)
            assert isinstance(stages[1], PairwiseCosineSimilarityStage)

            # Setup and run the similarity stage
            similarity_stage = stages[1]
            similarity_stage.setup()

            # Create task
            task = FileGroupTask(
                dataset_name="test",
                data=[str(input_file) for input_file in input_files],
                _metadata={"centroid_id": 0, "filetype": "parquet"},
            )

            # Process task
            result = similarity_stage.process(task)

            # Verify result structure
            assert isinstance(result, FileGroupTask)
            output_file = output_dir / "cluster_0.parquet"
            assert output_file.exists()

            # Read and verify output
            result_df = cudf.read_parquet(output_file)
            assert len(result_df) == len(embeddings_df)

            # Check columns
            expected_columns = {"id", "max_id", "cosine_sim_score"}
            assert set(result_df.columns) == expected_columns

            # Extract results for validation
            result_ids = result_df["id"].to_arrow().to_pylist()
            result_max_ids = result_df["max_id"].to_arrow().to_pylist()
            result_cosine_sim_scores = result_df["cosine_sim_score"].to_arrow().to_pylist()

            return result_ids, result_max_ids, result_cosine_sim_scores

    @pytest.mark.parametrize(
        ("which_to_keep", "sim_metric"),
        [
            ("hard", "cosine"),
            ("easy", "cosine"),
            ("hard", "l2"),
            ("easy", "l2"),
        ],
    )
    def test_distance_based_ranking_workflow(self, which_to_keep: str, sim_metric: str, tmp_path: Path) -> None:
        """Test distance-based ranking strategies (original semantic dedup behavior)."""
        # Setup ranking kwargs
        ranking_kwargs = {
            "which_to_keep": which_to_keep,
            "sim_metric": sim_metric,
        }

        # Run test and get results
        result_ids, result_max_ids, result_cosine_sim_scores = self._run_pairwise_stage_test(tmp_path, ranking_kwargs)

        # Verify output based on ranking configuration
        if which_to_keep == "hard":
            expected_ids = [3, 2, 1, 5, 4, 0]
            expected_max_ids = [3, 3, 2, 1, 5, 5]
            expected_cosine_sim_scores = np.array(
                [0.0000, 0.99961, 0.99819, 0.974631, 1.0000, 1.0000], dtype=np.float32
            )
        else:  # easy
            expected_ids = [0, 4, 5, 1, 2, 3]
            expected_max_ids = [0, 0, 0, 0, 1, 2]
            expected_cosine_sim_scores = np.array(
                [0.0000, 1.0000, 1.0000, 0.97464, 0.99819, 0.999618], dtype=np.float32
            )

        # Check exact output values
        assert result_ids == expected_ids, f"Expected IDs {expected_ids}, got {result_ids}"
        assert result_max_ids == expected_max_ids, f"Expected max IDs {expected_max_ids}, got {result_max_ids}"
        np.testing.assert_allclose(
            result_cosine_sim_scores,
            expected_cosine_sim_scores,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Cosine similarity scores don't match for {which_to_keep} {sim_metric}",
        )

        # Common validations
        self._validate_common_results(result_ids, result_max_ids, result_cosine_sim_scores)

    @pytest.mark.parametrize(
        ("columns", "ascending"),
        [
            (["custom_score"], False),
            (["priority", "custom_score"], [True, False]),
        ],
    )
    def test_metadata_based_ranking_workflow(
        self, columns: list[str], ascending: bool | list[bool], tmp_path: Path
    ) -> None:
        """Test custom metadata-based ranking strategies."""
        # Create custom ranking strategy
        custom_ranking = RankingStrategy.metadata_based(
            metadata_cols=columns,
            ascending=ascending,
        )
        ranking_kwargs = {"ranking_strategy": custom_ranking}

        # Run test and get results
        result_ids, result_max_ids, result_cosine_sim_scores = self._run_pairwise_stage_test(tmp_path, ranking_kwargs)

        # Verify the actual output matches expected ordering for metadata-based ranking
        if columns == ["custom_score"] and ascending is False:
            # Test data: custom_score values are [0.1, 0.8, 0.3, 0.9, 0.2, 0.15] for IDs [0,1,2,3,4,5]
            # Descending order by custom_score: 0.9(id=3), 0.8(id=1), 0.3(id=2), 0.2(id=4), 0.15(id=5), 0.1(id=0)
            expected_ids = [3, 1, 2, 4, 5, 0]
            assert result_ids == expected_ids, (
                f"Custom score descending ranking failed. Expected: {expected_ids}, got: {result_ids}"
            )
        elif columns == ["priority", "custom_score"] and ascending == [True, False]:
            # Test data:
            #   priority values: [3, 1, 2, 1, 3, 2] for IDs [0,1,2,3,4,5] (ascending)
            #   custom_score values: [0.1, 0.8, 0.3, 0.9, 0.2, 0.15] for IDs [0,1,2,3,4,5] (descending within priority)
            # Expected ordering:
            #   priority=1: IDs 1,3 -> sort by custom_score desc: 0.9(id=3), 0.8(id=1) -> [3, 1]
            #   priority=2: IDs 2,5 -> sort by custom_score desc: 0.3(id=2), 0.15(id=5) -> [2, 5]
            #   priority=3: IDs 0,4 -> sort by custom_score desc: 0.2(id=4), 0.1(id=0) -> [4, 0]
            # Final order: [3, 1, 2, 5, 4, 0]
            expected_ids = [3, 1, 2, 5, 4, 0]
            assert result_ids == expected_ids, (
                f"Priority + custom_score ranking failed. Expected: {expected_ids}, got: {result_ids}"
            )

        # Verify all IDs are present
        assert set(result_ids) == set(range(6)), "All IDs should be present in metadata ranking"

        # Common validations
        self._validate_common_results(result_ids, result_max_ids, result_cosine_sim_scores)

    def test_random_ranking_workflow(self, tmp_path: Path) -> None:
        """Test random ranking strategy."""
        # Create random ranking strategy
        random_ranking = RankingStrategy.random(random_seed=42)
        ranking_kwargs = {"ranking_strategy": random_ranking}

        # Run test and get results
        result_ids, result_max_ids, result_cosine_sim_scores = self._run_pairwise_stage_test(tmp_path, ranking_kwargs)

        # For random ranking, we can't predict exact order but verify all IDs are present
        assert set(result_ids) == set(range(6)), "All IDs should be present in random ranking"

        # Common validations
        self._validate_common_results(result_ids, result_max_ids, result_cosine_sim_scores)

    def _validate_common_results(self, result_ids: list, result_max_ids: list, result_cosine_sim_scores: list) -> None:
        """Helper method to perform common validations across all ranking types."""
        # Verify all similarity scores are valid (between 0 and 1)
        for score in result_cosine_sim_scores:
            assert 0.0 <= score <= 1.0, f"Cosine similarity score {score} should be between 0 and 1"

        # Verify all IDs are present
        assert set(result_ids) == set(range(6)), "All original IDs should be present in output"

        # Verify max_ids reference valid IDs
        for max_id in result_max_ids:
            assert max_id in result_ids, f"max_id {max_id} should reference a valid ID in the result"
