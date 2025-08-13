# ruff: noqa: E402

import os
from pathlib import Path

import pytest
import ray

cudf = pytest.importorskip("cudf", reason="MinHashStage tests require cudf")

from ray_curator.backends.experimental.ray_actor_pool import RayActorPoolExecutor
from ray_curator.pipeline import Pipeline
from ray_curator.stages.deduplication.fuzzy.lsh.stage import LSHStage
from ray_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
from ray_curator.tasks import FileGroupTask


@pytest.fixture(scope="module")
def ray_fixture():
    """Initialize and shutdown Ray for tests."""
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.mark.gpu
@pytest.mark.usefixtures("ray_fixture")
class TestLSHStage:
    @pytest.fixture(autouse=True)
    def minhash_data(self, tmp_path: Path) -> FileGroupTask:
        """Create test data with minhash signatures and save to parquet files."""
        df = cudf.DataFrame(
            {
                CURATOR_DEDUP_ID_STR: [1, 2, 3, 4, 5],
                "_minhash_signature": [
                    [1, 2, 1, 2, 1, 2],
                    [1, 2, 3, 4, 5, 6],
                    [3, 2, 1, 4, 5, 6],
                    [9, 8, 7, 6, 5, 4],
                    [3, 1, 2, 4, 5, 4],
                ],
            }
        )

        # Save to parquet file
        minhash_file = os.path.join(tmp_path, "minhash_data.parquet")
        df.to_parquet(minhash_file)

        # Create FileGroupTask
        return FileGroupTask(
            task_id="test_minhash_0",
            dataset_name="test_dataset",
            data=[minhash_file],
            _metadata={
                "partition_index": 0,
                "total_partitions": 1,
                "source_files": [minhash_file],
            },
        )

    @pytest.mark.parametrize("bands_per_iteration", [2, 3])
    def test_lsh(
        self,
        minhash_data: FileGroupTask,
        tmp_path: Path,
        bands_per_iteration: int,
    ) -> None:
        # Create LSHStage
        lsh_stage = LSHStage(
            output_dir=str(tmp_path / "lsh_output"),
            num_bands=3,
            minhashes_per_band=2,  # num_hashes=6 / num_buckets=3
            bands_per_iteration=bands_per_iteration,
            minhash_field="_minhash_signature",
            id_column=CURATOR_DEDUP_ID_STR,
        )

        # Create pipeline and executor
        pipeline = Pipeline(name="test_lsh", stages=[lsh_stage])
        executor = RayActorPoolExecutor()

        # Run the pipeline
        result_tasks = pipeline.run(executor, initial_tasks=[minhash_data])

        # Verify results
        assert len(result_tasks) >= 1  # Should have at least one output task

        # Read and verify the bucket data from all output files
        all_buckets = []
        for task in result_tasks:
            files = task.data if isinstance(task.data, list) else [task.data]
            for file in files:
                assert os.path.exists(file), f"File {file} does not exist"
                all_buckets.append(cudf.read_parquet(file))

        buckets_df = cudf.concat(all_buckets, ignore_index=True)

        grouped_docs = buckets_df[CURATOR_DEDUP_ID_STR].to_pandas().tolist()
        found_pairs = {tuple(sorted(group)) for group in grouped_docs}
        expected_pairs = {(1, 2), (2, 3), (4, 5)}
        assert found_pairs == expected_pairs, f"Expected pairs {expected_pairs} not found in {found_pairs}"

    def test_custom_column_names(
        self,
        tmp_path: Path,
    ) -> None:
        """Test LSH with custom ID and minhash column names."""
        # Create test data with custom column names
        df = cudf.DataFrame(
            {
                "document_id": [1, 2, 3, 4, 5],  # Custom ID column
                "signature": [  # Custom minhash column
                    [1, 2, 1, 2, 1, 2],
                    [1, 2, 3, 4, 5, 6],
                    [3, 2, 1, 4, 5, 6],
                    [9, 8, 7, 6, 5, 4],
                    [3, 1, 2, 4, 5, 4],
                ],
            }
        )

        # Save to parquet file
        minhash_file = os.path.join(tmp_path, "minhash_custom_cols.parquet")
        df.to_parquet(minhash_file)

        # Create FileGroupTask
        minhash_data = FileGroupTask(
            task_id="test_custom_cols_0",
            dataset_name="test_dataset",
            data=[minhash_file],
            _metadata={
                "partition_index": 0,
                "total_partitions": 1,
                "source_files": [minhash_file],
            },
        )

        # Create LSHStage with custom column names
        lsh_stage = LSHStage(
            output_dir=str(tmp_path / "lsh_custom_output"),
            num_bands=3,
            minhashes_per_band=2,
            bands_per_iteration=3,
            id_column="document_id",  # Custom ID column
            minhash_field="signature",  # Custom minhash column
        )

        # Create pipeline and executor
        pipeline = Pipeline(name="test_lsh_custom_cols", stages=[lsh_stage])
        executor = RayActorPoolExecutor()

        # Run the pipeline
        result_tasks = pipeline.run(executor, initial_tasks=[minhash_data])

        # Verify results
        assert len(result_tasks) >= 1

        # Read and verify the output contains custom column names
        all_buckets = []
        for task in result_tasks:
            files = task.data if isinstance(task.data, list) else [task.data]
            for file in files:
                assert os.path.exists(file), f"File {file} does not exist"
                all_buckets.append(cudf.read_parquet(file))

        buckets_df = cudf.concat(all_buckets, ignore_index=True)

        # Verify the custom ID column is present
        assert "document_id" in buckets_df.columns
        assert "_bucket_id" in buckets_df.columns
        assert len(buckets_df) == 3

    def test_no_duplicates(
        self,
        tmp_path: Path,
    ) -> None:
        # Create test data with no duplicates (unique minhashes)
        minhash_df = cudf.DataFrame(
            {
                CURATOR_DEDUP_ID_STR: [1, 2, 3, 4, 5],
                "_minhash_signature": [
                    [1, 2, 1, 2, 1],
                    [2, 3, 3, 4, 5],
                    [3, 4, 5, 5, 6],
                    [4, 8, 7, 6, 7],
                    [5, 10, 9, 7, 8],
                ],
            }
        )

        # Save to parquet file
        minhash_file = os.path.join(tmp_path, "minhash_no_dup.parquet")
        minhash_df.to_parquet(minhash_file)

        # Create FileGroupTask
        minhash_data = FileGroupTask(
            task_id="test_minhash_no_dup_0",
            dataset_name="test_dataset",
            data=[minhash_file],
            _metadata={
                "partition_index": 0,
                "total_partitions": 1,
                "source_files": [minhash_file],
            },
        )

        # Create LSHStage
        lsh_stage = LSHStage(
            output_dir=str(tmp_path / "lsh_no_dup_output"),
            num_bands=5,
            minhashes_per_band=1,  # num_hashes=5 / num_buckets=5
            bands_per_iteration=1,
            id_column=CURATOR_DEDUP_ID_STR,
            minhash_field="_minhash_signature",
        )

        # Create pipeline and executor
        pipeline = Pipeline(name="test_lsh_no_dup", stages=[lsh_stage])
        executor = RayActorPoolExecutor()

        # Run the pipeline
        result_tasks = pipeline.run(executor, initial_tasks=[minhash_data])

        total_bucket_entries = 0
        for task in result_tasks:
            files = task.data if isinstance(task.data, list) else [task.data]
            for file in files:
                assert os.path.exists(file), f"File {file} does not exist"
                df = cudf.read_parquet(file)
                total_bucket_entries += len(df)

        assert total_bucket_entries == 0, f"Expected no bucket entries, got {total_bucket_entries}"

    def test_partial_overlap(
        self,
        tmp_path: Path,
    ) -> None:
        # Create test data with partial overlaps
        minhash_df = cudf.DataFrame(
            {
                CURATOR_DEDUP_ID_STR: [1, 2, 3],
                "_minhash_signature": [
                    [1, 2, 1, 1, 1],
                    [2, 3, 1, 2, 2],
                    [3, 4, 2, 3, 1],
                ],
            }
        )

        # Save to parquet file
        minhash_file = os.path.join(tmp_path, "minhash_partial.parquet")
        minhash_df.to_parquet(minhash_file)

        # Create FileGroupTask
        minhash_data = FileGroupTask(
            task_id="test_minhash_partial_0",
            dataset_name="test_dataset",
            data=[minhash_file],
            _metadata={
                "partition_index": 0,
                "total_partitions": 1,
                "source_files": [minhash_file],
            },
        )

        # Create LSHStage
        lsh_stage = LSHStage(
            output_dir=str(tmp_path / "lsh_partial_output"),
            num_bands=5,
            minhashes_per_band=1,
            bands_per_iteration=1,
            id_column=CURATOR_DEDUP_ID_STR,
            minhash_field="_minhash_signature",
        )

        # Create pipeline and executor
        pipeline = Pipeline(name="test_lsh_partial", stages=[lsh_stage])
        executor = RayActorPoolExecutor()

        # Run the pipeline
        result_tasks = pipeline.run(executor, initial_tasks=[minhash_data])

        # Verify results
        assert len(result_tasks) >= 1

        # Collect all bucket data
        all_buckets = []
        for task in result_tasks:
            files = task.data if isinstance(task.data, list) else [task.data]
            for file in files:
                assert os.path.exists(file), f"File {file} does not exist"
                all_buckets.append(cudf.read_parquet(file))

        buckets_df = cudf.concat(all_buckets, ignore_index=True)
        assert len(buckets_df) == 2
        docs_list = buckets_df[CURATOR_DEDUP_ID_STR].to_pandas().tolist()
        found_pairs = {tuple(sorted(group)) for group in docs_list}
        expected_pairs = {(1, 2), (1, 3)}
        assert found_pairs == expected_pairs, f"Expected pairs {expected_pairs} not found in {found_pairs}"

    def test_actor_not_initialized(
        self,
        minhash_data: FileGroupTask,
        tmp_path: Path,
    ) -> None:
        """Test that proper error is raised when actor object is not initialized."""
        # Create LSHStage
        lsh_stage = LSHStage(
            output_dir=str(tmp_path / "lsh_no_actor"),
            num_bands=3,
            minhashes_per_band=2,
            bands_per_iteration=1,
            id_column=CURATOR_DEDUP_ID_STR,
            minhash_field="_minhash_signature",
        )

        # Try to call methods without actor object initialized
        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="Actor object not initialized"):
            lsh_stage.read_and_insert(minhash_data, (0, 1))

        with pytest.raises(RuntimeError, match="Actor object not initialized"):
            lsh_stage.insert_finished()

        with pytest.raises(RuntimeError, match="Actor object not initialized"):
            lsh_stage.extract_and_write()

        with pytest.raises(RuntimeError, match="Actor object not initialized"):
            lsh_stage.teardown()

        with pytest.raises(NotImplementedError, match="LSHProcessingStage does not support the process method"):
            lsh_stage.process(minhash_data)

        # Set actor object to wrong type
        lsh_stage._actor_obj = "not_an_actor"  # Wrong type
        with pytest.raises(RuntimeError, match="Actor object not initialized"):
            lsh_stage.read_and_insert(minhash_data, (0, 1))

    def test_invalid_init_args(
        self,
    ) -> None:
        """Test that proper errors are raised for invalid initialization arguments."""
        # Test invalid bands_per_iteration (less than 1)
        with pytest.raises(ValueError, match="Invalid bands_per_iteration"):
            LSHStage(
                num_bands=5,
                minhashes_per_band=2,
                bands_per_iteration=0,  # Invalid: less than 1
            )

        # Test invalid bands_per_iteration (greater than num_bands)
        with pytest.raises(ValueError, match="Invalid bands_per_iteration"):
            LSHStage(
                num_bands=5,
                minhashes_per_band=2,
                bands_per_iteration=10,  # Invalid: greater than num_bands
            )

        # Test negative bands_per_iteration
        with pytest.raises(ValueError, match="Invalid bands_per_iteration"):
            LSHStage(
                num_bands=5,
                minhashes_per_band=2,
                bands_per_iteration=-1,  # Invalid: negative
            )
