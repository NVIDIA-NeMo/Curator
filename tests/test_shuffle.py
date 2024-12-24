import dask.dataframe as dd
import pandas as pd
from dask.distributed import Client, LocalCluster

import nemo_curator as nc
from nemo_curator.datasets import DocumentDataset


def list_to_dataset(documents, col_name="text", npartitions=2):
    data = {col_name: documents}
    pdf = pd.DataFrame(data)

    return DocumentDataset(dd.from_pandas(pdf, npartitions=npartitions))


def all_equal(left_dataset, right_dataset):
    left_result = left_dataset.df.compute()
    right_result = right_dataset.df.compute()

    l_cols = set(left_result.columns)
    r_cols = set(right_result.columns)
    assert l_cols == r_cols
    for col in left_result.columns:
        left = left_result[col].reset_index(drop=True)
        right = right_result[col].reset_index(drop=True)
        assert all(left == right), f"Mismatch in {col} column.\n{left}\n{right}\n"


class TestShuffleNondeterministic:
    def test_shuffle(self):
        # Single threaded Dask is the only way to guarantee shuffle determinism
        # Docs: https://docs.dask.org/en/latest/generated/dask.dataframe.DataFrame.shuffle.html
        with LocalCluster(n_workers=1, threads_per_worker=1) as cluster:
            with Client(cluster):
                original_dataset = list_to_dataset(
                    ["one", "two", "three", "four", "five"]
                )
                expected_dataset = list_to_dataset(
                    ["two", "five", "three", "one", "four"]
                )
                shuffle = nc.Shuffle(seed=42)
                result_dataset = shuffle.shuffle_nondeterministic(original_dataset)
                all_equal(expected_dataset, result_dataset)

    def test_new_partitions(self):
        with LocalCluster(n_workers=1, threads_per_worker=1) as cluster:
            with Client(cluster):
                original_dataset = list_to_dataset(
                    ["one", "two", "three", "four", "five"], npartitions=3
                )
                expected_dataset = list_to_dataset(
                    ["two", "five", "three", "one", "four"], npartitions=3
                )
                shuffle = nc.Shuffle(seed=42, npartitions=2)
                result_dataset = shuffle.shuffle_nondeterministic(original_dataset)
                all_equal(expected_dataset, result_dataset)

    def test_filename(self):
        with LocalCluster(n_workers=1, threads_per_worker=1) as cluster:
            with Client(cluster):
                original_dataset = list_to_dataset(
                    ["one", "two", "three", "four", "five"], npartitions=1
                )
                original_dataset.df["file_name"] = "original.jsonl"

                expected_data = {
                    "text": ["one", "two", "three", "five", "four"],
                    "file_name": [
                        "file_0000000000.jsonl",
                        "file_0000000000.jsonl",
                        "file_0000000000.jsonl",
                        "file_0000000001.jsonl",
                        "file_0000000001.jsonl",
                    ],
                }
                pdf = pd.DataFrame(expected_data)
                expected_dataset = DocumentDataset(dd.from_pandas(pdf, npartitions=2))

                shuffle = nc.Shuffle(seed=42, npartitions=2)
                result_dataset = shuffle.shuffle_nondeterministic(original_dataset)
                all_equal(expected_dataset, result_dataset)

    def test_custom_filenames(self):
        with LocalCluster(n_workers=1, threads_per_worker=1) as cluster:
            with Client(cluster):
                original_dataset = list_to_dataset(
                    ["one", "two", "three", "four", "five"], npartitions=1
                )
                original_dataset.df["file_name"] = "original.jsonl"

                expected_data = {
                    "text": ["one", "two", "three", "five", "four"],
                    "file_name": [
                        "my_0.test",
                        "my_0.test",
                        "my_0.test",
                        "my_1.test",
                        "my_1.test",
                    ],
                }
                pdf = pd.DataFrame(expected_data)
                expected_dataset = DocumentDataset(dd.from_pandas(pdf, npartitions=2))

                def filename_fn(x):
                    return f"my_{x}.test"

                shuffle = nc.Shuffle(
                    seed=42, npartitions=2, partition_to_filename=filename_fn
                )
                result_dataset = shuffle.shuffle_nondeterministic(original_dataset)
                all_equal(expected_dataset, result_dataset)

    def test_shuffle_no_seed(self):
        original_dataset = list_to_dataset(["one", "two", "three", "four", "five"])
        shuffle = nc.Shuffle()
        result_dataset = shuffle(original_dataset)
        assert len(result_dataset.df.compute()) == 5


class TestShuffleDeterministic:
    def test_shuffle(self):
        original_dataset = list_to_dataset(["one", "two", "three", "four", "five"])
        expected_dataset = list_to_dataset(["five", "four", "three", "one", "two"])
        shuffle = nc.Shuffle(seed=42)
        result_dataset = shuffle(original_dataset)
        all_equal(expected_dataset, result_dataset)

    def test_new_partitions(self):
        original_dataset = list_to_dataset(
            ["one", "two", "three", "four", "five"], npartitions=3
        )
        expected_dataset = list_to_dataset(
            ["four", "three", "five", "one", "two"], npartitions=3
        )
        shuffle = nc.Shuffle(seed=42, npartitions=2)
        result_dataset = shuffle(original_dataset)
        all_equal(expected_dataset, result_dataset)

    def test_filename(self):
        original_dataset = list_to_dataset(
            ["one", "two", "three", "four", "five"], npartitions=1
        )
        original_dataset.df["file_name"] = "original.jsonl"

        expected_data = {
            "text": ["four", "five", "three", "one", "two"],
            "file_name": [
                "file_0000000000.jsonl",
                "file_0000000001.jsonl",
                "file_0000000001.jsonl",
                "file_0000000001.jsonl",
                "file_0000000001.jsonl",
            ],
        }
        pdf = pd.DataFrame(expected_data)
        expected_dataset = DocumentDataset(dd.from_pandas(pdf, npartitions=2))

        shuffle = nc.Shuffle(seed=42, npartitions=2)
        result_dataset = shuffle(original_dataset)
        all_equal(expected_dataset, result_dataset)

    def test_custom_filenames(self):
        original_dataset = list_to_dataset(
            ["one", "two", "three", "four", "five"], npartitions=1
        )
        original_dataset.df["file_name"] = "original.jsonl"

        expected_data = {
            "text": ["four", "five", "three", "one", "two"],
            "file_name": [
                "my_0.test",
                "my_1.test",
                "my_1.test",
                "my_1.test",
                "my_1.test",
            ],
        }
        pdf = pd.DataFrame(expected_data)
        expected_dataset = DocumentDataset(dd.from_pandas(pdf, npartitions=2))

        def filename_fn(x):
            return f"my_{x}.test"

        shuffle = nc.Shuffle(seed=42, npartitions=2, partition_to_filename=filename_fn)
        result_dataset = shuffle(original_dataset)
        all_equal(expected_dataset, result_dataset)
