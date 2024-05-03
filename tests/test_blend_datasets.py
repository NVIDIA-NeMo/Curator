import dask.dataframe as dd
import pandas as pd

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


class TestBlending:
    def test_blend_as_original(self):
        first_dataset = list_to_dataset(["one", "two", "three"])
        result_dataset = nc.blend_datasets(len(first_dataset), [first_dataset], [1.0])
        all_equal(first_dataset, result_dataset)

    def test_equal_blend(self):
        first_dataset = list_to_dataset(["a", "a"])
        second_dataset = list_to_dataset(["b", "b"])
        result_dataset = nc.blend_datasets(
            2, [first_dataset, second_dataset], [0.5, 0.5]
        )
        counts = result_dataset.df["text"].value_counts().compute()
        assert len(result_dataset) == 2
        assert counts["a"] == 1
        assert counts["b"] == 1

    def test_equal_blend_with_weights(self):
        first_dataset = list_to_dataset(["a", "a"])
        second_dataset = list_to_dataset(["b", "b"])
        result_dataset = nc.blend_datasets(
            2, [first_dataset, second_dataset], [2.0, 2.0]
        )
        counts = result_dataset.df["text"].value_counts().compute()
        assert len(result_dataset) == 2
        assert counts["a"] == 1
        assert counts["b"] == 1

    def test_uneven_blend(self):
        first_dataset = list_to_dataset(["a", "a"])
        second_dataset = list_to_dataset(["b", "b"])
        result_dataset = nc.blend_datasets(
            4, [first_dataset, second_dataset], [3.0, 1.0]
        )
        counts = result_dataset.df["text"].value_counts().compute()
        assert len(result_dataset) == 4
        assert counts["a"] == 3
        assert counts["b"] == 1

    def test_very_uneven_blend(self):
        first_dataset = list_to_dataset(["a", "a"])
        second_dataset = list_to_dataset(["b", "b"])
        result_dataset = nc.blend_datasets(
            4, [first_dataset, second_dataset], [1.0, 0.0]
        )
        counts = result_dataset.df["text"].value_counts().compute()
        assert len(result_dataset) == 4
        assert counts["a"] == 4
        assert "b" not in counts

    def test_proper_uneven_blend(self):
        first_dataset = list_to_dataset(["a", "b", "c", "d"])
        second_dataset = list_to_dataset(["e", "f"])
        result_dataset = nc.blend_datasets(
            8, [first_dataset, second_dataset], [1.0, 0.0]
        )
        counts = result_dataset.df["text"].value_counts().compute()
        assert len(result_dataset) == 8
        assert counts["a"] == 2
        assert counts["b"] == 2
        assert counts["c"] == 2
        assert counts["d"] == 2

    def test_four_dataset_blend(self):
        datasets = []
        datasets.append(list_to_dataset(["a", "a"]))
        datasets.append(list_to_dataset(["b", "b", "b"]))
        datasets.append(list_to_dataset(["c"]))
        datasets.append(list_to_dataset(["d", "d", "d", "d"]))
        result_dataset = nc.blend_datasets(8, datasets, [1.0, 2.0, 3.0, 4.0])
        counts = result_dataset.df["text"].value_counts().compute()
        assert len(result_dataset) == 10
        assert counts["a"] == 1
        assert counts["b"] == 2
        assert counts["c"] == 3
        assert counts["d"] == 4
