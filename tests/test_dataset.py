import dask.dataframe as dd
import pandas as pd

import nemo_curator as nc
from nemo_curator.datasets import DocumentDataset


def all_equal(left_result: pd.DataFrame, right_result: pd.DataFrame):
    l_cols = set(left_result.columns)
    r_cols = set(right_result.columns)
    assert l_cols == r_cols
    for col in left_result.columns:
        left = left_result[col].reset_index(drop=True)
        right = right_result[col].reset_index(drop=True)
        assert all(left == right), f"Mismatch in {col} column.\n{left}\n{right}\n"


class TestDocumentDataset:
    def test_to_from_pandas(self):
        original_df = pd.DataFrame(
            {"first_col": [1, 2, 3], "second_col": ["a", "b", "c"]}
        )
        dataset = DocumentDataset.from_pandas(original_df)
        converted_df = dataset.to_pandas()
        all_equal(original_df, converted_df)
