import pandas as pd

from nemo_curator.datasets import DocumentDataset


def test_to_from_pandas():
    original_df = pd.DataFrame({"first_col": [1, 2, 3], "second_col": ["a", "b", "c"]})
    dataset = DocumentDataset.from_pandas(original_df)
    converted_df = dataset.to_pandas()
    pd.testing.assert_frame_equal(original_df, converted_df)
