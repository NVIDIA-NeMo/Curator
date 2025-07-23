import fasttext
import numpy as np
import pandas as pd
from fasttext.FastText import _FastText
from huggingface_hub import hf_hub_download

from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import load_object_on_worker


class FastTextQualityClassifier:
    """
    A classifier that uses a fastText model to predict a confidence score for text.

    It appends one or two output columns to the data:
      - A float column representing the confidence score.
      - Optionally, an integer column (1 if the top label contains "hq", else 0).

    The model is loaded from the Hugging Face Hub during initialization.

    Args:
        pred_column (str): Name of the output column for the confidence score.
        int_column (str, optional): Name of the output column for the binary indicator.
                                    If not provided, only the pred_column is added.
    """

    def __init__(self, pred_column: str, int_column: str | None = None) -> None:
        self.pred_column: str = pred_column
        self.int_column: str | None = int_column

        self.repo_id: str = "mlfoundations/fasttext-oh-eli5"
        self.model_filename: str = "openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin"
        # Download the fastText model from Hugging Face Hub.
        self.model_path: str = hf_hub_download(repo_id=self.repo_id, filename=self.model_filename)
        self.model_identifier: str = f"{self.repo_id}/{self.model_filename}"

    def _load_fasttext_model(self) -> _FastText:
        """Load and return the fastText model."""
        return fasttext.load_model(self.model_path)

    def predict_text(self, text: str) -> tuple[float, int]:
        """
        Predict the confidence score and binary indicator for a given text.

        Args:
            text (str): The input text to classify.

        Returns:
            Tuple[float, int]: A tuple containing the confidence score (float) and binary indicator (int).
        """
        model = load_object_on_worker(self.model_identifier, self._load_fasttext_model, {})
        text = text.replace("\n", "")
        predictions = model.predict(text, k=2)
        # predictions[0]: labels, predictions[1]: scores
        # If the top predicted label contains "hq", return the first score; otherwise, use the second.
        if "hq" in predictions[0][0]:
            return predictions[1][0], 1
        else:
            return predictions[1][1], 0

    def _predict_on_partition(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply predictions to a pandas DataFrame partition.

        Assumes the DataFrame has a "text" column.

        Args:
            df (pd.DataFrame): Input DataFrame partition.

        Returns:
            pd.DataFrame: DataFrame with added prediction columns.
        """
        # Load the model on the worker.
        model = load_object_on_worker(self.model_identifier, self._load_fasttext_model, {})  # noqa: F841
        results = df["text"].apply(self.predict_text)
        df[self.pred_column] = results.apply(lambda x: x[0]).astype(np.float32)
        if self.int_column is not None:
            df[self.int_column] = results.apply(lambda x: x[1]).astype(np.int32)
        return df

    def __call__(self, dataset: DocumentDataset) -> DocumentDataset:
        """
        Apply the classifier to a distributed dataset.

        The dataset should have a "text" column. The classifier converts the dataset
        to a pandas backend, applies predictions to each partition, and then converts the result
        back to cudf.

        Args:
            dataset: A distributed DataFrame (e.g., a Dask DataFrame) containing a "text" column.

        Returns:
            DocumentDataset: The dataset with added prediction columns.
        """
        meta = dataset.df._meta  # noqa: SLF001
        if hasattr(meta, "to_pandas"):
            meta = meta.to_pandas()
        meta[self.pred_column] = np.float32(0.0)
        if self.int_column is not None:
            meta[self.int_column] = np.int32(0)

        processed_df = dataset.df.to_backend("pandas").map_partitions(self._predict_on_partition, meta=meta)
        processed_df = processed_df.to_backend("cudf")
        return DocumentDataset(processed_df)
