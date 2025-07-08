import json

import cudf
import cupy as cp
import numpy as np


def weighted_percentile(data: np.ndarray, percentiles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Compute weighted percentiles with the "inverted_cdf" method.

    Parameters:
      data : array-like, the data values.
      percentiles : scalar or array-like, percentiles in [0, 100].
      weights : array-like, the weights for each data value.

    Returns:
      The weighted percentile values.
    """
    data = np.asarray(data)
    weights = np.asarray(weights)

    # Sort data and associated weights
    sorter = np.argsort(data)
    data_sorted = data[sorter]
    weights_sorted = weights[sorter]

    # Compute the cumulative sum of weights and normalize it to [0, 1]
    cum_weights = np.cumsum(weights_sorted)
    total_weight = cum_weights[-1]
    normalized_cum_weights = cum_weights / total_weight

    # For each desired percentile, find the first data value where
    # the normalized cumulative weight is >= (percentile / 100).
    percentiles = np.atleast_1d(percentiles)
    results = []
    for p in percentiles:
        # np.searchsorted returns the index where (p/100) should be inserted
        # to maintain order.
        idx = np.searchsorted(normalized_cum_weights, p / 100.0, side="left")
        results.append(data_sorted[idx])

    return np.array(results)


def compute_thresholds(score_ar: np.ndarray, token_ar: np.ndarray) -> dict[str, float]:
    """
    Compute percentile-based thresholds for a given score column using weighted percentiles.

    Args:
        score_ar (np.ndarray): Array containing the scores.
        token_ar (np.ndarray): Array containing token counts for weighting.

    Returns:
        Dict[str, float]: Dictionary containing percentile thresholds.
    """
    percentiles = np.arange(5, 100, 5)
    # NumPy < 2.0 does not support the "inverted_cdf" method for computing percentiles
    # with weights directly via np.percentile (see commented-out equivalent code below).
    # To achieve the same result, we manually implement the weighted percentile computation
    # using NumPy primitives.
    # thresholds = np.percentile(cc_df_score, percentiles, weights=cc_df_tokens, method='inverted_cdf') # noqa: ERA001
    thresholds = weighted_percentile(score_ar, percentiles, weights=token_ar)
    return {int(percentile): float(thresh) for percentile, thresh in zip(percentiles, thresholds, strict=False)}


def compute_thresholds_for_score_columns(
    df: cudf.DataFrame, text_col_name: str, score_col_names: list[str]
) -> dict[str, dict[str, float]]:
    """
    Compute percentile-based thresholds for all specified score columns in a DataFrame.

    Args:
        df (cudf.DataFrame): The DataFrame containing the score columns and text column.
        text_col_name (str): The name of the text column used to derive token counts.
        score_col_names (List[str]): List of column names for which thresholds should be computed.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary mapping each score column to its percentile thresholds.
    """
    threshold_dict = {}
    token_series = df[text_col_name].str.byte_count()

    for score_col in score_col_names:
        threshold_dict[score_col] = compute_thresholds(df[score_col].values_host, token_series.values_host)

    return threshold_dict


def save_thresholds(threshold_dict: dict[str, dict[str, float]], file_name: str) -> None:
    """
    Save computed thresholds to a JSON file.

    Args:
        threshold_dict (Dict[str, Dict[str, float]]): The dictionary containing computed thresholds.
        file_name (str, optional): The name of the output JSON file. Defaults to "thresholds.json".
    Returns:
        None
    """
    with open(file_name, "w") as fout:
        json.dump(threshold_dict, fout, indent=4)
    print(f"Thresholds saved to {file_name}")


def map_scores(df: cudf.DataFrame, score_col_name: str, score_int_name: str, bins: list[float]) -> cudf.DataFrame:
    """
    Given a DataFrame df and a column of original scores,
    use cp.digitize to map them into integer bins using the given thresholds.
    """
    pred_orig_score = cp.array(df[score_col_name])
    pred_int_score = cp.digitize(pred_orig_score, bins)
    df[score_int_name] = pred_int_score.get()
    return df


def map_score_columns(
    df: cudf.DataFrame, score_col_names: list[str], threshold_dict: dict[str, dict]
) -> cudf.DataFrame:
    """
    For each score column in score_col_names, this function:
      1. Creates a new column name by appending '-int'
      2. Retrieves the corresponding thresholds from threshold_dict,
         sorts them (using the keys which are assumed to be strings of numbers),
      3. Passes the bins to map_scores to create the integer score column.
    """
    for score_col_name in score_col_names:
        # Build the new integer score column name.
        score_int_name = score_col_name + "-int"
        thresholds = threshold_dict.get(score_col_name)
        if thresholds is None:
            msg = f"No thresholds found for score column '{score_col_name}'"
            raise ValueError(msg)

        sorted_keys = sorted(thresholds.keys(), key=lambda x: int(x))
        # Use cp.array to create a CuPy array from the list of threshold values.
        bins = cp.array([thresholds[k] for k in sorted_keys])

        # Map the original score column to the new integer score column.
        df = map_scores(df, score_col_name, score_int_name, bins)
    return df
