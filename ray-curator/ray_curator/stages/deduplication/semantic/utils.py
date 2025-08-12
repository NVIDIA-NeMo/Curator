from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cudf
    import cupy as cp

from typing import Any

from loguru import logger


def get_array_from_df(df: "cudf.DataFrame", embedding_col: str) -> "cp.ndarray":
    """
    Convert a column of lists to a 2D array.
    """
    return df[embedding_col].list.leaves.values.reshape(len(df), -1)


def break_parquet_partition_into_groups(
    files: list[str], embedding_dim: int | None = None, storage_options: dict[str, Any] | None = None
) -> list[list[str]]:
    """Break parquet files into groups to avoid cudf 2bn row limit."""
    if embedding_dim is None:
        # Default aggressive assumption of 1024 dimensional embedding
        embedding_dim = 1024

    cudf_max_num_rows = 2_000_000_000  # cudf only allows 2bn rows
    cudf_max_num_elements = cudf_max_num_rows / embedding_dim  # cudf considers each element in an array to be a row

    import pyarrow.parquet as pq
    from fsspec.parquet import open_parquet_file

    # Load the first file and get the number of rows to estimate
    with open_parquet_file(files[0], storage_options=storage_options) as f:
        # Multiply by 1.5 to adjust for skew
        avg_num_rows = pq.read_metadata(f).num_rows * 1.5

    max_files_per_subgroup = int(cudf_max_num_elements / avg_num_rows)
    max_files_per_subgroup = max(1, max_files_per_subgroup)  # Ensure at least 1 file per subgroup

    # Break files into subgroups
    subgroups = [files[i : i + max_files_per_subgroup] for i in range(0, len(files), max_files_per_subgroup)]

    logger.debug(
        f"Broke {len(files)} files into {len(subgroups)} subgroups with max {max_files_per_subgroup} files per subgroup"
    )
    return subgroups
