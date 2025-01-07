import tempfile

import pandas as pd
import pytest

from nemo_curator._compat import DASK_CUDF_PARQUET_READ_INCONSISTENT_SCHEMA
from nemo_curator.utils.distributed_utils import (
    read_data,
    read_data_blocksize,
    read_data_files_per_partition,
)
from nemo_curator.utils.file_utils import get_all_files_paths_under

NUM_FILES = 5
NUM_RECORDS = 100


# Fixture to create multiple small JSONL files
@pytest.fixture
def mock_multiple_jsonl_files(tmp_path):
    file_paths = []
    for file_id in range(NUM_FILES):
        jsonl_file = tmp_path / f"test_{file_id}.jsonl"
        with open(jsonl_file, "w") as f:
            for record_id in range(NUM_RECORDS):
                # 100 rows are ~5kb
                f.write(
                    f'{{"id": "id_{file_id}_{record_id}", "text": "A longish string {file_id}_{record_id}"}}\n'
                )
        file_paths.append(str(jsonl_file))
    return file_paths


# Fixture to create multiple small Parquet files
@pytest.fixture
def mock_multiple_parquet_files(tmp_path):
    file_paths = []
    for file_id in range(NUM_FILES):
        # 100 rows are ~5kb
        parquet_file = tmp_path / f"test_{file_id}.parquet"
        df = pd.DataFrame(
            [
                {
                    "id": f"id_{file_id}_{record_id}",
                    "text": f"A string {file_id}_{record_id}",
                }
                for record_id in range(NUM_RECORDS)
            ]
        )
        # We specify row_group_size so that we can test splitting a single big file into smaller chunks
        df.to_parquet(parquet_file, compression=None, row_group_size=10)
        file_paths.append(str(parquet_file))
    return file_paths


@pytest.fixture
def mock_multiple_jsonl_files_different_cols(tmp_path):
    file_paths = []
    for file_id in range(NUM_FILES):
        jsonl_file = tmp_path / f"different_cols_test_{file_id}.jsonl"

        def make_record_without_meta(file_id, record_id):
            return {
                "id": f"id_{file_id}_{record_id}",
                "text": f"A string {file_id}_{record_id}",
            }

        def make_record_with_meta(file_id, record_id):
            return {
                "text": f"A string {file_id}_{record_id}",
                "meta1": [
                    {"field1": "field_one", "field2": "field_two"},
                ],
                "id": f"id_{file_id}_{record_id}",
            }

        df = pd.DataFrame(
            [
                (
                    make_record_without_meta(file_id, record_id)
                    if file_id == 0
                    else make_record_with_meta(file_id, record_id)
                )
                for record_id in range(NUM_RECORDS)
            ]
        )

        df.to_json(jsonl_file, orient="records", lines=True)
        file_paths.append(str(jsonl_file))
    return file_paths


# Fixture to create multiple small Parquet files
@pytest.fixture
def mock_multiple_parquet_files_different_cols(tmp_path):
    file_paths = []
    for file_id in range(NUM_FILES):
        # 100 rows are ~5kb
        parquet_file = tmp_path / f"test_diff_cols_{file_id}.parquet"

        def make_record_without_meta(file_id, record_id):
            return {
                "id": f"id_{file_id}_{record_id}",
                "text": f"A string {file_id}_{record_id}",
            }

        def make_record_with_meta(file_id, record_id):
            return {
                "text": f"A string {file_id}_{record_id}",
                "meta1": [
                    {"field1": "field_one", "field2": "field_two"},
                ],
                "id": f"id_{file_id}_{record_id}",
            }

        df = pd.DataFrame(
            [
                (
                    make_record_without_meta(file_id, record_id)
                    if file_id == 0
                    else make_record_with_meta(file_id, record_id)
                )
                for record_id in range(NUM_RECORDS)
            ]
        )
        df.to_parquet(parquet_file, compression=None, row_group_size=10)
        file_paths.append(str(parquet_file))
    return file_paths


@pytest.mark.gpu
@pytest.mark.parametrize("file_type", ["jsonl", "parquet"])
@pytest.mark.parametrize("blocksize", ["1kb", "5kb", "10kb"])
def test_cudf_read_data_blocksize_partitioning(
    mock_multiple_jsonl_files, mock_multiple_parquet_files, file_type, blocksize
):
    import cudf

    input_files = (
        mock_multiple_jsonl_files
        if file_type == "jsonl"
        else mock_multiple_parquet_files
    )

    df = read_data_blocksize(
        input_files=input_files,
        backend="cudf",
        file_type=file_type,
        blocksize=blocksize,
        add_filename=False,
        input_meta=None,
        columns=None,
    )

    # Compute the number of partitions in the resulting DataFrame
    num_partitions = df.optimize().npartitions
    # Assert that we have two partitions (since we have ~15KB total data and a blocksize of 10KB)
    if blocksize == "1kb":
        assert (
            num_partitions > NUM_FILES
        ), f"Expected > {NUM_FILES} partitions but got {num_partitions}"
    elif blocksize == "5kb":
        assert (
            num_partitions == NUM_FILES
        ), f"Expected {NUM_FILES} partitions but got {num_partitions}"
    elif blocksize == "10kb":
        assert (
            num_partitions < NUM_FILES
        ), f"Expected < {NUM_FILES} partitions but got {num_partitions}"
    else:
        raise ValueError(f"Invalid blocksize: {blocksize}")
    total_rows = len(df)
    assert (
        total_rows == NUM_FILES * NUM_RECORDS
    ), f"Expected {NUM_FILES * NUM_RECORDS} rows but got {total_rows}"

    assert isinstance(df["id"].compute(), cudf.Series)


@pytest.mark.parametrize("file_type", ["jsonl", "parquet"])
@pytest.mark.parametrize("blocksize", ["1kb", "5kb", "10kb"])
def test_pandas_read_data_blocksize_partitioning(
    mock_multiple_jsonl_files, mock_multiple_parquet_files, file_type, blocksize
):
    input_files = (
        mock_multiple_jsonl_files
        if file_type == "jsonl"
        else mock_multiple_parquet_files
    )

    df = read_data_blocksize(
        input_files=input_files,
        backend="pandas",
        file_type=file_type,
        blocksize=blocksize,
        add_filename=False,
        input_meta=None,
        columns=None,
    )

    # Compute the number of partitions in the resulting DataFrame
    num_partitions = df.npartitions
    # Our total data is ~25kb where each file is 5kb
    if blocksize == "1kb":
        assert (
            num_partitions > NUM_FILES
        ), f"Expected > {NUM_FILES} partitions but got {num_partitions}"
    elif blocksize == "5kb":
        assert (
            num_partitions == NUM_FILES
        ), f"Expected {NUM_FILES} partitions but got {num_partitions}"
    elif blocksize == "10kb":
        # Because pandas doesn't suppport reading json files together, a partition will only be as big as a single file
        if file_type == "jsonl":
            assert (
                num_partitions == NUM_FILES
            ), f"Expected {NUM_FILES} partitions but got {num_partitions}"
        # Parquet files can be read together
        elif file_type == "parquet":
            assert (
                num_partitions < NUM_FILES
            ), f"Expected > {NUM_FILES} partitions but got {num_partitions}"
    else:
        raise ValueError(f"Invalid blocksize: {blocksize}")
    total_rows = len(df)
    assert (
        total_rows == NUM_FILES * NUM_RECORDS
    ), f"Expected {NUM_FILES * NUM_RECORDS} rows but got {total_rows}"

    assert isinstance(df["id"].compute(), pd.Series)


@pytest.mark.parametrize(
    "backend",
    ["pandas", pytest.param("cudf", marks=pytest.mark.gpu)],
)
@pytest.mark.parametrize("file_type", ["jsonl", "parquet"])
@pytest.mark.parametrize("fpp", [1, NUM_FILES // 2, NUM_FILES, NUM_FILES * 2])
def test_read_data_fpp_partitioning(
    mock_multiple_jsonl_files, mock_multiple_parquet_files, backend, file_type, fpp
):
    input_files = (
        mock_multiple_jsonl_files
        if file_type == "jsonl"
        else mock_multiple_parquet_files
    )

    df = read_data_files_per_partition(
        input_files=input_files,
        backend=backend,
        file_type=file_type,
        files_per_partition=fpp,
        add_filename=False,
        input_meta=None,
        columns=None,
    )

    # Compute the number of partitions in the resulting DataFrame
    num_partitions = df.npartitions
    # Assert that we have two partitions (since we have ~15KB total data and a blocksize of 10KB)
    if fpp == 1:
        assert (
            num_partitions == NUM_FILES
        ), f"Expected {NUM_FILES} partitions but got {num_partitions}"
    elif fpp == NUM_FILES // 2:
        assert (
            num_partitions < NUM_FILES
        ), f"Expected {NUM_FILES} partitions but got {num_partitions}"
    elif fpp >= NUM_FILES:
        assert num_partitions == 1, f"Expected 1 partition but got {num_partitions}"
    else:
        raise ValueError(f"Invalid fpp: {fpp}")
    total_rows = len(df)
    assert (
        total_rows == NUM_FILES * NUM_RECORDS
    ), f"Expected {NUM_FILES * NUM_RECORDS} rows but got {total_rows}"
    if backend == "cudf":
        import cudf

        assert isinstance(df["id"].compute(), cudf.Series)
    elif backend == "pandas":
        assert isinstance(df["id"].compute(), pd.Series)


@pytest.mark.parametrize(
    "backend",
    [
        "pandas",
        pytest.param("cudf", marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize("filename_arg", [True, "some_filename"])
def test_read_data_blocksize_add_filename_jsonl(
    mock_multiple_jsonl_files, backend, filename_arg
):
    df = read_data_blocksize(
        input_files=mock_multiple_jsonl_files,
        backend=backend,
        file_type="jsonl",
        blocksize="128Mib",
        add_filename=filename_arg,
        input_meta=None,
        columns=None,
    )

    filename_str = "file_name" if filename_arg is True else filename_arg
    assert filename_str in df.columns
    file_names = df[filename_str].unique().compute()
    if backend == "cudf":
        file_names = file_names.to_pandas()

    assert len(file_names) == NUM_FILES
    assert set(file_names.values) == {
        f"test_{file_id}.jsonl" for file_id in range(NUM_FILES)
    }


@pytest.mark.parametrize(
    "backend",
    [
        "pandas",
        pytest.param("cudf", marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize("filename_arg", [True, "some_filename"])
def test_read_data_blocksize_add_filename_parquet(
    mock_multiple_parquet_files, backend, filename_arg
):
    with pytest.raises(
        ValueError,
        match="add_filename and blocksize cannot be set at the same time for Parquet files",
    ):
        read_data_blocksize(
            input_files=mock_multiple_parquet_files,
            backend=backend,
            file_type="parquet",
            blocksize="128Mib",
            add_filename=filename_arg,
            input_meta=None,
            columns=None,
        )


@pytest.mark.parametrize(
    "backend,file_type",
    [
        pytest.param("cudf", "jsonl", marks=pytest.mark.gpu),
        pytest.param("cudf", "parquet", marks=pytest.mark.gpu),
        ("pandas", "jsonl"),
        ("pandas", "parquet"),
    ],
)
@pytest.mark.parametrize("filename_arg", [True, "some_filename"])
def test_read_data_fpp_add_filename(
    mock_multiple_jsonl_files,
    mock_multiple_parquet_files,
    backend,
    file_type,
    filename_arg,
):
    input_files = (
        mock_multiple_jsonl_files
        if file_type == "jsonl"
        else mock_multiple_parquet_files
    )

    df = read_data_files_per_partition(
        input_files=input_files,
        backend=backend,
        file_type=file_type,
        files_per_partition=NUM_FILES,
        add_filename=filename_arg,
        input_meta=None,
        columns=None,
    )

    filename_str = "file_name" if filename_arg is True else filename_arg
    assert filename_str in df.columns
    assert list(df.columns) == list(df.head().columns)
    assert set(df.columns) == {filename_str, "id", "text"}
    file_names = df[filename_str].unique().compute()
    if backend == "cudf":
        file_names = file_names.to_pandas()

    assert len(file_names) == NUM_FILES
    assert set(file_names.values) == {
        f"test_{file_id}.{file_type}" for file_id in range(NUM_FILES)
    }


@pytest.mark.parametrize(
    "backend",
    [
        "pandas",
        pytest.param("cudf", marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize(
    "file_type,add_filename,function_name",
    [
        *[
            ("jsonl", True, func)
            for func in ["read_data_blocksize", "read_data_files_per_partition"]
        ],
        *[
            ("jsonl", False, func)
            for func in ["read_data_blocksize", "read_data_files_per_partition"]
        ],
        *[
            ("parquet", False, func)
            for func in ["read_data_blocksize", "read_data_files_per_partition"]
        ],
        *[("parquet", True, "read_data_files_per_partition")],
    ],
)
@pytest.mark.parametrize(
    "cols_to_select", [None, ["id"], ["text", "id"], ["id", "text"]]
)
def test_read_data_select_columns(
    mock_multiple_jsonl_files,
    mock_multiple_parquet_files,
    backend,
    file_type,
    add_filename,
    function_name,
    cols_to_select,
):
    input_files = (
        mock_multiple_jsonl_files
        if file_type == "jsonl"
        else mock_multiple_parquet_files
    )
    if function_name == "read_data_files_per_partition":
        func = read_data_files_per_partition
        read_kwargs = {"files_per_partition": 1}
    elif function_name == "read_data_blocksize":
        func = read_data_blocksize
        read_kwargs = {"blocksize": "128Mib"}

    df = func(
        input_files=input_files,
        backend=backend,
        file_type=file_type,
        add_filename=add_filename,
        input_meta=None,
        columns=list(cols_to_select) if cols_to_select else None,
        **read_kwargs,
    )
    if not cols_to_select:
        cols_to_select = ["id", "text"]

    assert list(df.columns) == list(df.head().columns)
    if not add_filename:
        assert list(df.columns) == sorted(cols_to_select)
    else:
        assert list(df.columns) == sorted(cols_to_select + ["file_name"])


@pytest.mark.parametrize(
    "backend",
    [
        "pandas",
        pytest.param("cudf", marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize(
    "function_name", ["read_data_blocksize", "read_data_files_per_partition"]
)
@pytest.mark.parametrize(
    "input_meta", [{"id": "str"}, {"text": "str"}, {"id": "str", "text": "str"}]
)
def test_read_data_input_meta(
    mock_multiple_jsonl_files, backend, function_name, input_meta
):
    if function_name == "read_data_files_per_partition":
        func = read_data_files_per_partition
        read_kwargs = {"files_per_partition": 1}
    elif function_name == "read_data_blocksize":
        func = read_data_blocksize
        read_kwargs = {"blocksize": "128Mib"}

    df = func(
        input_files=mock_multiple_jsonl_files,
        backend=backend,
        file_type="jsonl",
        add_filename=False,
        input_meta=input_meta,
        columns=None,
        **read_kwargs,
    )

    assert list(df.columns) == list(input_meta.keys())


def xfail_inconsistent_schema_jsonl():
    return pytest.mark.xfail(
        reason="inconsistent schemas are not supported with jsonl files, "
        "see https://github.com/dask/dask/issues/11595"
    )


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("pandas"),
        pytest.param("cudf", marks=[pytest.mark.gpu]),
    ],
)
@pytest.mark.parametrize("file_type", ["jsonl", "parquet"])
@pytest.mark.parametrize("fpp", [1, 3, 5])
def test_read_data_different_columns_files_per_partition(
    mock_multiple_jsonl_files_different_cols,
    mock_multiple_parquet_files_different_cols,
    backend,
    file_type,
    fpp,
):
    read_kwargs = {"columns": ["id", "text"]}
    if file_type == "jsonl":
        input_files = mock_multiple_jsonl_files_different_cols
        read_kwargs["input_meta"] = {"id": "str", "text": "str"}
    elif file_type == "parquet":
        input_files = mock_multiple_parquet_files_different_cols
        if backend == "cudf":
            read_kwargs["allow_mismatched_pq_schemas"] = True

    df = read_data(
        input_files=input_files,
        file_type=file_type,
        backend=backend,
        add_filename=False,
        files_per_partition=fpp,
        blocksize=None,
        **read_kwargs,
    )
    assert list(df.columns) == ["id", "text"]
    assert list(df.compute().columns) == ["id", "text"]
    with tempfile.TemporaryDirectory() as tmpdir:
        df.to_parquet(tmpdir)
    assert len(df) == NUM_FILES * NUM_RECORDS


@pytest.mark.parametrize(
    "backend,file_type",
    [
        pytest.param(
            "cudf", "jsonl", marks=[pytest.mark.gpu, xfail_inconsistent_schema_jsonl()]
        ),
        pytest.param("pandas", "jsonl", marks=[xfail_inconsistent_schema_jsonl()]),
        pytest.param(
            "cudf",
            "parquet",
            marks=[pytest.mark.gpu]
            + (
                [xfail_inconsistent_schema_jsonl()]
                if not DASK_CUDF_PARQUET_READ_INCONSISTENT_SCHEMA
                else []
            ),
        ),
        pytest.param("pandas", "parquet"),
    ],
)
@pytest.mark.parametrize("blocksize", ["1kb", "5kb", "10kb"])
def test_read_data_different_columns_blocksize(
    mock_multiple_jsonl_files_different_cols,
    mock_multiple_parquet_files_different_cols,
    backend,
    file_type,
    blocksize,
):
    read_kwargs = {"columns": ["id", "text"]}
    read_kwargs["columns"] = ["id", "text"]
    if file_type == "jsonl":
        input_files = mock_multiple_jsonl_files_different_cols
        read_kwargs["input_meta"] = {"id": "str", "text": "str"}
    elif file_type == "parquet":
        input_files = mock_multiple_parquet_files_different_cols
        if backend == "cudf":
            read_kwargs["allow_mismatched_pq_schemas"] = True

    df = read_data(
        input_files=input_files,
        file_type=file_type,
        blocksize=blocksize,
        files_per_partition=None,
        backend=backend,
        add_filename=False,
        **read_kwargs,
    )
    assert list(df.columns) == ["id", "text"]
    assert list(df.compute().columns) == ["id", "text"]
    with tempfile.TemporaryDirectory() as tmpdir:
        df.to_parquet(tmpdir)
    assert len(df) == NUM_FILES * NUM_RECORDS
