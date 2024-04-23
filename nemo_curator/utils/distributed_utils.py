# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from __future__ import annotations

import os

os.environ["RAPIDS_NO_INITIALIZE"] = "1"
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Union

import dask.dataframe as dd
import pandas as pd
from dask.distributed import Client, LocalCluster, get_worker, performance_report

from nemo_curator.utils.gpu_utils import GPU_INSTALL_STRING, is_cudf_type
from nemo_curator.utils.import_utils import gpu_only_import, gpu_only_import_from

cudf = gpu_only_import("cudf")
LocalCUDACluster = gpu_only_import_from("dask_cuda", "LocalCUDACluster")


class DotDict:
    def __init__(self, d):
        self.__dict__["_data"] = d

    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        else:
            raise AttributeError(f"No attribute '{name}'")


class NoWorkerError(Exception):
    pass


def start_dask_gpu_local_cluster(args) -> Client:
    """
    This function sets up a Dask cluster across all the
    GPUs present on the machine.

    """
    # Setting conservative defaults
    # which should work across most systems
    nvlink_only = getattr(args, "nvlink_only", False)
    protocol = getattr(args, "protocol", "tcp")
    rmm_pool_size = getattr(args, "rmm_pool_size", "1024M")
    enable_spilling = getattr(args, "enable_spilling", True)
    set_torch_to_use_rmm = getattr(args, "set_torch_to_use_rmm", True)

    extra_kwargs = (
        {
            "enable_tcp_over_ucx": True,
            "enable_nvlink": True,
            "enable_infiniband": False,
            "enable_rdmacm": False,
        }
        if nvlink_only and protocol == "ucx"
        else {}
    )

    cluster = LocalCUDACluster(
        rmm_pool_size=rmm_pool_size,
        protocol=protocol,
        rmm_async=True,
        **extra_kwargs,
    )
    client = Client(cluster)

    if enable_spilling:
        _enable_spilling()
        client.run(_enable_spilling)

    if set_torch_to_use_rmm:
        _set_torch_to_use_rmm()
        client.run(_set_torch_to_use_rmm)
        print("Torch is using RMM memory pool", flush=True)
    return client


def start_dask_cpu_local_cluster(args) -> Client:
    """
    This function sets up a Dask cluster across all the
    CPUs present on the machine.

    """
    # TODO: expand args as needed for CPU clusters
    n_workers = getattr(args, "n_workers", None)
    if n_workers is None:
        # Setting threads_per_worker
        # to 1 ensure 1 worker per CPU
        # by default
        threads_per_worker = getattr(args, "threads_per_worker", 1)
    else:
        threads_per_worker = None
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)
    client = Client(cluster)
    return client


def get_client(args, cluster_type) -> Client:
    """
    This function sets up a Dask cluster across n GPUs if not already set up.
    It also ensures maximum memory efficiency for the GPU by:
        1. Ensuring the PyTorch memory pool is the same as the RAPIDS memory pool.
        2. Enabling spilling for cuDF.

    Args:
        args: An argparse Namespace object.
        cluster_type="cpu": The type of cluster to set up. Either "cpu" or "gpu".
    Returns:
        A Dask client object.

    """
    if cluster_type not in ["cpu", "gpu"]:
        raise ValueError(f"Unknown cluster type: {cluster_type}")

    # Helper class allows the user to pass in a dict instead of a parser
    if type(args) == dict:
        args = DotDict(args)

    if args.scheduler_address:
        if args.scheduler_file:
            raise ValueError(
                "Only one of scheduler_address or scheduler_file can be provided"
            )
        else:
            return Client(address=args.scheduler_address, timeout="30s")
    elif args.scheduler_file:
        return Client(scheduler_file=args.scheduler_file, timeout="30s")
    else:
        if cluster_type == "gpu":
            return start_dask_gpu_local_cluster(args)
        else:
            return start_dask_cpu_local_cluster(args)


def _set_torch_to_use_rmm():
    """
    This function sets up the PyTorch memory pool to be the same as the RAPIDS memory pool.
    This helps avoid OOM errors when using both PyTorch and RAPIDS on the same GPU.

    See article:
    https://medium.com/rapids-ai/pytorch-rapids-rmm-maximize-the-memory-efficiency-of-your-workflows-f475107ba4d4

    """
    import torch
    from rmm.allocators.torch import rmm_torch_allocator

    torch.cuda.memory.change_current_allocator(rmm_torch_allocator)


def _enable_spilling():
    """
    Setting this environment variable enables automatic spilling (and "unspilling")
    of buffers from device to host to enable out-of-memory computation,
    i.e., computing on objects that occupy more memory than is available on the GPU.

    """
    import cudf

    cudf.set_option("spill", True)


def read_single_partition(
    files, backend="cudf", filetype="jsonl", add_filename=False
) -> Union[cudf.DataFrame, pd.DataFrame]:
    """
    This function reads a file with cuDF, sorts the columns of the DataFrame
    and adds a "filename" column.

    Args:
        files: The path to the jsonl files to read.
        backend: The backend to use for reading the data. Either "cudf" or "pandas".
        add_filename: Whether to add a "filename" column to the DataFrame.
    Returns:
        A cudf DataFrame or a pandas DataFrame.

    """
    if filetype == "jsonl":
        read_kwargs = {"lines": True}
        if backend == "cudf":
            read_f = cudf.read_json
        else:
            read_kwargs["dtype"] = False
            read_f = pd.read_json
    elif filetype == "parquet":
        read_kwargs = {}
        if backend == "cudf":
            read_f = cudf.read_parquet
        else:
            read_f = pd.read_parquet
    else:
        raise RuntimeError("Could not read data, please check file type")

    if add_filename:
        read_files_one_at_a_time = True
    else:
        if backend == "cudf":
            # cuDF supports reading multiple files at once
            read_files_one_at_a_time = False
        else:
            # pandas does not support reading multiple files at once
            read_files_one_at_a_time = True

    if read_files_one_at_a_time:
        if backend == "cudf":
            concat_f = cudf.concat
        else:
            concat_f = pd.concat
        df_ls = []
        for file in files:
            df = read_f(file, **read_kwargs)
            if add_filename:
                df["filename"] = os.path.basename(file)
            df_ls.append(df)
        df = concat_f(df_ls, ignore_index=True)
    else:
        df = read_f(files, **read_kwargs)
    df = df[sorted(df.columns)]
    return df


def read_pandas_pickle(file, add_filename=False) -> pd.DataFrame:
    """
    This function reads a pickle file with pandas and adds a "filename" column.

    Args:
        file: The path to the pickle file to read.
        add_filename: Whether to add a "filename" column to the DataFrame.
    Returns:
        A pandas DataFrame.

    """
    if add_filename:
        warnings.warn("add_filename is not supported for pickle files")
    return pd.read_pickle(file)


def read_data(
    input_files,
    file_type="pickle",
    backend="cudf",
    files_per_partition=1,
    add_filename=False,
) -> Union[dd.DataFrame, dask_cudf.DataFrame]:
    """
    This function can read multiple data formats and returns a Dask-cuDF DataFrame.

    Args:
        input_files: The path of the input file(s).
        file_type: The type of the input file(s).
        backend: The backend to use for reading the data.
        files_per_partition: The number of files to read per partition.
        add_filename: Whether to add a "filename" column to the DataFrame.

    Returns:
        A Dask-cuDF or a Dask-pandas DataFrame.

    """
    if backend == "cudf":
        # Try using cuDF. If not availible will throw an error.
        test_obj = cudf.Series

    if file_type == "pickle":
        df = read_pandas_pickle(input_files[0], add_filename=add_filename)
        df = dd.from_pandas(df, npartitions=16)
        if backend == "cudf":
            df = df.to_backend("cudf")

    elif file_type in ["jsonl", "parquet"]:
        print(f"Reading {len(input_files)} files", flush=True)
        input_files = sorted(input_files)
        if files_per_partition > 1:
            input_files = [
                input_files[i : i + files_per_partition]
                for i in range(0, len(input_files), files_per_partition)
            ]
        else:
            input_files = [[file] for file in input_files]
        return dd.from_map(
            read_single_partition,
            input_files,
            filetype=file_type,
            backend=backend,
            add_filename=add_filename,
            enforce_metadata=False,
        )
    else:
        raise RuntimeError("Could not read data, please check file type")
    return df


def process_batch(
    load_model_function, load_model_kwargs, run_inference_function, run_inference_kwargs
):
    """
    This function loads a model on a Dask worker and then runs inference on a batch of data.

    Args:
        load_model_function: A user-provided function for loading a classifier.
        load_model_kwargs: A dictionary of arguments necessary for `load_model_function`.
        run_inference_function: A user-provided function for running inference, which has a "model" argument.
        run_inference_kwargs: A dictionary of arguments necessary for `run_inference_function`.
    Returns:
        Whatever `run_inference_function` returns, such as a list or tensor of predictions.

    """
    model = load_object_on_worker("model", load_model_function, load_model_kwargs)
    return run_inference_function(**run_inference_kwargs, model=model)


def process_all_batches(
    loader_valid,
    load_model_function,
    load_model_kwargs,
    run_inference_function,
    run_inference_kwargs,
):
    """
    This function iterates over batches of data, loading a model and running inference per batch.

    Args:
        loader_valid: An iterable data object, such as a PyTorch DataLoader.
        load_model_function: A user-provided function for loading a classifier.
        load_model_kwargs: A dictionary of arguments necessary for `load_model_function`.
        run_inference_function: A user-provided function for running inference, which has "model" and "batch" arguments.
        run_inference_kwargs: A dictionary of arguments necessary for `run_inference_function`.
    Returns:
        A tensor of predictions for all batches of the data.

    """
    import torch

    return torch.cat(
        [
            process_batch(
                load_model_function,
                load_model_kwargs,
                run_inference_function,
                dict(run_inference_kwargs, batch=batch),
            )
            for batch in loader_valid
        ]
    )


def single_partition_write_with_filename(df, output_file_dir, output_type="jsonl"):
    """
    This function processes a DataFrame and writes it to disk

    Args:
        df: A DataFrame.
        output_file_dir: The output file path.
        output_type="jsonl": The type of output file to write.
    Returns:
        If the DataFrame is non-empty, return a Series containing a single element, True.
        If the DataFrame is empty, return a Series containing a single element, False.

    """
    assert "filename" in df.columns

    if len(df) > 0:
        empty_partition = True
    else:
        warnings.warn(f"Empty partition found")
        empty_partition = False

    if is_cudf_type(df):
        import cudf

        success_ser = cudf.Series([empty_partition])
    else:
        success_ser = pd.Series([empty_partition])

    if empty_partition:
        filename = df.filename.iloc[0]
        num_filenames = len(df.filename.unique())
        if num_filenames > 1:
            raise ValueError(
                f"More than one filename found in partition: {num_filenames}"
            )
        filename = Path(filename).stem
        output_file_path = os.path.join(output_file_dir, filename)
        if output_type == "jsonl":
            output_file_path = output_file_path + ".jsonl"
            if isinstance(df, pd.DataFrame):
                df.to_json(
                    output_file_path, orient="records", lines=True, force_ascii=False
                )
            else:
                # See open issue here: https://github.com/rapidsai/cudf/issues/15211
                # df.to_json(
                #     output_file_path, orient="records", lines=True, engine="cudf", force_ascii=False
                # )
                df.to_json(
                    output_file_path, orient="records", lines=True, force_ascii=False
                )
        elif output_type == "parquet":
            output_file_path = output_file_path + ".parquet"
            df.to_parquet(output_file_path)
        else:
            raise ValueError(f"Unknown output type: {output_type}")

    return success_ser


def write_to_disk(df, output_file_dir, write_to_filename=False, output_type="jsonl"):
    """
    This function writes a Dask DataFrame to the specified file path.
    If write_to_filename is True, then it expects the
    DataFrame to have a "filename" column that specifies where to write the document.

    Args:
        df: A Dask DataFrame.
        output_file_dir: The output file path.
        write_to_filename: Whether to write the filename using the "filename" column.
        output_type="jsonl": The type of output file to write.

    """
    if write_to_filename and "filename" not in df.columns:
        raise ValueError(
            "write_using_filename is True but no filename column found in df"
        )

    if write_to_filename:
        if is_cudf_type(df):
            import cudf

            output_meta = cudf.Series([True])
        else:
            output_meta = pd.Series([True], dtype="bool")

        os.makedirs(output_file_dir, exist_ok=True)
        output = df.map_partitions(
            single_partition_write_with_filename,
            output_file_dir,
            output_type=output_type,
            meta=output_meta,
            enforce_metadata=False,
        )
        output = output.compute()
    else:
        if output_type == "jsonl":
            if is_cudf_type(df):
                # See open issue here: https://github.com/rapidsai/cudf/issues/15211
                # df.to_json(output_file_dir, orient="records", lines=True, engine="cudf", force_ascii=False)
                df.to_json(
                    output_file_dir, orient="records", lines=True, force_ascii=False
                )
            else:
                df.to_json(
                    output_file_dir, orient="records", lines=True, force_ascii=False
                )
        elif output_type == "parquet":
            df.to_parquet(output_file_dir, write_index=False)
        else:
            raise ValueError(f"Unknown output type: {output_type}")

    print(f"Writing to disk complete for {df.npartitions} partitions", flush=True)


def load_object_on_worker(attr, load_object_function, load_object_kwargs):
    """
    This function checks if a Dask worker has a specified attribute for storing an object.
    If it does, then fetch the object and return it.
    If it does not, then load the object, set it as an attribute, and return it.

    Args:
        attr: A string representing an attribute of a Dask worker.
        load_object_function: A user-provided function for how to load the object.
        load_object_kwargs: A dictionary of arguments necessary for `load_object_function`.
    Returns:
        The object of interest according to the `attr`.

    """
    # get_worker will fail during type inference of Dask functions
    try:
        worker = get_worker()
    except ValueError as e:
        raise NoWorkerError(str(e))

    if hasattr(worker, attr):
        obj = getattr(worker, attr)
    else:
        obj = load_object_function(**load_object_kwargs)
        setattr(worker, attr, obj)
    return obj


def offload_object_on_worker(attr):
    """
    This function deletes an existing attribute from a Dask worker.

    Args:
        attr: The name of the attribute to delete.
    Returns:
        True.

    """
    worker = get_worker()
    if hasattr(worker, attr):
        delattr(worker, attr)
    return True


def get_num_workers(client):
    """
    Returns the number of workers in the cluster
    """
    if client is None:
        return None
    worker_list = list(client.scheduler_info()["workers"].keys())
    return len(worker_list)


def get_current_client():
    """
    Returns the context-local or latest initailised client.
    If no Client instances exist, returns None
    """
    try:
        return Client.current()
    except ValueError:
        return None


def performance_report_if(path=None, report_name="dask-profile.html"):
    if path is not None:
        return performance_report(os.path.join(path, report_name))
    else:
        return nullcontext()
