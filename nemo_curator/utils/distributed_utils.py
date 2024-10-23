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

import ast
import os

os.environ["RAPIDS_NO_INITIALIZE"] = "1"
import random
import warnings
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import dask.dataframe as dd
import numpy as np
import pandas as pd
import psutil
from dask.distributed import Client, LocalCluster, get_worker, performance_report

from nemo_curator.utils.gpu_utils import is_cudf_type
from nemo_curator.utils.import_utils import gpu_only_import, gpu_only_import_from

cudf = gpu_only_import("cudf")
LocalCUDACluster = gpu_only_import_from("dask_cuda", "LocalCUDACluster")
get_device_total_memory = gpu_only_import_from(
    "dask_cuda.utils", "get_device_total_memory"
)


class NoWorkerError(Exception):
    pass


def _enable_spilling():
    """
    Setting this environment variable enables automatic spilling (and "unspilling")
    of buffers from device to host to enable out-of-memory computation,
    i.e., computing on objects that occupy more memory than is available on the GPU.
    """
    # Workaround for below (which is missing in 24.08, but fixed in 24.10)
    # Remove this when we update to 24.10 or later dask-cuda
    # https://github.com/rapidsai/dask-cuda/pull/1369/files
    cudf.set_option("spill", True)


def start_dask_gpu_local_cluster(
    nvlink_only=False,
    protocol="tcp",
    rmm_pool_size="1024M",
    enable_spilling=True,
    set_torch_to_use_rmm=True,
    rmm_async=True,
    rmm_maximum_pool_size=None,
    rmm_managed_memory=False,
    rmm_release_threshold=None,
    **cluster_kwargs,
) -> Client:
    """
    This function sets up a Dask cluster across all the
    GPUs present on the machine.

    """
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
        enable_cudf_spill=enable_spilling,
        rmm_async=rmm_async,
        rmm_maximum_pool_size=rmm_maximum_pool_size,
        rmm_managed_memory=rmm_managed_memory,
        rmm_release_threshold=rmm_release_threshold,
        **extra_kwargs,
        **cluster_kwargs,
    )
    client = Client(cluster)

    if set_torch_to_use_rmm:
        _set_torch_to_use_rmm()
        client.run(_set_torch_to_use_rmm)
        print("Torch is using RMM memory pool", flush=True)

    if enable_spilling:
        _enable_spilling()
        print("cuDF Spilling is enabled", flush=True)

    assert get_num_workers(client) > 0, "No workers are currently connected."
    return client


def start_dask_cpu_local_cluster(
    n_workers=os.cpu_count(), threads_per_worker=1, **cluster_kwargs
) -> Client:
    """
    This function sets up a Dask cluster across all the
    CPUs present on the machine.

    """
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        **cluster_kwargs,
    )

    client = Client(cluster)
    assert get_num_workers(client) > 0, "No workers are currently connected."
    return client


def get_client(
    cluster_type="cpu",
    scheduler_address=None,
    scheduler_file=None,
    n_workers=os.cpu_count(),
    threads_per_worker=1,
    nvlink_only=False,
    protocol="tcp",
    rmm_pool_size="1024M",
    enable_spilling=True,
    set_torch_to_use_rmm=False,
    rmm_async=True,
    rmm_maximum_pool_size=None,
    rmm_managed_memory=False,
    rmm_release_threshold=None,
    **cluster_kwargs,
) -> Client:
    """
    Initializes or connects to a Dask cluster.
    The Dask cluster can be CPU-based or GPU-based (if GPUs are available).
    The intialization ensures maximum memory efficiency for the GPU by:
        1. Ensuring the PyTorch memory pool is the same as the RAPIDS memory pool. (If `set_torch_to_use_rmm` is True)
        2. Enabling spilling for cuDF. (If `enable_spilling` is True)

    Args:
        cluster_type: If scheduler_address and scheduler_file are None, sets up a local (single node) cluster of the specified type.
            Either "cpu" or "gpu". Defaults to "cpu". Many options in get_client only apply to CPU-based or GPU-based clusters.
            Make sure you check the description of the parameter.
        scheduler_address: Address of existing Dask cluster to connect to. This can be the address of a Scheduler server like a
            string '127.0.0.1:8786' or a cluster object like LocalCluster(). If specified, all other arguments are ignored and
            the client is connected to the existing cluster. The other configuration options should be done when setting up the
            Dask cluster.
        scheduler_file: Path to a file with scheduler information if available. If specified, all other arguments are ignored
            and the client is connected to the existing cluster. The other configuration options should be done when setting up the
            Dask cluster.
        n_workers: For CPU-based clusters only. The number of workers to start. Defaults to os.cpu_count(). For GPU-based clusters,
            the number of workers is locked to the number of GPUs in CUDA_VISIBLE_DEVICES.
        threads_per_worker: For CPU-based clusters only. The number of threads per each worker. Defaults to 1.
            Before increasing, ensure that your functions frequently release the GIL.
        nvlink_only: For GPU-based clusters only. Whether to use nvlink or not.
        protocol: For GPU-based clusters only. Protocol to use for communication. "tcp" or "ucx".
        rmm_pool_size: For GPU-based clusters only. RMM pool size to initialize each worker with. Can be an integer (bytes),
            float (fraction of total device memory), string (like "5GB" or "5000M"), or None to disable RMM pools.
        enable_spilling: For GPU-based clusters only. Enables automatic spilling (and "unspilling") of buffers from device to
            host to enable out-of-memory computation, i.e., computing on objects that occupy more memory than is available on the GPU.
        set_torch_to_use_rmm: For GPU-based clusters only. Sets up the PyTorch memory pool to be the same as the RAPIDS memory pool.
            This helps avoid OOM errors when using both PyTorch and RAPIDS on the same GPU.
        rmm_async: For GPU-based clusters only. Initializes each worker with RAPIDS Memory Manager (RMM)
            (see RMM documentation for more information: https://docs.rapids.ai/api/rmm/stable/)
            and sets it to use RMM's asynchronous allocator. Warning: The asynchronous allocator requires CUDA Toolkit 11.2 or newer.
            It is also incompatible with RMM pools and managed memory. Trying to enable both will result in an exception.
        rmm_maximum_pool_size: For GPU-based clusters only. When rmm_pool_size is set, this argument indicates the maximum pool size.
            Can be an integer (bytes), float (fraction of total device memory), string (like "5GB" or "5000M") or None.
            By default, the total available memory on the GPU is used.
            rmm_pool_size must be specified to use RMM pool and to set the maximum pool size.
            Note: When paired with --enable-rmm-async the maximum size cannot be guaranteed due to fragmentation.
            Note: This size is a per-worker configuration, and not cluster-wide.
        rmm_managed_memory: For GPU-based clusters only. Initialize each worker with RMM and set it to use managed memory.
            If disabled, RMM may still be used by specifying rmm_pool_size.
            Warning: Managed memory is currently incompatible with NVLink. Trying to enable both will result in an exception.
        rmm_release_threshold: For GPU-based clusters only. When rmm.async is True and the pool size grows beyond this value,
            unused memory held by the pool will be released at the next synchronization point.
            Can be an integer (bytes), float (fraction of total device memory), string (like "5GB" or "5000M") or None.
            By default, this feature is disabled.
            Note: This size is a per-worker configuration, and not cluster-wide.
        cluster_kwargs: Additional keyword arguments for the LocalCluster or LocalCUDACluster configuration.
            See API documentation https://docs.dask.org/en/stable/deploying-python.html#distributed.deploy.local.LocalCluster
            for all LocalCluster parameters, or https://docs.rapids.ai/api/dask-cuda/nightly/api/ for all LocalCUDACluster parameters.
    Returns:
        A Dask client object.

    """
    if cluster_type not in ["cpu", "gpu"]:
        raise ValueError(f"Unknown cluster type: {cluster_type}")

    if scheduler_address:
        if scheduler_file:
            raise ValueError(
                "Only one of scheduler_address or scheduler_file can be provided"
            )
        else:
            client = Client(address=scheduler_address, timeout="30s")
            assert get_num_workers(client) > 0, "No workers are currently connected."
            return client
    elif scheduler_file:
        client = Client(scheduler_file=scheduler_file, timeout="30s")
        assert get_num_workers(client) > 0, "No workers are currently connected."
        return client
    else:
        if cluster_type == "gpu":
            return start_dask_gpu_local_cluster(
                nvlink_only=nvlink_only,
                protocol=protocol,
                rmm_pool_size=rmm_pool_size,
                enable_spilling=enable_spilling,
                set_torch_to_use_rmm=set_torch_to_use_rmm,
                rmm_async=rmm_async,
                rmm_maximum_pool_size=rmm_maximum_pool_size,
                rmm_managed_memory=rmm_managed_memory,
                rmm_release_threshold=rmm_release_threshold,
                **cluster_kwargs,
            )
        else:
            return start_dask_cpu_local_cluster(
                n_workers=n_workers,
                threads_per_worker=threads_per_worker,
                **cluster_kwargs,
            )


def _set_torch_to_use_rmm():
    """
    This function sets up the PyTorch memory pool to be the same as the RAPIDS memory pool.
    This helps avoid OOM errors when using both PyTorch and RAPIDS on the same GPU.

    See article:
    https://medium.com/rapids-ai/pytorch-rapids-rmm-maximize-the-memory-efficiency-of-your-workflows-f475107ba4d4
    """

    import torch
    from rmm.allocators.torch import rmm_torch_allocator

    if torch.cuda.get_allocator_backend() == "pluggable":
        warnings.warn(
            "PyTorch allocator already plugged in, not switching to RMM. "
            "Please ensure you have not already swapped it."
        )
        return

    torch.cuda.memory.change_current_allocator(rmm_torch_allocator)


def read_single_partition(
    files,
    backend="cudf",
    filetype="jsonl",
    add_filename=False,
    input_meta: Union[str, dict] = None,
    columns: Optional[List[str]] = None,
    **kwargs,
) -> Union[cudf.DataFrame, pd.DataFrame]:
    """
    This function reads a file with cuDF, sorts the columns of the DataFrame
    and adds a "filename" column.

    Args:
        files: The path to the jsonl files to read.
        backend: The backend to use for reading the data. Either "cudf" or "pandas".
        add_filename: Whether to add a "filename" column to the DataFrame.
        input_meta: A dictionary or a string formatted as a dictionary, which outlines
            the field names and their respective data types within the JSONL input file.
        columns: If not None, only these columns will be read from the file.
            There is a significant performance gain when specifying columns for Parquet files.

    Returns:
        A cudf DataFrame or a pandas DataFrame.

    """
    if input_meta is not None and filetype != "jsonl":
        warnings.warn(
            "input_meta is only valid for JSONL files and will be ignored for other "
            " file formats.."
        )

    if filetype in ["jsonl", "json"]:
        read_kwargs = {"lines": filetype == "jsonl"}
        if backend == "cudf":
            read_f = cudf.read_json
            if input_meta is not None:
                read_kwargs["prune_columns"] = True
        else:
            read_kwargs["dtype"] = False
            read_f = pd.read_json

        if input_meta is not None:
            read_kwargs["dtype"] = (
                ast.literal_eval(input_meta) if type(input_meta) == str else input_meta
            )

    elif filetype == "parquet":
        read_kwargs = {"columns": columns}
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
            # Pandas does not support reading multiple files at once
            read_files_one_at_a_time = True

    if read_files_one_at_a_time:
        if backend == "cudf":
            concat_f = cudf.concat
        else:
            concat_f = pd.concat
        df_ls = []
        for file in files:
            df = read_f(file, **read_kwargs, **kwargs)
            if add_filename:
                df["filename"] = os.path.basename(file)
            df_ls.append(df)
        df = concat_f(df_ls, ignore_index=True)
    else:
        df = read_f(files, **read_kwargs, **kwargs)

    if filetype in ["jsonl", "json"] and columns is not None:
        if add_filename and "filename" not in columns:
            columns.append("filename")
        df = df[columns]

    df = df[sorted(df.columns)]
    return df


def read_pandas_pickle(
    file, add_filename=False, columns=None, **kwargs
) -> pd.DataFrame:
    """
    This function reads a pickle file with Pandas.

    Args:
        file: The path to the pickle file to read.
        add_filename: Whether to add a "filename" column to the DataFrame.
    Returns:
        A Pandas DataFrame.

    """
    if add_filename:
        warnings.warn("add_filename is not supported for pickle files")

    if columns is not None:
        return pd.read_pickle(file, **kwargs)[columns]
    else:
        return pd.read_pickle(file, **kwargs)


def read_data(
    input_files,
    file_type: str = "pickle",
    backend: str = "cudf",
    files_per_partition: int = 1,
    add_filename: bool = False,
    input_meta: Union[str, dict] = None,
    columns: Optional[List[str]] = None,
    **kwargs,
) -> Union[dd.DataFrame, dask_cudf.DataFrame]:
    """
    This function can read multiple data formats and returns a Dask-cuDF DataFrame.

    Args:
        input_files: The path of the input file(s).
        file_type: The type of the input file(s).
        backend: The backend to use for reading the data.
        files_per_partition: The number of files to read per partition.
        add_filename: Whether to add a "filename" column to the DataFrame.
        input_meta: A dictionary or a string formatted as a dictionary, which outlines
            the field names and their respective data types within the JSONL input file.
        columns: If not None, only these columns will be read from the file.
            There is a significant performance gain when specifying columns for Parquet files.

    Returns:
        A Dask-cuDF or a Dask-pandas DataFrame.

    """
    if backend == "cudf":
        # Try using cuDF. If not availible will throw an error.
        test_obj = cudf.Series

    if file_type == "pickle":
        df = read_pandas_pickle(
            input_files[0], add_filename=add_filename, columns=columns, **kwargs
        )
        df = dd.from_pandas(df, npartitions=16)
        if backend == "cudf":
            df = df.to_backend("cudf")

    elif file_type in ["json", "jsonl", "parquet"]:
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
            input_meta=input_meta,
            enforce_metadata=False,
            columns=columns,
            **kwargs,
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


def single_partition_write_with_filename(
    df,
    output_file_dir,
    keep_filename_column=False,
    output_type="jsonl",
):
    """
    This function processes a DataFrame and writes it to disk

    Args:
        df: A DataFrame.
        output_file_dir: The output file path.
        keep_filename_column: Whether to keep or drop the "filename" column, if it exists.
        output_type="jsonl": The type of output file to write.
    Returns:
        If the DataFrame is non-empty, return a Series containing a single element, True.
        If the DataFrame is empty, return a Series containing a single element, False.

    """
    assert "filename" in df.columns

    if len(df) > 0:
        empty_partition = False
    else:
        warnings.warn(f"Empty partition found")
        empty_partition = True

    if is_cudf_type(df):
        import cudf

        success_ser = cudf.Series([empty_partition])
    else:
        success_ser = pd.Series([empty_partition])

    if not empty_partition:
        filenames = df.filename.unique()
        filenames = list(filenames.values_host) if is_cudf_type(df) else list(filenames)
        num_files = len(filenames)

        for filename in filenames:
            out_df = df[df.filename == filename] if num_files > 1 else df
            if not keep_filename_column:
                out_df = out_df.drop("filename", axis=1)

            filename = Path(filename).stem
            output_file_path = os.path.join(output_file_dir, filename)

            if output_type == "jsonl":
                output_file_path = output_file_path + ".jsonl"

                if isinstance(df, pd.DataFrame):
                    out_df.to_json(
                        output_file_path,
                        orient="records",
                        lines=True,
                        force_ascii=False,
                    )
                else:
                    # See open issue here: https://github.com/rapidsai/cudf/issues/15211
                    # df.to_json(
                    #     output_file_path, orient="records", lines=True, engine="cudf", force_ascii=False
                    # )
                    out_df.to_json(
                        output_file_path,
                        orient="records",
                        lines=True,
                        force_ascii=False,
                    )

            elif output_type == "parquet":
                output_file_path = output_file_path + ".parquet"
                out_df.to_parquet(output_file_path)

            else:
                raise ValueError(f"Unknown output type: {output_type}")

    return success_ser


def write_to_disk(
    df,
    output_file_dir,
    write_to_filename=False,
    keep_filename_column=False,
    output_type="jsonl",
):
    """
    This function writes a Dask DataFrame to the specified file path.
    If write_to_filename is True, then it expects the
    DataFrame to have a "filename" column that specifies where to write the document.

    Args:
        df: A Dask DataFrame.
        output_file_dir: The output file path.
        write_to_filename: Whether to write the filename using the "filename" column.
        keep_filename_column: Whether to keep or drop the "filename" column, if it exists.
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
            keep_filename_column=keep_filename_column,
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


def performance_report_if_with_ts_suffix(
    path: Optional[str] = None, report_name: str = "dask-profile"
):
    """Suffixes the report_name with the timestamp"""
    return performance_report_if(
        path=path,
        report_name=f"{report_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
    )


def seed_all(seed: int = 42):
    """
    Function to set seed for random number generators for reproducibility.

    Args:
        seed: The seed value to use for random number generators. Default is 42.

    Returns:
        None
    """
    ## Imporing torch to help with context issues
    import torch

    # Set seed values for various random number generators
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for CUDA algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_network_interfaces() -> List[str]:
    """
    Gets a list of all valid network interfaces on a machine

    Returns:
        A list of all valid network interfaces on a machine
    """
    return list(psutil.net_if_addrs().keys())


def get_gpu_memory_info() -> Dict[str, int]:
    """
    Get the total GPU memory for each Dask worker.
    Returns:
        dict: A dictionary mapping Dask worker addresses ('IP:PORT') to their
        respective GPU memory (in bytes).
    Example:
        {'192.168.0.100:9000': 3.2e+10, '192.168.0.101:9000': 3.2e+10}
    Note:
        If there is no active Dask client, an empty dictionary is returned.
    """
    client = get_current_client()
    if client is None:
        return {}
    return client.run(get_device_total_memory)
