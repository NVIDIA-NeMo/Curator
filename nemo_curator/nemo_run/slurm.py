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

from dataclasses import dataclass
from typing import Dict

from nemo_curator.utils.import_utils import safe_import

run = safe_import("nemo_run")


@dataclass
class SlurmJobConfig:
    """
    Configuration for running a NeMo Curator script on a Slurm cluster using
    NeMo Run

    Args:
        job_dir: The base directory where all the files related to setting up
            the Dask cluster for NeMo Curator will be written
        container_entrypoint: A path to the container-entrypoint.sh script
            on the cluster. container-entrypoint.sh is found in the repo
            here: https://github.com/NVIDIA/NeMo-Curator/blob/main/examples/slurm/container-entrypoint.sh
        script_command: The NeMo Curator CLI tool to run. Pass any additional arguments
            needed directly in this string.
        device: The type of script that will be running, and therefore the type
            of Dask cluster that will be created. Must be either "cpu" or "gpu".
        interface: The network interface the Dask cluster will communicate over.
            Use nemo_curator.get_network_interfaces() to get a list of available ones.
        protocol: The networking protocol to use. Can be either "tcp" or "ucx".
            Setting to "ucx" is recommended for GPU jobs if your cluster supports it.
        cpu_worker_memory_limit: The maximum memory per process that a Dask worker can use.
            "5GB" or "5000M" are examples. "0" means no limit.
        rapids_no_initialize: Will delay or disable the CUDA context creation of RAPIDS libraries,
            allowing for improved compatibility with UCX-enabled clusters and preventing runtime warnings.
        cudf_spill: Enables automatic spilling (and “unspilling”) of buffers from device to host to
            enable out-of-memory computation, i.e., computing on objects that occupy more memory
            than is available on the GPU.
        rmm_scheduler_pool_size: Sets a small pool of GPU memory for message transfers when
            the scheduler is using ucx
        rmm_worker_pool_size: The amount of GPU memory each GPU worker process may use.
            Recommended to set at 80-90% of available GPU memory. 72GiB is good for A100/H100
        libcudf_cufile_policy: Allows reading/writing directly from storage to GPU.
    """

    job_dir: str
    container_entrypoint: str
    script_command: str
    device: str = "cpu"
    interface: str = "eth0"
    protocol: str = "tcp"
    cpu_worker_memory_limit: str = "0"
    rapids_no_initialize: str = "1"
    cudf_spill: str = "1"
    rmm_scheduler_pool_size: str = "1GB"
    rmm_worker_pool_size: str = "72GiB"
    libcudf_cufile_policy: str = "OFF"

    def to_script(self, add_scheduler_file: bool = True, add_device: bool = True):
        """
        Converts to a script object executable by NeMo Run
        Args:
            add_scheduler_file: Automatically appends a '--scheduler-file' argument to the
                script_command where the value is job_dir/logs/scheduler.json. All
                scripts included in NeMo Curator accept and require this argument to scale
                properly on Slurm clusters.
            add_device: Automatically appends a '--device' argument to the script_command
                where the value is the member variable of device. All scripts included in
                NeMo Curator accept and require this argument.
        Returns:
            A NeMo Run Script that will intialize a Dask cluster, and run the specified command.
            It is designed to be executed on a Slurm cluster
        """
        env_vars = self._build_env_vars()

        if add_scheduler_file:
            env_vars[
                "SCRIPT_COMMAND"
            ] += f" --scheduler-file={env_vars['SCHEDULER_FILE']}"
        if add_device:
            env_vars["SCRIPT_COMMAND"] += f" --device={env_vars['DEVICE']}"

        # Surround the command in quotes so the variable gets set properly
        env_vars["SCRIPT_COMMAND"] = f"\"{env_vars['SCRIPT_COMMAND']}\""

        return run.Script(path=self.container_entrypoint, env=env_vars)

    def _build_env_vars(self) -> Dict[str, str]:
        env_vars = vars(self)
        # Convert to uppercase to match container_entrypoint.sh
        env_vars = {key.upper(): val for key, val in env_vars.items()}

        env_vars["LOGDIR"] = f"{self.job_dir}/logs"
        env_vars["PROFILESDIR"] = f"{self.job_dir}/profiles"
        env_vars["SCHEDULER_FILE"] = f"{env_vars['LOGDIR']}/scheduler.json"
        env_vars["SCHEDULER_LOG"] = f"{env_vars['LOGDIR']}/scheduler.log"
        env_vars["DONE_MARKER"] = f"{env_vars['LOGDIR']}/done.txt"

        return env_vars
