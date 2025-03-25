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

import os
from unittest.mock import MagicMock, patch

import pytest

from nemo_curator.nemo_run.slurm import SlurmJobConfig


class TestSlurmJobConfig:

    @pytest.fixture
    def basic_config(self):
        """Returns a basic SlurmJobConfig with required parameters"""
        return SlurmJobConfig(
            job_dir="/path/to/job",
            container_entrypoint="/path/to/container-entrypoint.sh",
            script_command="nemo_curator_tool arg1 arg2",
        )

    @pytest.fixture
    def custom_config(self):
        """Returns a SlurmJobConfig with custom parameters"""
        return SlurmJobConfig(
            job_dir="/path/to/custom/job",
            container_entrypoint="/path/to/custom/container-entrypoint.sh",
            script_command="custom_tool --arg1=val1 --arg2=val2",
            device="gpu",
            interface="ib0",
            protocol="ucx",
            cpu_worker_memory_limit="8GB",
            rapids_no_initialize="0",
            cudf_spill="0",
            rmm_scheduler_pool_size="2GB",
            rmm_worker_pool_size="80GiB",
            libcudf_cufile_policy="ON",
        )

    def test_initialize_with_defaults(self, basic_config):
        """Test initializing SlurmJobConfig with default values"""
        # Check required parameters
        assert basic_config.job_dir == "/path/to/job"
        assert basic_config.container_entrypoint == "/path/to/container-entrypoint.sh"
        assert basic_config.script_command == "nemo_curator_tool arg1 arg2"

        # Check default values
        assert basic_config.device == "cpu"
        assert basic_config.interface == "eth0"
        assert basic_config.protocol == "tcp"
        assert basic_config.cpu_worker_memory_limit == "0"
        assert basic_config.rapids_no_initialize == "1"
        assert basic_config.cudf_spill == "1"
        assert basic_config.rmm_scheduler_pool_size == "1GB"
        assert basic_config.rmm_worker_pool_size == "72GiB"
        assert basic_config.libcudf_cufile_policy == "OFF"

    def test_initialize_with_custom_values(self, custom_config):
        """Test initializing SlurmJobConfig with custom values"""
        # Check required parameters
        assert custom_config.job_dir == "/path/to/custom/job"
        assert (
            custom_config.container_entrypoint
            == "/path/to/custom/container-entrypoint.sh"
        )
        assert custom_config.script_command == "custom_tool --arg1=val1 --arg2=val2"

        # Check custom values
        assert custom_config.device == "gpu"
        assert custom_config.interface == "ib0"
        assert custom_config.protocol == "ucx"
        assert custom_config.cpu_worker_memory_limit == "8GB"
        assert custom_config.rapids_no_initialize == "0"
        assert custom_config.cudf_spill == "0"
        assert custom_config.rmm_scheduler_pool_size == "2GB"
        assert custom_config.rmm_worker_pool_size == "80GiB"
        assert custom_config.libcudf_cufile_policy == "ON"

    def test_build_env_vars(self, basic_config):
        """Test the _build_env_vars method"""
        env_vars = basic_config._build_env_vars()

        # Check that keys are uppercased
        assert "JOB_DIR" in env_vars
        assert "CONTAINER_ENTRYPOINT" in env_vars
        assert "SCRIPT_COMMAND" in env_vars
        assert "DEVICE" in env_vars

        # Check derived variables
        assert env_vars["LOGDIR"] == "/path/to/job/logs"
        assert env_vars["PROFILESDIR"] == "/path/to/job/profiles"
        assert env_vars["SCHEDULER_FILE"] == "/path/to/job/logs/scheduler.json"
        assert env_vars["SCHEDULER_LOG"] == "/path/to/job/logs/scheduler.log"
        assert env_vars["DONE_MARKER"] == "/path/to/job/logs/done.txt"

        # Check values
        assert env_vars["JOB_DIR"] == "/path/to/job"
        assert env_vars["CONTAINER_ENTRYPOINT"] == "/path/to/container-entrypoint.sh"
        assert env_vars["SCRIPT_COMMAND"] == "nemo_curator_tool arg1 arg2"
        assert env_vars["DEVICE"] == "cpu"

    def test_to_script_with_defaults(self, basic_config):
        """Test to_script method with default arguments"""
        # Mock the run.Script class
        with patch("nemo_curator.nemo_run.slurm.run") as mock_run:
            mock_script = MagicMock()
            mock_run.Script.return_value = mock_script

            script = basic_config.to_script()

            # Verify Script was created with correct parameters
            mock_run.Script.assert_called_once()
            call_args = mock_run.Script.call_args[1]
            assert call_args["path"] == "/path/to/container-entrypoint.sh"

            # Check env variables
            env_vars = call_args["env"]
            assert (
                env_vars["SCRIPT_COMMAND"]
                == '"nemo_curator_tool arg1 arg2 --scheduler-file=/path/to/job/logs/scheduler.json --device=cpu"'
            )

            # Check that the script object was returned
            assert script == mock_script

    def test_to_script_with_custom_options(self, basic_config):
        """Test to_script method with custom add_scheduler_file and add_device arguments"""
        # Mock the run.Script class
        with patch("nemo_curator.nemo_run.slurm.run") as mock_run:
            mock_script = MagicMock()
            mock_run.Script.return_value = mock_script

            # Don't add scheduler file or device arguments
            script = basic_config.to_script(add_scheduler_file=False, add_device=False)

            # Verify Script was created with correct parameters
            mock_run.Script.assert_called_once()
            call_args = mock_run.Script.call_args[1]
            assert call_args["path"] == "/path/to/container-entrypoint.sh"

            # Check env variables - should not contain scheduler file or device
            env_vars = call_args["env"]
            assert env_vars["SCRIPT_COMMAND"] == '"nemo_curator_tool arg1 arg2"'

            # Check that it doesn't have the scheduler file or device arguments
            assert "--scheduler-file" not in env_vars["SCRIPT_COMMAND"]
            assert "--device" not in env_vars["SCRIPT_COMMAND"]

    def test_to_script_with_gpu_device(self, custom_config):
        """Test to_script method with a GPU device configuration"""
        # Mock the run.Script class
        with patch("nemo_curator.nemo_run.slurm.run") as mock_run:
            mock_script = MagicMock()
            mock_run.Script.return_value = mock_script

            script = custom_config.to_script()

            # Verify Script was created with correct parameters
            call_args = mock_run.Script.call_args[1]
            env_vars = call_args["env"]

            # Check that the device is set to GPU
            assert "--device=gpu" in env_vars["SCRIPT_COMMAND"]
            assert env_vars["DEVICE"] == "gpu"

            # Check UCX-specific settings
            assert env_vars["PROTOCOL"] == "ucx"
            assert env_vars["RMM_WORKER_POOL_SIZE"] == "80GiB"
