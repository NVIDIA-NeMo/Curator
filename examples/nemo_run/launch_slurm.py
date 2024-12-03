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

import nemo_run as run
from nemo_run.core.execution import SlurmExecutor

from nemo_curator.nemo_run import SlurmJobConfig


@run.factory
def nemo_curator_slurm_executor() -> SlurmExecutor:
    """
    Configure the following function with the details of your Slurm cluster
    """
    return SlurmExecutor(
        job_name_prefix="nemo-curator",
        account="my-account",
        nodes=2,
        exclusive=True,
        time="04:00:00",
        container_image="nvcr.io/nvidia/nemo:dev",
        container_mounts=["/path/on/machine:/path/in/container"],
    )


def main():
    # Path to NeMo-Curator/examples/slurm/container_entrypoint.sh on the Slurm cluster
    container_entrypoint = "/cluster/path/slurm/container_entrypoint.sh"
    # The NeMo Curator command to run
    # This command can be susbstituted with any NeMo Curator command
    curator_command = "text_cleaning --input-data-dir=/path/to/data --output-clean-dir=/path/to/output"
    curator_job = SlurmJobConfig(
        job_dir="/home/user/jobs",
        container_entrypoint=container_entrypoint,
        script_command=curator_command,
    )

    executor = run.resolve(SlurmExecutor, "nemo_curator_slurm_executor")
    with run.Experiment("example_nemo_curator_exp", executor=executor) as exp:
        exp.add(curator_job.to_script(), tail_logs=True)
        exp.run(detach=False)


if __name__ == "__main__":
    main()
