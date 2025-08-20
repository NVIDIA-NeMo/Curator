---
description: "Deploy NeMo Curator for all modalities on Slurm clusters with job scripts, Dask cluster setup, and Python-based job submission"
categories: ["how-to-guides"]
tags: ["slurm", "deployment", "dask-cluster", "job-scripts", "container", "shared-filesystem", "multi-modal"]
personas: ["admin-focused", "devops-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "universal"
---

(admin-deployment-slurm-general)=
# Deploy All Modalities on Slurm

## Prerequisites

* Access to a Slurm cluster with a shared filesystem (for example, NFS, Lustre) mounted on all nodes
* [Dask](https://docs.dask.org/en/stable/) and [dask-cuda](https://docs.rapids.ai/api/dask-cuda/stable/) (for GPU jobs) installed in your environment or container
* Python 3.8+ environment (virtualenv, conda, or container)
* (Optional) [Singularity/Apptainer](https://apptainer.org/) or Docker for containerized execution
* Sufficient permissions to submit jobs with `sbatch`/`srun`

## Storage

NeMo Curator requires a shared filesystem accessible from all compute nodes. Place your input data and output directories on this shared storage.

```{admonition} Note
Unlike Kubernetes, Slurm does not manage storage. Ensure your data is accessible to all nodes via a shared filesystem.
```

## Set Up Python Environment

You can use a Python virtual environment or a container. For a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .[all]  # Install NeMo Curator and dependencies
```

Or use a container image (recommended for reproducibility):

- Build or pull a container with NeMo Curator and Dask installed
- Mount your shared storage into the container at runtime

```{seealso}
For details on available container environments and configurations, see [Container Environments](reference-infrastructure-container-environments).

**Configuration**: For Slurm-specific environment variables and performance tuning, see {doc}`Deployment Environment Configuration <../../config/deployment-environments>` and {doc}`Environment Variables Reference <../../config/environment-variables>`.
```

## Example Slurm Job Script

The repository provides example scripts in `examples/slurm/`:

- `start-slurm.sh`: Slurm job script for launching a Dask cluster and running a NeMo Curator module
- `container-entrypoint.sh`: Entrypoint script that starts the Dask scheduler/workers and runs your command

Below is a simplified example based on `start-slurm.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=nemo-curator-job
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --time=04:00:00

# Set up job directories
export BASE_JOB_DIR=$PWD/nemo-curator-jobs
export JOB_DIR=$BASE_JOB_DIR/$SLURM_JOB_ID
export LOGDIR=$JOB_DIR/logs
export PROFILESDIR=$JOB_DIR/profiles
export SCHEDULER_FILE=$LOGDIR/scheduler.json
export SCHEDULER_LOG=$LOGDIR/scheduler.log
export DONE_MARKER=$LOGDIR/done.txt

# Main script to run (update this for your use case)
export DEVICE='cpu'  # or 'gpu'
export SCRIPT_PATH=/path/to/your_script.py
export SCRIPT_COMMAND="python $SCRIPT_PATH --scheduler-file $SCHEDULER_FILE --device $DEVICE"

# Container parameters (if using containers)
export CONTAINER_IMAGE=/path/to/container
export BASE_DIR=$PWD
export MOUNTS="$BASE_DIR:$BASE_DIR"
export CONTAINER_ENTRYPOINT=$BASE_DIR/examples/slurm/container-entrypoint.sh

# Network interface (update as needed)
export INTERFACE=eth0
export PROTOCOL=tcp

# Start the container and entrypoint
srun \
    --container-mounts=${MOUNTS} \
    --container-image=${CONTAINER_IMAGE} \
    ${CONTAINER_ENTRYPOINT}
```

```{admonition} Note
You must update `SCRIPT_PATH`, `CONTAINER_IMAGE`, and mount paths for your environment. See `examples/slurm/start-slurm.sh` for a full template.

**Storage & Credentials**: If your job requires cloud storage access, see {doc}`Storage & Credentials Configuration <../../config/storage-credentials>` for setting up AWS, Azure, or GCS credentials in your Slurm environment.
```

```{seealso}
For complete details on Slurm environment variables and their defaults, see [Slurm Environment Variables](reference-infrastructure-container-environments-slurm).
```

## Example Entrypoint Script

The `container-entrypoint.sh` script (see `examples/slurm/container-entrypoint.sh`) starts the Dask scheduler and workers, then runs your command:

```bash
#!/bin/bash
# ... (see repo for full script)

# Start scheduler on rank 0
if [[ -z "$SLURM_NODEID" ]] || [[ $SLURM_NODEID == 0 ]]; then
  mkdir -p $LOGDIR
  mkdir -p $PROFILESDIR
  dask scheduler \
    --scheduler-file $SCHEDULER_FILE \
    --protocol $PROTOCOL \
    --interface $INTERFACE >> $SCHEDULER_LOG 2>&1 &
fi
sleep 30
# Start workers on all nodes
export WORKER_LOG=$LOGDIR/worker_${SLURM_NODEID}-${SLURM_LOCALID}.log
dask worker \
    --scheduler-file $SCHEDULER_FILE \
    --memory-limit $CPU_WORKER_MEMORY_LIMIT \
    --nworkers -1 \
    --interface $INTERFACE >> $WORKER_LOG 2>&1 &
sleep 60
# Run the main script on rank 0
if [[ -z "$SLURM_NODEID" ]] || [[ $SLURM_NODEID == 0 ]]; then
  bash -c "$SCRIPT_COMMAND"
  touch $DONE_MARKER
fi
# Wait for completion
while [ ! -f $DONE_MARKER ]; do sleep 15; done
```

## Upload Data to Shared Filesystem

Copy your input data to the shared filesystem accessible by all nodes. For example:

```bash
cp -r /local/path/my_dataset /shared/path/my_dataset
```

## Running a NeMo Curator Module

To run a NeMo Curator module (for example, fuzzy deduplication), update `SCRIPT_PATH` and `SCRIPT_COMMAND` in your job script. For example:

```bash
export SCRIPT_PATH=/path/to/nemo_curator/scripts/fuzzy_deduplication/jaccard_compute.py
export SCRIPT_COMMAND="python $SCRIPT_PATH --input-data-dirs /shared/path/my_dataset --output-dir /shared/path/output --scheduler-file $SCHEDULER_FILE --device $DEVICE"
```

Your script should use the `get_client` function from `nemo_curator.utils.distributed_utils` to connect to the Dask cluster. Example (from `jaccard_compute.py`):

```python
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.utils.script_utils import ArgumentHelper
# ...
client = get_client(**ArgumentHelper.parse_client_args(args))
```

## Adapting Your Scripts for Slurm/Dask

- Add distributed arguments using `ArgumentHelper.add_distributed_args()` or `parse_gpu_dedup_args()`
- Use `get_client(**ArgumentHelper.parse_client_args(args))` to connect to the Dask cluster
- Pass `--scheduler-file $SCHEDULER_FILE` and `--device $DEVICE` as command-line arguments

## Monitoring and Logs

- Scheduler and worker logs are written to `$LOGDIR` (see job script)
- Output and intermediate files should be written to the shared filesystem

## Cleaning Up

After your job completes, clean up any temporary files or job directories as needed:

```bash
rm -rf $BASE_JOB_DIR
```

## Advanced: Python-Based Slurm Job Submission

You can also launch jobs programmatically using the `nemo_run` package. See `examples/nemo_run/launch_slurm.py`:

```python
import nemo_run as run
from nemo_run.core.execution import SlurmExecutor
from nemo_curator.nemo_run import SlurmJobConfig

# Configure the Slurm executor
executor = SlurmExecutor(
    job_name_prefix="nemo-curator",
    account="my-account",
    nodes=2,
    exclusive=True,
    time="04:00:00",
    container_image="/path/to/container",
    container_mounts=["/shared/path:/shared/path"],
)

# Define the job
curator_job = SlurmJobConfig(
    job_dir="/shared/path/jobs",
    container_entrypoint="/shared/path/examples/slurm/container-entrypoint.sh",
    script_command="python /path/to/your_script.py --scheduler-file $SCHEDULER_FILE --device $DEVICE",
)

with run.Experiment("example_nemo_curator_exp", executor=executor) as exp:
    exp.add(curator_job.to_script(), tail_logs=True)
    exp.run(detach=False)
```

## Deleting Output and Cleaning Up

After your job is finished, you can remove job directories and outputs as needed:

```bash
rm -rf $BASE_JOB_DIR
```

## References

- See `examples/slurm/` in the repository for full job and entrypoint script templates
- See the [Kubernetes deployment guide](admin-deployment-kubernetes) for a conceptual comparison
- For more on Dask with Slurm: [Dask Jobqueue Slurm Docs](https://jobqueue.dask.org/en/latest/generated/dask_jobqueue.SlurmCluster.html) 