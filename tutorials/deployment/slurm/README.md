# Ray SLURM Deployment Script

The `ray-sbatch-job.sh` script automates deploying a Ray cluster on SLURM-managed HPC systems with container support (e.g., clusters using Enroot & Pyxis). It handles multi-node Ray cluster initialization and job submission.

## Quick Start

Basic usage pattern:

```bash
RUN_COMMAND="python my_script.py" \
IMAGE="nvcr.io/nvidia/nemo-curator:latest" \
CONTAINER_MOUNTS="/path/to/data:/data" \
sbatch --nodes=2 --account=my_account --partition=batch --time=01:00:00 \
    ray-sbatch-job.sh
```

## Configuration Variables

### Required Variables (Usually Set by User)

| Variable | Description | Default |
|----------|-------------|---------|
| `RUN_COMMAND` | The command to execute on the Ray cluster | Prints cluster resources |
| `IMAGE` | Container image path. Local squashfs recommended for performance (see [Slow Container Loading](#slow-container-loading)) | `nvcr.io/nvidia/nemo-curator:latest` |
| `CONTAINER_MOUNTS` | Comma-separated mount mappings (`host:container`) | Empty |

### Resource Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `NUM_CPUS_PER_NODE` | CPUs to allocate per node | Auto-detected from SLURM |
| `NUM_GPUS_PER_NODE` | GPUs to allocate per node | `8` |
| `OBJECT_STORE_MEMORY` | Ray object store memory in bytes | Ray's default (auto-calculated) |

### Optional Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `NODE_INIT_COMMAND` | Command to run on each node before starting Ray | Empty |
| `LOGURU_LEVEL` | Logging level for NeMo Curator | `INFO` |

### Port Configuration

These ports are used by Ray services. Change them if you encounter port conflicts:

| Variable | Default | Description |
|----------|---------|-------------|
| `GCS_PORT` | `6379` | Ray Global Control Store |
| `CLIENT_PORT` | `10001` | Ray client server |
| `DASH_PORT` | `8265` | Ray dashboard |
| `NODE_MANAGER_PORT` | `6800` | Node manager |
| `OBJECT_MANAGER_PORT` | `6801` | Object manager |
| `RUNTIME_ENV_AGENT_PORT` | `6802` | Runtime environment agent |
| `DASHBOARD_AGENT_GRPC_PORT` | `6803` | Dashboard agent gRPC |
| `METRICS_EXPORT_PORT` | `6804` | Metrics export |

## Examples

### GPU Job (Multi-Node)

Run a GPU-accelerated curation pipeline on 2 nodes:

```bash
CONTAINER_MOUNTS="$HOME:/workspace,/data/datasets:/datasets" \
NUM_CPUS_PER_NODE=64 \
RUN_COMMAND="python /workspace/curation_pipeline.py --input /datasets/raw --output /datasets/processed" \
IMAGE=/path/to/container.sqsh \
sbatch \
    --nodes=2 \
    --account=my_account \
    --partition=batch \
    --gpus-per-node=8 \
    --time=04:00:00 \
    --output="gpu-job-%j.out" \
    ray-sbatch-job.sh
```

### CPU-Only Job

For CPU-only workloads, set `NUM_GPUS_PER_NODE=0` and use a CPU partition:

```bash
CONTAINER_MOUNTS="$HOME:/workspace,/data/datasets:/datasets" \
NUM_CPUS_PER_NODE=56 \
NUM_GPUS_PER_NODE=0 \
RUN_COMMAND="python /workspace/cpu_pipeline.py --input /datasets/raw --output /datasets/processed" \
IMAGE=/path/to/container.sqsh \
sbatch \
    --nodes=8 \
    --account=my_account \
    --partition=cpu \
    --time=03:00:00 \
    --output="cpu-job-%j.out" \
    ray-sbatch-job.sh
```

### Using NODE_INIT_COMMAND

The `NODE_INIT_COMMAND` variable runs a command on each node before Ray starts. This is useful for:

- Cleaning up leftover Ray processes from previous jobs
- Applying patches or fixes
- Setting up environment variables
- Running pre-flight checks

Example:

```bash
NODE_INIT_COMMAND="ray stop --force && rm -rf /tmp/ray/* && export MY_VAR=value"
```

## SLURM Directives

Common SLURM options can be passed via `sbatch` flags or uncommented in the script:

| Flag | Description |
|------|-------------|
| `--nodes=N` | Number of nodes to allocate |
| `--time=HH:MM:SS` | Maximum job runtime |
| `--account=NAME` | Account/project to charge |
| `--partition=NAME` | Partition/queue to use |
| `--gpus-per-node=N` | GPUs per node (for GPU partitions) |
| `--output=FILE` | Stdout file (`%j` expands to job ID) |
| `--error=FILE` | Stderr file |

## How It Works

1. **SLURM allocates nodes** based on your `sbatch` parameters
2. **Script starts Ray head node** on the first allocated node
3. **Worker nodes connect** to the head node to form a cluster
4. **Your command runs** via the Ray Job Submission API (`ray job submit`) on the cluster
5. **Job output** is captured in the SLURM output file

## Troubleshooting

### Slow Container Loading

If container startup is slow due to network latency when pulling from a remote registry, pre-download the container image to a local squashfs file. Run this once from a compute node before submitting jobs:

```bash
ENROOT_CONNECT_TIMEOUT=0 \
ENROOT_TRANSFER_TIMEOUT=0 \
ENROOT_TRANSFER_RETRIES=5 \
enroot import -o /path/to/local/nemo-curator.sqsh docker://nvcr.io/nvidia/nemo-curator:latest
```

Then use the local path in your job submission:

```bash
IMAGE=/path/to/local/nemo-curator.sqsh sbatch ... ray-sbatch-job.sh
```

This eliminates network overhead during job execution and ensures consistent container availability across all nodes.

### Container Mount Issues

Ensure mount paths exist and use the format `host_path:container_path`:

```bash
CONTAINER_MOUNTS="/lustre/data:/data,/home/user:/workspace"
```

**Note:** Paths used in `RUN_COMMAND`, `NODE_INIT_COMMAND`, and other variables refer to paths *inside the container*. Make sure to mount your host directories to the appropriate container paths and reference the container paths in your commands.

### Ray Startup Conflicts

If Ray startup fails due to leftover processes or state from a previous job that didn't clean up properly, use `NODE_INIT_COMMAND` to clean up before starting Ray:

```bash
NODE_INIT_COMMAND="ray stop --force && rm -rf /tmp/ray/*"
```

### Port Conflicts

If you see errors about ports already in use, another job may be using the default ports. Override them:

```bash
GCS_PORT=7379 DASH_PORT=9265 sbatch ... ray-sbatch-job.sh
```

## Non-Container Environments

This script has been tested with Pyxis/Enroot container setups. For SLURM environments without container support, the script can be adapted as follows:

1. Remove `--container-image` and `--container-mounts` flags from `srun` commands
2. Optionally use `module load` commands and/or source a virtual environment in each `srun` call
