# Running NeMo Curator on SLURM

This tutorial shows how to scale a NeMo Curator pipeline from a single laptop to a multi-node SLURM cluster with a **one-line change**.

## Contents

| File | Purpose |
|------|---------|
| `pipeline.py` | A simple CPU-only pipeline (word-count + node-tag) that runs locally or on SLURM |
| `submit.sh` | `sbatch` script for bare-metal clusters with a shared virtualenv |
| `submit_container.sh` | `sbatch` script using the official NGC container (Pyxis/enroot) |

---

## The key concept: RayClient vs SlurmRayClient

NeMo Curator uses a `RayClient` to manage the Ray cluster lifecycle. The `SlurmRayClient` is a drop-in replacement that handles the multi-process SLURM model automatically.

```python
# Local development — Ray starts on the current machine
ray_client = RayClient()

# SLURM multi-node — Ray spans all allocated nodes automatically
ray_client = SlurmRayClient()

# One-liner to auto-detect the environment:
ray_client = SlurmRayClient() if os.environ.get("SLURM_JOB_ID") else RayClient()
```

That is the **only change** needed to go from a local run to a distributed SLURM job. Everything else — pipeline stages, executor, `pipeline.run()` — is identical.

### How SlurmRayClient works

When `srun` launches one Python process per node, `SlurmRayClient.start()` behaves differently on each node:

```
srun --ntasks-per-node=1 python pipeline.py --slurm
         │
         ├─ Node 0 (SLURM_NODEID=0) — HEAD
         │    start() → ray start --head
         │            → writes GCS port to shared file
         │            → waits for all workers to join
         │            → returns  ← pipeline runs here
         │
         ├─ Node 1 — WORKER
         │    start() → reads port file from Node 0
         │            → ray start --block --address=<head>:<port>
         │            → blocks here (serving Ray tasks)
         │
         └─ Node N — WORKER  (same as Node 1)
```

Worker nodes never return from `start()`. They serve Ray remote tasks dispatched by the Xenna executor running on the head. When `ray_client.stop()` is called on the head, the `ray stop` signal propagates and worker `srun` tasks exit.

---

## Quick start — local run

No SLURM needed. This is useful for iterating on pipeline logic.

```bash
# Install NeMo Curator
pip install nemo-curator

# Run locally (RayClient, single machine)
python tutorials/slurm/pipeline.py

# Expected output:
# Tasks processed by 1 distinct node(s): ['your-hostname']
```

---

## SLURM run — NGC container (Pyxis/enroot)

The recommended approach on clusters that support it. The official NeMo Curator image from NGC provides a stable Python environment; the local virtualenv (on your shared filesystem) is activated inside the container to pick up any unreleased code from your checkout.

### Prerequisites

Check that your cluster has the Pyxis SLURM plugin:

```bash
srun --help | grep container-image
# Should print: --container-image=...
```

If this flag is missing, ask your cluster admin or see the [bare-metal section](#slurm-run--bare-metal-shared-virtualenv) below.

### 1. Build the virtualenv on a shared filesystem

```bash
# From the NeMo Curator root on a login node (or wherever the shared FS is mounted)
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Submit the job

```bash
# Default: 2 nodes, 2 GPUs each, nvcr.io/nvidia/nemo-curator:26.02
sbatch tutorials/slurm/submit_container.sh

# Override container image
export CONTAINER_IMAGE="nvcr.io/nvidia/nemo-curator:25.06"
sbatch tutorials/slurm/submit_container.sh

# Override mounts (default: /lustre:/lustre)
export CONTAINER_MOUNTS="/scratch:/scratch,/data:/data"
sbatch tutorials/slurm/submit_container.sh
```

Override resources without editing the script:

```bash
sbatch --nodes=1 --gpus-per-node=8 tutorials/slurm/submit_container.sh
sbatch --nodes=4 --cpus-per-task=32 --time=00:30:00 tutorials/slurm/submit_container.sh
```

### 3. Check the output

```bash
tail -f logs/slurm_demo_container_<JOB_ID>.log
```

On a 2-node run you should see both hostnames in the processed-by summary:

```
Tasks processed by 2 distinct node(s):
  node-001: 2 GPU(s): NVIDIA A100-SXM4-80GB, 81251 MiB; NVIDIA A100-SXM4-80GB, 81251 MiB
  node-002: 2 GPU(s): NVIDIA A100-SXM4-80GB, 81251 MiB; NVIDIA A100-SXM4-80GB, 81251 MiB
```

### Singularity / Apptainer

If your cluster uses Singularity or Apptainer instead of Pyxis:

```bash
# Pull the image once (on the login node)
singularity pull nemo-curator.sif docker://nvcr.io/nvidia/nemo-curator:26.02

# In your sbatch script, replace the srun flags with:
srun singularity exec \
    --nv \
    --bind /lustre:/lustre \
    nemo-curator.sif \
    bash -c "source /path/to/Curator/.venv/bin/activate && python pipeline.py --slurm"
```

---

## SLURM run — bare metal (shared virtualenv)

Use this if your cluster does not have a container runtime.

### 1. Install on shared filesystem

Build a virtualenv on a **shared filesystem** (Lustre, NFS, GPFS) so every node sees the same Python environment:

```bash
# On the login node, from the NeMo Curator root
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Submit the job

```bash
sbatch tutorials/slurm/submit.sh
```

Override resources without editing the script:

```bash
sbatch --nodes=4 --cpus-per-task=32 --time=00:30:00 tutorials/slurm/submit.sh
```

### 3. Check the output

```bash
tail -f logs/slurm_demo_<JOB_ID>.log
```

---

## Configuration reference

### SlurmRayClient parameters

```python
SlurmRayClient(
    # Ray GCS port — defaults to a random free port
    ray_port=6379,

    # Shared directory for Ray temp files (logs, sockets)
    # Must be visible to all nodes
    ray_temp_dir="/tmp/ray",

    # Resource overrides (auto-detected from SLURM env vars if not set)
    num_gpus=8,   # GPUs per node
    num_cpus=64,  # CPUs per node

    # How long to wait for all worker nodes to join (seconds)
    worker_connect_timeout_s=300,
)
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RAY_PORT_BROADCAST_DIR` | `/tmp` | Directory for the port-broadcast file. **Set to a shared filesystem path when `/tmp` is not shared across nodes.** |
| `RAY_TMPDIR` | `/tmp/ray` | Ray temp directory. Recommend setting to `/tmp/ray_${SLURM_JOB_ID}` to avoid cross-job collisions. |
| `SLURM_JOB_ID` | set by SLURM | Used to name the port-broadcast file. Set manually if testing outside SLURM. |

> **Important**: If your cluster's `/tmp` is local to each node (the common case), set
> `RAY_PORT_BROADCAST_DIR` to a Lustre/NFS path so all nodes can read the port file:
>
> ```bash
> export RAY_PORT_BROADCAST_DIR=/lustre/my-project/ray_ports
> ```

---

## Adapting to your own pipeline

Switching any existing pipeline from `RayClient` to `SlurmRayClient` is the same one-line change shown in `pipeline.py`:

```python
# Before (local only):
from nemo_curator.core.client import RayClient
ray_client = RayClient()

# After (works locally AND on SLURM):
from nemo_curator.core.client import RayClient, SlurmRayClient
ray_client = SlurmRayClient() if os.environ.get("SLURM_JOB_ID") else RayClient()
```

Then wrap your `pipeline.run()` call in `srun`:

```bash
# In your sbatch script:
srun --ntasks-per-node=1 python my_pipeline.py
```

No other changes to stages, executor, or pipeline logic are required.

---

## Troubleshooting

**Workers not joining the cluster**

The most common cause is that `/tmp` is node-local so workers cannot read the port file written by the head. Fix:

```bash
export RAY_PORT_BROADCAST_DIR=/shared/filesystem/path
```

**`TimeoutError: ray.init timed out`**

The GCS port file exists but `ray.init()` hung. This usually means a firewall is blocking inter-node communication. Verify that the GCS port (default: random in 20000–30000) is open between nodes, or pin a known-open port:

```python
SlurmRayClient(ray_port=6379)
```

**Jobs finish too quickly / no tasks processed**

Ensure `--num-tasks` is larger than the number of workers × 2, otherwise all tasks may be completed before workers connect. The script will warn you:

```
Job allocated 2 nodes but only 1 node(s) processed tasks.
Check that --num-tasks is large enough to distribute across all workers.
```

**Container image not found**

```bash
# Pull manually and verify
docker pull nvcr.io/nvidia/nemo-curator:26.02
# or with enroot:
enroot import docker://nvcr.io/nvidia/nemo-curator:26.02
```

**`ImportError: cannot import name 'SlurmRayClient'`**

The container image has an older NeMo Curator without `SlurmRayClient`. Activating the local virtualenv (`source .venv/bin/activate`) inside the container overrides the container's installed version with your local checkout. Make sure the virtualenv was built from a source tree that includes `SlurmRayClient`.
