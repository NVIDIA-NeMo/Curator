# Running NeMo Curator on SLURM

This tutorial shows how to scale a NeMo Curator pipeline from a single laptop to a multi-node SLURM cluster with a **one-line change**.

## Contents

| File | Purpose |
|------|---------|
| `pipeline.py` | A simple CPU-only pipeline (word-count + node-tag) that runs locally or on SLURM |
| `submit.sh` | `sbatch` script for bare-metal clusters with a shared virtualenv |
| `submit_container.sh` | `sbatch` script using the official NGC container (Pyxis/enroot) |
| `array_pipeline.py` | Generic JSONL/Parquet pipeline that processes one Slurm array shard |
| `submit_array.sh` | `sbatch --array` script for splitting many input files across independent jobs |

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

## SLURM job arrays — JSONL or Parquet file sharding

Use `submit_array.sh` when you already have a large directory of text data files and want to split the file set across many independent Slurm jobs. Each array task starts its own Curator pipeline, hashes the input file partitions deterministically, and processes only the partitions assigned to that task.

This pattern is useful when the dataset is naturally represented as many JSONL or Parquet files and you want simple horizontal scaling without coordination between jobs.

### 1. Build the virtualenv on a shared filesystem

The array example uses the official NGC container for the base environment, then activates your local checkout inside the container so unreleased source changes are picked up:

```bash
cd /path/to/Curator
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Make sure `CURATOR_DIR`, `INPUT_DIR`, and `OUTPUT_DIR` are visible from every compute node, either because they are on a shared filesystem or because you set `CONTAINER_MOUNTS` to expose the right host paths inside the container.

### 2. Submit a JSONL array job

By default, `submit_array.sh` reads JSONL files and writes JSONL output:

```bash
export CURATOR_DIR=/path/to/Curator
export INPUT_DIR=/shared/data/my-jsonl-dataset
export OUTPUT_DIR=/shared/output/my-jsonl-dataset

# 20 array tasks, task IDs 0-19
sbatch --array=0-19 tutorials/slurm/submit_array.sh
```

For example, if the input directory contains 2000 files and `FILES_PER_PARTITION=1`, each of the 20 array tasks receives roughly 100 file partitions. Assignment is hash-based rather than contiguous, so work remains stable if Slurm retries a task.

Single-node array tasks use `RayClient`. If you override the allocation to use more than one node per array task, `submit_array.sh` automatically passes `--slurm` to `array_pipeline.py`, which switches that task to `SlurmRayClient` so the nodes form one Ray cluster:

```bash
sbatch --array=0-9 --nodes=2 --cpus-per-task=32 tutorials/slurm/submit_array.sh
```

### 3. Use Parquet instead

Set the input and output file types to `parquet`:

```bash
export INPUT_DIR=/shared/data/my-parquet-dataset
export OUTPUT_DIR=/shared/output/my-parquet-dataset
export INPUT_FILE_TYPE=parquet
export OUTPUT_FILE_TYPE=parquet

sbatch --array=0-19 tutorials/slurm/submit_array.sh
```

### 4. Edit sharding logic

If your array does not start at zero, set `MINIMUM_SHARD_INDEX` to the first task ID:

```bash
MINIMUM_SHARD_INDEX=1 sbatch --array=1-20 tutorials/slurm/submit_array.sh
```

If your cluster limits the number of tasks in a single Slurm array, you can still use a larger logical shard count by overriding `TOTAL_SHARDS` and submitting the shard ID range in multiple windows. For example, if you want 10,000 logical shards but the cluster allows only 1,000 array tasks per submission:

```bash
export TOTAL_SHARDS=10000

sbatch --array=0-999 tutorials/slurm/submit_array.sh
sbatch --array=1000-1999 tutorials/slurm/submit_array.sh
sbatch --array=2000-2999 tutorials/slurm/submit_array.sh
# ...
sbatch --array=9000-9999 tutorials/slurm/submit_array.sh
```

In this mode, keep `MINIMUM_SHARD_INDEX=0` because the Slurm array task IDs are already the global shard IDs. Each partition is assigned by `hash(partition) % TOTAL_SHARDS`, so the full set of windowed submissions covers shards `0` through `9999` exactly once. Some individual tasks may receive no files if `TOTAL_SHARDS` is larger than the number of file partitions.

Some clusters enforce the maximum array index rather than just the number of tasks per submitted array. If `--array=1000-1999` is rejected, this windowing pattern needs an explicit shard-index offset in the submission script rather than higher Slurm task IDs.

### 5. Retry failed array tasks only

When `submit_array.sh` launches `array_pipeline.py`, it passes `--checkpoint-path`. At startup, the driver process for each array task creates a pending retry manifest under:

```bash
${CHECKPOINT_PATH:-$OUTPUT_DIR}/.nemo_curator_metadata/.slurm_array_retry/
```

In other words, retries are tracked at `checkpoint_path/.nemo_curator_metadata/.slurm_array_retry/`.

If the shard completes successfully, that shard's matching retry manifests are removed. If the process fails, is preempted, or reaches the Slurm time limit before cleanup runs, the manifest remains in the retry directory. Caught Python exceptions update the manifest with `status="failed"` and the error message; hard termination may leave `status="pending"`, which should still be treated as retryable after the original Slurm array has finished.

Retry manifests are uniquely named JSON files written with an atomic rename, so multiple array tasks can write to the same retry directory without coordinating through a shared database.

Each manifest records the failed `shard_index`, plus the `total_shards` and `minimum_shard_index` values used for the original run. To retry only the failed shards, rebuild a Slurm array list from those manifests and preserve the original shard settings. For example, using `jq`:

```bash
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-$OUTPUT_DIR}"
RETRY_DIR="${CHECKPOINT_PATH}/.nemo_curator_metadata/.slurm_array_retry"

FAILED_SHARDS=$(jq -r '.shard_index' "${RETRY_DIR}"/manifest_*.json | sort -n -u | paste -sd, -)
TOTAL_SHARDS_VALUES=$(jq -r '.total_shards' "${RETRY_DIR}"/manifest_*.json | sort -n -u)
MINIMUM_SHARD_INDEX_VALUES=$(jq -r '.minimum_shard_index' "${RETRY_DIR}"/manifest_*.json | sort -n -u)

if [[ -z "${FAILED_SHARDS}" ]]; then
    echo "No failed shards found in ${RETRY_DIR}" >&2
    exit 1
fi

if [[ "${TOTAL_SHARDS_VALUES}" == *$'\n'* || "${MINIMUM_SHARD_INDEX_VALUES}" == *$'\n'* ]]; then
    echo "Retry manifests contain multiple shard configurations; split them by run." >&2
    exit 1
fi

export TOTAL_SHARDS="${TOTAL_SHARDS_VALUES}"
export MINIMUM_SHARD_INDEX="${MINIMUM_SHARD_INDEX_VALUES}"

sbatch --array="${FAILED_SHARDS}" tutorials/slurm/submit_array.sh
```

The `TOTAL_SHARDS` override is important. On a retry array like `--array=3,17,42`, Slurm sets `SLURM_ARRAY_TASK_COUNT=3`, but the data was originally assigned using the full logical shard count. Reusing the original `TOTAL_SHARDS` keeps `hash(partition) % total_shards` identical to the first run.

Run this retry collection after the original Slurm array has finished, otherwise still-running tasks will still have pending manifests. Use one `CHECKPOINT_PATH` per logical array run, or move old retry manifests aside after building `FAILED_SHARDS`, so later retries do not include failures that already succeeded.

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
