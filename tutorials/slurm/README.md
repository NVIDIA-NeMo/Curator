# Running NeMo Curator on SLURM

This tutorial shows how to scale a NeMo Curator pipeline from a single laptop to a multi-node SLURM cluster with a **one-line change**.

## Contents

| File | Purpose |
|------|---------|
| `pipeline.py` | A simple CPU-only pipeline (word-count + node-tag) that runs locally or on SLURM |
| `submit.sh` | `sbatch` script for bare-metal clusters with a shared virtualenv |
| `submit_container.sh` | `sbatch` script using the official NGC container (Pyxis/enroot) |
| `array_pipeline.py` | Generic JSONL/Parquet pipeline that processes one Slurm array shard |
| `retry_array.py` | Finds shards without completion manifests and prints one or more retry array configurations |
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

Use `submit_array.sh` when you already have a large directory of text data files and want to split the file set across many independent Slurm jobs. Each array task starts its own Curator pipeline; source stages still produce the full deterministic task list, and the backend adapter filters that list to only the tasks assigned to the current Slurm task.

This pattern is useful when the dataset is naturally represented as many JSONL or Parquet files and you want simple horizontal scaling without coordination between jobs.

### 1. Build the virtualenv on a shared filesystem

The array example uses the official NGC container for the base environment, then activates your local checkout inside the container so unreleased source changes are picked up:

```bash
cd /path/to/Curator
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Make sure `CURATOR_DIR`, `INPUT_DIR`, `OUTPUT_DIR`, and `CHECKPOINT_PATH` are visible from every compute node, either because they are on a shared filesystem or because you set `CONTAINER_MOUNTS` to expose the right host paths inside the container.

### 2. Submit a JSONL array job

By default, `submit_array.sh` reads JSONL files and writes JSONL output:

```bash
export CURATOR_DIR=/path/to/Curator
export INPUT_DIR=/shared/data/my-jsonl-dataset
export OUTPUT_DIR=/shared/output/my-jsonl-dataset
export CHECKPOINT_PATH=/shared/checkpoints/my-jsonl-run

# 20 array tasks, task IDs 0-19
sbatch --array=0-19 tutorials/slurm/submit_array.sh
```

Use a checkpoint directory dedicated to this logical array run. It stores the original shard configuration and durable completion manifests used to discover incomplete shards. The same `CHECKPOINT_PATH` must be reused for every retry of this run.

For example, if the input directory contains 2000 files and `FILES_PER_PARTITION=1`, each of the 20 array tasks receives roughly 100 source tasks. `submit_array.sh` computes the shard index and total shard count from Slurm and exports the `NEMO_CURATOR_SLURM_ARRAY_*` shard variables consumed by `array_pipeline.py` and the backend. Assignment is deterministic SHA-256-based rather than contiguous, so work remains stable if Slurm retries a task.

Slurm array filtering is enabled by default whenever Curator finds a shard index and total shard count in the Curator or Slurm environment. No opt-in flag is required. A power user can explicitly disable filtering with:

```bash
export NEMO_CURATOR_SLURM_ARRAY_ENABLED=0
```

When no array configuration is present, Curator leaves filtering inactive, so ordinary local and non-array Slurm runs are unaffected.

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

In this mode, keep `MINIMUM_SHARD_INDEX=0` because the Slurm array task IDs are already the global shard IDs. Each source task is assigned by a deterministic SHA-256 digest of `task_id` modulo `TOTAL_SHARDS`, so the full set of windowed submissions covers shards `0` through `9999` exactly once. Some individual tasks may receive no files if `TOTAL_SHARDS` is larger than the number of source tasks.

Some clusters enforce the maximum array index rather than just the number of tasks per submitted array. If `--array=1000-1999` is rejected, use `SHARD_INDEX_OFFSET` instead of higher Slurm task IDs.

For those clusters, submit each window with Slurm task IDs `0-999` and set `SHARD_INDEX_OFFSET` so the script computes the global shard ID as `SLURM_ARRAY_TASK_ID + SHARD_INDEX_OFFSET`:

```bash
export TOTAL_SHARDS=10000

SHARD_INDEX_OFFSET=0    sbatch --array=0-999 tutorials/slurm/submit_array.sh
SHARD_INDEX_OFFSET=1000 sbatch --array=0-999 tutorials/slurm/submit_array.sh
SHARD_INDEX_OFFSET=2000 sbatch --array=0-999 tutorials/slurm/submit_array.sh
# ...
SHARD_INDEX_OFFSET=9000 sbatch --array=0-999 tutorials/slurm/submit_array.sh
```

Keep `MINIMUM_SHARD_INDEX=0` for this offset mode too. `SHARD_INDEX_OFFSET` changes the logical shard ID exported for adapter-level filtering; `MINIMUM_SHARD_INDEX` changes the assignable shard range used by the source-stage adapter.

For retries on a cluster with a maximum array size, pass that limit once and let `retry_array.py` group every missing logical shard into physical array indices and derive each `SHARD_INDEX_OFFSET`. For example, a cluster with `MaxArraySize=1000` accepts physical indices from `0` through `999`:

```bash
if ! retry_fields="$(
    python tutorials/slurm/retry_array.py \
        --checkpoint-path "${CHECKPOINT_PATH}" \
        --max-array-size 1000 \
        --format fields
)"; then
    echo "Failed to discover retryable shards." >&2
    exit 1
fi

while read -r RETRY_ARRAY SHARD_INDEX_OFFSET MINIMUM_SHARD_INDEX TOTAL_SHARDS; do
    [[ -z "${RETRY_ARRAY}" ]] && continue
    CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
    MINIMUM_SHARD_INDEX="${MINIMUM_SHARD_INDEX}" \
    TOTAL_SHARDS="${TOTAL_SHARDS}" \
    SHARD_INDEX_OFFSET="${SHARD_INDEX_OFFSET}" \
    sbatch \
        --array="${RETRY_ARRAY}" \
        tutorials/slurm/submit_array.sh
done <<< "${retry_fields}"
```

The maximum array size is a cluster submission limit rather than part of the logical Curator run, so it is not stored in `run.json`. The logical minimum index and total shard count are read from that manifest; all offsets are calculated by the helper.

### 5. Retry incomplete array tasks only

When completion tracking is initialized, the array task atomically creates or validates the logical run configuration at:

```bash
${CHECKPOINT_PATH:-$OUTPUT_DIR}/.nemo_curator_metadata/.slurm_array_completion/run.json
```

The run configuration stores `MINIMUM_SHARD_INDEX` and the original `TOTAL_SHARDS`, so retry discovery still knows the full expected range when zero shards complete. A shard writes one completion manifest only after the pipeline finishes cleanly with no `FailedTask` results.

| Outcome | Completion manifest | Retry behavior |
|---------|---------------------|----------------|
| Pipeline raises, crashes, times out, or is preempted | Not written | Rerun the shard |
| Pipeline finishes but emits one or more `FailedTask` results | Not written | Rerun the shard |
| Pipeline finishes with no `FailedTask` results | Written with shard ID and original shard configuration | Do not rerun the shard |

Both pipeline failures and `FailedTask` results therefore appear as missing completion records and use the same retry command and checkpoint directory. Retries happen at shard granularity: the full owning shard runs again, not only an individual failed Curator task.

After all array submissions that make up the logical run have finished, read the outstanding retry configuration from the login node:

```bash
python tutorials/slurm/retry_array.py \
    --checkpoint-path "${CHECKPOINT_PATH}" \
    --format fields
```

Without `--max-array-size`, the output is always at most one line (all missing logical shards are emitted as a single submission). For example, this output means that shards `1`, `2`, `5` through `10`, and `99` should be retried with offset `0`, using the original logical shard range of `0` through `99`:

```text
1-2,5-10,99 0 0 100
```

The four fields are the retry array expression, `SHARD_INDEX_OFFSET`, `MINIMUM_SHARD_INDEX`, and the original `TOTAL_SHARDS`. Capture and pass them when submitting the new array:

```bash
if ! retry_fields="$(
    python tutorials/slurm/retry_array.py \
        --checkpoint-path "${CHECKPOINT_PATH}" \
        --format fields
)"; then
    echo "Failed to discover retryable shards." >&2
    exit 1
fi

if [[ -z "${retry_fields}" ]]; then
    echo "No shards need retrying."
else
    read -r RETRY_ARRAY SHARD_INDEX_OFFSET MINIMUM_SHARD_INDEX TOTAL_SHARDS <<< "${retry_fields}"

    CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
    SHARD_INDEX_OFFSET="${SHARD_INDEX_OFFSET}" \
    MINIMUM_SHARD_INDEX="${MINIMUM_SHARD_INDEX}" \
    TOTAL_SHARDS="${TOTAL_SHARDS}" \
    sbatch \
        --array="${RETRY_ARRAY}" \
        tutorials/slurm/submit_array.sh
fi
```

You can change resources directly on the retry submission:

```bash
CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
SHARD_INDEX_OFFSET="${SHARD_INDEX_OFFSET}" \
MINIMUM_SHARD_INDEX="${MINIMUM_SHARD_INDEX}" \
TOTAL_SHARDS="${TOTAL_SHARDS}" \
sbatch \
    --array="${RETRY_ARRAY}" \
    --time=02:00:00 \
    --cpus-per-task=32 \
    tutorials/slurm/submit_array.sh
```

`TOTAL_SHARDS` must be the original logical shard count, not the number of shards in the retry array. For example, the retry array above contains nine tasks, but `TOTAL_SHARDS` must remain `100` so deterministic task assignment does not change.

#### Checkpoint and FailedTask directories during retry

Always reuse the original `CHECKPOINT_PATH` for both pipeline-failure retries and `FailedTask` retries. A new checkpoint directory would lose the original run configuration and completed-shard history, causing previously completed shards to appear incomplete.

Do not reuse a previous attempt's `NEMO_CURATOR_FAILED_TASKS_DIR`. Before Ray starts, `array_pipeline.py` automatically derives a fresh directory from `CHECKPOINT_PATH`, `SLURM_JOB_ID`, `SLURM_ARRAY_TASK_ID`, `SLURM_RESTART_COUNT`, and the logical shard index. This keeps the durable run configuration and completion manifests in the shared checkpoint directory while isolating at most one FailedTask manifest for every Slurm job, array task, and restart:

```text
${CHECKPOINT_PATH}/.nemo_curator_metadata/
├── .slurm_array_completion/
│   ├── run.json                         # original minimum index and total shards
│   └── completed_slurm_array_*.json     # one per successfully completed shard
└── .failed_tasks/
    ├── slurm_job_<original-job-id>/.../restart_0/.../failed_tasks.json
    └── slurm_job_<retry-job-id>/.../restart_0/.../failed_tasks.json
```

Each attempt writes `failed_tasks.json` at most once, as soon as any `FailedTask` is detected; additional failures require only an existence check and do not create more files. Old FailedTask manifests are retained for inspection, but they cannot make a later successful attempt look failed because each attempt checks only its fresh directory. Custom submission scripts can set `NEMO_CURATOR_FAILED_TASKS_DIR` directly to choose a different manifest tree; keep the directory attempt-scoped so an old manifest cannot affect a retry.

Run retry discovery only after all array submissions that make up the logical run have finished; a still-running or not-yet-submitted shard has no completion manifest and therefore appears retryable. Use a new `CHECKPOINT_PATH` for a new, unrelated logical pipeline run, and reuse the existing path only for retries of that run.

Source stages may not emit `FailedTask` sentinels, regardless of whether Slurm array filtering is enabled. Source-stage `FailedTask`s do not carry enough stable source identity to guarantee that downstream processing or a shard retry can recover the same work, so Curator raises an error instead. Source stages should raise an exception and, in the Slurm array workflow, leave the shard without a completion manifest.

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

> **Important**: If your cluster's `/tmp` is local to each node (the common case), set `RAY_PORT_BROADCAST_DIR` to a Lustre/NFS path so all nodes can read the port file:
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
