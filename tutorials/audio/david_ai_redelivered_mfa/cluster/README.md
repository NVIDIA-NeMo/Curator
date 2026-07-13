# Multi-node SLURM execution

The cluster launcher uses a SLURM job array with one task per node. Session
discovery is deterministic, and task `i` processes sessions whose sorted index
satisfies:

```text
session_index % NUM_NODES == i
```

Each node runs multiple session processes (`WORKERS_PER_NODE`), while each MFA
recording may use multiple MFA jobs (`MFA_NUM_JOBS`).

## Requirements

- `DATA_ROOT`, `WORK_DIR`, the repository, MFA environment, and MFA models must
  already be visible at identical absolute paths on every node.
- `$MFA_ENV/bin/python`, `$MFA_ENV/bin/mfa`, and ffmpeg must be executable on
  compute nodes.
- `WORK_DIR` must be writable by all array tasks.
- Scratch uses `$SLURM_TMPDIR` when available, otherwise `/tmp`.
- No input/output copying is performed by these scripts.
- Optional containers must mount every required shared path explicitly.

## Submit

From the tutorial root:

```bash
cd ~/Curator_my_fork/tutorials/audio/david_ai_redelivered_mfa

VARIANT=wav \
DATA_ROOT=/shared/data/david_ai_sessions \
WORK_DIR=/shared/output/david_ai_wav \
MFA_ENV=/shared/envs/david-ai-mfa \
MFA_ROOT_DIR=/shared/models/MFA_models \
NUM_NODES=8 \
MAX_CONCURRENT_NODES=8 \
CPUS_PER_NODE=64 \
WORKERS_PER_NODE=16 \
MFA_NUM_JOBS=2 \
SEG_EXTRACT_WORKERS=8 \
SLURM_ACCOUNT=my-account \
SLURM_PARTITION=cpu \
TIME_LIMIT=08:00:00 \
bash cluster/run_multinode.sh
```

Use `VARIANT=opus` for the Opus pipeline.

To process a subset, add a shared absolute list path:

```bash
SESSIONS_FILE=/shared/config/session_subset.txt \
... \
bash cluster/run_multinode.sh
```

## Optional container

```bash
CONTAINER_IMAGE=/shared/containers/pipeline.sqsh \
CONTAINER_MOUNTS=/shared:/shared \
... \
bash cluster/run_multinode.sh
```

The image must have access to the submitted repository path, data, outputs, MFA
environment, models, and node-local scratch.

## Parallelism

Approximate MFA process slots per node:

```text
WORKERS_PER_NODE × MFA_NUM_JOBS
```

Start with that product at or below `CPUS_PER_NODE`. Leave CPU headroom for
ffmpeg extraction, pause masking, mixing, and filesystem work.

Example for a 64-CPU node:

```text
WORKERS_PER_NODE=16
MFA_NUM_JOBS=2
```

This creates up to 32 MFA slots and leaves headroom for other stages.

## MFA directory isolation

`MFA_ROOT_DIR` is treated as a read-only source for pretrained models. Runtime
state is never written there.

Isolation hierarchy:

```text
<node-local-scratch>/
└── david_ai_<variant>_<slurm-job>_<array-task>/
    ├── model_source/         # one shard-local copy from shared storage
    └── mfa_workers/
        └── worker_<process-pid>/
            ├── models/       # private worker model copies
            ├── mfa_root/     # private MFA config/database root
            └── align_temp/
                └── <session-id>/
```

- Every array shard has a unique scratch root based on job and task IDs.
- Every shard stages the shared dictionary, acoustic, and G2P source once into
  its own node-local `model_source`.
- Every session worker process has a private MFA root and private model copies.
- Every worker pre-extracts the G2P archive and validates that it contains one
  non-empty `model.fst` before launching MFA, avoiding concurrent MFA extraction.
- A worker processes its speaker recordings sequentially.
- Every session uses a separate alignment temp directory.
- MFA subprocesses receive the private worker root through `MFA_ROOT_DIR`.

Consequently, nodes and concurrent session workers do not share writable MFA
model, database, or temporary directories.

## Logs and status

SLURM logs:

```text
<work-dir>/logs/slurm/<job-name>_<array-job-id>_<task-id>.out
<work-dir>/logs/slurm/<job-name>_<array-job-id>_<task-id>.err
```

Per-shard pipeline logs include the SLURM job and array task IDs, preventing
multiple nodes from writing the same log filename.

Monitor:

```bash
squeue -j <job-id>
sacct -j <job-id> --format=JobID,State,Elapsed,ExitCode
```

Successful sessions write:

```text
<work-dir>/.done/sessions/<session-id>.done
```

If an array task fails or times out, submit the same command again. Every node
recomputes its deterministic shard, skips sessions with done flags, and runs only
unfinished sessions. Changing `NUM_NODES` is also safe because done flags are
checked after the new deterministic sharding assignment.

## Security

`run_multinode.sh` exports only explicitly required, non-secret variables. It
does not pass the submit shell's full environment. Do not place credentials in
these variables, command arguments, logs, or container mounts.
