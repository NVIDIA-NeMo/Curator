# Running the Slurm array tutorial on Hugging Face Jobs

This tutorial runs the same array-sharded pipeline as [`tutorials/slurm`](../slurm) on [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs). Hugging Face Jobs provide compute for AI and data workflows, allowing you to run workloads on Hugging Face infrastructure with a familiar UV & Docker-like interface.

The pipeline and retry planner are the unmodified files from the Slurm tutorial; only the scheduler-side plumbing changes.

## Contents

| File              | Purpose                                                                    |
| ----------------- | -------------------------------------------------------------------------- |
| `run_shard.py`    | HF Jobs shim: all scheduler-side plumbing, the analog of `submit_array.sh` |
| `launch_array.sh` | Fires one independent Job per shard                                        |

The pipeline itself is [`../slurm/array_pipeline.py`](../slurm/array_pipeline.py) and the retry planner is [`../slurm/retry_array.py`](../slurm/retry_array.py) — both used as-is. (`hf jobs uv run` uploads a single script file, so the shim downloads `array_pipeline.py` inside the Job at the release tag matching its pinned `nemo-curator` dependency and verifies the sha256 before executing it.)

## Prerequisites

- [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/guides/cli) installed and logged in (`hf auth login`) — Jobs and bucket access run under your account
- A checkout of this repository (for `launch_array.sh` and `retry_array.py`; nothing is installed from it)

---

## The key concept: only the scheduler file changes

In the Slurm tutorial, `submit_array.sh` is the one scheduler-specific file: it reads `SLURM_ARRAY_TASK_ID`, exports the `NEMO_CURATOR_SLURM_ARRAY_*` shard variables, and invokes `array_pipeline.py`. On HF Jobs that role is played by `launch_array.sh` (sets the same variables per Job) plus `run_shard.py` (runs the unmodified pipeline inside the Job's container).

```bash
hf jobs uv run --flavor cpu-basic \
  -e NEMO_CURATOR_SLURM_ARRAY_ENABLED=1 \
  -e NEMO_CURATOR_SLURM_ARRAY_SHARD_INDEX=$k \
  -e NEMO_CURATOR_SLURM_ARRAY_TOTAL_SHARDS=$K \
  -v hf://buckets/<user>/<bucket>:/mnt \
  run_shard.py
```

`array_pipeline.py` and `retry_array.py` do not change at all. Curator's array sharding is driven entirely by the `NEMO_CURATOR_SLURM_ARRAY_*` environment variables — each shard builds the full deterministic task list and keeps a task if `sha256(task_id) % TOTAL_SHARDS == SHARD_INDEX` — so the contract is scheduler-agnostic. The shared filesystem is an HF bucket FUSE-mounted at `/mnt` in every Job; Jobs never communicate directly, exactly as in the Slurm array workflow. Completion manifests land on the bucket, so retry discovery works unchanged.

---

## 1. Put input files on a bucket

Create a bucket and upload your JSONL files under `tutorial/input/`:

```bash
hf buckets create <user>/<bucket>
hf buckets sync ./my-jsonl-dir hf://buckets/<user>/<bucket>/tutorial/input
```

No data handy? Generate a few small files first and the tutorial runs end to end as-is:

```bash
mkdir -p my-jsonl-dir
for i in $(seq 0 7); do
  printf '{"id": "doc-%d-0", "text": "hello from file %d"}\n{"id": "doc-%d-1", "text": "hf jobs array tutorial"}\n' \
    "$i" "$i" "$i" > "my-jsonl-dir/part_$i.jsonl"
done
```

## 2. Launch the array

```bash
export BUCKET=hf://buckets/<user>/<bucket>

# 3 shards, one cpu-basic Job each
bash tutorials/hf_jobs/launch_array.sh 3
```

Each Job prints its id on launch; follow logs with `hf jobs logs <JOB_ID>`. Expected output for a healthy shard (dependency install and pipeline logs omitted):

```text
[shim] shard 0/3 share=/mnt/tutorial
[shim] fetched array_pipeline.py @ v1.3.0 sha256-verified
...
[shim] published manifest completed_slurm_array_3800bd4ecff9e7f6.json
[shim] published manifest run.json
SHIM_OK 0
```

A shard ends with `SHIM_OK <k>` on success.

## 3. Check the output

On the bucket you should see one output file per source task and one completion manifest per finished shard:

```text
tutorial/
├── input/*.jsonl
├── out/<deterministic-name>.jsonl          # idempotent re-runs overwrite in place
└── ckpt/.nemo_curator_metadata/.slurm_array_completion/
    ├── run.json                            # original shard configuration
    └── completed_slurm_array_*.json        # one per completed shard
```

## 4. Retry incomplete shards only

`retry_array.py` reads the completion-manifest directory. The simplest way to run it is in a small Job with the bucket mounted, so it reads the manifests in place — the planner file runs unmodified:

```bash
hf jobs uv run --flavor cpu-basic -s HF_TOKEN \
    --with "nemo-curator==1.3.0" \
    -v hf://buckets/<user>/<bucket>:/mnt \
    tutorials/slurm/retry_array.py --checkpoint-path /mnt/tutorial/ckpt --format fields
```

(On a Linux machine you can instead run the planner locally, exactly as in the Slurm tutorial: `hf buckets sync hf://buckets/<user>/<bucket>/tutorial/ckpt ./ckpt-mirror`, then `uv run --no-project --with "nemo-curator==1.3.0" python tutorials/slurm/retry_array.py --checkpoint-path ./ckpt-mirror --format fields`. NeMo Curator itself supports Linux only, so on macOS/Windows use the Job form above.)

The first field is the missing-shard expression (e.g. `2` or `0,2`). Relaunch only those shards, keeping the original total shard count — and, as in the Slurm tutorial, you can change resources on the retry:

```bash
bash tutorials/hf_jobs/launch_array.sh 3 2                    # retry shard 2 of a 3-shard run
FLAVOR=cpu-performance bash tutorials/hf_jobs/launch_array.sh 3 2   # same, on a bigger flavor
```

As on Slurm, retries happen at shard granularity and must reuse the same checkpoint path (here: the same bucket prefix) so completed shards are not rerun.

---

## HF Jobs container notes

Just as `submit_array.sh` absorbs Slurm quirks (task-ID arithmetic, `SHARD_INDEX_OFFSET`, node-local `/tmp`), `run_shard.py` absorbs the Jobs-container quirks so the pipeline file stays clean:

1. **Work from local disk, not the mount.** The container starts with its working directory on the FUSE bucket mount; Ray's runtime-env packaging hashes the cwd and hangs there. The shim does `os.chdir` to a `/tmp` workdir before starting Ray.
2. **CPU autodetection.** Ray detects the cgroup limit (1 CPU on `cpu-basic`), below the Xenna executor's streaming floor. The shim passes `RayClient(num_cpus=8)` as a logical overcommit (override with `HF_JOBS_RAY_NUM_CPUS`). Only needed on `cpu-basic` — on larger flavors (verified on `cpu-upgrade`) Ray's autodetected CPU count already clears the floor.
3. **No POSIX rename on the mount.** Curator writes completion manifests atomically (write + rename), which bucket mounts reject. The shim checkpoints to local disk and publishes the manifest JSONs to the bucket with plain writes after the run.

---

## Scaling beyond the tutorial

The same launch pattern scales in two independent directions:

- **More shards**: pass a larger total to `launch_array.sh` — shard assignment is deterministic, so any subset can be launched, retried, or re-run at any time.
- **Bigger flavors**: set `FLAVOR` (e.g. `cpu-upgrade`, or GPU flavors for classifier stages) — the shim and sharding contract are flavor-agnostic.
