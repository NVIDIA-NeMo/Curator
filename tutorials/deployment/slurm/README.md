# Ray Cluster SLURM Script (Singularity / Apptainer)

This example shows how to launch a **multi-node Ray cluster** on a SLURM-based HPC system using **Singularity / Apptainer**.

The script:

- Starts a **Ray head** process on the first allocated SLURM node.
- Starts **Ray worker** processes on the remaining nodes.
- Runs a user-provided **Python command** inside the same container on the head node.
- Is designed to work on **air-gapped** clusters (no internet on compute nodes) by default.

---

## Prerequisites

You need:

- A SLURM cluster.
- Singularity or Apptainer available on the compute nodes.
- A container image (`.sif`) which includes nemo-curator. You can create one using a command like:
    `singularity pull nemo-curator_25.09.sif docker://nvcr.io/nvidia/nemo-curator:25.09`

Typical example:

```bash
singularity exec /path/to/nemo-curator_25.09.sif python -c "import ray; print(ray.__version__)"
```

should work on your system before using the script.

---

## How the script works

High-level flow:

1. SLURM allocates one or more nodes for the job.
2. The script:

   * Determines the node list via `SLURM_JOB_NODELIST`.
   * Picks the **first node** as the Ray head node.
   * Computes basic resource counts (`NUM_CPUS_PER_NODE`, `NUM_GPUS_PER_NODE`).
3. It creates temporary directories under `${SCRATCH_ROOT}` (or the current directory by default) for:

   * `ray_tmp/$JOB_ID` – per-job temp dir
   * `ray_workers_tmp/$JOB_ID` – per-worker temp dirs
   * `ray_spill/$JOB_ID` – Ray object spill directory
4. It configures environment variables and propagates them into the container using `SINGULARITYENV_…`.
5. It launches:

   * The **Ray head** on the first node.
   * **Ray workers** on each remaining node.
6. It runs `RUN_COMMAND` inside the container on the head node.
7. On exit, it cleans up the temporary directories.

---

## Quick start

1. **Copy the script** into your project, e.g.:

```bash
cp ray-singularity-sbatch.sh examples/ray-cluster.sbatch
```

2. **Edit the SBATCH header** to match your cluster:

```bash
#SBATCH --job-name=ray-cluster-example
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --account=your_account
#SBATCH --partition=your_partition
```

3. **Set the container image** at the top of the script or via an environment variable:

```bash
export IMAGE=/path/to/nemo-curator_25.09.sif
```

4. **Submit a test job**:

   ```bash
   RUN_COMMAND="python -c 'import ray; ray.init(); print(ray.cluster_resources())'" \
   sbatch --nodes=2 --gres=gpu:4 examples/ray-cluster.sbatch
   ```

   This should print Ray cluster resources from inside the container on the head node.

---

## Environment variables

Most behaviour can be changed **without editing the script**, just by setting environment variables before calling `sbatch`.

### Core parameters

* **`RUN_COMMAND`**
  Command to run on the head node after the Ray cluster is up.

  ```bash
  RUN_COMMAND="python my_nemo_curator_pipeline.py" sbatch ray-singularity-sbatch.sh
  ```

* **`IMAGE`**
  Path to the Singularity/Apptainer image with Ray installed.

  ```bash
  export IMAGE=/path/to/nemo-curator_25.09.sif
  ```

* **`CONTAINER_CMD`**
  Container runtime command. Defaults to `singularity`, can be set to `apptainer`:

  ```bash
  export CONTAINER_CMD=apptainer
  ```

### Ports

* **`GCS_PORT`** (default: `6379`)
  Ray GCS port.

* **`CLIENT_PORT`** (default: `10001`)
  Ray client port (for `ray://` connections).

* **`DASH_PORT`** (default: `8265`)
  Ray dashboard port.

If these ports conflict on your system, you can override them:

```bash
export GCS_PORT=16379
export CLIENT_PORT=11001
export DASH_PORT=18265
```

### Paths & scratch

* **`SCRATCH_ROOT`**
  Base directory for job-specific Ray temp data:

  * `${SCRATCH_ROOT}/ray_tmp/$JOB_ID`
  * `${SCRATCH_ROOT}/ray_workers_tmp/$JOB_ID`
  * `${SCRATCH_ROOT}/ray_spill/$JOB_ID`

  By default:

  ```bash
  SCRATCH_ROOT="${SCRATCH:-$(pwd)}"
  ```

  On many clusters you’ll want something like:

  ```bash
  export SCRATCH_ROOT=/scratch/$USER
  ```

* **`CONTAINER_MOUNTS`**
  Comma-separated list of host paths to mount into the container.
  Defaults to the current directory (`$PWD`), and the script automatically appends:

  * `HF_HOME`
  * Ray temp and spill directories (head node)

  Example:

  ```bash
  export CONTAINER_MOUNTS="/project/myapp,/data/datasets"
  ```

  > Note: When a bare path (`/foo/bar`) is passed to `--bind`, Singularity/Apptainer mounts it at the *same* path inside the container.

### Resources

* **`NUM_CPUS_PER_NODE`**
  Auto-detected from SLURM or `getconf` / `nproc`. Override if needed:

  ```bash
  export NUM_CPUS_PER_NODE=32
  ```

* **`NUM_GPUS_PER_NODE`**
  Auto-detected from `SLURM_GPUS_ON_NODE` or defaults to `1`. Override if needed:

  ```bash
  export NUM_GPUS_PER_NODE=4
  ```

### Hugging Face (offline by default)

* **`HF_HOME`**
  Hugging Face cache directory (mounted into the container).

  ```bash
  export HF_HOME=/scratch/$USER/hf_cache
  ```

* **`HF_HUB_OFFLINE`** *(default: `1`)*
  Controls offline mode for `huggingface_hub`:

  * `1` – **offline**: do not attempt internet downloads (recommended for air-gapped clusters).
  * `0` – allow network access (if your nodes have connectivity).

  ```bash
  export HF_HUB_OFFLINE=0   # only if your compute nodes have internet
  ```

### Startup waits

* **`HEAD_STARTUP_WAIT`** (default: `10`)
* **`WORKER_STARTUP_WAIT`** (default: `60`)

Wait times (seconds) after launching head/worker processes to give Ray time to start up before running the user command.

---

## Modifying SBATCH directives

The SBATCH header is intentionally minimal:

```bash
#SBATCH --job-name=ray-cluster-example
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=ray-job-%j.out
#SBATCH --error=ray-job-%j.err
```

You should adjust it to fit your site’s policies and your workload. Common modifications:

* Set the **number of nodes** and **GPUs per node**:

```bash
  #SBATCH --nodes=4
  #SBATCH --gres=gpu:4
```

* Specify your **account** and **partition**:

```bash
  #SBATCH --account=my_account
  #SBATCH --partition=gpu
```

* Use `--mem` or `--mem-per-cpu` if your cluster requires explicit memory requests.

---

## Connecting to the Ray dashboard (optional)

If the dashboard port (`DASH_PORT`, default `8265`) is open on the head node, you can access it via SSH tunneling from your local machine.

Example (pattern):

```bash
# From your local machine:
ssh -L 8265:HEAD_NODE_HOSTNAME:8265 user@login-node.your.cluster
```

Then open in a browser:

* [http://localhost:8265](http://localhost:8265)

Exact details depend on your cluster’s network and login setup; consult your HPC documentation.

---

## Connecting via `ray://` (optional)

The script exposes the Ray client server on `CLIENT_PORT` (default `10001`) on the head node:

* Ray address: `ray://HEAD_NODE_IP:CLIENT_PORT`

Inside the container on the head node, your Python code can use:

```python
from nemo_curator.pipeline import Pipeline

# Define your pipeline
pipeline = Pipeline(...)

# Add stages
pipeline.add_stage(...)
executor = XennaExecutor()
results = pipeline.run(executor=executor)
```

---

## Typical usage patterns

### 1. Run a self-contained Ray script in the container

```bash
export IMAGE=/path/to/nemo-curator_25.09.sif

RUN_COMMAND="python curator_pipeline.py" \
sbatch --nodes=2 --gres=gpu:4 ray-singularity-sbatch.sh
```

Where `curator_pipeline.py` contains:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.backends.xenna.executor import XennaExecutor

# Define your pipeline
pipeline = Pipeline(...)

# Add stages
pipeline.add_stage(...)
executor = XennaExecutor()
results = pipeline.run(executor=executor)
```

### 2. Use a project directory and datasets

```bash
export IMAGE=/path/to/nemo-curator_25.09.sif
export CONTAINER_MOUNTS="/project/myapp,/data/my-dataset"

RUN_COMMAND="cd /project/myapp && python curator_pipeline.py" \
sbatch --nodes=4 --gres=gpu:4 ray-singularity-sbatch.sh
```

Make sure the paths you use in `RUN_COMMAND` match the mounted paths inside the container.

### 3. Air-gapped cluster with pre-downloaded HF models

```bash
export IMAGE=/path/to/nemo-curator_25.09.sif
export SCRATCH_ROOT=/scratch/$USER
export HF_HOME=/scratch/$USER/hf_cache
export HF_HUB_OFFLINE=1

RUN_COMMAND="python ray_script_using_hf.py" \
sbatch --nodes=2 --gres=gpu:4 ray-singularity-sbatch.sh
```
