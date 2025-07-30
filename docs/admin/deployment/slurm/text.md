(admin-deployment-slurm-text)=
# Deploy Text Curation on Slurm

This workflow covers the full text curation pipeline on Slurm, including model download, text cleaning, deduplication, classification, and PII redaction. 

```{seealso}
For details on text container environments and Slurm environment variables, see [Container Environments](reference-infrastructure-container-environments).
```

## Prerequisites

- Create required directories for AWS credentials, NeMo Curator configuration, and local workspace:

  ```bash
  mkdir $HOME/.aws
  mkdir -p $HOME/.config/nemo_curator
  mkdir $HOME/nemo_curator_local_workspace
  ```

- Prepare configuration files:

  :::: {tab-set}

  ::: {tab-item} AWS Credentials
  `$HOME/.aws/credentials` (for S3 access)

  ```{literalinclude} _assets/.aws/eg.creds
  :language: ini
  ```

  :::

  ::: {tab-item} NeMo Curator Configuration
  `$HOME/.config/nemo_curator/config.yaml` (for HuggingFace API key)

  ```{literalinclude} _assets/.config/nemo_curator/config.yaml
  :language: yaml
  ```

  :::

  ::: {tab-item} Text Processing Configuration
  `$HOME/nemo_curator_local_workspace/text_config.yaml` (for advanced semantic deduplication)

  ```yaml
  # Semantic deduplication configuration (for programmatic API usage)
  # Note: CLI scripts use command-line parameters, not this configuration file
  cache_dir: "/tmp/semdedup_cache"
  
  # Embedding settings
  embedding_model_name_or_path: "sentence-transformers/all-MiniLM-L6-v2"
  embedding_batch_size: 128
  embeddings_save_loc: "embeddings"
  
  # Clustering settings  
  max_iter: 100
  n_clusters: 1000
  clustering_save_loc: "clustering_results"
  
  # Deduplication settings
  eps_to_extract: 0.01
  sim_metric: "cosine"
  which_to_keep: "hard"
  batched_cosine_similarity: 1024
  ```

  :::

  ::::

---

## Model Download

1. Copy the following script for downloading all required text processing models into the Slurm cluster.

   ```{note}
   This script downloads models required for the quality classifier and other components. You may need to adjust the model list based on your specific workflow requirements.
   ```

   ```bash
   #!/bin/bash

   #SBATCH --job-name=download_text_models
   #SBATCH -p defq
   #SBATCH --nodes=1
   #SBATCH --ntasks-per-node=1
   #SBATCH --exclusive

   # Update Me!
   #SBATCH --output=/home/<username>/logs/%x_%j.log
   USER_DIR="/home/${USER}"
   CONTAINER_IMAGE="${USER_DIR}/path-to/curator.sqsh"
   #

   LOCAL_WORKSPACE="${USER_DIR}/nemo_curator_local_workspace"
   LOCAL_WORKSPACE_MOUNT="${LOCAL_WORKSPACE}:/config"
   NEMO_CONFIG_MOUNT="${HOME}/.config/nemo_curator/config.yaml:/nemo_curator/config/nemo_curator.yaml"
   CONTAINER_MOUNTS="${LOCAL_WORKSPACE_MOUNT},${NEMO_CONFIG_MOUNT}"

   export NEMO_CURATOR_RAY_SLURM_JOB=1
   export NEMO_CURATOR_LOCAL_DOCKER_JOB=1

   # Download Text Processing Models
   srun \
     --mpi=none \
     --container-writable \
     --no-container-remap-root \
     --export=NEMO_CURATOR_RAY_SLURM_JOB,NEMO_CURATOR_LOCAL_DOCKER_JOB \
     --container-image "${CONTAINER_IMAGE}" \
     --container-mounts "${CONTAINER_MOUNTS}" \
       --  python3 -c "
   from nemo_curator.utils.download_utils import download_models
   models = [
       'sentence-transformers/all-MiniLM-L6-v2',
       'huggingface/fasttext-language-identification',
       'nvidia/quality-classifier-deberta',
       'nvidia/domain-classifier-fineweb'
   ]
   download_models(models, '/config/models')
   "
   ```

2. Update the `SBATCH` parameters and paths to match your username and environment.
3. Run the script.

   ```bash
   sbatch 1_curator_download_text_models.sh
   ```

## Text Processing Pipeline

The workflow consists of four main Slurm scripts, to be run in order:

1. `curator_text_clean.sh`: Cleans and preprocesses raw text data.
2. `curator_text_classify.sh`: Applies quality and domain classification.
3. `curator_text_dedup.sh`: Performs exact and fuzzy deduplication (semantic deduplication requires separate setup).
4. `curator_text_redact.sh`: Identifies and redacts personally identifiable information.

```{note}
**Semantic Deduplication**: The current scripts implement exact and fuzzy deduplication. For semantic deduplication, you'll need to use the programmatic API or the specialized scripts in `nemo_curator.scripts.semdedup`. See the {ref}`Semantic Deduplication documentation <text-process-data-format-sem-dedup>` for complete setup instructions.
```

:::: {tab-set}

::: {tab-item} 1. Cleaning
`curator_text_clean.sh` - Cleans and preprocesses raw text data.

```bash
#!/bin/bash

#SBATCH --job-name=text-clean
#SBATCH -p defq
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=04:00:00

# Update Me!
#SBATCH --output=/home/<username>/logs/%x_%j.log
#SBATCH --error=/home/<username>/logs/%x_%j.log
USER_DIR="/home/${USER}"
CONTAINER_IMAGE="${USER_DIR}/path-to/curator.sqsh"
INPUT_DATA_PATH="s3://your-bucket/raw-text-data/"
OUTPUT_DATA_PATH="s3://your-bucket/cleaned-text-data/"
#

LOCAL_WORKSPACE="${USER_DIR}/nemo_curator_local_workspace"
LOCAL_WORKSPACE_MOUNT="${LOCAL_WORKSPACE}:/config"
NEMO_CONFIG_MOUNT="${HOME}/.config/nemo_curator/config.yaml:/nemo_curator/config/nemo_curator.yaml"
AWS_MOUNT="${HOME}/.aws:/root/.aws"
CONTAINER_MOUNTS="${LOCAL_WORKSPACE_MOUNT},${NEMO_CONFIG_MOUNT},${AWS_MOUNT}"

export NEMO_CURATOR_RAY_SLURM_JOB=1

srun \
  --mpi=none \
  --container-writable \
  --no-container-remap-root \
  --export=NEMO_CURATOR_RAY_SLURM_JOB \
  --container-image "${CONTAINER_IMAGE}" \
  --container-mounts "${CONTAINER_MOUNTS}" \
    -- python3 -m nemo_curator.scripts.text_cleaning \
      --input-data-dir "${INPUT_DATA_PATH}" \
      --output-clean-dir "${OUTPUT_DATA_PATH}" \
      --scheduler-address "127.0.0.1" \
      --scheduler-port 8786 \
      --device gpu \
      --normalize-newlines \
      --remove-urls
```

:::

::: {tab-item} 2. Classification
`curator_text_classify.sh` - Applies quality and domain classification.

```bash
#!/bin/bash

#SBATCH --job-name=text-classify
#SBATCH -p defq
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=06:00:00

# Update Me!
#SBATCH --output=/home/<username>/logs/%x_%j.log
#SBATCH --error=/home/<username>/logs/%x_%j.log
USER_DIR="/home/${USER}"
CONTAINER_IMAGE="${USER_DIR}/path-to/curator.sqsh"
INPUT_DATA_PATH="s3://your-bucket/cleaned-text-data/"
OUTPUT_DATA_PATH="s3://your-bucket/classified-text-data/"
#

LOCAL_WORKSPACE="${USER_DIR}/nemo_curator_local_workspace"
LOCAL_WORKSPACE_MOUNT="${LOCAL_WORKSPACE}:/config"
NEMO_CONFIG_MOUNT="${HOME}/.config/nemo_curator/config.yaml:/nemo_curator/config/nemo_curator.yaml"
AWS_MOUNT="${HOME}/.aws:/root/.aws"
CONTAINER_MOUNTS="${LOCAL_WORKSPACE_MOUNT},${NEMO_CONFIG_MOUNT},${AWS_MOUNT}"

export NEMO_CURATOR_RAY_SLURM_JOB=1

srun \
  --mpi=none \
  --container-writable \
  --no-container-remap-root \
  --export=NEMO_CURATOR_RAY_SLURM_JOB \
  --container-image "${CONTAINER_IMAGE}" \
  --container-mounts "${CONTAINER_MOUNTS}" \
    -- python3 -m nemo_curator.scripts.classifiers.quality_classifier_inference \
      --input-data-dir "${INPUT_DATA_PATH}" \
      --output-data-dir "${OUTPUT_DATA_PATH}" \
      --scheduler-address "127.0.0.1" \
      --scheduler-port 8786 \
      --device gpu \
      --input-text-field "text" \
      --batch-size 256
```

:::

::: {tab-item} 3. Deduplication
`curator_text_dedup.sh` - Performs exact and fuzzy deduplication (semantic deduplication requires separate setup).

```bash
#!/bin/bash

#SBATCH --job-name=text-dedup
#SBATCH -p defq
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=12:00:00

# Update Me!
#SBATCH --output=/home/<username>/logs/%x_%j.log
#SBATCH --error=/home/<username>/logs/%x_%j.log
USER_DIR="/home/${USER}"
CONTAINER_IMAGE="${USER_DIR}/path-to/curator.sqsh"
INPUT_DATA_PATH="s3://your-bucket/classified-text-data/"
OUTPUT_DATA_PATH="s3://your-bucket/deduplicated-text-data/"
CACHE_DIR="s3://your-bucket/dedup-cache/"
#

LOCAL_WORKSPACE="${USER_DIR}/nemo_curator_local_workspace"
LOCAL_WORKSPACE_MOUNT="${LOCAL_WORKSPACE}:/config"
NEMO_CONFIG_MOUNT="${HOME}/.config/nemo_curator/config.yaml:/nemo_curator/config/nemo_curator.yaml"
AWS_MOUNT="${HOME}/.aws:/root/.aws"
CONTAINER_MOUNTS="${LOCAL_WORKSPACE_MOUNT},${NEMO_CONFIG_MOUNT},${AWS_MOUNT}"

export NEMO_CURATOR_RAY_SLURM_JOB=1

# Step 1: Exact Deduplication
srun \
  --mpi=none \
  --container-writable \
  --no-container-remap-root \
  --export=NEMO_CURATOR_RAY_SLURM_JOB \
  --container-image "${CONTAINER_IMAGE}" \
  --container-mounts "${CONTAINER_MOUNTS}" \
    -- python3 -m nemo_curator.scripts.find_exact_duplicates \
      --input-data-dirs "${INPUT_DATA_PATH}" \
      --output-dir "${CACHE_DIR}/exact_dedup" \
      --input-json-id-field "id" \
      --input-json-text-field "text" \
      --scheduler-address "127.0.0.1" \
      --scheduler-port 8786 \
      --device gpu

# Step 2: Fuzzy Deduplication - Compute MinHashes
srun \
  --mpi=none \
  --container-writable \
  --no-container-remap-root \
  --export=NEMO_CURATOR_RAY_SLURM_JOB \
  --container-image "${CONTAINER_IMAGE}" \
  --container-mounts "${CONTAINER_MOUNTS}" \
    -- python3 -m nemo_curator.scripts.fuzzy_deduplication.compute_minhashes \
      --input-data-dirs "${CACHE_DIR}/exact_dedup" \
      --output-minhash-dir "${CACHE_DIR}/fuzzy_dedup/minhashes" \
      --input-json-id-field "id" \
      --input-json-text-field "text" \
      --scheduler-address "127.0.0.1" \
      --scheduler-port 8786 \
      --device gpu \
      --char-ngram 24 \
      --minhash-length 260

# Step 3: Fuzzy Deduplication - LSH
srun \
  --mpi=none \
  --container-writable \
  --no-container-remap-root \
  --export=NEMO_CURATOR_RAY_SLURM_JOB \
  --container-image "${CONTAINER_IMAGE}" \
  --container-mounts "${CONTAINER_MOUNTS}" \
    -- python3 -m nemo_curator.scripts.fuzzy_deduplication.minhash_lsh \
      --input-data-dirs "${CACHE_DIR}/fuzzy_dedup/minhashes" \
      --output-bucket-dir "${CACHE_DIR}/fuzzy_dedup/buckets" \
      --input-json-id-field "id" \
      --scheduler-address "127.0.0.1" \
      --scheduler-port 8786 \
      --device gpu \
      --num-bands 20 \
      --buckets-per-shuffle 5

# Note: Semantic deduplication requires a separate configuration file and multi-step process
# See the semantic deduplication documentation for the complete workflow
```

:::

::: {tab-item} 4. PII Redaction
`curator_text_redact.sh` - Identifies and redacts personally identifiable information.

```bash
#!/bin/bash

#SBATCH --job-name=text-redact
#SBATCH -p defq
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=08:00:00

# Update Me!
#SBATCH --output=/home/<username>/logs/%x_%j.log
#SBATCH --error=/home/<username>/logs/%x_%j.log
USER_DIR="/home/${USER}"
CONTAINER_IMAGE="${USER_DIR}/path-to/curator.sqsh"
INPUT_DATA_PATH="s3://your-bucket/deduplicated-text-data/"
OUTPUT_DATA_PATH="s3://your-bucket/final-text-data/"
#

LOCAL_WORKSPACE="${USER_DIR}/nemo_curator_local_workspace"
LOCAL_WORKSPACE_MOUNT="${LOCAL_WORKSPACE}:/config"
NEMO_CONFIG_MOUNT="${HOME}/.config/nemo_curator/config.yaml:/nemo_curator/config/nemo_curator.yaml"
AWS_MOUNT="${HOME}/.aws:/root/.aws"
CONTAINER_MOUNTS="${LOCAL_WORKSPACE_MOUNT},${NEMO_CONFIG_MOUNT},${AWS_MOUNT}"

export NEMO_CURATOR_RAY_SLURM_JOB=1

srun \
  --mpi=none \
  --container-writable \
  --no-container-remap-root \
  --export=NEMO_CURATOR_RAY_SLURM_JOB \
  --container-image "${CONTAINER_IMAGE}" \
  --container-mounts "${CONTAINER_MOUNTS}" \
    -- python3 -m nemo_curator.scripts.find_pii_and_deidentify \
      --input-data-dir "${INPUT_DATA_PATH}" \
      --output-data-dir "${OUTPUT_DATA_PATH}" \
      --scheduler-address "127.0.0.1" \
      --scheduler-port 8786 \
      --device gpu \
      --supported-entities "PERSON,EMAIL_ADDRESS,PHONE_NUMBER" \
      --anonymize-action replace \
      --batch-size 1000 \
      --text-field "text"
```

:::

::::

1. **Update** all `# Update Me!` sections in the scripts for your environment (paths, usernames, S3 buckets, etc).
2. Submit each job with `sbatch`:

  ```sh
  sbatch curator_text_clean.sh
  sbatch curator_text_classify.sh
  sbatch curator_text_dedup.sh
  sbatch curator_text_redact.sh
  ```

## Monitoring and Logs

1. Check job status:

   ```bash
   squeue
   ```

2. View logs:

   ```bash
   tail -f /path/to/logs/<jobname>-<jobid>.log
   ```

## Performance Considerations

- **Memory Requirements**: Text deduplication can be memory-intensive. Consider increasing node memory or reducing batch sizes for large datasets.
- **Storage**: Intermediate deduplication files can be large. Ensure sufficient storage in your cache directory.
- **Scaling**: Semantic deduplication benefits from more GPU nodes. Consider 8+ nodes for datasets >1TB. 