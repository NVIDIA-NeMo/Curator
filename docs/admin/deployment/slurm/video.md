---
only: not ga
---

(admin-deployment-slurm-video)=
#  Deploy Video Curation on Slurm

This workflow covers the full video curation pipeline on SLURM, including model download, video splitting, deduplication, and sharding.

```{seealso}
For details on video container environments and Slurm environment variables, see [Container Environments](reference-infrastructure-container-environments).
```

## Prerequisites

- Create required directories for AWS credentials, NeMo Curator config, and local workspace:

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

  ::: {tab-item} NeMo Curator Config
  `$HOME/.config/nemo_curator/config.yaml` (for HuggingFace API key)

  ```{literalinclude} _assets/.config/nemo_curator/config.yaml
  :language: yaml
  ```

  :::

  ::: {tab-item} Model Download Config
  `$HOME/nemo_curator_local_workspace/model_download.yaml` (lists required models)

  ```{literalinclude} _assets/nemo_curator_local_workspace/model_download.yaml
  :language: yaml
  ```

  :::

  ::::

---

## Model Download

1. Copy the following script for downloading all required model weights into the Slurm cluster.

   ```{literalinclude} _assets/1_curator_download_models.sh
   :language: sh
   ```

2. Update the `SBATCH` parameters and paths to match your username and environment.
3. Run the script.

   ```bash
   sbatch 1_curator_download_models.sh
   ```

## Video Processing Pipeline

The workflow consists of three main SLURM scripts, to be run in order:

1. `curator_split.sh`: Splits, transcodes, annotates, and generates embeddings for raw video data.
2. `curator_dedup.sh`: Deduplicates video clips using generated embeddings.
3. `curator_shard.sh`: Packages a sharded dataset, removing deduplicated clips.

:::: {tab-set}

::: {tab-item} 1. Splitting
`curator_split.sh` - Splits, transcodes, annotates, and generates embeddings for raw video data.

```{literalinclude} _assets/2_curator_split.sh
:language: sh
```

:::

::: {tab-item} 2. Deduplication
`curator_dedup.sh` - Deduplicates video clips using generated embeddings.

```{literalinclude} _assets/3_curator_dedup.sh
:language: sh
```

:::

::: {tab-item} 3. Sharding
`curator_shard.sh` - Packages a sharded dataset, removing deduplicated clips.

```{literalinclude} _assets/4_curator_shard.sh
:language: sh
```

:::

::::

1. **Update** all `# Update Me!` sections in the scripts for your environment (paths, usernames, S3 buckets, etc).
2. Submit each job with `sbatch`:

  ```sh
  sbatch curator_split.sh
  sbatch curator_dedup.sh
  sbatch curator_shard.sh
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
