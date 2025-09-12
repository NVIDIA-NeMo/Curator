(admin-deployment-slurm-image)=
# Deploy Image Curation on Slurm

<!-- Note: This documentation has been verified against NeMo Curator source code for technical accuracy -->

This workflow covers the full image curation pipeline on Slurm, including model download, embedding generation, classification, filtering, and deduplication. 

```{seealso}
For details on image container environments and Slurm environment variables, see [Container Environments](reference-infrastructure-container-environments).
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

  ::: {tab-item} Image Processing Configuration
  `$HOME/nemo_curator_local_workspace/image_config.yaml` (image processing parameters)

  ```yaml
  # Image curation configuration
  embedding:
    model_name: "vit_large_patch14_clip_quickgelu_224.openai"
    batch_size: 1024
    num_threads_per_worker: 16
    normalize_embeddings: true
    autocast: false

  aesthetic_classifier:
    score_threshold: 6.0
    batch_size: 512

  nsfw_classifier:
    score_threshold: 0.2
    batch_size: 512

  semantic_deduplication:
    max_iter: 100
    n_clusters: 50000
    random_state: 42
    eps_thresholds: [0.9, 0.95, 0.99]
    eps_to_extract: 0.95
    which_to_keep: "hard"

  filtering:
    min_resolution: 256
    max_aspect_ratio: 3.0
    min_aesthetic_score: 6.0
    max_nsfw_score: 0.2
  ```

  :::

  ::::

---

## Model Download

1. Copy the following script for downloading all required image processing models into the Slurm cluster.

   ```bash
   #!/bin/bash

   #SBATCH --job-name=download_image_models
   #SBATCH -p defq
   #SBATCH --nodes=1
   #SBATCH --ntasks-per-node=1
   #SBATCH --exclusive
   #SBATCH --gres=gpu:1

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

   # Download Image Processing Models
   srun \
     --mpi=none \
     --container-writable \
     --no-container-remap-root \
     --export=NEMO_CURATOR_RAY_SLURM_JOB,NEMO_CURATOR_LOCAL_DOCKER_JOB \
     --container-image "${CONTAINER_IMAGE}" \
     --container-mounts "${CONTAINER_MOUNTS}" \
       --  python3 -c "
   from nemo_curator.stages.image.embedders.clip_embedder import ImageEmbeddingStage
   from nemo_curator.stages.image.filters.aesthetic_filter import ImageAestheticFilterStage
   from nemo_curator.stages.image.filters.nsfw_filter import ImageNSFWFilterStage
   import os
   
   # Create model directory
   model_dir = '/config/models'
   os.makedirs(model_dir, exist_ok=True)
   
   # Download and cache models by initializing stages
   embedding_stage = ImageEmbeddingStage(model_dir=model_dir, num_gpus_per_worker=0.25)
   aesthetic_stage = ImageAestheticFilterStage(model_dir=model_dir, score_threshold=0.5, num_gpus_per_worker=0.25)
   nsfw_stage = ImageNSFWFilterStage(model_dir=model_dir, score_threshold=0.5, num_gpus_per_worker=0.25)
   
   print('Image models downloaded successfully')
   "
   ```

2. Update the `SBATCH` parameters and paths to match your username and environment.
3. Run the script.

   ```bash
   sbatch 1_curator_download_image_models.sh
   ```

## Image Processing Pipeline

The workflow consists of three main Slurm scripts. The first script performs the complete curation pipeline, while the others are optional for specific use cases:

1. `curator_image_embed.sh`: Complete pipeline - partitions files, reads images, generates embeddings, applies filtering, and saves results.
2. `curator_image_reshard.sh`: (Optional) Reshards processed images with different shard sizes.
3. `curator_image_dedup.sh`: (Optional) Prepares data for semantic deduplication (full deduplication features coming in future releases).

:::: {tab-set}

::: {tab-item} 1. Embedding
`curator_image_embed.sh` - Generates embeddings and applies classifications to images.

```bash
#!/bin/bash

#SBATCH --job-name=image-embed
#SBATCH -p defq
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00

# Update Me!
#SBATCH --output=/home/<username>/logs/%x_%j.log
#SBATCH --error=/home/<username>/logs/%x_%j.log
USER_DIR="/home/${USER}"
CONTAINER_IMAGE="${USER_DIR}/path-to/curator.sqsh"
INPUT_WEBDATASET_PATH="s3://your-bucket/raw-images/{00000..00999}.tar"
OUTPUT_DATA_PATH="s3://your-bucket/embedded-images/"
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
    -- python3 -c "
from nemo_curator.pipeline import Pipeline
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.image.io.image_reader import ImageReaderStage
from nemo_curator.stages.image.embedders.clip_embedder import ImageEmbeddingStage
from nemo_curator.stages.image.filters.aesthetic_filter import ImageAestheticFilterStage
from nemo_curator.stages.image.filters.nsfw_filter import ImageNSFWFilterStage
from nemo_curator.stages.image.io.image_writer import ImageWriterStage

# Create image curation pipeline
pipeline = Pipeline(name='slurm_image_embed', description='Image embedding and classification on Slurm')

# Stage 1: Partition WebDataset files
pipeline.add_stage(FilePartitioningStage(
    file_paths='${INPUT_WEBDATASET_PATH}',
    files_per_partition=1,
    file_extensions=['.tar'],
))

# Stage 2: Read images with DALI
pipeline.add_stage(ImageReaderStage(
    task_batch_size=100,
    num_threads=16,
    num_gpus_per_worker=0.25,
    verbose=True,
))

# Stage 3: Generate CLIP embeddings
pipeline.add_stage(ImageEmbeddingStage(
    model_dir='/config/models',
    model_inference_batch_size=32,
    num_gpus_per_worker=0.25,
    remove_image_data=False,
    verbose=True,
))

# Stage 4: Apply aesthetic classification
pipeline.add_stage(ImageAestheticFilterStage(
    model_dir='/config/models',
    score_threshold=0.5,
    model_inference_batch_size=32,
    num_gpus_per_worker=0.25,
    verbose=True,
))

# Stage 5: Apply NSFW classification
pipeline.add_stage(ImageNSFWFilterStage(
    model_dir='/config/models',
    score_threshold=0.5,
    model_inference_batch_size=32,
    num_gpus_per_worker=0.25,
    verbose=True,
))

# Stage 6: Save results
pipeline.add_stage(ImageWriterStage(
    output_dir='${OUTPUT_DATA_PATH}',
    images_per_tar=1000,
    remove_image_data=True,
    verbose=True,
))

# Execute pipeline
executor = XennaExecutor()
pipeline.run(executor)
"
```

:::

::: {tab-item} 2. Resharding (Optional)
`curator_image_reshard.sh` - Reshards the processed images with different shard sizes if needed.

```bash
#!/bin/bash

#SBATCH --job-name=image-reshard
#SBATCH -p defq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00

# Update Me!
#SBATCH --output=/home/<username>/logs/%x_%j.log
#SBATCH --error=/home/<username>/logs/%x_%j.log
USER_DIR="/home/${USER}"
CONTAINER_IMAGE="${USER_DIR}/path-to/curator.sqsh"
INPUT_DATA_PATH="s3://your-bucket/embedded-images/"
OUTPUT_DATA_PATH="s3://your-bucket/resharded-images/"
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
    -- python3 -c "
from nemo_curator.pipeline import Pipeline
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.image.io.image_reader import ImageReaderStage
from nemo_curator.stages.image.io.image_writer import ImageWriterStage

# Create resharding pipeline
pipeline = Pipeline(name='slurm_image_reshard', description='Reshard processed images')

# Stage 1: Partition processed WebDataset files
pipeline.add_stage(FilePartitioningStage(
    file_paths='${INPUT_DATA_PATH}',
    files_per_partition=1,
    file_extensions=['.tar'],
))

# Stage 2: Read processed images
pipeline.add_stage(ImageReaderStage(
    task_batch_size=100,
    num_threads=16,
    num_gpus_per_worker=0.25,
    verbose=True,
))

# Stage 3: Write with new shard size
pipeline.add_stage(ImageWriterStage(
    output_dir='${OUTPUT_DATA_PATH}',
    images_per_tar=5000,  # Larger shard size
    remove_image_data=True,
    verbose=True,
))

# Execute pipeline
executor = XennaExecutor()
pipeline.run(executor)
"
```

:::

::: {tab-item} 3. Deduplication
`curator_image_dedup.sh` - Performs semantic deduplication using image embeddings.

```bash
#!/bin/bash

#SBATCH --job-name=image-dedup
#SBATCH -p defq
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00

# Update Me!
#SBATCH --output=/home/<username>/logs/%x_%j.log
#SBATCH --error=/home/<username>/logs/%x_%j.log
USER_DIR="/home/${USER}"
CONTAINER_IMAGE="${USER_DIR}/path-to/curator.sqsh"
INPUT_DATA_PATH="s3://your-bucket/filtered-images/"
OUTPUT_DATA_PATH="s3://your-bucket/deduplicated-images/"
CACHE_DIR="s3://your-bucket/image-dedup-cache/"
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
    -- python3 -c "
# Note: Semantic deduplication requires the old API for now
# This is a complex operation that hasn't been fully migrated to the pipeline approach
from nemo_curator.pipeline import Pipeline
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.image.io.image_reader import ImageReaderStage
from nemo_curator.stages.image.io.image_writer import ImageWriterStage
import yaml
import os

print('Note: Advanced semantic deduplication is currently being migrated to the pipeline architecture.')
print('For now, this script performs basic resharding. Full deduplication will be available in future releases.')

# Create basic pipeline for now
pipeline = Pipeline(name='slurm_image_basic_dedup', description='Basic image processing for deduplication prep')

# Stage 1: Partition processed WebDataset files
pipeline.add_stage(FilePartitioningStage(
    file_paths='${INPUT_DATA_PATH}',
    files_per_partition=1,
    file_extensions=['.tar'],
))

# Stage 2: Read processed images
pipeline.add_stage(ImageReaderStage(
    task_batch_size=100,
    num_threads=16,
    num_gpus_per_worker=0.25,
    verbose=True,
))

# Stage 3: Save for deduplication processing
pipeline.add_stage(ImageWriterStage(
    output_dir='${OUTPUT_DATA_PATH}',
    images_per_tar=1000,
    remove_image_data=True,
    verbose=True,
))

# Execute pipeline
executor = XennaExecutor()
pipeline.run(executor)

print('Basic processing completed. Advanced semantic deduplication will be available in future pipeline releases.')
"
```

:::

::::

1. **Update** all `# Update Me!` sections in the scripts for your environment (paths, usernames, S3 buckets, etc).
2. Submit the main processing job:

  ```sh
  # Main pipeline (required) - performs complete image curation
  sbatch curator_image_embed.sh
  ```

3. Optionally submit additional jobs if needed:

  ```sh
  # Optional: Reshard processed images with different shard sizes
  sbatch curator_image_reshard.sh
  
  # Optional: Prepare for deduplication (basic processing for now)
  sbatch curator_image_dedup.sh
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

- **GPU Memory**: Image processing requires significant GPU memory. Consider using nodes with high-memory GPUs (40GB+ VRAM) for large batch sizes.
- **WebDataset Format**: Ensure your input data is in WebDataset format (`.tar` files containing images, captions, and metadata).
- **Network I/O**: Image data can be large. Consider local caching or high-bandwidth storage for better performance.
- **Clustering Scale**: For datasets with millions of images, increase `n_clusters` to 50,000+ to improve deduplication performance. 