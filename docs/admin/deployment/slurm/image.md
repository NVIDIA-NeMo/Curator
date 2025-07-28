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
   import timm
   from nemo_curator.image.embedders import TimmImageEmbedder
   from nemo_curator.image.classifiers import AestheticClassifier, NsfwClassifier
   
   # Download and cache CLIP model
   embedder = TimmImageEmbedder('vit_large_patch14_clip_quickgelu_224.openai', pretrained=True)
   
   # Download aesthetic and NSFW classifiers
   aesthetic = AestheticClassifier()
   nsfw = NsfwClassifier()
   
   print('Image models downloaded successfully')
   "
   ```

2. Update the `SBATCH` parameters and paths to match your username and environment.
3. Run the script.

   ```bash
   sbatch 1_curator_download_image_models.sh
   ```

## Image Processing Pipeline

The workflow consists of three main Slurm scripts, to be run in order:

1. `curator_image_embed.sh`: Generates embeddings and applies classifications to images.
2. `curator_image_filter.sh`: Filters images based on quality, aesthetic, and NSFW scores.
3. `curator_image_dedup.sh`: Performs semantic deduplication using image embeddings.

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
from nemo_curator.datasets import ImageTextPairDataset
from nemo_curator.image.embedders import TimmImageEmbedder
from nemo_curator.image.classifiers import AestheticClassifier, NsfwClassifier
from nemo_curator.utils.distributed_utils import get_client

# Initialize Dask client
client = get_client(cluster_type='gpu')

# Load dataset
dataset = ImageTextPairDataset.from_webdataset('${INPUT_WEBDATASET_PATH}', id_col='key')

# Generate embeddings
embedder = TimmImageEmbedder(
    'vit_large_patch14_clip_quickgelu_224.openai',
    pretrained=True,
    batch_size=1024,
    num_threads_per_worker=16,
    normalize_embeddings=True,
    autocast=False
)
dataset = embedder(dataset)

# Apply aesthetic classification
aesthetic_classifier = AestheticClassifier()
dataset = aesthetic_classifier(dataset)

# Apply NSFW classification
nsfw_classifier = NsfwClassifier()
dataset = nsfw_classifier(dataset)

# Save results
dataset.to_webdataset('${OUTPUT_DATA_PATH}')
client.close()
"
```

:::

::: {tab-item} 2. Filtering
`curator_image_filter.sh` - Filters images based on quality, aesthetic, and NSFW scores.

```bash
#!/bin/bash

#SBATCH --job-name=image-filter
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
OUTPUT_DATA_PATH="s3://your-bucket/filtered-images/"
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
from nemo_curator.datasets import ImageTextPairDataset
from nemo_curator.utils.distributed_utils import get_client
import yaml

# Initialize Dask client
client = get_client(cluster_type='gpu')

# Load configuration
with open('/config/image_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load dataset
dataset = ImageTextPairDataset.from_webdataset('${INPUT_DATA_PATH}', id_col='key')

# Apply filters based on configuration
filter_params = config['filtering']
filters = []

# Aesthetic score filter
if 'min_aesthetic_score' in filter_params:
    filters.append(f\"aesthetic_score >= {filter_params['min_aesthetic_score']}\")

# NSFW score filter  
if 'max_nsfw_score' in filter_params:
    filters.append(f\"nsfw_score <= {filter_params['max_nsfw_score']}\")

# Apply combined filter
if filters:
    filter_expression = ' and '.join(filters)
    dataset.metadata['passes_filter'] = dataset.metadata.eval(filter_expression)
    
    # Save filtered dataset
    dataset.to_webdataset('${OUTPUT_DATA_PATH}', filter_column='passes_filter')
else:
    dataset.to_webdataset('${OUTPUT_DATA_PATH}')

print(f'Filtering completed. Applied filters: {filter_expression if filters else \"None\"}')
client.close()
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
from nemo_curator.datasets import ImageTextPairDataset, DocumentDataset
from nemo_curator import ClusteringModel, SemanticClusterLevelDedup
from nemo_curator.utils.distributed_utils import get_client
import yaml
import os

# Initialize Dask client
client = get_client(cluster_type='gpu')

# Load configuration
with open('/config/image_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load dataset
dataset = ImageTextPairDataset.from_webdataset('${INPUT_DATA_PATH}', id_col='key')
embeddings_dataset = DocumentDataset(dataset.metadata)

# Semantic deduplication parameters
dedup_config = config['semantic_deduplication']
clustering_output = '${CACHE_DIR}/cluster_output'
os.makedirs(clustering_output, exist_ok=True)

# Run clustering
clustering_model = ClusteringModel(
    id_column='key',
    embedding_column='image_embedding',
    max_iter=dedup_config['max_iter'],
    n_clusters=dedup_config['n_clusters'],
    random_state=dedup_config['random_state'],
    clustering_output_dir=clustering_output,
)
clustered_dataset = clustering_model(embeddings_dataset)

if clustered_dataset:
    print('Clustering completed successfully')
    
    # Run cluster-level deduplication
    emb_by_cluster_output = os.path.join(clustering_output, 'embs_by_nearest_center')
    duplicate_output = '${CACHE_DIR}/duplicates'
    
    semantic_dedup = SemanticClusterLevelDedup(
        n_clusters=dedup_config['n_clusters'],
        emb_by_clust_dir=emb_by_cluster_output,
        id_column='key',
        which_to_keep=dedup_config['which_to_keep'],
        embedding_column='image_embedding',
        batched_cosine_similarity=1024,
        output_dir=duplicate_output,
    )
    
    semantic_dedup.compute_semantic_match_dfs()
    deduplicated_dataset_ids = semantic_dedup.extract_dedup_data(
        eps_to_extract=dedup_config['eps_to_extract']
    )
    
    # Mark unique images and save
    dataset.metadata['is_unique'] = dataset.metadata['key'].isin(
        deduplicated_dataset_ids.df['key'].compute()
    )
    dataset.to_webdataset('${OUTPUT_DATA_PATH}', filter_column='is_unique')
    
    print(f'Deduplication completed. Unique images saved to ${OUTPUT_DATA_PATH}')
else:
    print('Clustering failed')

client.close()
"
```

:::

::::

1. **Update** all `# Update Me!` sections in the scripts for your environment (paths, usernames, S3 buckets, etc).
2. Submit each job with `sbatch`:

  ```sh
  sbatch curator_image_embed.sh
  sbatch curator_image_filter.sh
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