---
description: "Guide for migrating NeMo Curator workflows from Dask to Ray-based pipeline architecture"
categories: ["migration", "release-notes"]
tags: ["migration", "dask", "ray", "pipeline", "text", "image", "video", "audio"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "guide"
modality: "universal"
---

(migration-guide)=

# NeMo Curator Migration Guide: Dask to Ray

This guide explains how to transition existing Dask-based NeMo Curator workflows to the new Ray-based pipeline architecture.

```{seealso}
For broader NeMo Framework migration topics, refer to the [NeMo Framework 2.0 Migration Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/migration/index.html).
```

## Overview

NeMo Curator previously used Dask as its primary execution engine for distributed data processing. The latest Curator architecture transitions to Ray as a unified backend, enabling all modalities—text, image, video, and audio—to use a single, consistent execution engine.

Workflows built as sequential function calls will need to be refactored into pipelines composed of modular stages. This migration guide explains how to transition existing workflows to the new modular, Ray-based Curator Pipeline structure.

### Previous Approach: Dask-Based Sequential Processing

The example below is a skeleton Dask-based data loading workflow. Each operation is represented as a function and applied to the entire dataset at once.

```python
# Old workflow: Sequential Dask-based processing
dataset = DocumentDataset.read_json()
processor = Processor()
processed = processor(dataset)
result = processed.to_parquet("output")
```

### New Approach: Ray-Based Modular Pipelines

The example below implements the same skeleton workflow using the new Curator Pipeline architecture. Each stage is a standalone component focused on a specific operation and can be flexibly combined within a Pipeline object. In this new system, data flows through the pipeline as discrete tasks, each containing a batch of data (such as a `DocumentBatch` for text or `ImageBatch` for images). Each stage operates independently and in parallel on its assigned batch.

```python
# New workflow: Modular, Ray-based pipeline
pipeline = Pipeline(name="my_pipeline")
pipeline.add_stage(ReaderStage())
pipeline.add_stage(ProcessingStage())
pipeline.add_stage(WriterStage())
results = pipeline.run(executor)
```

```{seealso}
For more details about the new design, refer to the [Curator Ray API Design](https://github.com/NVIDIA-NeMo/Curator/blob/31cdc6f5ded46d059ad28fa24ff32afad2d430fd/api-design.md) documentation.
```

---

## Migrating Text Curation Workflows

Previously, NeMo Curator loaded and processed text data as standardized `DocumentDataset` objects. These objects could then be used for further curation steps, including additional processing, filtering, and generation steps.

In the new release, this same functionality is available through a pipeline architecture, which uses stages to handle each discrete curation task.

The following example data loading pipeline showcases the differences between the Dask-based (previous) and Ray-based (current) curation approaches.

### Step 1: Start a Distributed Computing Client

The script begins by initializing the distributed computing client. This client manages execution of tasks across multiple workers.

**Previous: Dask Cluster**

Initialize a local Dask cluster, specifying `cluster_type="gpu"` to leverage GPU resources.

```python
# Old: Dask
from nemo_curator.utils.distributed_utils import get_client
dask_client = get_client(cluster_type="gpu")
```

**New: Ray Cluster**

Connect to a Ray cluster, which can manage tasks across CPU or GPU-backed nodes. For cluster setup details, refer to {ref}`Production Deployment Requirements <admin-deployment-requirements>` and the [Ray documentation](https://docs.ray.io/en/latest/cluster/getting-started.html).

```python
# New: Ray
from nemo_curator.core.client import RayClient
ray_client = RayClient()
ray_client.start()
```

### Step 2: Define Operations

In this step, core data curation operations—such as loading, cleaning, filtering, and deduplication—are defined. In Dask-based workflows, each processing step is written as a sequential call (often as Python functions or chained operations). In Ray-based workflows, each operation is expressed as a modular, declarative stage.

Example operations:

- Download the dataset and convert it to JSONL format
- Clean and unify the dataset (remove quotation marks, Unicode)
- Filter the dataset based on various criteria (word count, completeness)
- Remove exact duplicates from the dataset (deduplication)

**Previous: Sequential Operations**

In the previous version of NeMo Curator, the data loading and formatting process could be run sequentially, as individual functions or within `main()`, as in the code snippet below.

```python
# Old: Define curation logic
def main():
    dataset = DocumentDataset.read_json(files)
    
    # Clean and unify
    cleaners = Sequential([
        Modify(QuotationUnifier()),
        Modify(UnicodeReformatter()),
    ])
    dataset = cleaners(dataset)

    # Filter
    filters = Sequential([
        ScoreFilter(WordCountFilter(min_words=80)),
        ScoreFilter(IncompleteStoryFilter()),
    ])
    dataset = filters(dataset)

    # Deduplicate
    deduplicator = ExactDuplicates()
    duplicates = deduplicator(dataset)
    dataset = deduplicator.remove(dataset, duplicates)

    # Write results
    dataset.to_json(out_path, write_to_filename=True)
```

**New: Modular Stages**

In the new version, these operations are defined as discrete stages that operate on batches of data. Each stage can specify resources such as GPU count or CPU threads. For details on {ref}`available filters <text-process-data-filter-heuristic>`, {ref}`content cleaning operations <about-concepts-text-data-processing>`, and {ref}`pipeline concepts <about-concepts-video-abstractions>`, refer to the linked documentation.

```python
# New: Define stages
stages = [
    TinyStoriesDownloadExtractStage(raw_dir, split=args.split),
    Modify(modifier_fn=QuotationUnifier()),
    Modify(modifier_fn=UnicodeReformatter()),
    ScoreFilter(filter_obj=WordCountFilter(min_words=80)),
    ScoreFilter(filter_obj=IncompleteStoryFilter()),
    JsonlWriter(curated_dir),
]
```

```{note}
In the new version, deduplication should be run as a separate workflow using classes like `ExactDeduplicationWorkflow`, not embedded directly as a pipeline stage. For details and usage, refer to the {ref}`text deduplication documentation <text-process-data-dedup>`.
```

### Step 3: Create and Run Pipeline

After defining all the required processing steps, you can assemble and execute your workflow.

In the new version, a pipeline object can be created using the previously defined stages. The pipeline can then be run using the Curator Pipeline `run()` function. The pipeline is run using the Xenna executor.

```python
# New: Create pipeline with stages
pipeline = Pipeline(
    name="tinystories",
    description="Download and curation pipeline for the TinyStories dataset.",
    stages=stages,
)

# Create executor (see {ref}`Pipeline Execution Backends <reference-execution-backends>` for configuration options)
executor = XennaExecutor()

# Execute pipeline
pipeline.run(executor)
```

### Step 4: Stop the Client

As a final step, stop the distributed computing client to release resources and cleanly terminate your session.

**Previous: Dask Client**

```python
# Close Dask client
dask_client.close()
```

**New: Ray Client**

```python
# Stop Ray client
ray_client.stop()
```

```{note}
This is a high-level example, and exact implementation details may vary. For more in-depth information about setting up text curation pipelines, refer to the {ref}`text curation quickstart <gs-text>`.
```

---

## Migrating Image Curation Workflows

This section demonstrates how to transition Dask-based image curation to the new Ray-based modular pipeline.

The following steps walk through constructing and running an image curation workflow in the new release, highlighting differences and adjustments compared to the old workflow.

### Step 1: Start a Distributed Computing Client

First, start your distributed computing client.

**Previous: Dask Client**

The previous version relied on a Dask client, specifying `cluster_type="gpu"` to leverage GPU resources.

```python
# Old: Dask
from nemo_curator.utils.distributed_utils import get_client
dask_client = get_client(cluster_type="gpu")
```

**New: Ray Client**

The new version uses Ray, which can be initialized with the following code:

```python
# New: Ray
from nemo_curator.core.client import RayClient
ray_client = RayClient()
ray_client.start()
```

### Step 2: Load and Preprocess Data

Next, load your image data. This step reads image files and prepares them for downstream processing.

**Previous: Dataset-Based Loading**

In the previous version of NeMo Curator, data loading was performed using helper functions from Curator dataset classes such as `ImageTextPairDataset`. This approach required users to directly manage dataset construction and often involved chaining Dask-based operations for filtering or transformation.

```python
# Old: Load Dataset
dataset = ImageTextPairDataset.from_webdataset(dataset_path, id_col)
```

**New: Stage-Based Loading**

In the new version, data loading is encapsulated in a dedicated pipeline stage (see {ref}`Image Processing Concepts <about-concepts-image-data-processing>` for details). Instead of directly creating a dataset, users define an `ImageReaderStage` that handles reading from WebDataset `.tar` files.

```python
# New: Read images from webdataset tar files
read_stage = ImageReaderStage(
    task_batch_size=args.task_batch_size,
    num_threads=16,
    num_gpus_per_worker=0.25,
)
```

### Step 3: Generate CLIP Embeddings

Once the image-text data has been loaded, the next step is to convert it into vector representations using a CLIP (Contrastive Language-Image Pre-training) model. This allows the data to be used in tasks such as filtering, clustering, deduplication, and similarity search.

**Previous: Direct Model Application**

In the previous NeMo Curator version, embeddings were generated by instantiating an embedding model and applying it directly to the dataset object.

```python
# Old: Generate CLIP embeddings for images
from nemo_curator.image.embedders import TimmImageEmbedder

embedding_model = TimmImageEmbedder(
    "vit_large_patch14_clip_quickgelu_224.openai",
    pretrained=True,
)

dataset = embedding_model(dataset)

dataset.save_metadata()
```

**New: Embedding Stage**

In the new version, embedding generation is handled by a dedicated `ImageEmbeddingStage` pipeline stage with configurable resource parameters (see {ref}`CLIP Embedding Stage <image-process-data-embeddings-clip>` for details).

```python
# New: Generate CLIP embeddings for images
img_embedding_stage = ImageEmbeddingStage(
    model_dir=args.model_dir,
    num_gpus_per_worker=args.embedding_gpus_per_worker,
    model_inference_batch_size=args.embedding_batch_size,
    remove_image_data=False,
    verbose=args.verbose,
)
```

### Step 4: Aesthetic Scoring

Aesthetic scoring assigns a quality score to each image based on its visual appeal. This score can be used to filter out poor-quality images from a dataset.

**Previous: Classifier-Based Filtering**

In the previous version, aesthetic scoring was performed by applying an `AestheticClassifier` directly to the dataset. This added a new column with scores and a boolean filter for high-quality images. The filtered dataset could then be saved using `to_webdataset()`.

```python
# Old: Generate aesthetic quality scores and filter
from nemo_curator.image.classifiers import AestheticClassifier

aesthetic_classifier = AestheticClassifier()
dataset = aesthetic_classifier(dataset)

dataset.to_webdataset(aesthetic_dataset_path, filter_column="passes_aesthetic_check")
```

**New: Aesthetic Filter Stage**

In the new version, aesthetic scoring and filtering are handled by the `ImageAestheticFilterStage` (see {ref}`Aesthetic Filter <image-process-data-filters-aesthetic>` for details). This stage scores each image using a pretrained model and filters out images below a configured threshold.

```python
# New: Generate aesthetic quality scores and filter
aesthetic_filter_stage = ImageAestheticFilterStage(
    model_dir=args.model_dir,
    num_gpus_per_worker=args.aesthetic_gpus_per_worker,
    model_inference_batch_size=args.aesthetic_batch_size,
    score_threshold=args.aesthetic_threshold,
    verbose=args.verbose,
)
```

### Step 5: Semantic Deduplication

Semantic deduplication removes visually or semantically similar images from the dataset by clustering embeddings and eliminating near-duplicates based on similarity.

**Previous: Multi-Step Clustering and Deduplication**

The previous NeMo Curator version required two separate steps. First, image embeddings were clustered to group similar images. Second, deduplication, based on cosine similarity, was performed within clusters.

```python
# Old: Semantic Deduplication

# Cluster image embeddings
clustering_model = ClusteringModel(
    id_column=id_col,
    embedding_column="image_embedding",
    clustering_output_dir=clustering_output,
)
clustered_dataset = clustering_model(embeddings_dataset)

# Run cluster-level dedup
emb_by_cluster_output = os.path.join(clustering_output, "embs_by_nearest_center")
duplicate_output = os.path.join(semantic_dedup_outputs, "duplicates")

semantic_dedup = SemanticClusterLevelDedup(
    n_clusters=1,
    emb_by_clust_dir=emb_by_cluster_output,
    id_column=id_col,
    which_to_keep="hard",
    sim_metric="cosine",
    embedding_column="image_embedding",
    batched_cosine_similarity=1024,
    output_dir=duplicate_output,
)
semantic_dedup.compute_semantic_match_dfs()
deduplicated_dataset_ids = semantic_dedup.extract_dedup_data(eps_to_extract=0.01)

# Remove duplicates
deduplicated_dataset_path = "./deduplicated_dataset"
dataset.metadata["is_unique"] = ~dataset.metadata["key"].isin(
    deduplicated_dataset_ids.df["key"].compute(),
)
dataset.to_webdataset(deduplicated_dataset_path, "is_unique")
```

**New: Single Deduplication Stage**

In the new version, semantic deduplication is encapsulated in a single stage, `SemanticDeduplicationStage` (see {ref}`Deduplication Concepts <about-concepts-deduplication>` for comprehensive documentation). This stage handles clustering and duplicate removal internally, using the configured number of clusters and similarity threshold.

```python
# New: Semantic Deduplication
semantic_dedup_stage = SemanticDeduplicationStage(
    id_field="image_id",
    embedding_field="image_embedding",
    n_clusters=100,
    eps=0.01,
)
```

### Step 6: Create and Run Pipeline

In the previous version, each command could be run directly; assembling the defined functions into a "pipeline" format was optional.

In the new version, once all the required stages are defined (data reading, embedding generation, aesthetic filtering, and deduplication), you can assemble them into a pipeline and run it using a Ray-based executor.

```python
# New: Define pipeline
pipeline = Pipeline(
    name="image_curation",
    description="Curate images with embeddings and quality scoring",
    stages=[
        read_stage,
        img_embedding_stage,
        semantic_dedup_stage,
    ],
)

# Create executor
executor = XennaExecutor()

# Execute pipeline
pipeline.run(executor)
```

### Step 7: Stop the Client

As a final step, stop the distributed computing client to release resources and cleanly terminate your session.

**Previous: Dask Client**

```python
# Old: Close Dask client
dask_client.close()
```

**New: Ray Client**

```python
# New: Stop Ray client
ray_client.stop()
```

```{note}
This is a high-level example, and exact implementation details may vary. For more in-depth information about setting up image curation pipelines, refer to the {ref}`image curation quickstart <gs-image>`.
```

---

## Additional Resources

For questions about migration or other topics, refer to the {ref}`Migration FAQ <migration-faq>`.
