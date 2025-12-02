---
description: "Core concepts for saving and exporting curated image datasets including metadata, filtering, and resharding"
categories: ["concepts-architecture"]
tags: ["data-export", "tar-files", "parquet", "filtering", "resharding", "metadata"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "concept"
modality: "image-only"
---

(about-concepts-image-data-export)=

# Data Export Concepts (Image)

This page covers the core concepts for saving and exporting curated image datasets in NeMo Curator.

## Key Topics

- Saving metadata to Parquet files
- Exporting filtered datasets as tar archives
- Configuring output sharding
- Understanding output format structure
- Preparing data for downstream training or analysis

## Saving Results

After processing through the pipeline, you can save the curated images and metadata using the `ImageWriterStage`.

**Example:**

```python
from nemo_curator.stages.image.io.image_writer import ImageWriterStage

# Add writer stage to pipeline
pipeline.add_stage(ImageWriterStage(
    output_dir="/output/curated_dataset",
    images_per_tar=1000,
    remove_image_data=True,
    verbose=True,
    deterministic_name=True,  # Use deterministic naming for reproducible output
))
```

- The writer stage creates tar files with curated images
- Metadata for each image (including paths, IDs, scores, and processing metadata) is always stored in separate Parquet files alongside tar archives
- Configurable images per tar file for optimal sharding
- `deterministic_name=True` ensures reproducible file naming based on input content

## Pipeline-Based Filtering

Filtering happens automatically within the pipeline stages. Each filter stage (aesthetic, NSFW) removes images that don't meet the configured thresholds, so only curated images reach the final `ImageWriterStage`.

**Example Pipeline Flow:**

```python
from nemo_curator.pipeline.pipeline import Pipeline
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.image.io.image_reader import ImageReaderStage
from nemo_curator.stages.image.embedders.clip_embedder import ImageEmbeddingStage
from nemo_curator.stages.image.filters.aesthetic_filter import ImageAestheticFilterStage
from nemo_curator.stages.image.filters.nsfw_filter import ImageNSFWFilterStage
from nemo_curator.stages.image.io.image_writer import ImageWriterStage

# Complete pipeline with filtering
pipeline = Pipeline(name="image_curation")

# Load images from tar archives
pipeline.add_stage(FilePartitioningStage(
    file_paths="/input/image_dataset",
    files_per_partition=1,
    file_extensions=[".tar"],
))

pipeline.add_stage(ImageReaderStage(
    batch_size=100,
    num_threads=16,
    num_gpus_per_worker=0.25,
))

# Generate CLIP embeddings (required for filters)
pipeline.add_stage(ImageEmbeddingStage(
    model_dir="/models",
    model_inference_batch_size=32,
    num_gpus_per_worker=0.25,
))

# Filter by quality (keeps images with aesthetic_score >= 0.5)
pipeline.add_stage(ImageAestheticFilterStage(
    model_dir="/models",
    score_threshold=0.5,
    num_gpus_per_worker=0.25,
))

# Filter NSFW content (keeps images with nsfw_score < 0.5)
pipeline.add_stage(ImageNSFWFilterStage(
    model_dir="/models",
    score_threshold=0.5,
    num_gpus_per_worker=0.25,
))

# Save curated results
pipeline.add_stage(ImageWriterStage(
    output_dir="/output/curated",
    images_per_tar=1000,
    remove_image_data=True,
))

# Execute the pipeline
results = pipeline.run()
```

- Filtering is built into the stages - no separate filtering step needed
- Images passing all filters reach the output
- Thresholds are configurable per stage
- **Note:** Aesthetic filter keeps images with `score >= threshold` (higher is better), while NSFW filter keeps images with `score < threshold` (lower is safer)

## Output Format

The `ImageWriterStage` creates tar archives containing curated images with accompanying metadata files:

**Output Structure:**

```bash
output/
├── images-{hash}-000000.tar    # Contains JPEG images
├── images-{hash}-000000.parquet # Metadata for corresponding tar
├── images-{hash}-000001.tar
├── images-{hash}-000001.parquet
```

**Format Details:**

- **Tar contents**: JPEG images with sequential or ID-based filenames
- **Metadata storage**: Separate Parquet files containing image paths, IDs, and processing metadata
- **Naming**: Deterministic or random naming based on configuration
- **Sharding**: Configurable number of images per tar file for optimal performance

## Configuring Output Sharding

The `ImageWriterStage` parameters control how images get distributed across output tar files.

**Example:**

```python
# Configure output sharding
pipeline.add_stage(ImageWriterStage(
    output_dir="/output/curated_dataset",
    images_per_tar=5000,  # Images per tar file
    remove_image_data=True,
    deterministic_name=True,
))
```

- Adjust `images_per_tar` to balance I/O, parallelism, and storage efficiency
- Smaller values create more files but enable better parallelism
- Larger values reduce file count but may impact loading performance

## Preparing for Downstream Use

- Ensure your exported dataset matches the requirements of your training or analysis pipeline.
- Use consistent naming and metadata fields for compatibility.
- Document any filtering or processing steps for reproducibility.
- Test loading the exported dataset before large-scale training.

<!-- Detailed content to be added here. --> 