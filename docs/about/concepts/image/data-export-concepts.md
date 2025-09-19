---
description: "Core concepts for saving and exporting curated image datasets including metadata, filtering, and resharding"
categories: ["concepts-architecture"]
tags: ["data-export", "webdataset", "parquet", "filtering", "resharding", "metadata"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "concept"
modality: "image-only"
---

(about-concepts-image-data-export)=
# Data Export Concepts (Image)

This page covers the core concepts for saving and exporting curated image datasets in NeMo Curator.

## Key Topics
- Saving metadata to Parquet
- Exporting filtered datasets
- Resharding WebDatasets
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
))
```
- The writer stage creates new WebDataset tar files with curated images
- Metadata (including scores) is preserved in the output structure
- Configurable images per tar file for optimal sharding

## Pipeline-Based Filtering

Filtering happens automatically within the pipeline stages. Each filter stage (aesthetic, NSFW) removes images that don't meet the configured thresholds, so only curated images reach the final `ImageWriterStage`.

**Example Pipeline Flow:**
```python
# Complete pipeline with filtering
pipeline = Pipeline(name="image_curation")

# Load images
pipeline.add_stage(FilePartitioningStage(...))
pipeline.add_stage(ImageReaderStage(...))

# Generate embeddings
pipeline.add_stage(ImageEmbeddingStage(...))

# Filter by quality (removes low aesthetic scores)
pipeline.add_stage(ImageAestheticFilterStage(score_threshold=0.5))

# Filter NSFW content (removes high NSFW scores)  
pipeline.add_stage(ImageNSFWFilterStage(score_threshold=0.5))

# Save curated results
pipeline.add_stage(ImageWriterStage(output_dir="/output/curated"))
```
- Filtering is built into the stages - no separate filtering step needed
- Only images passing all filters reach the output
- Thresholds are configurable per stage

## Resharding WebDatasets

Resharding is controlled by the `ImageWriterStage` parameters, which determine how many images are packed into each output tar file.

**Example:**
```python
# Configure output sharding
pipeline.add_stage(ImageWriterStage(
    output_dir="/output/resharded_dataset",
    images_per_tar=5000,  # New shard size
    remove_image_data=True,
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