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

## Saving Metadata

After processing, you can save the dataset's metadata (including embeddings, classifier scores, and other fields) to Parquet files for easy analysis or further processing.

**Example:**
```python
# Save all metadata columns to the original path
dataset.save_metadata()

# Save only selected columns to a custom path
dataset.save_metadata(path="/output/metadata", columns=["id", "aesthetic_score", "nsfw_score"])
```
- Parquet format is efficient and compatible with many analytics tools.
- You can choose to save all or only specific columns.

## Exporting Filtered Datasets

To export a filtered version of your dataset (e.g., after removing low-quality or NSFW images), use the `to_webdataset` method. This writes new `.tar` and `.parquet` files containing only the filtered samples.

**Example:**
```python
# Filter your metadata (e.g., keep only high-quality images)
filtered_col = (dataset.metadata["aesthetic_score"] > 0.5) & (dataset.metadata["nsfw_score"] < 0.2)
dataset.metadata["keep"] = filtered_col

dataset.to_webdataset(
    path="/output/filtered_webdataset",  # Output directory
    filter_column="keep",                # Boolean column indicating which samples to keep
    samples_per_shard=10000,              # Number of samples per tar shard
    max_shards=5                         # Number of digits for shard IDs
)
```
- The output directory will contain new `.tar` files (with images, captions, and metadata) and matching `.parquet` files for each shard.
- Adjust `samples_per_shard` and `max_shards` to control sharding granularity and naming.

## Resharding WebDatasets

Resharding changes the number of samples per shard, which can optimize data loading or prepare data for specific workflows.

**Example:**
```python
# Reshard the dataset without filtering (keep all samples)
dataset.metadata["keep"] = True

dataset.to_webdataset(
    path="/output/resharded_webdataset",
    filter_column="keep",
    samples_per_shard=20000,  # New shard size
    max_shards=6
)
```
- Use resharding to balance I/O, parallelism, and storage efficiency.

## Preparing for Downstream Use
- Ensure your exported dataset matches the requirements of your training or analysis pipeline.
- Use consistent naming and metadata fields for compatibility.
- Document any filtering or processing steps for reproducibility.
- Test loading the exported dataset before large-scale training.

<!-- Detailed content to be added here. --> 