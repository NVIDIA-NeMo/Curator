---
description: "Core concepts for saving and exporting curated image datasets including metadata and resharding"
categories: ["concepts-architecture"]
tags: ["data-export", "tar-files", "parquet", "resharding", "metadata"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "concept"
modality: "image-only"
---

(about-concepts-image-data-export)=

# Data Export Concepts (Image)

This page covers the core concepts for saving and exporting curated image datasets in NeMo Curator.

## Key Topics

- Saving curated images and metadata
- Understanding output format structure
- Configuring output sharding
- Preparing data for downstream training or analysis

## Saving Results

After processing through the pipeline, you can save the curated images and metadata using the `ImageWriterStage`.

**Example:**

```python
from nemo_curator.stages.image.io.image_writer import ImageWriterStage

# Add writer stage to pipeline
pipeline.add_stage(ImageWriterStage(
    output_dir="/output/curated_dataset",
    images_per_tar=1000,  # Images per tar file
    remove_image_data=True,
    verbose=True,
    deterministic_name=True,  # Use deterministic naming for reproducible output
))
```

**Key Parameters:**

- `output_dir`: Directory where tar archives and metadata files are written
- `images_per_tar`: Number of images per tar file for optimal sharding
- `remove_image_data`: Whether to remove image data from memory after writing
- `deterministic_name`: Ensures reproducible file naming based on input content

**Behavior:**

- The writer stage creates tar files with curated images
- Metadata for each image (including paths, IDs, scores, and processing metadata) is always stored in separate Parquet files alongside tar archives
- Adjust `images_per_tar` to balance I/O, parallelism, and storage efficiency
- Smaller values create more files but enable better parallelism
- Larger values reduce file count but may impact loading performance

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

## Preparing for Downstream Use

- Ensure your exported dataset matches the requirements of your training or analysis pipeline.
- Use consistent naming and metadata fields for compatibility.
- Document any filtering or processing steps for reproducibility.
- Test loading the exported dataset before large-scale training.

<!-- Detailed content to be added here. --> 