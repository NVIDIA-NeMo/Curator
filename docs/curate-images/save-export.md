---
description: "Save metadata, export filtered datasets, and configure output sharding for downstream use after image curation"
categories: ["how-to-guides"]
tags: ["data-export", "parquet", "tar-files", "filtering", "sharding", "metadata"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "image-only"
---

(image-save-export)=

# Saving and Exporting Image Datasets

After processing and filtering your image datasets using NeMo Curator's pipeline stages, you can save results and export curated data for downstream use. The pipeline-based approach provides flexible options for saving and exporting your curated image data.

## Saving Results with ImageWriterStage

The `ImageWriterStage` is the primary method for saving curated images and metadata to tar archives with accompanying Parquet files. This stage is typically the final step in your image curation pipeline.

```python
from nemo_curator.stages.image.io.image_writer import ImageWriterStage

# Add ImageWriterStage to your pipeline
pipeline.add_stage(ImageWriterStage(
    output_dir="/output/curated_images",    # Output directory for tar files and metadata
    images_per_tar=1000,                    # Number of images per tar file
    remove_image_data=True,                 # Remove image data from memory after writing
    verbose=True,                           # Enable progress logging
))
```

### Parameters

```{list-table}
:header-rows: 1
:widths: 20 15 15 50

* - Parameter
  - Type
  - Default
  - Description
* - `output_dir`
  - str
  - Required
  - Output directory for tar files and metadata
* - `images_per_tar`
  - int
  - 1000
  - Number of images per tar file (controls sharding)
* - `verbose`
  - bool
  - False
  - Enable verbose logging for debugging
* - `deterministic_name`
  - bool
  - True
  - Use deterministic hash-based naming for output files
* - `remove_image_data`
  - bool
  - False
  - Remove image data from memory after writing (saves memory)
```

## Pipeline-Based Filtering and Export

Filtering is built into the classification stages (`ImageAestheticFilterStage`, `ImageNSFWFilterStage`). Images that don't meet the criteria are automatically filtered out before reaching the `ImageWriterStage`.

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.image.io.image_reader import ImageReaderStage
from nemo_curator.stages.image.embedders.clip_embedder import ImageEmbeddingStage
from nemo_curator.stages.image.filters.aesthetic_filter import ImageAestheticFilterStage
from nemo_curator.stages.image.filters.nsfw_filter import ImageNSFWFilterStage
from nemo_curator.stages.image.io.image_writer import ImageWriterStage

# Create pipeline with filtering and export
pipeline = Pipeline(name="filter_and_export")

# Load and process images
pipeline.add_stage(FilePartitioningStage(
    file_paths="/input/tar_archives",
    files_per_partition=1,
    file_extensions=[".tar"],
))

pipeline.add_stage(ImageReaderStage(
    task_batch_size=100,
    num_threads=16,
    num_gpus_per_worker=0.25,
))

pipeline.add_stage(ImageEmbeddingStage(
    model_dir="/models",
    num_gpus_per_worker=0.25,
))

# Filter by aesthetic quality (score >= 0.5)
pipeline.add_stage(ImageAestheticFilterStage(
    model_dir="/models",
    score_threshold=0.5,
    num_gpus_per_worker=0.25,
))

# Filter NSFW content (score < 0.5)
pipeline.add_stage(ImageNSFWFilterStage(
    model_dir="/models",
    score_threshold=0.5,
    num_gpus_per_worker=0.25,
))

# Save filtered results
pipeline.add_stage(ImageWriterStage(
    output_dir="/output/filtered_dataset",
    images_per_tar=1000,
    remove_image_data=True,
))
```

## Configuring Output Sharding

To change the sharding of your dataset (create larger or smaller shards), adjust the `images_per_tar` parameter in `ImageWriterStage`:

```python
# Create larger shards (20,000 images per tar file)
pipeline.add_stage(ImageWriterStage(
    output_dir="/output/resharded_dataset",
    images_per_tar=20000,  # Larger shard size
    remove_image_data=True,
))
```

## Output Format

The `ImageWriterStage` creates:

* **Tar Archives**: `.tar` files containing JPEG images
* **Parquet Files**: `.parquet` files with metadata for each corresponding tar file
* **Deterministic Naming**: Files named with content-based hashes for reproducibility
* **Preserved Metadata**: All scores and metadata from processing stages stored in Parquet files

**Output Structure:**
```bash
output/
├── images-{hash}-000000.tar     # Contains JPEG images
├── images-{hash}-000000.parquet # Metadata for corresponding tar
├── images-{hash}-000001.tar
├── images-{hash}-000001.parquet
```

Each tar file contains JPEG images with sequential or ID-based filenames, while metadata (including `aesthetic_score`, `nsfw_score`, and other processing data) is stored in the accompanying Parquet files.

---

For more details on stage parameters and customization options, see the [ImageWriterStage documentation](process-data/index.md) and the [Complete Tutorial](https://github.com/NVIDIA-NeMo/Curator/blob/main/tutorials/image/getting-started/image_curation_example.py).

<!-- More details and examples will be added here. -->