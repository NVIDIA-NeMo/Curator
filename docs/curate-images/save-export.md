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