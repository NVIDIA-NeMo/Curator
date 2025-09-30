---
description: "Load and process JPEG images from tar archives using DALI-powered GPU acceleration with distributed processing"
categories: ["how-to-guides"]
tags: ["tar-archives", "data-loading", "dali", "gpu-acceleration", "distributed"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "image-only"
---

(image-load-data-tar-archives)=

# Loading Images from Tar Archives

Load and process JPEG images from tar archives using NeMo Curator's DALI-powered `ImageReaderStage`.

The `ImageReaderStage` uses NVIDIA DALI for high-performance image decoding with GPU acceleration and automatic CPU fallback, designed for processing large collections of images stored in tar files.

## How it Works

The `ImageReaderStage` processes directories containing `.tar` files with JPEG images. While tar files may contain other file types (text, JSON, etc.), the stage extracts only JPEG images for processing.

**Directory Structure Example**

```text
dataset/
├── 00000.tar
│   ├── 000000000.jpg
│   ├── 000000001.jpg
│   ├── 000000002.jpg
│   ├── ...
├── 00001.tar
│   ├── 000001000.jpg
│   ├── 000001001.jpg
│   ├── ...
```

**What gets processed:**
- **JPEG images**: All `.jpg` files within tar archives

**What gets ignored:**
- Text files (`.txt`), JSON files (`.json`), and other non-JPEG content within tar archives
- Any files outside the tar archives (like standalone Parquet files)

---

## Usage

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.image.io.image_reader import ImageReaderStage

# Create pipeline
pipeline = Pipeline(name="image_loading", description="Load images from tar archives")

# Stage 1: Partition tar files for parallel processing
pipeline.add_stage(FilePartitioningStage(
    file_paths="/path/to/tar_dataset",
    files_per_partition=1,
    file_extensions=[".tar"],
))

# Stage 2: Read JPEG images from tar files using DALI
pipeline.add_stage(ImageReaderStage(
    task_batch_size=100,
    verbose=True,
    num_threads=16,
    num_gpus_per_worker=0.25,
))

# Run the pipeline (uses XennaExecutor by default)
results = pipeline.run()
```

**Parameters:**

- `file_paths`: Path to directory containing tar files
- `files_per_partition`: Number of tar files to process per partition (controls parallelism)
- `task_batch_size`: Number of images per ImageBatch for processing

---

## ImageReaderStage Details

The `ImageReaderStage` is the core component that handles tar archive loading with the following capabilities:

### DALI Integration

- **Automatic Device Selection**: Uses GPU decoding when CUDA is available, CPU decoding otherwise
- **Tar Archive Reader**: Leverages DALI's tar archive reader to process tar files
- **Batch Processing**: Processes images in configurable batch sizes for memory efficiency
- **JPEG-Only Processing**: Extracts only JPEG files (`ext=["jpg"]`) from tar archives

### Image Processing

- **Format Support**: Reads only JPEG images (`.jpg`) from tar files
- **Size Preservation**: Maintains original image dimensions (no automatic resizing)
- **RGB Output**: Converts images to RGB format for consistent downstream processing
- **Metadata Extraction**: Creates ImageObject instances with image paths and generated IDs

### Error Handling

- **Missing Components**: Skips missing or corrupted images with `missing_component_behavior="skip"`
- **Graceful Fallback**: Automatically falls back to CPU processing if GPU is unavailable
- **Validation**: Validates tar file paths and provides clear error messages
- **Non-JPEG Filtering**: Silently ignores non-JPEG files within tar archives

---

## Parameters

### FilePartitioningStage Parameters

```{list-table}
:header-rows: 1
:widths: 20 15 15 50

* - Parameter
  - Type
  - Default
  - Description
* - `file_paths`
  - str | list[str]
  - Required
  - Path to directory containing tar files, or list of file paths
* - `files_per_partition`
  - int | None
  - None
  - Number of tar files to process per partition (controls parallelism). Defaults to 1 if both `files_per_partition` and `blocksize` are not provided
* - `file_extensions`
  - list[str] | None
  - `[".jsonl", ".json", ".parquet"]`
  - List of file extensions to include (for example, `[".tar"]`)
* - `blocksize`
  - int | str | None
  - None
  - Target size of the partitions. If provided, `files_per_partition` is ignored
* - `limit`
  - int | None
  - None
  - Maximum number of partitions to create
```

### ImageReaderStage Parameters

```{list-table}
:header-rows: 1
:widths: 20 15 15 50

* - Parameter
  - Type
  - Default
  - Description
* - `task_batch_size`
  - int
  - 100
  - Number of images per ImageBatch for processing
* - `verbose`
  - bool
  - True
  - Enable verbose logging for debugging
* - `num_threads`
  - int
  - 8
  - Number of threads for DALI operations
* - `num_gpus_per_worker`
  - float
  - 0.25
  - GPU allocation per worker (0.25 = 1/4 GPU)
```

---

## Output Format

The pipeline produces `ImageBatch` objects containing `ImageObject` instances for downstream curation tasks. Each `ImageObject` contains:

- `image_data`: Raw image pixel data as numpy array (H, W, C) in RGB format
- `image_path`: Path to the original image file in the tar
- `image_id`: Unique identifier extracted from the filename
- `metadata`: Additional metadata dictionary

**Example ImageObject structure:**

```python
ImageObject(
    image_path="00000.tar/000000031.jpg",
    image_id="000000031", 
    image_data=np.array(...),  # Shape: (H, W, 3)
    metadata={}
)
```

**Note**: Only JPEG images are extracted from tar files. Other content (text files, JSON metadata, etc.) within the tar archives is ignored during processing.

---

## Performance Optimization

### Hardware-Specific Configuration

**GPU-Enabled Environments (Recommended)**

```python
# Optimal configuration for GPU acceleration
pipeline.add_stage(ImageReaderStage(
    task_batch_size=256,        # Larger batches for GPU throughput
    num_threads=16,             # More threads for I/O parallelism
    num_gpus_per_worker=0.5,    # Allocate more GPU memory
    verbose=True,
))
```

**CPU Environments**

```python
# Optimized for CPU decoding
pipeline.add_stage(ImageReaderStage(
    task_batch_size=64,         # Smaller batches to avoid memory pressure
    num_threads=8,              # Fewer threads for CPU processing
    num_gpus_per_worker=0,      # No GPU allocation
    verbose=True,
))
```

## Customization Options & Performance Tips

- **GPU Acceleration**: Use a GPU-enabled environment for optimal performance. The stage automatically detects CUDA availability and uses GPU decoding when possible.
- **Parallelism Control**: Adjust `files_per_partition` to control how many tar files are processed together. Lower values increase parallelism but may increase overhead.
- **Batch Size Tuning**: Increase `task_batch_size` for better throughput, but ensure sufficient memory is available.
- **Thread Configuration**: Adjust `num_threads` for I/O operations based on your storage system's characteristics.

---

<!-- More advanced usage and troubleshooting tips can be added here. -->
