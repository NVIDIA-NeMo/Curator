---
description: "Load and process image-text pair datasets in WebDataset format with sharded storage and distributed processing"
categories: ["how-to-guides"]
tags: ["webdataset", "data-loading", "sharding", "distributed", "cloud-storage", "dali"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "image-only"
---

(image-load-data-webdataset)=

# WebDataset

Load and process image-text pair datasets in WebDataset format using NeMo Curator's DALI-powered `ImageReaderStage`.

WebDataset is a sharded, metadata-rich file format that enables scalable, distributed image curation. The `ImageReaderStage` uses NVIDIA DALI for high-performance image decoding with automatic GPU/CPU fallback.

## How it Works

A WebDataset directory contains sharded `.tar` files, each holding image-text pairs and metadata, along with corresponding `.parquet` files for tabular metadata. Optionally, `.idx` index files can be provided for fast DALI-based loading. Each record is identified by a unique ID, which is used as the prefix for all files belonging to that record.

**Directory Structure Example**

```text
dataset/
├── 00000.tar
│   ├── 000000000.jpg
│   ├── 000000000.txt
│   ├── 000000000.json
│   ├── ...
├── 00001.tar
│   ├── ...
├── 00000.parquet
├── 00001.parquet
├── 00000.idx  # optional
├── 00001.idx  # optional
```

- `.tar` files: Contain images (`.jpg`), captions (`.txt`), and metadata (`.json`)
- `.parquet` files: Tabular metadata for each record
- `.idx` files: (Optional) Index files for fast DALI-based loading

Each record is identified by a unique ID (for example, `000000031`), which is used as the prefix for all files belonging to that record.

---

## Usage

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.image.io.image_reader import ImageReaderStage

# Create pipeline
pipeline = Pipeline(name="image_loading", description="Load images from WebDataset")

# Stage 1: Partition tar files for parallel processing
pipeline.add_stage(FilePartitioningStage(
    file_paths="/path/to/webdataset",
    files_per_partition=1,
    file_extensions=[".tar"],
))

# Stage 2: Read images from webdataset tar files
pipeline.add_stage(ImageReaderStage(
    task_batch_size=100,
    verbose=True,
    num_threads=16,
    num_gpus_per_worker=0.25,
))

# Run the pipeline (uses XennaExecutor by default)
results = pipeline.run()


```

- `file_paths`: Path to the root of the WebDataset directory (local or cloud storage)
- `files_per_partition`: Number of tar files to process per partition (controls parallelism)
- `task_batch_size`: Number of images per ImageBatch for processing

---

## ImageReaderStage Details

The `ImageReaderStage` is the core component that handles WebDataset loading with the following capabilities:

### DALI Integration

- **Automatic Device Selection**: Uses GPU decoding when CUDA is available, CPU decoding otherwise
- **WebDataset Reader**: Leverages DALI's native WebDataset reader for optimal performance
- **Batch Processing**: Processes images in configurable batch sizes for memory efficiency

### Image Processing

- **Format Support**: Reads JPEG images (`.jpg`) from tar files
- **Size Preservation**: Maintains original image dimensions (no automatic resizing)
- **RGB Output**: Converts images to RGB format for consistent downstream processing

### Error Handling

- **Missing Components**: Skips missing or corrupted images with `missing_component_behavior="skip"`
- **Graceful Fallback**: Automatically falls back to CPU processing if GPU is unavailable
- **Validation**: Validates tar file paths and provides clear error messages

---

## Parameters

```{list-table} WebDataset Loading Parameters
:header-rows: 1
:widths: 20 20 40 20

* - Parameter
  - Type
  - Description
  - Default
* - `file_paths`
  - str
  - Path to the WebDataset directory (local or cloud storage)
  - Required
* - `files_per_partition`
  - int
  - Number of tar files to process per partition (controls parallelism)
  - 1
* - `task_batch_size`
  - int
  - Number of images per ImageBatch for processing
  - 100
* - `num_threads`
  - int
  - Number of threads for I/O operations
  - 8
* - `num_gpus_per_worker`
  - float
  - GPU allocation per worker (0.25 = 1/4 GPU)
  - 0.25
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

The WebDataset directory structure remains:
- Sharded `.tar` files with images, captions, and metadata
- `.parquet` files with tabular metadata
- (Optional) `.idx` files for DALI-based loading

---

## Key Features

### DALI-Powered Performance

- **GPU Acceleration**: Automatically uses DALI GPU decoding when CUDA is available, falls back to CPU decoding otherwise
- **High Throughput**: Optimized for processing large-scale image datasets with minimal I/O overhead
- **Memory Efficient**: Streams images in batches without loading entire datasets into memory

### Flexible Processing

- **Automatic Fallback**: Works in both GPU and CPU environments
- **Batch Processing**: Configurable batch sizes for optimal memory usage and throughput
- **Parallel Loading**: Supports multiple tar files processed in parallel via `FilePartitioningStage`

### Format Support

- **WebDataset Native**: Built specifically for WebDataset `.tar` format with DALI webdataset reader
- **Image Formats**: Supports JPEG images (`.jpg` extension) within tar files
- **Metadata Preservation**: Extracts image IDs and paths for downstream processing

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

### Performance Benchmarks

The DALI-based `ImageReaderStage` provides significant performance improvements over traditional image loading:

- **GPU Decoding**: Up to 3-5x faster than CPU-only image decoding for large batches
- **Memory Efficiency**: Streams images without loading entire datasets into memory
- **I/O Optimization**: DALI's WebDataset reader minimizes file system overhead

**Recommended Batch Sizes by Hardware:**

```{list-table} Optimal Batch Size Recommendations
:header-rows: 1
:widths: 30 25 25 20

* - Hardware Configuration
  - Batch Size
  - GPU Memory
  - Expected Throughput
* - **Single GPU (16GB+)**
  - 256-512
  - 0.5-1.0
  - ~2000-4000 images/sec
* - **Multi-GPU Setup**
  - 128-256
  - 0.25-0.5
  - ~1000-2000 images/sec per GPU
* - **CPU Only (32GB+ RAM)**
  - 32-64
  - 0
  - ~200-500 images/sec
* - **CPU Only (16GB RAM)**
  - 16-32
  - 0
  - ~100-300 images/sec
```

### Advanced Performance Tuning

**Index Files for Large Datasets**

```bash
# Generate DALI index files for faster loading
python -c "
from nvidia.dali.tools import wds2idx
wds2idx('/path/to/dataset/00000.tar', '/path/to/dataset/00000.idx')
"
```

**Parallelism Configuration**

```python
# For datasets with many small tar files
pipeline.add_stage(FilePartitioningStage(
    files_per_partition=4,      # Process multiple tars together
    file_extensions=[".tar"],
))

# For datasets with large tar files
pipeline.add_stage(FilePartitioningStage(
    files_per_partition=1,      # Process one tar per partition
    file_extensions=[".tar"],
))
```

**Ray Executor Selection for Image Workloads**

```python
from nemo_curator.backends.experimental.ray_actor_pool import RayActorPoolExecutor

# Use Ray Actor Pool for better resource management
pipeline = Pipeline(
    name="image_loading",
    executor=RayActorPoolExecutor(config={
        "num_actors_per_stage": 4,
        "cleanup_actors": True,     # Automatic resource cleanup
    })
)
```

## Customization Options & Performance Tips

- **Cloud Storage Support**: You can use local paths or cloud storage URLs (for example, S3, GCS, Azure) in `file_paths`. Make sure your environment is configured with the appropriate credentials.
- **DALI Index Files**: For large datasets, provide `.idx` files for each `.tar` to enable fast DALI-based loading (see [NVIDIA DALI documentation](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/data_loading/dataloading_webdataset.html#Creating-an-index)).
- **GPU Acceleration**: Use a GPU-enabled environment for best performance. The stage automatically detects CUDA availability and uses GPU decoding when possible.
- **Parallelism Control**: Adjust `files_per_partition` to control how many tar files are processed together. Lower values increase parallelism but may increase overhead.
- **Batch Size Tuning**: Increase `task_batch_size` for better throughput, but ensure sufficient memory is available.
- **Thread Configuration**: Adjust `num_threads` for I/O operations based on your storage system's characteristics.

---

<!-- More advanced usage and troubleshooting tips can be added here. -->