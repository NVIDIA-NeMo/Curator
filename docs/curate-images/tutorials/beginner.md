---
description: "Beginner-friendly tutorial for creating your first image curation pipeline with quality filtering and NSFW detection"
categories: ["image-curation"]
tags: ["beginner", "tutorial", "quickstart", "pipeline", "image-processing", "quality-filtering", "nsfw-detection"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "beginner"
content_type: "tutorial"
modality: "image-only"

---

(image-tutorials-beginner)=
# Create an Image Curation Pipeline

Learn the basics of creating an image curation pipeline in Curator by following a complete workflow that filters images by aesthetic quality and NSFW content.

```{contents} Tutorial Steps:
:local:
:depth: 2
```

## Before You Start

- Follow the [Get Started guide](gs-image) to install the package, prepare the model directory, and set up your data paths.

### Concepts and Mental Model

Use this overview to understand how stages pass data through the pipeline.

```{mermaid}
flowchart LR
  I[Images] --> P[Partition files]
  P --> R[ImageReader]
  R --> E[CLIP Embeddings]
  E --> A[Aesthetic Filter]
  A --> N[NSFW Filter]
  N --> W[Write filtered images]
  classDef dim fill:#f6f8fa,stroke:#d0d7de,color:#24292f;
  class P,R,E,A,N,W dim;
```

- **Pipeline**: An ordered list of stages that process data.
- **Stage**: A modular operation (for example, read, embed, filter, write).
- **Executor**: Runs the pipeline (XennaExecutor backend).
- **Data units**: Input images → embeddings → quality scores → filtered output.
- **Common choices**:
  - **Embeddings**: CLIP ViT-L/14 for semantic understanding
  - **Quality filters**: Aesthetic predictor and NSFW classifier
  - **Thresholds**: Configurable scoring thresholds for filtering
- **Outputs**: Filtered WebDataset with embeddings, quality scores, and image data.

For more information, refer to the [Image Concepts](about-concepts-image) section.

---

## 1. Define Imports and Paths

Import required classes and define paths used throughout the example.

```python
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.image.embedders.clip_embedder import ImageEmbeddingStage
from nemo_curator.stages.image.filters.aesthetic_filter import ImageAestheticFilterStage
from nemo_curator.stages.image.filters.nsfw_filter import ImageNSFWFilterStage
from nemo_curator.stages.image.io.image_reader import ImageReaderStage
from nemo_curator.stages.image.io.image_writer import ImageWriterStage

INPUT_WDS_DIR = "/path/to/input/webdataset"
OUTPUT_DIR = "/path/to/output/dataset"
MODEL_DIR = "/path/to/models"
```

## 2. Create the Pipeline

Instantiate a named pipeline to orchestrate the stages.

```python
pipeline = Pipeline(
    name="image_curation", 
    description="Curate images with embeddings and quality scoring"
)
```

## 3. Define Stages

Add modular stages to partition, read, embed, filter, and write images.

### Partition Input Files

Divide tar files across workers for parallel processing.

```python
pipeline.add_stage(FilePartitioningStage(
    file_paths=INPUT_WDS_DIR,
    files_per_partition=1,  # Adjust based on file sizes
    file_extensions=[".tar"],
))
```

### Read Images

Load images from WebDataset tar files and extract metadata.

```python
pipeline.add_stage(ImageReaderStage(
    task_batch_size=100,  # Images per batch
    verbose=True,
    num_threads=16,       # I/O threads
    num_gpus_per_worker=0.25,
))
```

### Generate CLIP Embeddings

Create semantic embeddings for each image using CLIP ViT-L/14.

```python
pipeline.add_stage(ImageEmbeddingStage(
    model_dir=MODEL_DIR,
    num_gpus_per_worker=0.25,
    model_inference_batch_size=32,  # Adjust for GPU memory
    remove_image_data=False,        # Keep images for filtering
    verbose=True,
))
```

### Filter by Aesthetic Quality

Score and filter images based on aesthetic quality using a trained predictor.

```python
pipeline.add_stage(ImageAestheticFilterStage(
    model_dir=MODEL_DIR,
    num_gpus_per_worker=0.25,
    model_inference_batch_size=32,
    score_threshold=0.5,  # Keep images with score >= 0.5
    verbose=True,
))
```

### Filter NSFW Content

Detect and filter out NSFW (Not Safe For Work) content.

```python
pipeline.add_stage(ImageNSFWFilterStage(
    model_dir=MODEL_DIR,
    num_gpus_per_worker=0.25,
    model_inference_batch_size=32,
    score_threshold=0.5,  # Filter images with NSFW score >= 0.5
    verbose=True,
))
```

### Write Filtered Dataset

Save the curated images and metadata to output directory.

```python
pipeline.add_stage(ImageWriterStage(
    output_dir=OUTPUT_DIR,
    images_per_tar=1000,   # Images per output tar file (default: 1000)
    remove_image_data=True, # Remove raw image data to save space
    verbose=True,
))
```

## 4. Run the Pipeline

Execute the configured pipeline. The pipeline will use XennaExecutor by default if no executor is specified.

```python
# Option 1: Use default executor (XennaExecutor)
pipeline.run()

# Option 2: Explicitly specify executor
executor = XennaExecutor()
pipeline.run(executor)
```

## Complete Example

Here's the full pipeline code:

```python
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.image.embedders.clip_embedder import ImageEmbeddingStage
from nemo_curator.stages.image.filters.aesthetic_filter import ImageAestheticFilterStage
from nemo_curator.stages.image.filters.nsfw_filter import ImageNSFWFilterStage
from nemo_curator.stages.image.io.image_reader import ImageReaderStage
from nemo_curator.stages.image.io.image_writer import ImageWriterStage

def create_image_curation_pipeline():
    """Create image curation pipeline with quality filtering."""
    
    # Define paths
    INPUT_WDS_DIR = "/path/to/input/webdataset"
    OUTPUT_DIR = "/path/to/output/dataset" 
    MODEL_DIR = "/path/to/models"
    
    # Create pipeline
    pipeline = Pipeline(
        name="image_curation",
        description="Curate images with embeddings and quality scoring"
    )
    
    # Add stages
    pipeline.add_stage(FilePartitioningStage(
        file_paths=INPUT_WDS_DIR,
        files_per_partition=1,
        file_extensions=[".tar"],
    ))
    
    pipeline.add_stage(ImageReaderStage(
        task_batch_size=100,
        verbose=True,
        num_threads=16,
        num_gpus_per_worker=0.25,
    ))
    
    pipeline.add_stage(ImageEmbeddingStage(
        model_dir=MODEL_DIR,
        num_gpus_per_worker=0.25,
        model_inference_batch_size=32,
        remove_image_data=False,
        verbose=True,
    ))
    
    pipeline.add_stage(ImageAestheticFilterStage(
        model_dir=MODEL_DIR,
        num_gpus_per_worker=0.25,
        model_inference_batch_size=32,
        score_threshold=0.5,
        verbose=True,
    ))
    
    pipeline.add_stage(ImageNSFWFilterStage(
        model_dir=MODEL_DIR,
        num_gpus_per_worker=0.25,
        model_inference_batch_size=32,
        score_threshold=0.5,
        verbose=True,
    ))
    
    pipeline.add_stage(ImageWriterStage(
        output_dir=OUTPUT_DIR,
        images_per_tar=1000,  # Default value
        remove_image_data=True,
        verbose=True,
    ))
    
    return pipeline

# Run the pipeline
if __name__ == "__main__":
    pipeline = create_image_curation_pipeline()
    pipeline.run()  # Uses XennaExecutor by default
```

## Performance Tuning

### Batch Size Guidelines

The batch sizes used in this workflow are conservative limits set for typical GPUs with 24-48 GB of VRAM (such as RTX 4090, A6000, RTX A5000). You can tune these based on your available GPU memory:

- **High-memory GPUs (80+ GB)** like H100, B200, or A100 80GB: Increase batch sizes for better performance:
  ```python
  model_inference_batch_size=500
  ```

- **Lower-memory GPUs (16 GB or less)**: Reduce batch sizes:
  ```python
  model_inference_batch_size=16
  ```

### GPU Allocation

Adjust `num_gpus_per_worker` based on your cluster setup:
- **Single GPU**: Set to `1.0` for each stage
- **Multi-GPU systems**: Use fractional values like `0.25` to run several workers per GPU
- **CPU-only**: Set to `0` (embedding and filtering stages require GPUs for optimal performance)

## Next Steps

After running this basic curation pipeline, you can:

1. **Adjust thresholds**: Experiment with different aesthetic and NSFW score thresholds
2. **Add custom filters**: Create domain-specific quality filters
3. **Run duplicate removal**: Remove similar or duplicate images using [semantic duplicate removal](image-tutorials-dedup)
4. **Export for training**: Prepare curated data for downstream ML training tasks

For more advanced workflows, refer to the [Image Duplicate Removal Tutorial](image-tutorials-dedup).
