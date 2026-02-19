---
description: "End-to-end workflow tutorial covering the image curation process from embedding generation through semantic deduplication"
categories: ["image-curation"]
tags: ["workflow", "pipeline", "image-embeddings", "deduplication", "semantic", "clustering"]
personas: ["mle-focused", "data-scientist-focused"]
difficulty: "intermediate"
content_type: "tutorial"
modality: "image-only"

---

(image-tutorials-dedup)=

# Image Duplicate Removal Workflow

Learn how to run a complete image duplicate removal workflow that generates embeddings, identifies semantic duplicates, and removes similar images from your dataset.

```{contents} Tutorial Steps:
:local:
:depth: 2
```

## Before You Start

- Complete the [Get Started guide](gs-image).
- Understand basic pipeline concepts from the [Image Beginner Tutorial](image-tutorials-beginner).

---

## 1. Generate Image Embeddings

Create CLIP embeddings for all images in your dataset. This pipeline reads images, generates embeddings, and saves them to Parquet format for duplicate removal processing.

### Define the Embedding Pipeline

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.image.embedders.clip_embedder import ImageEmbeddingStage
from nemo_curator.stages.image.io.convert import ConvertImageBatchToDocumentBatchStage
from nemo_curator.stages.image.io.image_reader import ImageReaderStage
from nemo_curator.stages.text.io.writer.parquet import ParquetWriter

def create_image_embedding_pipeline(input_dir, embeddings_dir, model_dir):
    """Create pipeline to generate embeddings for duplicate removal."""
    
    pipeline = Pipeline(
        name="image_embedding", 
        description="Generate CLIP embeddings for image duplicate removal"
    )
    
    # Partition tar files for parallel processing
    pipeline.add_stage(FilePartitioningStage(
        file_paths=input_dir,
        files_per_partition=1,
        file_extensions=[".tar"],
    ))
    
    # Read images from tar archives
    pipeline.add_stage(ImageReaderStage(
        batch_size=100,
        verbose=True,
        num_threads=16,
        num_gpus_per_worker=0.25,
    ))
    
    # Generate CLIP embeddings
    pipeline.add_stage(ImageEmbeddingStage(
        model_dir=model_dir,
        num_gpus_per_worker=0.25,
        model_inference_batch_size=32,
        verbose=True,
    ))
    
    # Convert to document format for deduplication
    pipeline.add_stage(ConvertImageBatchToDocumentBatchStage(
        fields=["image_id", "embedding"]
    ))
    
    # Save embeddings to Parquet
    pipeline.add_stage(ParquetWriter(path=embeddings_dir))
    
    return pipeline
```

### Run Embedding Generation

```python
# Set your paths
INPUT_TAR_DIR = "/path/to/input/tar_dataset"
EMBEDDINGS_DIR = "/path/to/embeddings"
MODEL_DIR = "/path/to/models"

# Create and run pipeline
embedding_pipeline = create_image_embedding_pipeline(
    INPUT_TAR_DIR, EMBEDDINGS_DIR, MODEL_DIR
)
embedding_pipeline.run()  # Uses XennaExecutor by default
```

### Embedding Format Example

The pipeline writes embeddings to Parquet with two columns:

- **image_id**: String identifier for the image
- **embedding**: List of float values with length 768 (CLIP ViT-L/14 dimension)

::::{tab-set}

:::{tab-item} Directory layout

```text
/path/to/embeddings/
  part-00000-....parquet
  part-00001-....parquet
  part-00002-....parquet
```

:::

:::{tab-item} Schema

```text
image_id: string
embedding: list<float32>  # length = 768 for CLIP ViT-L/14
```

:::

:::{tab-item} Sample row

```
{"image_id": "00001234", "embedding": [0.0123, -0.0456, 0.0031, 0.1279, ...]}
```

:::

:::{tab-item} Read example

```python
import pyarrow.parquet as pq

table = pq.read_table("/path/to/embeddings")
df = table.to_pandas()
print(df.head())  # columns: image_id, embedding (list[float])
print(f"Embedding dimension: {len(df.iloc[0]['embedding'])}")
```

:::

::::

---

## 2. Run Semantic Duplicate Removal

Use the semantic duplicate removal workflow to identify and mark duplicate images based on embedding similarity.

```python
from nemo_curator.stages.deduplication.semantic import SemanticDeduplicationWorkflow

def create_deduplication_workflow(embeddings_dir, removal_dir):
    """Create semantic deduplication workflow."""
    
    return SemanticDeduplicationWorkflow(
        input_path=embeddings_dir,
        output_path=removal_dir,
        id_field="image_id",
        embedding_field="embedding",
        n_clusters=100,          # Number of clusters for grouping
        eps=0.01,               # Similarity threshold (lower = more strict)
        verbose=True,
    )

# Set paths
EMBEDDINGS_DIR = "/path/to/embeddings"
REMOVAL_DIR = "/path/to/removal_ids"

# Run deduplication
dedup_workflow = create_deduplication_workflow(EMBEDDINGS_DIR, REMOVAL_DIR)
dedup_workflow.run()
```

---

## 3. Remove Duplicate Images

After identifying duplicates, use `ImageDuplicatesRemovalStage` to filter them from your dataset.

Filter the original dataset to remove identified duplicates and create the final deduplicated dataset.

```python
from nemo_curator.stages.image.deduplication.removal import ImageDuplicatesRemovalStage
from nemo_curator.stages.image.io.image_writer import ImageWriterStage

def create_image_removal_pipeline(input_dir, removal_dir, output_dir):
    """Create pipeline to remove duplicate images."""
    
    pipeline = Pipeline(
        name="image_deduplication",
        description="Remove duplicate images from dataset"
    )
    
    # Partition input files
    pipeline.add_stage(FilePartitioningStage(
        file_paths=input_dir,
        files_per_partition=1,
        file_extensions=[".tar"],
    ))
    
    # Read original images
    pipeline.add_stage(ImageReaderStage(
        batch_size=100,
        verbose=True,
        num_threads=16,
        num_gpus_per_worker=0.25,
    ))
    
    # Remove duplicates based on removal list
    pipeline.add_stage(ImageDuplicatesRemovalStage(
        removal_parquets_dir=removal_dir + "/duplicates",
        duplicate_id_field="id",
        verbose=True,
    ))
    
    # Write deduplicated dataset
    pipeline.add_stage(ImageWriterStage(
        output_dir=output_dir,
        remove_image_data=True,
        verbose=True,
    ))
    
    return pipeline
```

### Run the Removal Pipeline

```python
# Set paths
INPUT_TAR_DIR = "/path/to/input/tar_dataset"
REMOVAL_DIR = "/path/to/removal_ids"
OUTPUT_DIR = "/path/to/deduplicated/dataset"

# Run removal pipeline
removal_pipeline = create_image_removal_pipeline(
    INPUT_TAR_DIR, REMOVAL_DIR, OUTPUT_DIR
)
removal_pipeline.run()  # Uses XennaExecutor by default
```

---

## 4. Inspect Results

After deduplication, examine the results to understand what was removed:

### Check Removal Statistics

```python
import pandas as pd
from glob import glob

# Read removal results
removal_files = glob(f"{REMOVAL_DIR}/duplicates/*.parquet")
removal_dfs = [pd.read_parquet(f) for f in removal_files]
all_removals = pd.concat(removal_dfs, ignore_index=True)

print(f"Total images marked for removal: {len(all_removals)}")
print(f"Unique images marked for removal: {all_removals['id'].nunique()}")

# Show sample of removed images
print("\nSample removed image IDs:")
print(all_removals['id'].head(10).tolist())
```

### Compare Dataset Sizes

```python
import os

def count_tar_files(directory):
    """Count tar files in a directory."""
    tar_files = glob(os.path.join(directory, "*.tar"))
    return len(tar_files)

original_count = count_tar_files(INPUT_TAR_DIR)
deduplicated_count = count_tar_files(OUTPUT_DIR)

print(f"Original dataset: {original_count} tar files")
print(f"Deduplicated dataset: {deduplicated_count} tar files")
print(f"Reduction: {original_count - deduplicated_count} files ({(1 - deduplicated_count/original_count)*100:.1f}%)")
```

---

## 5. Complete Workflow Script

Here's the complete workflow that combines all steps:

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.deduplication.semantic import SemanticDeduplicationWorkflow
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.image.deduplication.removal import ImageDuplicatesRemovalStage
from nemo_curator.stages.image.embedders.clip_embedder import ImageEmbeddingStage
from nemo_curator.stages.image.io.convert import ConvertImageBatchToDocumentBatchStage
from nemo_curator.stages.image.io.image_reader import ImageReaderStage
from nemo_curator.stages.image.io.image_writer import ImageWriterStage
from nemo_curator.stages.text.io.writer.parquet import ParquetWriter

def run_image_deduplication_workflow():
    """Run complete image deduplication workflow."""
    
    # Define paths
    INPUT_TAR_DIR = "/path/to/input/tar_dataset"
    EMBEDDINGS_DIR = "/path/to/embeddings"
    REMOVAL_DIR = "/path/to/removal_ids"
    OUTPUT_DIR = "/path/to/deduplicated/dataset"
    MODEL_DIR = "/path/to/models"
    
    print("Step 1: Generating embeddings...")
    
    # Step 1: Generate embeddings
    embedding_pipeline = Pipeline(name="embedding", description="Generate embeddings")
    
    embedding_pipeline.add_stage(FilePartitioningStage(
        file_paths=INPUT_TAR_DIR, files_per_partition=1, file_extensions=[".tar"]
    ))
    embedding_pipeline.add_stage(ImageReaderStage(
        batch_size=100, verbose=True, num_threads=16, num_gpus_per_worker=0.25
    ))
    embedding_pipeline.add_stage(ImageEmbeddingStage(
        model_dir=MODEL_DIR, num_gpus_per_worker=0.25, 
        model_inference_batch_size=32, verbose=True
    ))
    embedding_pipeline.add_stage(ConvertImageBatchToDocumentBatchStage(
        fields=["image_id", "embedding"]
    ))
    embedding_pipeline.add_stage(ParquetWriter(path=EMBEDDINGS_DIR))
    
    embedding_pipeline.run()  # Uses XennaExecutor by default
    
    print("Step 2: Running semantic deduplication...")
    
    # Step 2: Semantic deduplication
    dedup_workflow = SemanticDeduplicationWorkflow(
        input_path=EMBEDDINGS_DIR,
        output_path=REMOVAL_DIR,
        id_field="image_id",
        embedding_field="embedding",
        n_clusters=100,
        eps=0.01,
        verbose=True,
    )
    dedup_workflow.run()
    
    print("Step 3: Removing duplicate images...")
    
    # Step 3: Remove duplicates
    removal_pipeline = Pipeline(name="removal", description="Remove duplicates")
    
    removal_pipeline.add_stage(FilePartitioningStage(
        file_paths=INPUT_TAR_DIR, files_per_partition=1, file_extensions=[".tar"]
    ))
    removal_pipeline.add_stage(ImageReaderStage(
        batch_size=100, verbose=True, num_threads=16, num_gpus_per_worker=0.25
    ))
    removal_pipeline.add_stage(ImageDuplicatesRemovalStage(
        removal_parquets_dir=REMOVAL_DIR + "/duplicates",
        duplicate_id_field="id",
        verbose=True,
    ))
    removal_pipeline.add_stage(ImageWriterStage(
        output_dir=OUTPUT_DIR, remove_image_data=True, verbose=True
    ))
    
    removal_pipeline.run()  # Uses XennaExecutor by default
    
    print(f"Deduplication complete! Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_image_deduplication_workflow()
```

## Next Steps

After running image deduplication:

1. **Quality assessment**: Manually review a sample of removed duplicates
2. **Combine with filtering**: Run aesthetic/NSFW filtering on deduplicated data
3. **Export for training**: Prepare final curated dataset for ML training
4. **Monitor metrics**: Track deduplication rates across different image types
