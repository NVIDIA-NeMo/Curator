---
description: "Save metadata, export filtered datasets, and reshard WebDatasets for downstream use after image curation"
categories: ["how-to-guides"]
tags: ["data-export", "parquet", "webdataset", "filtering", "resharding", "metadata"]
personas: ["data-scientist-focused", "mle-focused"]
difficulty: "intermediate"
content_type: "how-to"
modality: "image-only"
---

(image-save-export)=
# Saving and Exporting Image Datasets

After processing and filtering your image datasets, you can save metadata, export results, and reshard WebDatasets for downstream use. NeMo Curator provides flexible options for saving and exporting your curated image data.

## Saving Metadata to Parquet

You can save the metadata (such as classifier scores, embeddings, or other columns) of your `ImageTextPairDataset` to Parquet files for easy analysis or further processing.

```python
# Assume 'dataset' is an ImageTextPairDataset
# Save all metadata columns to the original path
# (or specify a different path if desired)
dataset.save_metadata()

# Save only selected columns to a custom path
dataset.save_metadata(path="/output/metadata", columns=["id", "aesthetic_score", "nsfw_score"])
```

## Exporting Filtered Datasets

To export a filtered version of your dataset (for example, after removing low-quality or NSFW images), use the `to_webdataset` method. This will write new WebDataset shards and Parquet files containing only the filtered samples.

```python
# Filter your metadata (e.g., keep only high-quality images)
filtered_col = (dataset.metadata["aesthetic_score"] > 0.5) & (dataset.metadata["nsfw_score"] < 0.2)
dataset.metadata["keep"] = filtered_col

# Export the filtered dataset to a new directory as WebDataset shards
dataset.to_webdataset(
    path="/output/filtered_webdataset",  # Output directory
    filter_column="keep",                # Boolean column indicating which samples to keep
    samples_per_shard=10000,              # Number of samples per tar shard
    max_shards=5                         # Number of digits for shard IDs
)
```

- The output directory will contain new `.tar` files (with images, captions, and metadata) and matching `.parquet` files for each shard.
- You can adjust `samples_per_shard` and `max_shards` to control sharding granularity and naming.

## Resharding WebDatasets

If you want to change the sharding of your dataset (for example, to create larger or smaller shards), you can use the same `to_webdataset` method without filtering:

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

This is useful for optimizing data loading performance or preparing data for specific downstream workflows.

---

For more details on the available methods and options, see the `ImageTextPairDataset` class in the NeMo Curator codebase.

<!-- More details and examples will be added here. --> 