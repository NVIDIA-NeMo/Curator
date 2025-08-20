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

Load and process image-text pair datasets in WebDataset format using NeMo Curator.

WebDataset is a sharded, metadata-rich file format that enables scalable, distributed image curation. It is the primary and currently only supported format for image data loading in NeMo Curator.

## How it Works

A WebDataset directory contains sharded `.tar` files, each holding image-text pairs and metadata, along with corresponding `.parquet` files for tabular metadata. Optionally, `.idx` index files can be provided for fast DALI-based loading. Each record is identified by a unique ID, which is used as the prefix for all files belonging to that record.

**Directory Structure Example**

```
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
from nemo_curator.datasets import ImageTextPairDataset

dataset = ImageTextPairDataset.from_webdataset(
    path="/path/to/webdataset",
    id_col="key"  # or the name of your unique ID column
)
```

- `path`: Path to the root of the WebDataset directory (local or cloud storage)
- `id_col`: Name of the unique identifier column in the metadata (commonly `key`)

---

## Parameters

```{list-table} WebDataset Loading Parameters
:header-rows: 1
:widths: 20 20 40 20

* - Parameter
  - Type
  - Description
  - Default
* - `path`
  - str
  - Path to the WebDataset directory (local or cloud storage)
  - Required
* - `id_col`
  - str
  - Name of the unique identifier column in the metadata (for example, `key`)
  - Required
```

---

## Output Format

The loaded `ImageTextPairDataset` object provides access to metadata, images, and captions for downstream curation tasks. The directory contains:

- Sharded `.tar` files with images, captions, and metadata
- `.parquet` files with tabular metadata
- (Optional) `.idx` files for DALI-based loading

**Example record structure:**

- `000000031.jpg`: Image file
- `000000031.txt`: Caption file
- `000000031.json`: Metadata file

The `ImageTextPairDataset.metadata` attribute is a Dask-cuDF DataFrame containing all metadata fields, including the unique ID column.

---

## Customization Options & Performance Tips

- **Cloud Storage Support**: You can use local paths or cloud storage URLs (for example, S3, GCS, Azure) thanks to `fsspec` integration. Make sure your environment is configured with the appropriate credentials.
- **DALI Index Files**: For large datasets, provide `.idx` files for each `.tar` to enable fast DALI-based loading (see [NVIDIA DALI documentation](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/data_loading/dataloading_webdataset.html#Creating-an-index)).
- **GPU Acceleration**: Use a GPU-enabled environment for best performance.
- **Saving Metadata**: Use `ImageTextPairDataset.save_metadata()` to export metadata as Parquet files.
- **Resharding/Filtering**: Use `ImageTextPairDataset.to_webdataset()` to reshard or filter the dataset and write a new WebDataset directory.

---

<!-- More advanced usage and troubleshooting tips can be added here. --> 