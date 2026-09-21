# Interleaved Pipeline

Row-wise interleaved multimodal ingestion and write path for WebDataset tar shards (MINT-1T style), with materialization support for local, remote, and tar-archived binary content.

## Architecture

```
WebDataset tar shards          Parquet files
        |                            |
        v                            v
┌──────────────────────────┐  ┌──────────────────────────┐
│ InterleavedWebdataset-   │  │ InterleavedParquetReader  │  Both are CompositeStages:
│ Reader (io/reader.py)    │  │ (io/reader.py)            │  FilePartitioningStage +
│                          │  │                           │  <modality>ReaderStage
└──────────┬───────────────┘  └────────────┬─────────────┘
           └──────────┬────────────────────┘
                      |  InterleavedBatch (Arrow/Pandas)
                      v
         ┌─────────────────────────┐
         │  Filter Stages          │  e.g. InterleavedAspectRatioFilterStage
         │  (stages.py)            │  Row-wise filtering with optional materialization
         └────────┬────────────────┘
                  |
        ┌─────────┴──────────┐
        v                    v
┌───────────────┐   ┌──────────────────────────┐
│ Interleaved-  │   │ InterleavedWebdataset-    │
│ ParquetWriter │   │ WriterStage               │
│ Stage         │   │ (io/writers/webdataset.py)│
│ (tabular.py)  │   │ MINT-1T-style tar shards  │
└───────────────┘   └──────────────────────────┘
```

## Schema (`INTERLEAVED_SCHEMA`)

Defined in `nemo_curator/tasks/interleaved.py`. Columns are split into **reserved** (managed by the pipeline) and **user** (passthrough from source data).

### Reserved columns (`RESERVED_COLUMNS`)

These are set and managed by pipeline stages. Users should not write to them directly.

| Column | Type | Category | Description |
|--------|------|----------|-------------|
| `sample_id` | string (required) | Identity | Unique document/sample identifier |
| `position` | int32 (required) | Identity | Position within sample (-1 for metadata rows) |
| `modality` | string (required) | Identity | Row modality: `text`, `image`, `metadata` built-in; extensible to `audio`, `table`, `generated_image`, etc. |
| `content_type` | string | Content | MIME type (e.g. `text/plain`, `image/jpeg`) |
| `text_content` | string | Content | Text payload for text rows |
| `binary_content` | large_binary | Content | Image bytes (populated by materialization) |
| `source_ref` | FILE | Internal | Parquet FILE-compatible reference: `uri`, `offset`, `size`, `content_type`, `checksum`, `inline` |
| `source_member` | string | Internal | Archive member metadata adjacent to FILE |
| `source_frame_index` | int32 | Internal | Multi-frame index metadata adjacent to FILE |
| `materialize_error` | string | Internal | Error message if materialization failed |

### User columns (passthrough)

Extra fields from the source data flow through the pipeline as additional columns. Specify them with the `fields` parameter on the reader:

```python
reader = InterleavedWebdatasetReader(
    file_paths="/data/shards/",
    fields=("p_hash", "score", "aux"),  # These become extra columns
)
```

If `fields` is `None` (default), all non-reserved fields from the source JSON are passed through. If specified explicitly, only the listed fields are included -- and the reader validates they exist and don't collide with reserved names.

## Key Concepts

### InterleavedBatch

The task type for interleaved multimodal data (`nemo_curator/tasks/interleaved.py`). Wraps either a PyArrow Table or Pandas DataFrame.

Class attributes:
- `REQUIRED_COLUMNS` -- frozenset of columns that must always be present (non-nullable schema fields)

Key methods:
- `build_source_ref(uri, offset, size)` -- build a FILE-compatible external reference
- `with_parsed_source_ref_columns(prefix)` -- expand source_ref into DataFrame columns
- `to_pyarrow()` / `to_pandas()` -- conversion between formats

### source_ref

`source_ref` uses all six fields from the closed Parquet FILE specification.
PyArrow cannot yet emit the new FILE footer annotation, so current output uses
the compatible physical group. Archive member and TIFF frame metadata remain
in the adjacent `source_member` and `source_frame_index` columns.

### Materialization

Binary content (images) can be loaded lazily. Two I/O strategies dispatch automatically based on `source_ref` content (`utils/materialization.py`):

| Strategy | When | How |
|----------|------|-----|
| **FILE read** | External FILE reference | Deduplicated `fs.cat_ranges()` calls, batched per filesystem |
| **Tar extract** | `source_member` present, no byte range | Open tar once, `extractfile()` per member |

When `source_frame_index` is set, materialization extracts a single frame from a multi-frame TIFF and returns it as a standalone TIFF. Non-TIFF content is returned unchanged regardless of `frame_index`.

Materialization can happen at read time (`materialize_on_read=True`) or write time (`materialize_on_write=True`).

## Usage

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.interleaved.io import InterleavedWebdatasetReader, InterleavedParquetWriterStage
from nemo_curator.stages.interleaved.stages import InterleavedAspectRatioFilterStage

pipeline = Pipeline(name="mint1t_pipeline")
pipeline.add_stage(InterleavedWebdatasetReader(
    file_paths="/data/mint1t/shards/",
))
pipeline.add_stage(InterleavedAspectRatioFilterStage(drop_invalid_rows=True))
pipeline.add_stage(InterleavedParquetWriterStage(
    path="/output/parquet/",
    materialize_on_write=True,
    mode="overwrite",
))
pipeline.run()
```

## File Layout

```
stages/interleaved/
├── __init__.py                     # Exports filter/annotator stages
├── stages.py                       # BaseInterleavedAnnotatorStage, BaseInterleavedFilterStage,
│                                   # InterleavedAspectRatioFilterStage
├── io/
│   ├── __init__.py                 # Exports InterleavedWebdatasetReader, InterleavedParquetReader,
│   │                               # InterleavedParquetWriterStage, InterleavedWebdatasetWriterStage
│   ├── reader.py                   # InterleavedWebdatasetReader, InterleavedParquetReader (CompositeStages)
│   ├── readers/
│   │   ├── base.py                 # BaseInterleavedReader
│   │   ├── parquet.py              # InterleavedParquetReaderStage (ProcessingStage)
│   │   └── webdataset.py           # InterleavedWebdatasetReaderStage (ProcessingStage)
│   └── writers/
│       ├── base.py                 # BaseInterleavedWriter (filesystem + materialization + process)
│       ├── tabular.py              # InterleavedParquetWriterStage
│       └── webdataset.py           # InterleavedWebdatasetWriterStage
└── utils/
    ├── constants.py                # Default file extensions
    ├── materialization.py          # Three-strategy materialization dispatch
    └── validation_utils.py         # Field validation, storage options resolution
```
