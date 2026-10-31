# Proof of Concept: cc-img-dl → NeMo Curator

This directory demonstrates how the `cc-img-dl` image download pipeline maps onto
NeMo Curator's stage/pipeline architecture. It is self-contained and does **not**
modify any existing Curator source code.

## Directory Structure

```
poc_image_download/
├── nemo_curator_stages/image/download/   # New Curator stages
│   ├── __init__.py
│   ├── manifest.py          # ManifestReaderStage (URLGenerator + reader)
│   ├── downloader.py        # ImageDownloaderStage (HTTP download with retries)
│   ├── stage.py             # ImageDownloadCompositeStage (ties everything together)
│   ├── compliance/
│   │   ├── __init__.py
│   │   ├── cache.py         # Per-worker compliance cache
│   │   ├── robots.py        # Robots.txt checking
│   │   ├── tdm.py           # TDMRep checking
│   │   └── filter.py        # ComplianceFilterStage (ProcessingStage)
│   └── storage.py           # Storage utilities
├── tutorial/
│   └── image_download_example.py   # End-to-end example
├── tests/
│   └── test_stages.py       # Unit tests for the new stages
└── README.md
```

## How It Maps

| cc-img-dl Module | Curator Stage | What Changed |
|---|---|---|
| `manifest.py` (iter_records, write_shards) | `ManifestReaderStage` | Sharding removed — Curator handles task distribution |
| `compliance/` (robots, tdm, checker) | `ComplianceFilterStage` | Rewritten as a ProcessingStage filter |
| `downloader.py` | `ImageDownloaderStage` | async httpx → sync requests with ThreadPool |
| `worker.py` (orchestration) | `Pipeline.run()` | Entire module eliminated |
| `checkpoint.py` | Xenna fault tolerance | Entire module eliminated |
| `storage.py` | Curator writers / fsspec | Simplified to direct file write in download stage |
| `config.py` | Stage constructor params | Config flattened into stage parameters |
| `cli.py` | Tutorial script | CLI replaced by Pipeline API |

## What Was Eliminated (~40% of cc-img-dl code)

- `worker.py` — orchestration logic (Curator pipeline handles this)
- `checkpoint.py` — resume logic (Xenna handles task-level retries)
- `manifest.write_shards()` / `shard_id_for()` — sharding (Curator distributes tasks)
- `storage.py` — storage abstraction (download stage writes directly)
- `cli.py` — CLI entrypoint (replaced by Pipeline API / tutorial)

## Running the Example

```python
# Assuming Curator is installed and Ray is available
python tutorial/image_download_example.py \
    --manifest url_manifest.parquet \
    --output-dir /data/images
```
