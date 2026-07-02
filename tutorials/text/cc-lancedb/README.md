# CC LanceDB Index — NeMo Curator

Downloads Common Crawl WARC data, extracts HTML bytes and text, and writes a LanceDB table to PBSS/SwiftStack on pdx.s8k.io.

## LanceDB Schema

| Column | Type | Description |
|---|---|---|
| cc_snapshot_id | string | WARC filename (e.g. CC-MAIN-2025-26-00000-of-90000.warc.gz) |
| cc_url | string | Target URL of the web page |
| cc_html_bytes | large_binary | Raw HTML bytes from WARC response |
| cc_extracted_text | string | Text extracted by the configured extractor |
| warc_id | string | WARC-Record-ID for deduplication |
| language | string | Detected language code |
| extractor_lib | string | Extractor used: trafilatura, justext, or resiliparse |

## Architecture

```
CC WARC URLs -> CommonCrawlWARCDownloader -> CommonCrawlWarcIterator -> CCHTMLBytesExtractor -> LanceDBWriter -> PBSS
```

## Quick Start

### Environment setup

```bash
export AWS_ENDPOINT_URL_S3=https://pdx.s8k.io
export AWS_ACCESS_KEY_ID=<your-pbss-key>
export AWS_SECRET_ACCESS_KEY=<your-pbss-secret>
export AWS_DEFAULT_REGION=us-east-1
```

### Install dependencies

```bash
pip install lancedb pyarrow warcio trafilatura nemo-curator
```

### Run (single snapshot, local Ray)

```bash
python pipeline.py \
  --start-snapshot CC-MAIN-2025-26 \
  --end-snapshot   CC-MAIN-2025-26 \
  --download-dir   /tmp/cc_warcs \
  --lancedb-uri    s3://YOUR-BUCKET/cc_lancedb \
  --table-name     cc_snapshot_index \
  --extractor-lib  trafilatura
```

### Query the LanceDB table from Python

```python
import lancedb, os

db = lancedb.connect(
    "s3://YOUR-BUCKET/cc_lancedb",
    storage_options={
        "endpoint": os.environ["AWS_ENDPOINT_URL_S3"],
        "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
        "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
        "aws_region": "us-east-1",
    }
)
tbl = db.open_table("cc_snapshot_index")
df = tbl.to_pandas()
```

## SLURM

See `slurm/submit.sh` for a ready-made batch script targeting Nebius (nb-hel-cs-001) or DFW (cw-dfw-cs-001).

## Extractor options

- **trafilatura** (default): best recall on boilerplate-heavy pages
- **justext**: high precision, good for multilingual; memory-limited to 2 workers
- **resiliparse**: fastest, best for very large WARC runs
