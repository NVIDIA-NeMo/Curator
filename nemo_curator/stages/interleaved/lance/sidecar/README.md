# Lance URL Sidecars

Interleaved document datasets usually store image references as URLs in
`source_ref`, while the image bytes live in a separate Lance image table. The
Lance materializer intentionally does not perform URL lookup: it expects direct
Lance coordinates such as `lance_row_id`, or `lance_fragment_id` plus
`lance_row_offset`.

The sidecar bridges those two datasets. It is a local, sharded SQLite index that
maps each image URL to its Lance row id and packed row address. At pipeline time,
the resolver stage performs fast local URL lookup and appends the coordinate
columns that `LanceRowIdImageMaterializationStage` consumes.

## Why This Exists

Large remote Lance scalar-index lookups by URL can be too slow for image
materialization pipelines. The sidecar moves the URL lookup into a local
precomputed index:

```text
document source_ref URL
  -> local sidecar lookup
  -> lance_row_id, lance_fragment_id, lance_row_offset
  -> LanceRowIdImageMaterializationStage(address_mode="row_address")
  -> image bytes
```

This keeps URL resolution separate from image-byte retrieval. The sidecar is
built once for an image Lance table version and reused by many document
pipelines.

## On-Disk Format

The builder creates:

```text
sidecar_dir/
  manifest.json
  build_report.json
  progress.json
  sample_urls.jsonl
  shards/
    shard-00000.sqlite
    shard-00001.sqlite
    ...
```

Each shard contains a compact `kv` table:

```text
url_hash BLOB PRIMARY KEY
row_id   BLOB NOT NULL
rowaddr  BLOB NOT NULL
```

The URL key is `blake2b-128(url)`. `row_id` and `rowaddr` are little-endian
`uint64`. The packed Lance row address is decoded as:

```python
fragment_id = rowaddr >> 32
row_offset = rowaddr & 0xFFFFFFFF
```

The resolver emits split row-address columns by default because that is the
native input expected by the row-address materializer.

## Build A Sidecar

Use the generic tutorial script:

```bash
python tutorials/interleaved/lance/build_url_lance_sidecar.py \
  --image-uri /path/to/images.lance \
  --image-version 4 \
  --key-column url \
  --output-dir /path/to/url_lance_sidecar \
  --shard-count 512
```

For remote S3-compatible storage, either pass an AWS profile:

```bash
python tutorials/interleaved/lance/build_url_lance_sidecar.py \
  --image-uri s3://bucket/path/to/images.lance \
  --image-version 4 \
  --key-column url \
  --output-dir /local/path/url_lance_sidecar \
  --aws-profile con
```

or pass Lance storage options directly:

```bash
python tutorials/interleaved/lance/build_url_lance_sidecar.py \
  --image-uri s3://bucket/path/to/images.lance \
  --output-dir /local/path/url_lance_sidecar \
  --storage-option aws_endpoint=https://pdx.s8k.io \
  --storage-option aws_region=us-east-1
```

For a smoke build, limit scanned rows:

```bash
python tutorials/interleaved/lance/build_url_lance_sidecar.py \
  --image-uri /path/to/images.lance \
  --output-dir /tmp/url_lance_sidecar_smoke \
  --max-rows 100000 \
  --overwrite
```

## Use In A Pipeline

Resolve URLs before materialization:

```python
from nemo_curator.stages.interleaved.lance import (
    LanceRowIdImageMaterializationStage,
    LanceTableConfig,
    ShardedSqliteUrlLanceAddressResolutionStage,
)

resolver = ShardedSqliteUrlLanceAddressResolutionStage(
    sidecar_dir="/path/to/url_lance_sidecar",
    input_url_column="source_ref",
)

materializer = LanceRowIdImageMaterializationStage(
    dataset=LanceTableConfig(uri="/path/to/images.lance", version=4),
    address_mode="row_address",
    input_fragment_id_column="lance_fragment_id",
    input_row_offset_column="lance_row_offset",
    presence_column="lance_materialized_present",
)
```

The resolver appends:

```text
lance_row_id
lance_fragment_id
lance_row_offset
lance_address_present
lance_lookup_error
```

`lance_lookup_error` is `missing_url` when the input URL is empty or null, and
`not_found_in_sidecar` when the URL is not present in the sidecar.

## Module Layout

```text
format.py    shared sidecar schema, hashing, encoding, manifest helpers
resolver.py  Curator stage for URL -> Lance coordinate lookup
builder.py   offline Lance scan -> sharded SQLite sidecar builder
```

Keep runtime resolver changes separate from builder changes where possible. The
resolver is in the hot pipeline path; the builder is an offline preprocessing
utility.
