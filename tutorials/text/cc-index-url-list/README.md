# CC Index URL List

Create one global unique URL list from an explicit set of Common Crawl Index snapshots. This tutorial points Curator's `ParquetReader` at the PBSS-hosted CC Index parquet table, reads `url` and `warc_filename`, and uses NeMo Curator exact deduplication and duplicate removal workflows to write a Parquet dataset that can be shared with a team.

The default config uses these snapshots:

```text
CC-MAIN-2026-17
CC-MAIN-2026-12
CC-MAIN-2026-08
CC-MAIN-2026-04
CC-MAIN-2025-51
CC-MAIN-2025-47
CC-MAIN-2025-43
CC-MAIN-2025-38
CC-MAIN-2025-33
CC-MAIN-2025-30
```

`CC-MAIN-2025-26` and `CC-MAIN-2025-08` are not present because the config is authoritative: include only the crawls you want in `selected_crawls.yaml`.

## Install

Create a CUDA environment with Curator text deduplication dependencies and the S3 fsspec backend. Keep the environment and uv cache on Lustre so smoke tests and full jobs reuse the same install:

```bash
export UV_CACHE_DIR=/lustre/fsw/portfolios/llmservice/users/vjawa/uv_cache
uv venv /lustre/fsw/portfolios/llmservice/users/vjawa/cc-index-url-list-venv
source /lustre/fsw/portfolios/llmservice/users/vjawa/cc-index-url-list-venv/bin/activate
uv pip install -e ".[text_cuda12]" "s3fs>=2024.12.0"
```

For local smoke tests, keep Ray's temporary directory short enough for Unix socket paths:

```bash
export RAY_TMPDIR=/tmp/vjawa_ray
```

For iteration, run smoke commands on a short Slurm CPU allocation:

```bash
srun -A nemotron_n4_pre -p cpu_dataprocessing \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --time=00:20:00 --pty bash
```

Use CPU nodes for config checks, PBSS listing checks, and staging iteration. The exact deduplication phase uses RAPIDS/cuDF and should run in the CUDA/RAPIDS environment used for full Curator text deduplication jobs.

## PBSS Credentials

The tutorial reads from the `commoncrawl` PBSS namespace at `https://pdx.s8k.io`. Set PBSS credentials before running:

```bash
export PBSS_ACCESS_KEY_ID=commoncrawl
export PBSS_SECRET_ACCESS_KEY=<secret>
```

The script also accepts `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` as fallbacks, but PBSS-prefixed variables are preferred so they do not collide with unrelated AWS credentials.

## Config

Edit `selected_crawls.yaml` to choose the exact snapshots:

```yaml
cc_index_base_uri: s3://cc-index/table/cc-main/warc
endpoint_url: https://pdx.s8k.io
output_name: global_unique_urls
included_crawls:
  - CC-MAIN-2026-17
  - CC-MAIN-2026-12
```

Each crawl expands to:

```text
s3://cc-index/table/cc-main/warc/crawl=<crawl>/subset=warc/
```

## Dry Run

Print the configured crawl directories without launching the Curator pipeline. This mode only expands the YAML config, so it does not need PBSS credentials unless you also pass `--max-files-per-crawl`.

```bash
python tutorials/text/cc-index-url-list/create_cc_index_url_list.py \
  --config tutorials/text/cc-index-url-list/selected_crawls.yaml \
  --output /lustre/$USER/cc_index_url_list \
  --dry-run
```

For a small smoke test, cap each crawl to one source parquet file. This is the only mode that expands crawl directories to individual parquet files before calling `ParquetReader`.

```bash
python tutorials/text/cc-index-url-list/create_cc_index_url_list.py \
  --config tutorials/text/cc-index-url-list/selected_crawls.yaml \
  --output /lustre/$USER/cc_index_url_list_smoke \
  --max-files-per-crawl 1
```

## Full Run

Run the full configured list:

```bash
python tutorials/text/cc-index-url-list/create_cc_index_url_list.py \
  --config tutorials/text/cc-index-url-list/selected_crawls.yaml \
  --output /lustre/$USER/cc_index_url_list \
  --staging-blocksize 512MB \
  --dedup-blocksize 512MB
```

Output layout:

```text
/lustre/$USER/cc_index_url_list/
|-- _staging_url_column/       # URL/WARC filename parquet staged from CC Index
|-- _dedup_ids/                # ExactDeduplicationWorkflow duplicate IDs
`-- global_unique_urls/        # final URL/WARC filename Parquet dataset
```

The final Parquet files contain two columns:

| Column | Description |
|--------|-------------|
| `url` | Globally unique page URL across all configured snapshots |
| `warc_filename` | WARC object path; includes `crawl-data/CC-MAIN-*` snapshot ID |

Because the deduplication key is only `url`, a URL that appears in several snapshots is written once. `warc_filename` is a retained-row provenance hint for debugging, not a complete list of every snapshot that contained that URL. The CC snapshot ID can be extracted from the `crawl-data/CC-MAIN-*` segment of `warc_filename`.

## How It Works

1. Expand each configured crawl ID to its CC Index parquet directory.
2. Pass those directories to `ParquetReader(file_paths=..., fields=["url", "warc_filename"], blocksize=...)`.
3. Stage larger URL/WARC filename parquet shards locally with `ParquetWriter`.
4. Run `ExactDeduplicationWorkflow(text_field="url", assign_id=True)` to identify duplicates.
5. Run `TextDuplicatesRemovalWorkflow(output_fields=["url", "warc_filename"])` to write the global unique URL dataset.

`files_per_partition` is not set for the full run. Staging uses `--staging-blocksize` instead, defaulting to `512MB`, so Curator groups source parquet files into larger tasks before writing staged parquet.

Exact deduplication and duplicate removal are run as the clean Curator workflow pair used elsewhere in the repository. `ExactDeduplicationWorkflow` writes duplicate-ID side outputs plus an ID generator mapping, and `TextDuplicatesRemovalWorkflow` consumes those outputs to write the final parquet dataset.
