# Wikipedia batch processing

Pipeline:

1. Download Wikipedia HTML dumps
2. Sample 1M pages
3. Extract DocLayNet training data
4. Shuffle and split

## Download Wikipedia HTML dumps

Download from [Wikimedia dumps](https://dumps.wikimedia.org/other/enterprise_html/runs/20250301/).

> **Note:** please review and comply with the terms of use for Wikipedia dumps.

## Sample 1M pages from each language

```sh
for name in dewiki enwiki eswiki frwiki itwiki jawiki kowiki nlwiki ptwiki zhwiki; do
  docker run --rm \
    -v $PWD:/workspace \
    -v path/to/data:/data \
    latex-augment python3 scripts/wikipedia/sample_pages.py \
    /data/${name}-NS0-20250301-ENTERPRISE-HTML.json.tar.gz \
    /data/${name}-1m.jsonl.zst
done
```

## Extract DocLayNet training data

Each language is split across 10 parallel containers, each writing 10k-page shards. Chromium is installed on first run inside the container.

```sh
for name in dewiki enwiki eswiki frwiki itwiki jawiki kowiki nlwiki ptwiki zhwiki; do
  docker run --rm --shm-size=2g \
    -v $PWD:/workspace \
    -v path/to/data:/data \
    latex-augment python3 scripts/wikipedia/wikidump.py \
    /data/${name}-1m.jsonl.zst \
    /data/${name}/temp
done
```

## Shuffle and split

Shuffle data and split into `train` and `val` subsets.

```sh
for name in dewiki enwiki eswiki frwiki itwiki jawiki kowiki nlwiki ptwiki zhwiki; do
  docker run --rm \
    -v $PWD:/workspace \
    -v path/to/data:/data \
    latex-augment python3 scripts/wikipedia/shuffle_wds.py \
    /data/${name}/temp /data/${name}/train --first-page-idx 10000
  docker run --rm \
    -v $PWD:/workspace \
    -v path/to/data:/data \
    latex-augment python3 scripts/wikipedia/shuffle_wds.py \
    /data/${name}/temp /data/${name}/val --last-page-idx 9999
done
```
