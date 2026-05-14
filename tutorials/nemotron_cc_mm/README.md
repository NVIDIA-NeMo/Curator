# Nemotron-CC-MM WARC pipeline

End-to-end pipeline that turns a Common Crawl WARC into an interleaved
text + image Parquet dataset, using NeMo Curator's `InterleavedBatch`
schema. The recipe follows OmniCorpus (Li et al. 2024) for HTML
extraction + Gopher / C4 text filtering + MMC4 / LAION-5B image
filtering.

```
WARC  →  HTML extraction  →  text filters  →  image download + filters  →  Parquet
                ②                 ②                    ④
```

See `presets/omnicorpus.yaml` for the full recipe (including what's
deferred — search for `📌 TODO`).

## Setup

```bash
# One-off: make sure Curator's editable install + its venv have ray on PATH.
export PATH=$(pwd)/Curator/.venv/bin:$PATH
```

## Run

Minimal — local WARC dir + output dir + the OmniCorpus preset:

```bash
python Curator/tutorials/nemotron_cc_mm/run_warc_pipeline.py \
    --preset omnicorpus \
    --input-path  data/warc/ \
    --output-path data/out/
```

Quick smoke test (50 records, no image download):

```bash
python Curator/tutorials/nemotron_cc_mm/run_warc_pipeline.py \
    --preset omnicorpus \
    --input-path  data/warc/ \
    --output-path /tmp/smoke/ \
    --record-limit 50 \
    --no-image-download --no-image-geometry --no-image-aspect-ratio \
    --no-image-nsfw --no-image-aesthetic --no-image-count
```

## Presets

Each preset under `presets/*.yaml` configures the whole pipeline (which
filters, which thresholds). YAML keys map 1:1 to CLI flag names (with
underscores instead of dashes).

| Preset | Recipe |
|---|---|
| `omnicorpus` | OmniCorpus paper §3.1: 14 text filters + image NSFW + LAION aesthetic |
| `mint1t`     | MINT-1T-style: lighter Gopher subset, looser thresholds (approximate — MINT-1T relies on KenLM perplexity which we don't have yet) |
| `mmc4`       | Permissive baseline: lang-ID + word count + URL-NSFW only |

Precedence: **CLI flags > preset > argparse defaults**. So you can pick a
preset and tweak one knob:

```bash
python ... --preset omnicorpus --no-lang-id    # OmniCorpus, but with lang-ID off
```

`--preset` accepts either a name (resolved against `presets/`) or a
direct path to any `.yaml`.

## Output

A directory of Parquet shards following Curator's `INTERLEAVED_SCHEMA`:

| column | type | meaning |
|---|---|---|
| `sample_id` | str | doc ID (= WARC record ID) |
| `position` | int | row order within doc; `-1` = metadata row |
| `modality` | str | `text` / `image` / `metadata` |
| `content_type` | str | MIME (e.g. `image/jpeg`, `text/plain`) |
| `text_content` | str | text body or image alt-text |
| `binary_content` | bytes | image bytes (if `--image-download`) |
| `source_ref` | str | JSON: doc URL, WARC ID, image URL |
| `materialize_error` | str | populated when image fetch failed |

Each doc starts with one `metadata` row at `position=-1`, followed by
content rows in DOM order.

## Common knobs

- `--extractor naive | magic_html | hybrid` — HTML→rows implementation.
  `naive` walks raw HTML; `magic_html` cleans main content first;
  `hybrid` tries magic_html and falls back to naive on empty pages.
- `--record-limit N` — cap records per WARC (smoke testing).
- `--files-per-partition K` — one Ray batch per `K` WARC files.
- Filter on/off pairs: every Stage-3 / Stage-5 filter has a
  `--<name>` / `--no-<name>` toggle; thresholds use `--<name>-min` /
  `--<name>-max`. See `--help` for the full list.

## Inspect output

A Streamlit dashboard for browsing Parquet output (with optional
side-by-side compare against a second run) ships next to this script:

```bash
PATH=$(pwd)/Curator/.venv/bin:$PATH \
    streamlit run Curator/tutorials/nemotron_cc_mm/dashboard.py
```

Open the printed URL (usually `http://localhost:8501`) and paste a
Parquet directory into the sidebar.  To A/B two runs, fill in the
"Compare against" field with a second directory.

## Layout

```
Curator/
├── nemo_curator/stages/nemotron_cc_mm/   ← all custom stages
└── tutorials/nemotron_cc_mm/
    ├── run_warc_pipeline.py              ← this script
    ├── dashboard.py                      ← Streamlit output viewer
    ├── README.md                         ← this file
    └── presets/
        ├── omnicorpus.yaml
        ├── mint1t.yaml
        └── mmc4.yaml
```
