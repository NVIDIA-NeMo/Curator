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

**Monolithic presets** (run the whole pipeline in one job):

| Preset | Recipe |
|---|---|
| `omnicorpus`           | OmniCorpus paper §3.1: 14 text filters + image NSFW + LAION aesthetic. Needs GPU. *Trips the GPU idle reaper on `batch` — prefer the chained workflow.* |
| `omnicorpus_cpu`       | Same as above with NSFW + aesthetic disabled. Pure-CPU. |
| `omnicorpus_text_only` | Text filters only — no image download, no GPU. Deterministic; useful for reproducible filter funnels. |
| `mint1t`               | MINT-1T-style: lighter Gopher subset, looser thresholds (approximate — MINT-1T relies on KenLM perplexity which we don't have yet) |
| `mmc4`                 | Permissive baseline: lang-ID + word count + URL-NSFW only |

**Chained pipeline presets** (each is one stage of a 4-step chain — see the *Chained pipeline workflow* section below):

| Preset | Input | Recipe |
|---|---|---|
| `extract`        | WARC    | Resiliparse + magic-html → InterleavedBatch Parquet (no filtering) |
| `text_filter`    | Parquet | 14 doc-level text filters |
| `image_acquire`  | Parquet | Image downloader only (no quality filtering) |
| `image_quality`  | Parquet | Geometry + aspect ratio + NSFW + aesthetic + image_count + PII redactor |

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

- `--extractor magic_html | magic_traf` — HTML→rows implementation.
  `magic_html` runs magic-html only; `magic_traf` (default) tries magic-html
  first and falls back to `trafilatura(output_format='html')` on empty pages.
- `--record-limit N` — cap records per WARC (smoke testing).
- `--files-per-partition K` — one Ray batch per `K` WARC files.
- `--max-text-chars N` — cap per-row text length (default 50,000); guards
  against pathological docs that would trigger a `<U{maxlen}>` numpy upcast
  inside the filter chain.
- `--max-batch-bytes B` — split each WARC into sub-batches no larger than `B`
  Arrow bytes (default 256 MiB).  Smaller = more pipeline parallelism.
- Filter on/off pairs: every Stage-3 / Stage-5 filter has a
  `--<name>` / `--no-<name>` toggle; thresholds use `--<name>-min` /
  `--<name>-max`. See `--help` for the full list.

## Slurm array — fan-out over many WARCs

For a multi-WARC run on a Slurm cluster, drive submissions through
**`submit_array.sh`**.  One array task = one WARC; per-shard `_SUCCESS`
markers make retries idempotent.

### Quick start

```bash
WARC_DIR=$USER_DIR/CC-MAIN-…/segments/…/warc \
WARC_PATTERN="CC-MAIN-…-%05d.warc.gz" \
OUTPUT_PATH=$USER_DIR/out/50warcs_cpu \
PRESET=omnicorpus_cpu \
ARRAY_SIZE=50 \
./submit_array.sh submit
```

### Four modes

| mode | what it does |
|---|---|
| `submit`        | fresh `sbatch --array=0..N-1%MAX` |
| `status`        | reads `_SUCCESS/shard_*.json`; prints "completed / missing" |
| `retry-missing` | resubmits only the shards without markers |
| `worker`        | per-shard srun entrypoint (called by sbatch — never invoke directly) |

Run `status` / `retry-missing` repeatedly; the workflow is idempotent.

### Required env

| var | meaning |
|---|---|
| `WARC_DIR`     | directory holding the WARC files |
| `WARC_PATTERN` | printf pattern indexed by shard, e.g. `"CC-MAIN-…-%05d.warc.gz"` |
| `OUTPUT_PATH`  | array root — markers, logs, per-WARC subdirs land here |
| `PRESET`       | `omnicorpus_cpu` / `omnicorpus_text_only` / `omnicorpus` |
| `ARRAY_SIZE`   | number of WARCs |

### Common knobs (all optional)

| var | default | notes |
|---|---|---|
| `PARTITION`     | `cpu_short` | set to `batch` for the GPU pipeline |
| `TIME_LIMIT`    | `01:00:00`  | bump to `01:30:00` if 2-per-node packing causes timeouts |
| `CPUS_PER_TASK` | 16          | proved optimal in benchmarks; below 16 thrashes C-extension threads |
| `MEM`           | 200G (CPU) / 190G (GPU) | drop to 110G to pack 2 tasks per CPU node |
| `MAX_CONCURRENT`| `ARRAY_SIZE` | throttle for QOS or politeness |

### Output layout

```
$OUTPUT_PATH/
├── _SUCCESS/                          ← idempotency markers, root-level
│   ├── shard_00000.json               (per completed shard)
│   └── …
├── _logs/                             ← all logs outside the per-WARC data dirs
│   ├── slurm-<job>_<idx>.out          (sbatch stdout)
│   └── idx_NNNNN.log                  (python --log-path, has funnel lines)
├── idx_00000/                         ← per-WARC output
│   ├── <hash>.parquet
│   └── _run.json                      (manifest + funnel parsed from log)
└── …
```

Markers and logs live at the array root so the writer's `--mode overwrite`
rmtree of `idx_NNNNN/` doesn't delete them.

### What it handles under the hood

- **Ray isolation per task** — each task starts its own local Ray cluster
  (`ray.init(address="local", _temp_dir=/tmp/ray_ccmm_<job>_<idx>)`) and sets
  `RAY_ADDRESS` so `RayClient.start()` skips its own `ray start --head` subprocess.
  Result: Slurm can pack multiple tasks per node without `/tmp/ray` collisions.
- **Resume on retry** — `Shard.has_marker()` short-circuits at startup;
  rerunning the full array re-runs nothing extra.
- **Sparse-retry shard count** — `CURATOR_ORIGINAL_ARRAY_SIZE` is propagated
  so `--array=3,7,12` retries still see `num_shards=50` in env.

### GPU caveat — and the chained pipeline that fixes it

Running the *monolithic* `PRESET=omnicorpus PARTITION=batch` end-to-end
trips the cluster's **GPU idle reaper** (uid 146504, `svc-hwinf-cs-sched`)
at ~37 min: the GPU sits at 0 % for the first 30 minutes (extract +
text filters + image download — all CPU/network), so by the time NSFW
+ aesthetic actually start, the reaper's 30-min idle window has elapsed.
Roughly 30–50 % of tasks get SIGTERM'd mid-run.

**The proper fix is to split the pipeline into stages** that each fit
under the 30-min idle threshold and only hold GPU when actually using it.
See the next section.

## Chained pipeline workflow (4 stage groups)

Each stage group runs as its own `submit_array.sh` invocation against its
own output directory; the next group reads the previous group's output
as Parquet.  This gives us:

- **Failure isolation** — a crash in stage 3 doesn't lose stage 1+2 work
- **Independent retries** — re-run one stage with tweaked thresholds
  without redoing the others
- **Right-sized resources per stage** — CPU for stages 1–3, GPU only for
  stage 4 (and only ~5 min/task, well under the idle-reaper threshold)
- **Resumability** — a re-submitted stage short-circuits per-shard via
  the existing `_SUCCESS/shard_NNNNN.json` markers

### The four presets

| group | preset | input | what it does | per-shard wallclock (1 WARC) |
|---|---|---|---|---|
| 1 | `extract`        | WARC    | Resiliparse + magic-html → InterleavedBatch Parquet | ~20 min |
| 2 | `text_filter`    | Parquet | 14 doc-level text filters                            | ~6 min |
| 3 | `image_acquire`  | Parquet | downloader (~70 % URL success)                       | ~15 min |
| 4 | `image_quality`  | Parquet | geometry + aspect + NSFW + aesthetic + image_count + PII | ~5 min |

Stage 4 finishes in ~5 min wallclock — under the GPU idle-reaper threshold.

### Driver invocation

```bash
ROOT=$USER_DIR/out/50warcs_chain
WARCS=$USER_DIR/CC-MAIN-…/segments/…/warc
WARC_PATTERN="CC-MAIN-…-%05d.warc.gz"

# 1. WARC → extracted Parquet
PRESET=extract INPUT_TYPE=warc \
WARC_DIR=$WARCS WARC_PATTERN=$WARC_PATTERN \
OUTPUT_PATH=$ROOT/01_extract ARRAY_SIZE=50 \
TIME_LIMIT=02:00:00 ./submit_array.sh submit

# 2. extracted → text-filtered (multimodal preset: word_count_min=20,
#    lang_id off, lang_id_annotate on)
PRESET=text_filter_multimodal INPUT_TYPE=parquet INPUT_PATH=$ROOT/01_extract \
OUTPUT_PATH=$ROOT/02_text ARRAY_SIZE=50 \
TIME_LIMIT=00:30:00 ./submit_array.sh submit

# 3. text-filtered → with downloaded images
PRESET=image_acquire INPUT_TYPE=parquet INPUT_PATH=$ROOT/02_text \
OUTPUT_PATH=$ROOT/03_images ARRAY_SIZE=50 \
TIME_LIMIT=01:30:00 ./submit_array.sh submit

# 4. with images → final (GPU)
PRESET=image_quality INPUT_TYPE=parquet INPUT_PATH=$ROOT/03_images \
OUTPUT_PATH=$ROOT/04_quality ARRAY_SIZE=50 \
PARTITION=batch TIME_LIMIT=00:30:00 ./submit_array.sh submit
```

Each step blocks the next implicitly (the next `submit` can't process
shards whose marker isn't yet written by the prior stage).  Per-stage
`status` and `retry-missing` work as usual within each group's
`OUTPUT_PATH`.

### What `INPUT_TYPE=parquet` does

Worker derives this shard's input dir as `${INPUT_PATH%/}/idx_<pad>/`
(matching the convention the prior stage writes to) and passes
`--input-path <derived>` along with `--input-type parquet` to
`run_warc_pipeline.py`.  Python uses
`nemo_curator.stages.interleaved.io.reader.InterleavedParquetReader`
in place of the WARC reader+extractor; everything downstream is
identical.

### Comparison: monolithic vs chained

|  | monolithic (`omnicorpus`) | chained (4 groups) |
|---|---|---|
| total wallclock per shard | ~35 min (when not reaped) | ~46 min serial; ~25 min if you can run downstream stages in parallel as upstream finishes |
| GPU idle-reaper kills | ~30–50 % of tasks | none (stage 4 is short enough) |
| failure recovery | re-run whole shard from WARC | re-run only the affected stage |
| tunability | one big preset | tweak one stage in isolation |
| disk I/O overhead | none (everything in-memory) | each stage reads + writes Parquet (~5–10 % overhead) |
| per-shard `_run.json` funnel | one for whole pipeline | one per stage (richer trace) |

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
    ├── run_warc_pipeline.py              ← pipeline entrypoint (one WARC per invocation)
    ├── run_manifest.py                   ← writes _run.json (lineage + funnel)
    ├── shard.py                          ← per-shard success markers (Slurm array)
    ├── submit_array.sh                   ← Slurm-array driver: submit/status/retry-missing/worker
    ├── dashboard.py                      ← Streamlit output viewer
    ├── README.md                         ← this file
    ├── PERF_NOTES.md                     ← measured timings, scaling math, bug history
    └── presets/
        ├── omnicorpus.yaml               (GPU full pipeline)
        ├── omnicorpus_cpu.yaml           (no GPU stages)
        ├── omnicorpus_text_only.yaml     (no images, deterministic)
        ├── mint1t.yaml
        └── mmc4.yaml
```
