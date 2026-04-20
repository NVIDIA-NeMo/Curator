# Speaker ID Annotation for Granary Dataset

Speaker identification annotation stage for the
[Granary](https://docs.nvidia.com/nemo/curator/latest/) dataset, built as a
[NeMo Curator](https://github.com/NVIDIA-NeMo/Curator) processing stage.

This pipeline extracts per-utterance speaker embeddings from NeMo-tarred audio
data using a NeMo `EncDecSpeakerLabelModel` (default:
[TitaNet-Large](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/speakerverification_en_titanet_large)).
The resulting embedding vectors are the foundation
for downstream speaker clustering and ID assignment within Granary's audio
curation workflow. Supports multi-GPU parallel extraction out of the box.

## Code structure

```
# NeMo Curator stage (library code)
nemo_curator/stages/audio/speaker_id/
  speaker_embedding_lhotse.py       # SpeakerEmbeddingLhotseStage — GPU embedding extraction
                                    #   merge_shard_embeddings()  — combine per-shard outputs
  speaker_embedding_request.py      # Alternative stage for plain JSONL manifests
  speaker_clustering_and_scoring.py # SpeakerClusteringStage — CPU AHC + confidence scores

  # Reusable libraries (promoted from speaker_id_for_asr_data, Apr 2026)
  clustering/
    cluster_config.py                       # SCOTCH codename + cluster_config.json sidecar
    large_scale_clustering_and_scoring.py   # BIRCH + AHC for 20–30M utterances
    ahc.py                                  # plain N×N cosine AHC
  embedding/                                # WeSpeaker / NeMo model loaders + features
  data/                                     # tar / S3 / manifest helpers
  multigpu/                                 # multi-GPU launcher
  utils/io.py                               # NPZ shard merge / load / save

  # End-to-end YODAS / YTC / Granary driver (also moved from old repo)
  run_pipeline.py
  run_multigpu.py

# Tutorial / CLI entry points
tutorials/audio/speaker_id/
  run_pipeline.py                   # --direct | --merge | --cluster | (default Ray pipeline)
  tune_threshold_librispeech.py     # 2-D threshold sweep on LibriSpeech (BIRCH × AHC)
  run_tune_threshold_librispeech.sh # 2-GPU launcher for the above
  scotch_cluster_and_annotate.py    # SCOTCH-v1.large_scale cluster + annotate driver
  run_scotch_cluster_only.sh        # CS-OCI-ORD / DRACO-OCI-IAD launcher for SCOTCH
  PARAM_TUNE.md                     # full LibriSpeech tuning write-up + plots (SCOTCH params)
  README_yodas_pipeline.md          # YODAS / YTC end-to-end recipe
  scripts/                          # standalone CLI tools (extract / cluster / download / split)
  configs/                          # default.yaml + corpus configs
  examples/                         # cluster / Draco-OCI launch scripts
  curator_spk_id_scripts/           # Curator-flavoured launchers for clustering / TitaNet
  embedding_norm_stats/             # cohort mean/std for `external` normalization
  param_tune_assets/                # plots, sweep CSVs, confidence analysis
```

`run_pipeline.py` execution modes:

| Flag | Resource | What it does |
|------|----------|-------------|
| `--direct` | **GPU** (CUDA) | Runs `SpeakerEmbeddingLhotseStage` on the current GPU. **Recommended for multi-GPU** when driven by a shell wrapper. |
| `--merge` | CPU | Merges per-shard `.npz`/`.pt` files in `--output_dir` into one file. |
| `--cluster` | **CPU only** | Runs `SpeakerClusteringStage`: agglomerative clustering + `speaker_label` / `confidence_score` on manifests. |
| _(default)_ | Ray / GPU per stage | NeMo Curator `Pipeline` + `RayDataExecutor`. |

Embeddings are written **raw** from the model. **Mean subtraction** for cosine AHC is applied inside clustering (`--embedding_normalization`, default `center_global`), not during extraction.

## Input format

NeMo tarred audio: a **manifest pattern** + **tar pattern** using NeMo's
brace-expand syntax (`_OP_` = `{`, `_CL_` = `}`).

```
manifest__OP_0..49_CL_.json   →   manifest_0.json, manifest_1.json, ..., manifest_49.json
audio__OP_0..49_CL_.tar       →   audio_0.tar,     audio_1.tar,     ..., audio_49.tar
```

Each `manifest_K.json` is a NeMo JSONL file where every line is a JSON object
with at least `audio_filepath`, `duration`, and `shard_id` fields.
Each `audio_K.tar` contains the corresponding audio files referenced by that
manifest.

## Output format

One `.npz` (or `.pt`) file **per shard**, written to `--output_dir`:

```
output/
  embeddings_0.npz
  embeddings_1.npz
  ...
  embeddings_49.npz
```

Each file contains two arrays:

| Key | Shape | Description |
|-----|-------|-------------|
| `cut_ids` | `(N,)` | String identifiers for each utterance in this shard |
| `embeddings` | `(N, D)` | Float32 speaker embedding vectors (192-dim for TitaNet-Large) |

Load with:

```python
import numpy as np
data = np.load("output/embeddings_0.npz", allow_pickle=True)
print(data["cut_ids"].shape)    # (N,)
print(data["embeddings"].shape) # (N, 192)
```

To merge all shards into a single file:

```bash
python tutorials/audio/speaker_id/run_pipeline.py --merge --output_dir "${WORK_ROOT}/embeddings"
# → ${WORK_ROOT}/embeddings/embeddings_merged.npz
```

## Two-phase workflow: extraction (GPU) → clustering (CPU)

### Caveats for laptops, servers, and HPC

| Phase | Hardware | Notes |
|-------|----------|--------|
| **Embedding extraction** | **One or more NVIDIA GPUs** | NeMo speaker models run on CUDA. On Slurm/PBS/etc., request a **GPU partition** (`#SBATCH --gres=gpu:1`, `CUDA_VISIBLE_DEVICES`, …). |
| **Clustering (`--cluster`)** | **CPU only** | Uses SciPy hierarchical clustering; **no GPU**. Prefer a **CPU node**, a login node (if policy allows), or a small CPU job—**do not** spend GPU allocation on this step. |

Paths below use **placeholders**. Set them to your Curator checkout, dataset roots, and scratch/work directories (e.g. `$SLURM_SUBMIT_DIR`, `$SCRATCH`, project NFS).

```bash
# Example layout (adjust for your site)
export CURATOR_ROOT=/path/to/Curator
export NEMO_ROOT=/path/to/NeMo                    # optional; needed on PYTHONPATH for some installs
export DATA_ROOT=/path/to/nemo_tarred_dataset   # manifests + tar shards
export WORK_ROOT=/path/to/work_outputs          # embeddings + annotated manifests
export SCRIPTS_DIR=/path/to/curator_spk_id_scripts   # optional: host copy of shell launchers
export PYTHONPATH="${NEMO_ROOT}:${CURATOR_ROOT}:${PYTHONPATH:-}"
```

Reference shell launchers (same logic as this doc) are maintained alongside many projects as:

- `run_speaker_id_titanet.sh` — multi-GPU extraction
- `run_speaker_clustering.sh` — CPU clustering + manifest annotation

Copy or symlink them into `${SCRIPTS_DIR}` and **edit** the internal `BASE_DIR`, `MANIFEST_PREFIX`, `TAR_PREFIX`, and shard range (`SHARD_START` / `SHARD_END`) to match `${DATA_ROOT}`.

### Phase 1 — embedding extraction (GPUs)

**Launcher example** (multi-GPU; each worker runs `run_pipeline.py --direct`):

```bash
# After customizing paths inside run_speaker_id_titanet.sh (or export overrides):
export OUTPUT_DIR="${WORK_ROOT}/embeddings"
bash "${SCRIPTS_DIR}/run_speaker_id_titanet.sh" \
    --devices 0,1 \
    --batch_size 64 \
    --output_dir "${OUTPUT_DIR}"
```

Pass-through flags (e.g. `--max_cuts 500`) are forwarded to `run_pipeline.py`.

**Equivalent single-GPU command** (no wrapper):

```bash
CUDA_VISIBLE_DEVICES=0 python "${CURATOR_ROOT}/tutorials/audio/speaker_id/run_pipeline.py" \
    --direct \
    --input_manifest "${DATA_ROOT}/manifest__OP_0..49_CL_.json" \
    --input_tar "${DATA_ROOT}/audio__OP_0..49_CL_.tar" \
    --lhotse_mode nemo_tarred \
    --output_dir "${WORK_ROOT}/embeddings" \
    --output_format npz \
    --model_name nvidia/speakerverification_en_titanet_large \
    --batch_size 64
```

The multi-GPU wrapper:

1. Splits the shard index range across GPUs (e.g. GPU 0 → shards 0–24, GPU 1 → 25–49).
2. Launches one `python …/run_pipeline.py --direct` per GPU (background jobs).
3. Writes disjoint `embeddings_<shard>.npz` into the same `--output_dir`.
4. Waits for all workers and prints a file summary.

### Phase 2 — clustering (CPU only)

Reads the **same** manifest pattern as extraction plus all `embeddings_*.npz` under `--embedding_dir`. Writes new JSONL manifests with `speaker_label` and `confidence_score`.

**Launcher example:**

```bash
# CPU node / login node — no GPU required
bash "${SCRIPTS_DIR}/run_speaker_clustering.sh" \
    --embedding_dir "${WORK_ROOT}/embeddings" \
    --output_manifest_dir "${WORK_ROOT}/annotated_manifests"
```

Optional overrides (also as environment variables in the reference script):

| Override | Default | Meaning |
|----------|---------|---------|
| `--threshold` / `CLUSTER_THRESHOLD` | `0.292` | Cosine-similarity cutoff for AHC (TitaNet + batch mean, EER-aligned on a local VoxCeleb subset; raise toward `0.35`–`0.40` for fewer false merges). |
| `--embedding_normalization` / `EMBEDDING_NORM` | `center_global` | Subtract mean of embeddings in the clustered batch before cosine (`none` \| `center_global` \| `external` + `.npy` stats). |
| `--embedding_dir` | _(set in script)_ | Directory containing `embeddings_*.npz`. |
| `--output_manifest_dir` | _(set in script)_ | Where annotated manifests are written. |

**Equivalent direct `run_pipeline.py` call:**

```bash
python "${CURATOR_ROOT}/tutorials/audio/speaker_id/run_pipeline.py" \
    --cluster \
    --input_manifest "${DATA_ROOT}/manifest__OP_0..49_CL_.json" \
    --embedding_dir "${WORK_ROOT}/embeddings" \
    --output_manifest_dir "${WORK_ROOT}/annotated_manifests" \
    --threshold 0.292 \
    --embedding_normalization center_global \
    --linkage_method average
```

The default `0.292` threshold and TitaNet-vs-WeSpeaker context are documented in `tutorials/audio/speaker_id/TITANET_VS_WESPKResNet_benchmark.md` (in this folder).

## Speaker-ID confidence score

Every annotated manifest line gets a `confidence_score` in `[0, 1]` alongside
`speaker_label`. It is a **silhouette-style score evaluated in cosine-similarity
space** — a cosine-domain analogue of
[`sklearn.metrics.silhouette_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html),
implemented in `speaker_confidence()` inside
`nemo_curator/stages/audio/speaker_id/speaker_clustering_and_scoring.py`.

For each utterance `i` with assigned cluster `k` (after `--embedding_normalization`
and L2 normalization, so dot products are cosine similarities):

- **Cohesion `a(i)`** — mean cosine similarity between embedding `i` and the
  **other** members of its own cluster `k` (excludes the self-similarity of 1).
- **Separation `b(i)`** — for every rival cluster `k' ≠ k`, compute the mean
  cosine similarity between `i` and all members of `k'`; take the **maximum**
  (the nearest rival speaker).
- **Score**:
  `confidence_score(i) = clamp((a(i) − b(i)) / max(a(i), b(i)), 0, 1)`

Edge cases:

- **Singletons** (`|k| = 1`) get `confidence_score = 0.0` — there is no within-cluster
  similarity to estimate cohesion from.
- Utterances with no embedding match get `speaker_label = -1` and `confidence_score = 0.0`.
- When `max(a, b) ≤ 0` the score is set to `0.0`.

Relationship to the textbook silhouette:

| Aspect | sklearn silhouette | This score |
|--------|--------------------|-----------|
| Geometry | Euclidean distance `d` | Cosine **similarity** `s` |
| Per-sample `a` | Mean intra-cluster **distance** | Mean intra-cluster **similarity** (higher = tighter) |
| Per-sample `b` | Distance to **nearest** other cluster (min over means) | Similarity to **nearest** other cluster (**max** over means) |
| Formula | `(b − a) / max(a, b)` | `(a − b) / max(a, b)` |
| Range | `[−1, 1]` | Clamped to `[0, 1]` |
| Aggregation | Mean over samples | Reported **per utterance** (no global average) |

Sign flip and clamping notes:

- `a` and `b` are swapped vs. sklearn because in similarity space "good" means
  **larger** within-cluster value and **smaller** nearest-rival value — the
  opposite of distance space — so the numerator is `a − b`.
- Negative raw scores (utterance closer to a rival cluster than to its own)
  are clamped to `0`, since downstream filtering only cares about a
  monotonically increasing reliability signal.

Practical use — the score is intended as a **reliability filter** for downstream
ASR / speaker-conditioned training:

```python
import json

with open("annotated_manifest_0.json") as f:
    rows = [json.loads(l) for l in f]

reliable = [r for r in rows if r.get("confidence_score", 0.0) >= 0.2]
print(f"Reliable: {len(reliable)} / {len(rows)}")
```

A threshold around `0.2` typically retains the majority of utterances while
removing borderline / mis-clustered ones; tighten toward `0.4`–`0.5` if the
downstream task is highly sensitive to speaker-label noise.

## SCOTCH: large-scale clustering (BIRCH + AHC, 20–30M utterances)

`run_pipeline.py --cluster` runs the **standard** AHC backend
(`SpeakerClusteringStage`), which materialises an `N × N` cosine-similarity
matrix and is therefore practical only up to a few hundred thousand
utterances. For corpora in the **20–30M-utterance range** (LibriSpeech-train,
YODAS, full Granary, …) the tutorial also ships the **SCOTCH-v1.large_scale**
backend, which streams BIRCH leaves first and runs AHC over a few thousand
leaf centroids — peak memory drops by 3–4 orders of magnitude with no
accuracy regression on LibriSpeech.

> All parameter rationale, sweeps, plots, memory comparisons, and the
> silhouette-style `confidence_score` calibration live in
> [`PARAM_TUNE.md`](PARAM_TUNE.md). Read it before tweaking anything.

The SCOTCH preset is a versioned, self-documenting bundle of every knob the
backend exposes; the canonical preset `librispeech-2026-04` is the one
calibrated in `PARAM_TUNE.md`. Every SCOTCH run drops a `cluster_config.json`
sidecar next to its outputs that records *exactly* which preset and which
overrides produced each set of labels.

### Cluster-only run on CS-OCI-ORD or DRACO-OCI-IAD

The launcher `run_scotch_cluster_only.sh` auto-detects the cluster from the
hostname and resolves `DATA_ROOT` accordingly:

| Cluster | `DATA_ROOT` |
|---------|-------------|
| **CS-OCI-ORD** (`*cs-oci*`)       | `/lustre/fs11/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/ASR/librispeech/tarred_train` |
| **DRACO-OCI-IAD** (`*draco-oci*`) | `/lustre/fs12/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/ASR/librispeech/tarred_train` |

Typical chained invocation (extract + merge once, then cluster):

```bash
cd ${CURATOR_ROOT}/tutorials/audio/speaker_id

# Phase 1 + 2: GPU extract + CPU merge (writes embeddings_merged.npz under $WORK_DIR)
./run_tune_threshold_librispeech.sh extract merge

# Phase 3: SCOTCH cluster + annotate (CPU only; uses the same $WORK_DIR by default)
./run_scotch_cluster_only.sh
```

Common overrides (all via env vars):

```bash
PRESET=librispeech-2026-04 \
WORK_DIR=/lustre/fsw/portfolios/llmservice/users/${USER}/scotch_librispeech \
OUTPUT_DIR=${WORK_DIR}/scotch_speaker_clustering_results \
THRESHOLD=0.55 MIN_CLUSTER_SIZE=10 \
    ./run_scotch_cluster_only.sh
```

Outputs land in `$OUTPUT_DIR`:

| File | What's in it |
|------|--------------|
| `manifest_<sid>.json` | Each input shard manifest copied verbatim plus two extra fields per row: `speaker_label` (int, `-1` if dropped by `min_cluster_size`) and `confidence_score` (silhouette in `[0, 1]`; see [PARAM_TUNE.md §9](PARAM_TUNE.md)). |
| `clusters_summary.jsonl` | Flat `{audio_filepath, speaker_label, confidence_score}` index — one line per utterance, useful for `pandas.read_json(..., lines=True)`. |
| `cluster_config.json` | SCOTCH sidecar built by `build_cluster_config(...)`. Documents the preset, every effective parameter, and any CLI overrides. See [PARAM_TUNE.md §3](PARAM_TUNE.md). |

### Direct Python invocation (any host)

`scotch_cluster_and_annotate.py` is path-agnostic and works on any node where
`nemo_curator` is importable; the launcher is just a thin convenience layer
that wires up the cluster paths and `PYTHONPATH`. Run the driver yourself
when you need an unusual layout:

```bash
export PYTHONPATH=${CURATOR_ROOT}:${PYTHONPATH:-}

python ${CURATOR_ROOT}/tutorials/audio/speaker_id/scotch_cluster_and_annotate.py \
    --merged_npz   ${WORK_DIR}/embeddings/embeddings_merged.npz \
    --manifest_dir ${DATA_ROOT}/raw_sharded_manifests \
    --output_dir   ${WORK_DIR}/scotch_speaker_clustering_results \
    --preset       librispeech-2026-04
```

`python scotch_cluster_and_annotate.py --help` lists every overridable knob
(`--threshold`, `--linkage`, `--min_cluster_size`, `--birch_cosine_floor`,
`--branching_factor`, `--partial_fit_batch`, `--assign_tile`,
`--embedding_normalization`, `--no_confidence`). Every override is recorded
under `overrides` in `cluster_config.json` so the sidecar always reflects
what actually ran.

### Out-of-memory? Tune these knobs first

The SCOTCH preset is sized for a **256 GB CPU node clustering ~30M utterances**
of 192-dim TitaNet embeddings. If you OOM (or the node starts swapping), don't
panic — every stage has a single dominant memory term, and each one has a knob
you can dial down. Lower the knobs roughly **in the order below**, not all at
once: change one, re-run, see if it fits. The kernel `OOMKiller` log line
(`dmesg | tail`) usually tells you which stage exploded; map it via this
table:

| Stage (log prefix) | Dominant cost | Knob to lower | Rule of thumb |
|--------------------|---------------|---------------|---------------|
| `Stage 1: BIRCH partial_fit` | `partial_fit_batch × D × 4 B` per call **+** the CF-tree (`n_subclusters × D × 4 B`) | `--partial_fit_batch` (preset 50000 → try **20000** → **10000**) | Halving it ~halves the per-call peak with no accuracy impact, just slightly slower. |
| `Stage 1: BIRCH partial_fit` (CF-tree blowing up; "leaf subclusters" log line in millions) | Number of BIRCH leaves grows when leaves are too tight | `--birch_cosine_floor` (preset 0.95 → try **0.92** → **0.90**) | Looser leaves = fewer of them. Stay **well above** `--threshold` (default 0.50) so you don't pre-merge distinct speakers. |
| `Stage 2: assigning ... to leaves` | `assign_tile × n_subclusters × 4 B` | `--assign_tile` (preset 16384 → try **8192** → **4096**) | Pure peak-RAM lever, zero accuracy effect, ~2× slower per halving. |
| `Stage 3: AHC on ... leaf centroids` (this is the **most common** SCOTCH OOM) | `n_subclusters² × 8 B` for the full distance matrix — quadratic! At 100k leaves that's **80 GB** just for `dist_mat`. | `--birch_cosine_floor` (preset 0.95 → try **0.92** → **0.90**) and/or `--branching_factor` (preset 50 → try **30**) | Both reduce `n_subclusters`. Stage 3 RAM scales with the **square** of the leaf count, so this is where modest looseness pays the biggest dividend. |
| `Stage 6: per-utterance confidence` | `assign_tile × K × 4 B` (K = surviving speakers) | `--assign_tile` (as above) or just `--no_confidence` | Skipping confidence is the cheap escape hatch if you only need labels. |

Three concrete recipes you can copy-paste:

```bash
# Recipe A -- "Tight node" (≤128 GB RAM, ~30M utts).  Halve the streaming
# buffer and the assignment tile.  Safe; no accuracy impact.
PARTIAL_FIT_BATCH=20000 ASSIGN_TILE=8192 \
    ./run_scotch_cluster_only.sh

# Recipe B -- "Stage-3 OOM" (kernel killed during AHC on leaf centroids).
# Loosen BIRCH leaves so there are fewer of them; this shrinks the n_sub^2
# distance matrix.  Still well above the 0.50 speaker decision threshold.
BIRCH_COSINE_FLOOR=0.92 BRANCHING_FACTOR=30 \
    ASSIGN_TILE=8192 PARTIAL_FIT_BATCH=20000 \
    ./run_scotch_cluster_only.sh

# Recipe C -- "Just give me labels, I'll filter later."  Skip the silhouette
# pass entirely (saves the Stage-6 buffer + the per-utt loop).
NO_CONFIDENCE=1 ASSIGN_TILE=8192 \
    ./run_scotch_cluster_only.sh
```

A few more rules of thumb that have saved us debugging time:

- **The merged NPZ itself is ~22 GB at N=30M, D=192.** That's a `mmap`-able
  hard floor — there is no SCOTCH knob that helps below that. If your node
  has less RAM than `4 × N × D` bytes, split the corpus into language /
  source shards and cluster each shard independently.
- **Don't lower `--threshold`.** It's the cosine speaker-decision cutoff and
  governs *accuracy*, not memory. Lowering it merges more speakers, which
  is a quality regression, not a memory fix.
- **`--min_cluster_size` is also not a memory knob.** It only filters labels
  *after* clustering finishes; the OOM has already happened by then.
- **Sidecar tells the truth.** Whatever combination you ran with, the
  effective values land in `cluster_config.json` under `parameters` (and
  any deviations from the preset under `overrides`). Diff two sidecars to
  see exactly what changed between a fitting run and an OOM run.
- **If you're still stuck**, run on a smaller subset first
  (`--max_cuts 500000` during `--direct` extraction) and grow from there;
  the SCOTCH cost model is monotonic in `N`, so a working 500k run gives
  you a solid lower bound on what your full corpus will need.

### When to use which clustering path

| Backend | Driver | Capacity | When to use |
|---------|--------|----------|-------------|
| **Standard AHC** (`N × N` cosine matrix) | `run_pipeline.py --cluster` | up to ~500k utts (RAM-bound) | small datasets, ablations, when you want the legacy self-cosine `confidence_score`. |
| **SCOTCH-v1.large_scale** (BIRCH + AHC) | `scotch_cluster_and_annotate.py` (or `run_scotch_cluster_only.sh`) | tens of millions of utts | LibriSpeech-train, YODAS, full Granary, anything where the standard AHC OOMs. |

## CLI reference (`run_pipeline.py`)

### Embedding / Ray / merge

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_manifest` | _(required for direct/cluster)_ | NeMo manifest pattern (brace-expand) |
| `--input_tar` | `""` | NeMo tarred audio pattern |
| `--output_dir` | `embeddings` | Output directory for per-shard files |
| `--output_format` | `npz` | `npz` or `pt` |
| `--model_name` | `nvidia/speakerverification_en_titanet_large` | Pretrained speaker model |
| `--batch_size` | `64` | Inference batch size per GPU |
| `--max_cuts` | `None` | Cap utterances (debug) |
| `--direct` | off | Bypass Ray; run embedding stage on current GPU |
| `--merge` | off | Merge per-shard files in `output_dir` |
| `--lhotse_mode` | `nemo_tarred` | `nemo_tarred`, `lhotse_shar`, or `nemo_row` |

### Clustering (`--cluster`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--embedding_dir` | `""` | Directory with `embeddings_*.npz` (if empty, falls back to `--output_dir`) |
| `--output_manifest_dir` | `output_manifests` | Annotated JSONL output directory |
| `--threshold` | `0.292` | Cosine threshold for AHC |
| `--embedding_normalization` | `center_global` | `none`, `center_global`, or `external` |
| `--external_norm_mean_npy` | `""` | `cohort_mean.npy` for `external` mode |
| `--external_norm_std_npy` | `""` | Optional `cohort_std.npy` |
| `--norm_eps` | `1e-8` | Epsilon added to std before division |
| `--linkage_method` | `average` | `average`, `complete`, or `single` |
| `--shard_level_clustering` | off | Cluster each shard independently (faster; IDs not global) |

## Known issues and fixes

Issues encountered while integrating with NeMo Curator and NeMo ASR.

### 1. `cosmos_xenna` import crash in NeMo Curator

`nemo_curator/__init__.py` unconditionally imports
`cosmos_xenna.ray_utils.cluster`, which is not installed in standard
environments.

**Fix**: Wrapped the import in `try/except` so the package loads without Xenna.

### 2. Wrong import path for `RayDataExecutor`

The original tutorial imported from `nemo_curator.backends.experimental.ray_data`,
but the module has moved to `nemo_curator.backends.ray_data`.

**Fix**: Updated the import path.

### 3. `_EmptyTask()` missing constructor arguments

`ProcessingStage.process()` was expected to return `_EmptyTask()`, but
`_EmptyTask` is a dataclass requiring `task_id`, `dataset_name`, and `data`.

**Fix**: Return the pre-built `EmptyTask` singleton instead of constructing a
new instance.

### 4. NeMo `LazyNeMoTarredIterator` shard-index bug

`LazyNeMoTarredIterator.__iter__` indexes `self.paths[sid]` where `sid` is the
literal shard ID from the manifest (e.g., 25). Since `self.paths` is a
0-indexed list, this crashes with `IndexError` when the shard range doesn't
start at 0.

**Fix**: Added a `self.shard_id_to_path` dict in `__init__` and changed the
lookup to `self.shard_id_to_path[sid]`.

### 5. NeMo brace-expand pattern mismatch (`_OP_` vs `__OP_`)

NeMo's `expand_sharded_filepaths` replaces `_OP_` (4 chars) with `{`. Using
the double-underscore `__OP_` variant eats the filename's trailing underscore,
producing `manifest0.json` instead of `manifest_0.json` → `FileNotFoundError`.

**Fix**: Updated `_expand_nemo_path()` to replace `_OP_` (matching `(`, `[`,
`<`) consistent with NeMo's convention.
