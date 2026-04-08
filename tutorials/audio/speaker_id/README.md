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

# Tutorial / CLI entry point
tutorials/audio/speaker_id/
  run_pipeline.py                   # --direct | --merge | --cluster | (default Ray pipeline)
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

The default `0.292` threshold and TitaNet vs WeSpeaker context are documented in companion notes such as `speaker_id_for_asr_data/TITANET_VS_WESPKResNet_benchmark.md` when that tree is checked out next to your project.

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
