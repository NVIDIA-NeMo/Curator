# Parakeet segment WER and FastMSS manifest pipeline

This Curator pipeline evaluates completed masked per-speaker WAVs with NVIDIA
Parakeet and creates filtered per-speaker training manifests.

## Flow

1. Read ground-truth segments from each session transcript manifest.
2. Resolve the matching `audio_16k_masked/<recording>.wav`.
3. Read recording-global words from `<recording>_fastmss.TextGrid`.
4. Extract each exact ground-truth segment into node-local scratch.
   Clips shorter than 100 ms are context-padded to 100 ms for stable Parakeet
   features; manifest start/end values remain unchanged.
5. Run `nvidia/parakeet-tdt-0.6b-v2` with batched GPU inference.
6. Normalize reference and hypothesis text identically.
7. Compute segment-level substitutions, deletions, insertions, and WER percent.
8. Delete the temporary segment WAV immediately after scoring.
9. Generate WER histograms, percentiles, a Tukey upper fence, and a proposed threshold.
10. Exclude segments above the applied threshold (100% by default).
11. Optionally exclude segments without FastMSS word alignment (enabled by default).
12. Write one filtered JSONL manifest per masked speaker recording.
13. Build exact-0%, 0–10%, and 0–100% Lhotse CutSets for the shard.

## Requirements

- Completed WAV MFA pipeline outputs:
  - `audio_16k_masked/*.wav`
  - `textgrids/*_fastmss.TextGrid`
- Raw session transcript manifests
- NVIDIA GPU
- `nvidia-ml-py` so Xenna can discover local GPUs
- NeMo ASR / Parakeet environment
- Curator installed from this repository

Install the additional environment dependencies:

```bash
python -m pip install -r parakeet_wer/requirements.txt
```

The Parakeet model may require approved outbound access on first use. Set
`MODEL_CACHE_DIR` to a persistent pre-populated model cache for offline runs.

## Run one shard

```bash
cd tutorials/audio/david_ai_redelivered_mfa/parakeet_wer

DATA_ROOT=/path/to/raw/sessions \
MFA_WORK_DIR=/path/to/completed/wav-mfa-workdir \
OUTPUT_DIR=/path/to/parakeet-wer-output \
SESSIONS_FILE=/path/to/session-list.txt \
PARAKEET_MODEL=nvidia/parakeet-tdt-0.6b-v2 \
ASR_BATCH_SIZE=16 \
WER_THRESHOLD_PCT=100 \
SHARD_COUNT=1 \
SHARD_INDEX=0 \
bash run_parakeet_wer.sh
```

For an array, set `SHARD_COUNT` to the total task count and
`SHARD_INDEX=$SLURM_ARRAY_TASK_ID`. Each shard writes to a separate output
directory.

## Multi-node cluster run

The cluster launcher submits one one-GPU array task per shard. It also submits
a dependent CPU job that runs only after the complete array succeeds, merging
the dataset-wide WER report and all three Lhotse CutSets:

```bash
DATA_ROOT=/shared/path/to/raw/sessions \
MFA_WORK_DIR=/shared/path/to/completed/wav-mfa-workdir \
OUTPUT_DIR=/shared/path/to/parakeet-wer-output \
ASR_ENV=/shared/path/to/parakeet-conda-env \
MODEL_CACHE_DIR=/shared/path/to/model-cache \
SESSIONS_FILE=/shared/path/to/session-list.txt \
NUM_NODES=8 \
MAX_CONCURRENT_NODES=8 \
SLURM_ACCOUNT=<account> \
SLURM_PARTITION=<gpu-partition> \
bash parakeet_wer/cluster/run_multinode.sh
```

All paths must be visible at the same absolute location on every node. The
launcher preloads the model once before submitting the array to prevent
concurrent cache writes. Set `PRELOAD_MODEL=0` only when the shared cache is
already complete. Optional Pyxis settings are `CONTAINER_IMAGE` and
`CONTAINER_MOUNTS`. The scripts never copy dataset outputs between hosts.

## Local two-GPU run

The two-GPU launcher pre-caches the model once and starts one Curator/Xenna
pipeline with both GPUs visible. The ASR stage requests one GPU per worker, so
Xenna schedules concurrent Parakeet workers inside one Ray cluster without
cross-cluster conflicts:

```bash
DATA_ROOT=/path/to/raw/sessions \
MFA_WORK_DIR=/path/to/completed/wav-mfa-workdir \
OUTPUT_DIR=/path/to/parakeet-wer-output \
SESSIONS_FILE=/path/to/session-list.txt \
GPU_IDS=0,1 \
ASR_BATCH_SIZE=16 \
WER_THRESHOLD_PCT=100 \
bash run_local_2gpu.sh
```

Logs:

```text
<output-dir>/logs/local_2gpu.log
```

Both GPU workers share a pre-populated read-only model cache. Curator manages
their worker and GPU isolation. The launcher sets `ASR_WORKERS=2`, and each ASR
worker requests exactly one GPU.

## Threshold selection

The default applied threshold keeps segments with:

```text
WER <= 100%
```

`wer_distribution.json` reports:

- min, max, and mean WER
- P25, P50, P75, P90, P95, and P99
- histogram counts
- Tukey upper fence: `P75 + 1.5 × (P75 - P25)`
- proposed threshold:

```text
min(100%, max(25%, P95, Tukey upper fence))
```

Review distributions across all shards before changing the production
threshold. To apply each shard's proposal automatically, set:

```bash
USE_RECOMMENDED_THRESHOLD=1
```

For consistent filtering across a dataset, prefer a single threshold selected
from the merged distribution.

After all shards finish, merge their WER distributions:

```bash
python analyze_wer_distribution.py \
  --output-dir /path/to/parakeet-wer-output \
  --applied-threshold-pct 100
```

This writes `wer_distribution_merged.json` without loading transcript or audio
files again.

## Outputs

```text
<output-dir>/shards/shard_<index>/
├── segments_with_wer.jsonl
├── wer_distribution.json
├── manifests/
│   └── <recording-id>.jsonl
└── lhotse/
    ├── wer_000/cuts.jsonl.gz
    ├── wer_000_010/cuts.jsonl.gz
    └── wer_000_100/cuts.jsonl.gz
```

Each filtered manifest row contains:

```json
{
  "audio_filepath": "/path/to/masked-speaker.wav",
  "offset": 1.25,
  "duration": 2.5,
  "text": "normalized ground truth",
  "text_raw": "Ground truth",
  "pred_text": "parakeet hypothesis",
  "wer_pct": 12.5,
  "session_id": "<session>",
  "speaker_id": "<speaker>",
  "recording_id": "<recording>",
  "segment_index": 0,
  "alignment": [
    {"symbol": "word", "start": 0.1, "duration": 0.3}
  ],
  "alignment_source": "fastmss_textgrid"
}
```

Alignment starts are relative to the segment offset. The audit JSONL also
retains absolute FastMSS word times and rejection reasons.

## Lhotse WER variants

Every pipeline shard automatically builds three nested Lhotse CutSets from its
completed audit JSONL. Disable this only for diagnostics with
`run_pipeline.py --no-build-lhotse`.

To rebuild a shard manually:

```bash
python build_lhotse_variants.py \
  --audit-jsonl <output-dir>/shards/shard_00000/segments_with_wer.jsonl \
  --output-dir <output-dir>/shards/shard_00000/lhotse
```

Outputs:

```text
<output-dir>/shards/shard_00000/lhotse/
├── wer_000/cuts.jsonl.gz       # exact 0% WER
├── wer_000_010/cuts.jsonl.gz   # 0–10% WER
└── wer_000_100/cuts.jsonl.gz   # 0–100% WER
```

All variants require non-empty FastMSS word alignment. Each `MonoCut` references
the full masked speaker WAV using its segment offset/duration, and its
supervision stores relative word alignments plus Parakeet text and WER metadata.

The dependent cluster merge job writes dataset-level variants to:

```text
<output-dir>/lhotse_merged/<variant>/cuts.jsonl.gz
```
