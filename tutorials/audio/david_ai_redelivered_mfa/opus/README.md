# David AI on-the-fly MFA E2E

This directory contains the Opus pipeline. Every unfinished session is rebuilt
from its raw inputs:

1. Read `machine_generated_transcript.json` and resolve one supported WAV per speaker
   (`_postprocess`, `_postprocessed`, ordinary `.wav`, then `_preprocessed`).
2. Normalize transcript rows in memory; no normalized manifests are saved or read.
3. Align with the base `english_us_arpa` dictionary and runtime MFA G2P for OOV words.
4. Save session RTTM.
5. Replace pauses using original manifest boundaries protected by ±0.5 seconds.
   Noise amplitude is `0.0002`; boundary smoothing is 5 ms.
6. Save every masked per-speaker Opus and the mixed session Opus.
7. Save ordinary and FastMSS TextGrids at session and recording level.
8. Validate every required output and write `.done/sessions/<session>.done`.

The pipeline never reads shared lexicons, persisted manifests, cached RTTM,
previous alignment results, or partial mixed audio. Validated done flags are
used for resume: completed sessions are skipped and sessions without flags are
processed from raw inputs.

## Entrypoints

Local:

```bash
DATA_ROOT=/path/to/raw/sessions \
WORK_DIR=/path/to/output \
WORKERS=16 \
MFA_NUM_JOBS=2 \
bash run_david_ai_mfa_ram_session.sh
```

SLURM multi-node runs use the shared cluster launcher:

```bash
cd ..

VARIANT=opus \
DATA_ROOT=/shared/path/to/raw/sessions \
WORK_DIR=/shared/path/to/output \
NUM_NODES=8 \
WORKERS_PER_NODE=16 \
MFA_NUM_JOBS=2 \
bash cluster/run_multinode.sh
```

See `../cluster/README.md` for environment and scheduler options.

## Outputs

```text
<work-dir>/
├── audio_mixed/
│   ├── <session>.opus
│   ├── <session>.rttm
│   └── speakers/
│       └── <speaker>_<session>_postprocessed.opus
├── textgrids/
│   ├── <session>.TextGrid
│   ├── <session>_fastmss.TextGrid
│   ├── <recording>.TextGrid
│   └── <recording>_fastmss.TextGrid
├── .done/
│   └── sessions/
│       └── <session>.done
└── logs/
```

## Runtime controls

- `DATA_ROOT`, `WORK_DIR`: raw input and output roots.
- `WORKERS`: concurrent sessions.
- `MFA_NUM_JOBS`: MFA jobs per recording.
- `SEG_EXTRACT_WORKERS`: concurrent segment extraction per recording.
- `MIX_PREP_WORKERS`: concurrent speaker preparation per session.
- `RAM_DIR`: ephemeral node-local scratch.
- `FFMPEG_BIN`, `FFMPEG_TIMEOUT_S`: ffmpeg runtime controls.
- `MFA_ROOT_DIR`, `MFA_ENV`: MFA model and environment locations.
- `RAM_ARRAY_COUNT`: cluster scheduling shards.

There are deliberately no `FORCE`, `SKIP_LEXICON`, session-subset, persisted-manifest,
resume, stage, or cache flags.
