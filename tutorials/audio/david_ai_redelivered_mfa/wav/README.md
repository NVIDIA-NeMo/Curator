# David AI on-the-fly MFA E2E — 16 kHz WAV outputs

This is an independent copy of the David AI E2E pipeline. It never creates Opus
files or a raw `audio_16k` directory.

For every raw session it:

1. Reads `machine_generated_transcript.json` and resolves one supported WAV per
   speaker (`_postprocess`, `_postprocessed`, ordinary `.wav`, then `_preprocessed`).
2. Normalizes transcript rows in memory.
3. Runs MFA with the base `english_us_arpa` dictionary and runtime G2P.
4. Writes the session RTTM.
5. Creates 16 kHz speaker WAVs in node-local scratch.
6. Replaces pauses outside original manifest boundaries ±0.5 seconds with white
   noise (`0.0002`, 5 ms smoothing).
7. Saves every masked speaker WAV to `audio_16k_masked` and filters the session
   RTTM into a matching per-speaker RTTM.
8. Mixes the masked tracks and saves one mono 16 kHz PCM WAV for the session.
9. Writes ordinary (MFA + fallback) and FastMSS TextGrids.
10. Validates outputs and writes `.done/sessions/<session>.done`.

Validated done flags provide resume behavior. Re-running the same command skips
completed sessions and processes only sessions without done flags.

## Run

```bash
DATA_ROOT=/path/to/raw/sessions \
WORK_DIR=/path/to/output \
WORKERS=16 \
MFA_NUM_JOBS=2 \
bash run_david_ai_mfa_ram_session.sh
```

For a multi-node SLURM run:

```bash
cd ..

VARIANT=wav \
DATA_ROOT=/shared/path/to/raw/sessions \
WORK_DIR=/shared/path/to/output \
NUM_NODES=8 \
WORKERS_PER_NODE=16 \
MFA_NUM_JOBS=2 \
bash cluster/run_multinode.sh
```

See `../cluster/README.md` for environment and scheduler options.

## Persistent outputs

```text
<work-dir>/
├── audio_16k_masked/
│   ├── <speaker>_<session>_postprocessed.wav
│   └── <speaker>_<session>_postprocessed.rttm
├── audio_mixed/
│   ├── <session>.wav       # mono 16 kHz PCM s16le
│   └── <session>.rttm
├── textgrids/
│   ├── <session>.TextGrid
│   ├── <session>_fastmss.TextGrid
│   ├── <recording>.TextGrid
│   └── <recording>_fastmss.TextGrid
├── .done/
│   └── sessions/<session>.done
└── logs/
```
