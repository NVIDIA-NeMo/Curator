# David AI redelivered MFA tutorial

This tutorial provides two independent, on-the-fly end-to-end pipelines for
multi-speaker David AI sessions:

- [`opus/`](opus/README.md): writes per-speaker and mixed-session Opus audio.
- [`wav/`](wav/README.md): writes masked per-speaker and mixed-session mono
  16 kHz PCM WAV audio.

Both variants:

1. Read raw source WAVs and `machine_generated_transcript.json`.
2. Normalize transcript text in memory.
3. Run Montreal Forced Aligner (MFA) with the `english_us_arpa` acoustic model,
   base dictionary, and runtime G2P for OOV words.
4. Generate session RTTM.
5. Preserve speech inside the original manifest boundaries ±0.5 seconds.
6. Replace pauses outside those boundaries with white noise at amplitude
   `0.0002`, using a 5 ms boundary crossfade.
7. Mix all speakers on the common session timeline.
8. Write ordinary TextGrids (MFA + fallback) and FastMSS TextGrids (MFA only).
9. Validate all required outputs before writing
   `.done/sessions/<session_id>.done`.

The pipelines do not reuse persisted normalized manifests, shared lexicons, or
partial RTTM/TextGrid/audio outputs. A validated session done flag is the only
resume state: sessions with a done flag are skipped, while unfinished sessions
are rebuilt from raw inputs.

## Data privacy

No David AI dataset content is included in this tutorial. All paths, IDs, and
transcript values in the documentation and tests are synthetic placeholders.

Do not commit or publish raw manifests, audio, RTTM, TextGrid, logs, archives,
or generated outputs. The tutorial `.gitignore` excludes these artifact types,
but generated changes must still be reviewed before committing.

## Requirements

### Supported environment

- Linux x86_64
- Python 3.11–3.13 (Python 3.12 is recommended and tested)
- ffmpeg and ffprobe
- Montreal Forced Aligner 3.3.9
- NeMo Curator from this repository
- Approximately 1–2 GB of node-local scratch per concurrent session

GPU access is not required for this pipeline.

### Required MFA models

The following English US ARPA models must be installed:

- Dictionary: `english_us_arpa`
- Acoustic model: `english_us_arpa`
- G2P model: `english_us_arpa`

The pipeline expects this layout under `$MFA_ROOT_DIR`:

```text
MFA_ROOT_DIR/
└── pretrained_models/
    ├── acoustic/english_us_arpa.zip
    ├── dictionary/english_us_arpa.dict
    └── g2p/english_us_arpa.zip
```

## Environment setup

From the repository root:

```bash
cd ~/Curator_my_fork

conda env create \
  -f tutorials/audio/david_ai_redelivered_mfa/environment.yml

conda activate david-ai-mfa

python -m pip install -e .
python -m pip install \
  -r tutorials/audio/david_ai_redelivered_mfa/requirements.txt
```

Download the MFA models:

```bash
export MFA_ROOT_DIR="$HOME/MFA_models"
mkdir -p "$MFA_ROOT_DIR"

mfa model download dictionary english_us_arpa
mfa model download acoustic english_us_arpa
mfa model download g2p english_us_arpa
```

Confirm the tools:

```bash
python --version
mfa version
ffmpeg -version
ffprobe -version
```

For development and tests:

```bash
python -m pip install \
  -r tutorials/audio/david_ai_redelivered_mfa/requirements-dev.txt
```

## Input layout

`DATA_ROOT` must contain one directory per session:

```text
<data-root>/
└── <session-id>/
    ├── machine_generated_transcript.json
    ├── <speaker-id-1>_postprocess.wav
    ├── <speaker-id-2>.wav
    └── ...
```

The transcript JSON must contain a `transcript` list:

```json
{
  "transcript": [
    {
      "text": "Example utterance.",
      "start": 1.25,
      "end": 2.85,
      "speaker": "<speaker-id-1>"
    }
  ]
}
```

Segment times use the shared session timeline. Speaker IDs must match the source
WAV filename prefixes.

For each speaker, the pipeline selects the first existing WAV in this order:

1. `<speaker-id>_postprocess.wav`
2. `<speaker-id>_postprocessed.wav`
3. `<speaker-id>.wav`
4. `<speaker-id>_preprocessed.wav`

The session fails explicitly if none of these files exists for a speaker named
in the transcript.

## Choose an output format

### Opus

```bash
cd ~/Curator_my_fork/tutorials/audio/david_ai_redelivered_mfa/opus

DATA_ROOT=/path/to/raw/sessions \
WORK_DIR=/path/to/opus-output \
MFA_ROOT_DIR="$HOME/MFA_models" \
MFA_ENV="$CONDA_PREFIX" \
WORKERS=16 \
MFA_NUM_JOBS=2 \
SEG_EXTRACT_WORKERS=8 \
bash run_david_ai_mfa_ram_session.sh
```

See [`opus/README.md`](opus/README.md) for the complete output layout.

### WAV

```bash
cd ~/Curator_my_fork/tutorials/audio/david_ai_redelivered_mfa/wav

DATA_ROOT=/path/to/raw/sessions \
WORK_DIR=/path/to/wav-output \
MFA_ROOT_DIR="$HOME/MFA_models" \
MFA_ENV="$CONDA_PREFIX" \
WORKERS=16 \
MFA_NUM_JOBS=2 \
SEG_EXTRACT_WORKERS=8 \
bash run_david_ai_mfa_ram_session.sh
```

The WAV variant writes:

```text
<work-dir>/
├── audio_16k_masked/
│   ├── <speaker>_<session>_postprocessed.wav
│   └── <speaker>_<session>_postprocessed.rttm
├── audio_mixed/
│   ├── <session>.wav
│   └── <session>.rttm
├── textgrids/
├── logs/
└── .done/sessions/
```

All WAV outputs are mono, 16 kHz, signed 16-bit PCM.

## Multi-node cluster run

Cluster submission scripts are kept separately in [`cluster/`](cluster/README.md).
They support both output variants through `VARIANT=opus` or `VARIANT=wav`.

Example:

```bash
VARIANT=wav \
DATA_ROOT=/shared/data/david_ai_sessions \
WORK_DIR=/shared/output/david_ai_wav \
MFA_ENV=/shared/envs/david-ai-mfa \
MFA_ROOT_DIR=/shared/models/MFA_models \
NUM_NODES=8 \
CPUS_PER_NODE=64 \
WORKERS_PER_NODE=16 \
MFA_NUM_JOBS=2 \
SLURM_ACCOUNT=my-account \
SLURM_PARTITION=cpu \
bash cluster/run_multinode.sh
```

The launcher uses one SLURM array task per node and exports only explicitly
required variables. It does not copy data to or from the cluster.

## Runtime configuration

| Variable | Default | Purpose |
|---|---:|---|
| `DATA_ROOT` | variant-specific example path | Raw session root |
| `WORK_DIR` | local variant workdir | Persistent output root |
| `SESSIONS_FILE` | unset | Optional absolute session-ID list |
| `MFA_ROOT_DIR` | `~/MFA_models` | MFA model root |
| `MFA_ENV` | `~/miniconda3/envs/curator_pain_1` | Environment containing MFA |
| `WORKERS` | `4` | Concurrent sessions |
| `MFA_NUM_JOBS` | `2` | MFA jobs per speaker recording |
| `SEG_EXTRACT_WORKERS` | `8` | Parallel ffmpeg segment extraction |
| `MIX_PREP_WORKERS` | number of session speakers | Parallel speaker masking |
| `RAM_DIR` | unique `/tmp` directory | Node-local ephemeral scratch |
| `FFMPEG_BIN` | `ffmpeg` from `PATH` | Explicit ffmpeg executable |
| `FFMPEG_TIMEOUT_S` | `600` | Per-ffmpeg timeout |

Approximate peak MFA parallelism is `WORKERS × MFA_NUM_JOBS`. Keep that value
near or below the available CPU count when other stages need CPU concurrently.

Scratch must be node-local (`/tmp` or equivalent), not a shared network
filesystem.

## Run a session subset

Create a text file containing one session directory name per line. Empty lines
and lines beginning with `#` are ignored:

```text
# validation subset
session-a
session-b
```

Pass it to either local variant:

```bash
SESSIONS_FILE=/absolute/path/to/sessions.txt \
DATA_ROOT=/path/to/raw/sessions \
WORK_DIR=/path/to/output \
bash run_david_ai_mfa_ram_session.sh
```

For a cluster run, pass the same shared absolute `SESSIONS_FILE` path to
`cluster/run_multinode.sh`. The subset is applied before deterministic sharding
and done-flag filtering.

## Success and restart behavior

A done flag is written only after all expected outputs for the current session
exist and are non-empty:

```text
<work-dir>/.done/sessions/<session-id>.done
```

Done flags are the parallel resume authority. Starting the same local or
multi-node command again skips sessions that already have a validated flag and
processes only sessions without one.

To intentionally rebuild a completed session, remove only its flag:

```bash
rm <work-dir>/.done/sessions/<session-id>.done
```

## Tests

Tests are separated by output variant:

```bash
cd ~/Curator_my_fork/tutorials/audio/david_ai_redelivered_mfa

pytest tests/opus
pytest tests/wav
pytest tests/cluster
```

Run lint checks:

```bash
ruff check opus wav tests
```

## Troubleshooting

### `mfa` not found

Activate the environment and confirm:

```bash
conda activate david-ai-mfa
which mfa
mfa version
```

### MFA model not found

Verify `$MFA_ROOT_DIR/pretrained_models/` matches the required model layout
above.

### ffmpeg not found

Install ffmpeg in the conda environment or set an explicit executable:

```bash
export FFMPEG_BIN=/path/to/ffmpeg
```

### A session has no done flag

Search its ID in `<work-dir>/logs/run_e2e_*.log`. A missing flag means the
session failed or at least one required output was missing/empty.

### Unexpectedly high memory or CPU use

Reduce `WORKERS`, `MFA_NUM_JOBS`, or `SEG_EXTRACT_WORKERS`. Start with:

```bash
WORKERS=4 MFA_NUM_JOBS=2 SEG_EXTRACT_WORKERS=4
```

## Security and cluster use

The local pipeline requires no outbound network access after dependencies and
MFA models are installed. Do not pass credentials, `.env` files, or unnecessary
shell environment variables to the pipeline.

Cluster runners should receive only explicitly required variables. This
tutorial does not require or perform data-copy operations to a cluster.
