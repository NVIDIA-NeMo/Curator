# David AI redelivered — MFA alignment pipeline

Segment-constrained [Montreal Forced Aligner (MFA)](https://montreal-forced-aligner.readthedocs.io/) pipeline for David AI redelivered English conversational audio. Each transcript segment is aligned independently, then results are concatenated into **per-session TextGrids** and downstream deliverables (Lhotse cutset, mixed-session RTTM).

## Input data layout

```
${DATA_ROOT}/${session_id}/
├── machine_generated_transcript.json   # segment list with speaker, start, end, text
├── ${speaker_id}_postprocessed.wav     # one full-track WAV per speaker
└── ...
```

Segment `start` / `end` times are on a **shared session timeline** (multi-speaker).

## Quick start

```bash
cd tutorials/audio/david_ai_redelivered_mfa

# Install Python deps (Curator venv recommended)
pip install -r requirements.txt

# MFA must be on PATH (conda env with montreal-forced-aligner)
export PATH="$HOME/miniconda3/envs/curator_pain_1/bin:$PATH"
export MFA_ROOT_DIR="$HOME/MFA_models"

# Full pipeline (stages 0–7)
bash run_david_ai_mfa.sh

# Re-run alignment + outputs only
FORCE=1 STAGE=2 STAGE_END=7 bash run_david_ai_mfa.sh

# Single session
SESSION=<session_id> FORCE=1 STAGE=0 STAGE_END=7 bash run_david_ai_mfa.sh
```

### RAM-disk variant (recommended for large runs)

Stage 2 scratch (segment WAV/TXT, MFA DB, copied models) lives on **tmpfs** (`/dev/shm`) and is removed on exit. Final TextGrids and `alignments.jsonl` are still written to disk.

```bash
WORKERS=16 FORCE=1 STAGE=2 STAGE_END=7 bash run_david_ai_mfa_ram.sh
```

Equivalent: `RAM_DISK=1 bash run_david_ai_mfa.sh`

## Prerequisites

| Tool | Used in |
|------|---------|
| **ffmpeg** | Opus encode (stage 1), segment extract for MFA (stage 2), mixed audio (stage 6) |
| **MFA** (`mfa align`, `mfa g2p`) | Stage 0b (lexicon), stage 2 (alignment) |
| **Python 3.10+** with `nemo-curator`, `lhotse`, `textgrid`, `num2words` | All stages |

Pretrained MFA models under `MFA_ROOT_DIR` (default `~/MFA_models`):

- Dictionary: `english_us_arpa`
- Acoustic: `english_us_arpa`
- G2P: `english_us_arpa`

**Parallel MFA tip:** symlink `command_history.yaml` to `/dev/null` to avoid YAML corruption when running many workers:

```bash
ln -sf /dev/null ~/MFA_models/command_history.yaml
```

## Pipeline stages

| Stage | Script | Description |
|-------|--------|-------------|
| **0** | `stage0_build_manifests.py` | Build per-session JSONL manifests; normalize transcript text |
| **0b** | `stage0_build_lexicon.py` | Merge MFA dictionary + G2P for OOV words |
| **1** | `stage1_resample_audio.py` | Encode per-speaker **16 kHz mono Opus** (`audio_16k/`) |
| **2** | `stage2_mfa_align_textgrids.py` or `stage2_mfa_align_ramdisk.py` | MFA per segment → session TextGrids + `alignments.jsonl` |
| **3** | `stage3_build_recording_rttm.py` | *(legacy)* per-recording RTTM — only if `WRITE_TEXTGRIDS=1` |
| **4** | `stage4_build_final_outputs.py` | Lhotse cutset + **session RTTM** for mixed audio |
| **5** | `stage5_merge_session_rttm.py` | *(legacy)* — skipped; stage 4 writes session RTTM |
| **6** | `stage6_mix_session_audio.py` | Per-speaker pause=white noise using **`audio_mixed/{session_id}.rttm`** (by `speaker_id`); mix → `{session_id}.opus` |
| **7** | `stage7_export_deliverables.py` | Index deliverables in `deliverables/manifest.jsonl` |

Orchestrator: `run_david_ai_mfa.sh` (stages controlled with `STAGE` / `STAGE_END`, default `0`–`7`).

## Outputs (`workdir/`)

| Artifact | Path | Notes |
|----------|------|-------|
| Manifests | `manifests/{session_id}.jsonl`, `*_norm.jsonl` | Raw + normalized segments |
| Per-speaker 16 kHz Opus | `audio_16k/{speaker}_{session}_postprocessed.opus` | Stored format for Lhotse |
| Session TextGrid (ordinary) | `textgrids/{session_id}.TextGrid` | MFA words + `speech` fallbacks |
| Session TextGrid (FastMSS) | `textgrids/{session_id}_fastmss.TextGrid` | MFA words only; gaps where MFA failed |
| Alignments cache | `alignments.jsonl` | One JSON line per session (words + per-recording breakdown) |
| Mixed session audio | `audio_mixed/{session_id}.opus` | 16 kHz sum of all speakers (Opus) |
| Session RTTM | `audio_mixed/{session_id}.rttm` | Speaker-labelled intervals on session timeline |
| Lhotse | `lhotse/david_ai_aligned_cuts.jsonl.gz` (+ recordings/supervisions) | Per-speaker cuts with word alignment |
| Deliverables index | `deliverables/manifest.jsonl`, `summary.json` | Pointers to all of the above |
| Logs | `logs/run_david_ai_mfa_*.log`, `normalization.jsonl`, `mfa_segment_fallback.jsonl` | |

`rttm/` and `rttm_sessions/` are **legacy** directories (empty unless `WRITE_TEXTGRIDS=1`). Current session RTTM lives next to mixed Opus in `audio_mixed/`.

## TextGrid variants

- **Ordinary** (`{session_id}.TextGrid`): MFA-aligned words plus `speech` placeholder intervals for segments where MFA failed (manifest boundaries used as fallback).
- **FastMSS** (`{session_id}_fastmss.TextGrid`): MFA words only; failed segments appear as empty gaps (suitable for Lhotse / FastMSS).

When MFA succeeds on all segments, the two files are identical.

## Text normalization (stage 0)

Transcripts are normalized for MFA lexicon compatibility:

1. Strip digit grouping commas (`2,000` → `2000`)
2. **Verbalize numbers** (`3` → `three`, `2020` → `two thousand and twenty`, decades, feet/inches, etc.)
3. Lowercase, map punctuation/symbols to spaces, keep `'` and `-`
4. Unknown tokens → `spn` (MFA unknown-word placeholder)

Whisper normalization is **not** used.

Re-run stage 0 after changing normalization: `FORCE=1 STAGE=0 STAGE_END=0 bash run_david_ai_mfa.sh`

## MFA and Opus

MFA requires **PCM WAV + text** in its corpus directory; it does not align Opus directly.

This pipeline stores per-speaker audio as **Opus** (smaller on disk). Stage 2 decodes each segment to temporary `seg_*.wav` via ffmpeg before calling `mfa align`. Temp files are deleted after each recording (or kept only on RAM disk during the run).

## Parallelism

| Level | Setting | Effect |
|-------|---------|--------|
| Sessions | `WORKERS` (default 4) | Parallel worker subprocesses in stage 2 |
| MFA jobs | `MFA_NUM_JOBS` (default 4) | Threads inside each `mfa align` call |
| Final outputs | `FINAL_WORKERS` (default 2) | Stage 4 Lhotse + RTTM build |

Total MFA concurrency ≈ `WORKERS × MFA_NUM_JOBS`. Size workers to CPU count and available `/dev/shm` when using the RAM-disk script.

Each worker uses an **isolated MFA root** (copied dictionary + acoustic model) to avoid SQLite / model races.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_ROOT` | `~/FastMSS/DavidAI/d12/subser_251spk` | Input sessions |
| `WORK_DIR` | `./workdir` | All outputs |
| `WORKERS` | `4` | Parallel session workers (stage 2) |
| `MFA_NUM_JOBS` | `4` | MFA `-j` per alignment |
| `FINAL_WORKERS` | `2` | Stage 4 parallelism |
| `SEGMENT_PADDING` | `0.5` | Seconds of audio context around each segment for MFA |
| `STAGE` / `STAGE_END` | `0` / `7` | Inclusive stage range |
| `FORCE` | `0` | Set `1` to re-run and clear `.done` markers |
| `SESSION` | *(empty)* | Process one session only |
| `RAM_DISK` | `0` | Set `1` to use RAM-disk stage 2 (or use `run_david_ai_mfa_ram.sh`) |
| `RAM_DIR` | `/dev/shm` | tmpfs mount for RAM-disk mode |
| `MFA_ROOT_DIR` | `~/MFA_models` | MFA pretrained models |
| `NUM2WORDS_LANG` | `en` | Digit verbalization language |
| `WRITE_TEXTGRIDS` | `0` | Set `1` to enable legacy stages 3/5 |

## Project layout

```
david_ai_redelivered_mfa/
├── run_david_ai_mfa.sh          # Main orchestrator
├── run_david_ai_mfa_ram.sh      # RAM-disk wrapper (RAM_DISK=1)
├── david_ai_common.py           # Shared helpers
├── stage0_build_manifests.py
├── stage0_build_lexicon.py
├── stage1_resample_audio.py
├── stage2_mfa_align_textgrids.py
├── stage2_mfa_align_ramdisk.py
├── stage2_mfa_worker.py         # Stage 2 worker subprocess
├── stage4_build_final_outputs.py
├── stage6_mix_session_audio.py
├── stage7_export_deliverables.py
├── requirements.txt
└── workdir/                     # Default output tree (gitignored in practice)
```

## Troubleshooting

**`mfa_temp/` not empty during a run** — Expected. Up to `WORKERS` active temp dirs hold the current recording's segment WAV/TXT while MFA runs. They are removed per recording when using default cleanup. `mfa_temp/workers/` retains copied models for the whole run (~1 GB).

**`mfa_temp/` not empty after a finished run** — `workers/` model copies persist in disk mode. Safe to delete: `rm -rf workdir/mfa_temp`. Use `run_david_ai_mfa_ram.sh` to avoid disk scratch entirely.

**Empty `text_norm` for filler segments** (`"Um..."`, `"..."`) — Normalization may produce empty strings; those segments are skipped for MFA.

**`spn` in normalized text** — Token not in MFA dictionary/alphabet; add to lexicon (stage 0b) or fix source transcript.
