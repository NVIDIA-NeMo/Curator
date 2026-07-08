#!/bin/bash
# David AI redelivered MFA pipeline — RAM-disk variant.
#
# Identical to run_david_ai_mfa.sh, but Stage 2 runs stage2_mfa_align_ramdisk.py
# so ALL MFA scratch (segment .wav/.txt, corpus DB, per-worker MFA roots, MFA
# TextGrid output) lives on a tmpfs RAM disk (default /dev/shm) and is removed on
# exit. Only the final per-session TextGrids + alignments.jsonl are persisted.
#
# This is a thin wrapper that forces RAM_DISK=1 and delegates to the main
# orchestrator, so the two never drift out of sync.
#
# Usage (same env vars as run_david_ai_mfa.sh):
#   bash run_david_ai_mfa_ram.sh
#   WORKERS=16 FORCE=1 STAGE=2 STAGE_END=4 bash run_david_ai_mfa_ram.sh
#   RAM_DIR=/dev/shm bash run_david_ai_mfa_ram.sh   # override tmpfs mount

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RAM_DISK=1
exec bash "$SCRIPT_DIR/run_david_ai_mfa.sh" "$@"
