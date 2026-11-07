#!/usr/bin/env bash
# download_and_prepare.sh
#
# Downloads the otoSpeech-full-duplex-processed-141h dataset from HuggingFace,
# unpacks the WebDataset tar shards, resamples audio to 16kHz, and updates
# the lhotse manifest recording paths to point to the new location.
#
# Requirements: git-lfs, python3 (with datasets library), sox or ffmpeg
# Usage: bash download_and_prepare.sh <LHOTSE_MANIFESTS_DIR> [INSTALL_DIR]

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash download_and_prepare.sh <LHOTSE_MANIFESTS_DIR> [INSTALL_DIR]"
    echo ""
    echo "  LHOTSE_MANIFESTS_DIR  Path to the lhotse_manifests directory (from Google Drive)"
    echo "  INSTALL_DIR           Where to download and unpack the dataset (default: cwd)"
    exit 1
fi

REPO="otoearth/otoSpeech-full-duplex-processed-141h"
MANIFEST_DIR="$(cd "$1" && pwd)"
INSTALL_DIR="${2:-$(pwd)}"
UNPACKED_DIR="${INSTALL_DIR}/unpacked"
RESAMPLED_DIR="${INSTALL_DIR}/unpacked_16kHz"
TARGET_SR=16000
NUM_JOBS="${NUM_JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

# Preflight checks
for cmd in ffmpeg python3 hf; do
    if ! command -v "${cmd}" &>/dev/null; then
        echo "Error: ${cmd} not found. Please install it before running this script."
        exit 1
    fi
done

# Validate manifest dir
if [ ! -f "${MANIFEST_DIR}/recordings.jsonl" ] && [ ! -f "${MANIFEST_DIR}/recordings.jsonl.gz" ]; then
    echo "Error: No recordings.jsonl or recordings.jsonl.gz found in ${MANIFEST_DIR}"
    exit 1
fi

echo "=== otoSpeech-full-duplex-processed-141h Dataset Preparation ==="
echo "Manifest dir : ${MANIFEST_DIR}"
echo "Install dir  : ${INSTALL_DIR}"
echo "Parallel jobs: ${NUM_JOBS}"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Download the dataset from HuggingFace
# ---------------------------------------------------------------------------
echo "[1/5] Downloading dataset from HuggingFace..."

hf download "${REPO}" \
    --repo-type dataset \
    --local-dir "${INSTALL_DIR}" \
    --include "data/train/*.tar"

echo "Download complete."

# ---------------------------------------------------------------------------
# Step 2: Unpack the WebDataset tar shards
# ---------------------------------------------------------------------------
echo ""
echo "[2/5] Unpacking WebDataset tar shards..."

mkdir -p "${UNPACKED_DIR}"

for tarfile in "${INSTALL_DIR}"/data/train/*.tar; do
    [ -f "${tarfile}" ] || continue
    echo "  Extracting: $(basename "${tarfile}")"
    # WebDataset tars contain files like {key}.flac and {key}.json
    # Extract only flac and json, flatten into unpacked dir
    tar xf "${tarfile}" -C "${UNPACKED_DIR}" --strip-components=0 2>/dev/null || \
    tar xf "${tarfile}" -C "${UNPACKED_DIR}" 2>/dev/null
done

# Some WebDataset formats nest files — flatten if needed
# Move any nested flac/json to top level
find "${UNPACKED_DIR}" -mindepth 2 -name "*.flac" -exec mv {} "${UNPACKED_DIR}/" \;
find "${UNPACKED_DIR}" -mindepth 2 -name "*.json" -exec mv {} "${UNPACKED_DIR}/" \;
# Clean up empty subdirs
find "${UNPACKED_DIR}" -mindepth 1 -type d -empty -delete 2>/dev/null || true

TOTAL_FILES=$(find "${UNPACKED_DIR}" -name "*.flac" | wc -l | tr -d ' ')
echo "Unpacked ${TOTAL_FILES} FLAC files."

# ---------------------------------------------------------------------------
# Step 3: Resample audio to 16kHz
# ---------------------------------------------------------------------------
echo ""
echo "[3/5] Resampling audio to ${TARGET_SR} Hz..."

mkdir -p "${RESAMPLED_DIR}"

resample_file() {
    local src="$1"
    local dst_dir="$2"
    local target_sr="$3"
    local basename
    basename=$(basename "${src}")
    local dst="${dst_dir}/${basename}"

    # Skip if already resampled and non-empty
    if [ -f "${dst}" ] && [ -s "${dst}" ]; then
        return 0
    fi

    # Remove any corrupt/empty leftover
    rm -f "${dst}"

    if ! ffmpeg -hide_banner -loglevel error -y -i "${src}" -ar "${target_sr}" "${dst}"; then
        echo "WARNING: ffmpeg failed on ${src}" >&2
        rm -f "${dst}"
        return 1
    fi

    # Verify output is non-empty
    if [ ! -s "${dst}" ]; then
        echo "WARNING: empty output for ${src}" >&2
        rm -f "${dst}"
        return 1
    fi
}
export -f resample_file

# Use xargs for parallel resampling
find "${UNPACKED_DIR}" -name "*.flac" -print0 | \
    xargs -0 -P "${NUM_JOBS}" -I {} bash -c 'resample_file "$@"' _ {} "${RESAMPLED_DIR}" "${TARGET_SR}"

RESAMPLED_COUNT=$(find "${RESAMPLED_DIR}" -name "*.flac" | wc -l | tr -d ' ')
echo "Resampled ${RESAMPLED_COUNT} files to ${TARGET_SR} Hz."

# ---------------------------------------------------------------------------
# Step 4: Copy lhotse manifests and update recording paths
# ---------------------------------------------------------------------------
echo ""
echo "[4/5] Copying lhotse manifests and updating recording paths..."

LOCAL_MANIFEST_DIR="${INSTALL_DIR}/lhotse_manifests"
mkdir -p "${LOCAL_MANIFEST_DIR}"

# Copy all manifest files into install dir
cp -a "${MANIFEST_DIR}"/* "${LOCAL_MANIFEST_DIR}/"
echo "  Copied manifests to ${LOCAL_MANIFEST_DIR}"

RECORDINGS_FILE="${LOCAL_MANIFEST_DIR}/recordings.jsonl"

# Decompress if only .gz exists
if [ ! -f "${RECORDINGS_FILE}" ] && [ -f "${RECORDINGS_FILE}.gz" ]; then
    echo "  Decompressing recordings.jsonl.gz..."
    gzip -dk "${RECORDINGS_FILE}.gz"
fi

RECORDINGS_OUT="${RECORDINGS_FILE}.tmp"

python3 - "${RECORDINGS_FILE}" "${RECORDINGS_OUT}" "${RESAMPLED_DIR}" "${TARGET_SR}" <<'PYEOF'
import json
import os
import sys

recordings_in = sys.argv[1]
recordings_out = sys.argv[2]
resampled_dir = os.path.abspath(sys.argv[3])
target_sr = int(sys.argv[4])

count = 0
with open(recordings_in) as fin, open(recordings_out, "w") as fout:
    for line in fin:
        rec = json.loads(line)
        rec_id = rec["id"]

        # Rewrite source paths to local resampled directory
        for src in rec["sources"]:
            src["source"] = os.path.join(resampled_dir, f"{rec_id}.flac")

        # Update sampling rate and recompute num_samples if needed
        old_sr = rec["sampling_rate"]
        if old_sr != target_sr:
            ratio = target_sr / old_sr
            rec["sampling_rate"] = target_sr
            rec["num_samples"] = round(rec["num_samples"] * ratio)
            rec["duration"] = round(rec["num_samples"] / target_sr, 3)

        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        count += 1

print(f"Updated {count} recording entries.")
PYEOF

mv "${RECORDINGS_OUT}" "${RECORDINGS_FILE}"

# Recompress
if command -v gzip &>/dev/null; then
    gzip -kf "${RECORDINGS_FILE}"
    echo "Compressed to ${RECORDINGS_FILE}.gz"
fi

# ---------------------------------------------------------------------------
# Step 5: Verify manifest paths and audio integrity
# ---------------------------------------------------------------------------
echo ""
echo "[5/5] Verifying manifest recording paths and audio integrity..."

python3 - "${RECORDINGS_FILE}" <<'PYEOF'
import json
import os
import subprocess
import sys

recordings_file = sys.argv[1]
missing = []
corrupt = []
total = 0

with open(recordings_file) as f:
    for line in f:
        rec = json.loads(line)
        total += 1
        path = rec["sources"][0]["source"]

        if not os.path.isfile(path):
            missing.append(rec["id"])
            continue

        # Validate with ffprobe
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-i", path],
            capture_output=True, timeout=10,
        )
        if result.returncode != 0:
            corrupt.append(rec["id"])

if missing:
    print(f"ERROR: {len(missing)} recordings point to missing files:")
    for rid in missing[:10]:
        print(f"  {rid}")
    if len(missing) > 10:
        print(f"  ... and {len(missing) - 10} more")

if corrupt:
    print(f"ERROR: {len(corrupt)} recordings are corrupt (unreadable audio):")
    for rid in corrupt:
        print(f"  {rid}")

if missing or corrupt:
    sys.exit(1)
else:
    print(f"OK: All {total} recording paths and audio verified.")
PYEOF

echo ""
echo "=== Done ==="
echo "  Unpacked audio (44.1kHz): ${UNPACKED_DIR}"
echo "  Resampled audio (16kHz) : ${RESAMPLED_DIR}"
echo "  Lhotse manifests (local): ${LOCAL_MANIFEST_DIR}"
echo "  Recordings manifest     : ${RECORDINGS_FILE}"
