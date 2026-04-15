#!/bin/bash
# Check progress of speaker embedding extraction.
#
# Usage: bash check_status.sh

WORK_DIR="${WORK_DIR:?Set WORK_DIR to speaker_id working directory}"

CORPORA=(
    "ytc/ru|64"
    "yodas/0_from_captions/ru|256"
    "yodas/0_by_whisper/ru|1024"
    "yodas/1_by_whisper/ru|8192"
)

echo "=== Embedding Extraction Progress ==="
TOTAL_DONE=0
TOTAL_SHARDS=0

for corpus_def in "${CORPORA[@]}"; do
    IFS='|' read -r NAME NUM_SHARDS <<< "$corpus_def"
    EMB_DIR="${WORK_DIR}/embeddings/${NAME}"
    DONE=$(ls "${EMB_DIR}"/embeddings_*.npz 2>/dev/null | wc -l)
    TOTAL_DONE=$((TOTAL_DONE + DONE))
    TOTAL_SHARDS=$((TOTAL_SHARDS + NUM_SHARDS))
    PCT=$((DONE * 100 / NUM_SHARDS))
    printf "  %-30s %4d / %4d  (%3d%%)\n" "${NAME}" "${DONE}" "${NUM_SHARDS}" "${PCT}"
done
echo "  ---"
printf "  %-30s %4d / %4d\n" "TOTAL" "${TOTAL_DONE}" "${TOTAL_SHARDS}"

echo
echo "=== Clustering Progress ==="
for corpus_def in "${CORPORA[@]}"; do
    IFS='|' read -r NAME NUM_SHARDS <<< "$corpus_def"
    OUT_DIR="${WORK_DIR}/output_manifests/${NAME}"
    DONE=$(ls "${OUT_DIR}"/shard_*.jsonl 2>/dev/null | wc -l)
    printf "  %-30s %4d / %4d\n" "${NAME}" "${DONE}" "${NUM_SHARDS}"
done

echo
echo "=== Running Jobs ==="
squeue -u "$USER" -n "spkemb_*,spkclust_*" -o "%.10i %.20j %.8T %.10M %.6D %R" 2>/dev/null || echo "  (squeue not available)"
