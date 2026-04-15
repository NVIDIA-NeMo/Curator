#!/bin/bash
# Auto-submit remaining yodas_1bw chunks as queue space opens.
# Run on login node: nohup bash auto_submit_1bw.sh &

WORK="${WORK_DIR:?Set WORK_DIR to speaker_id working directory}"

for i in 0 1 2 3 4 5 6 7 8; do
    SCRIPT="${WORK}/logs/yodas_1bw_ru_embed_${i}.sbatch"
    while true; do
        RESULT=$(sbatch "${SCRIPT}" 2>&1)
        if echo "$RESULT" | grep -q "Submitted"; then
            echo "$(date): chunk ${i} -> ${RESULT}"
            break
        fi
        # Wait for queue space
        sleep 60
    done
done

echo "$(date): All yodas_1bw chunks submitted."
