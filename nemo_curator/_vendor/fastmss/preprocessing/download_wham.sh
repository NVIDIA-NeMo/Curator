#!/bin/bash

# WHAM!48kHz Noise Dataset Download Script (Split Files Method)
# Downloads 74 split files and concatenates them into the final dataset

set -e

# Configuration
BASE_URL="https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham48khz_noise"
FINAL_ZIP="high_res_wham_cat.zip"
EXPECTED_MD5="b11cff68963f24acdefc64aa42766fa2"
TGT_DIR="./wham_noise"

mkdir -p $TGT_DIR

echo "WHAM!48kHz Noise Dataset Downloader"
echo "Will download 74 files (~1GB each) and create ${FINAL_ZIP} (68.1GB)"
echo

# Check dependencies
for cmd in curl md5sum; do
    if ! command -v $cmd &> /dev/null; then
        echo "Error: $cmd is required but not installed."
        exit 1
    fi
done

# Check if final file already exists and is valid
if [[ -f "$FINAL_ZIP" ]]; then
    echo "Checking existing file..."
    actual_md5=$(md5sum "$FINAL_ZIP" | cut -d' ' -f1)
    if [[ "$actual_md5" == "$EXPECTED_MD5" ]]; then
        echo "✓ File already exists and is valid!"
        exit 0
    fi
    echo "Existing file is corrupted, re-downloading..."
fi

# Generate suffixes (aa, ab, ..., cv for 74 files)
suffixes=()
for i in {a..c}; do
    for j in {a..z}; do
        suffixes+=("$i$j")
        [[ ${#suffixes[@]} -eq 74 ]] && break 2
    done
done

echo "Downloading 74 split files..."
failed=0

# Progress tracking setup
PROGRESS_FILE=".download_progress"
rm -f "$PROGRESS_FILE"
touch "$PROGRESS_FILE"

show_progress() {
    while true; do
        local completed=$(wc -l < "$PROGRESS_FILE" 2>/dev/null || echo "0")
        local percent=$((completed * 100 / 74))
        local bar_length=50
        local filled_length=$((percent * bar_length / 100))

        printf "\r["
        printf "%*s" $filled_length | tr ' ' '='
        printf "%*s" $((bar_length - filled_length)) | tr ' ' '-'
        printf "] %d/74 (%d%%) " $completed 74 $percent

        [[ $completed -eq 74 ]] && break
        sleep 0.5
    done
    echo
}

# Download function with progress tracking
download_part() {
    local i=$1
    local suffix="${suffixes[$i]}"
    local part_file="${TGT_DIR}/high_res_wham.zip.$suffix"
    local part_url="$BASE_URL/high_res_wham.zip.$suffix"

    if curl -s -L -C - -o "$part_file" "$part_url"; then
        echo "$part_file" >> "$PROGRESS_FILE"
        return 0
    else
        echo "FAILED:$part_file" >> "$PROGRESS_FILE"
        return 1
    fi
}

# Export function and variables for parallel execution
export -f download_part
export BASE_URL suffixes PROGRESS_FILE

echo "Starting parallel downloads (max 8 concurrent)..."

# Start progress bar in background
show_progress &
progress_pid=$!

# Run downloads in parallel
if command -v parallel &> /dev/null; then
    parallel -j 8 download_part ::: "${!suffixes[@]}" >/dev/null 2>&1
else
    # Background jobs approach
    pids=()
    for i in "${!suffixes[@]}"; do
        download_part "$i" &
        pids+=($!)

        # Limit to 8 concurrent downloads
        if [[ ${#pids[@]} -eq 8 ]]; then
            wait "${pids[@]}"
            pids=()
        fi
    done
    # Wait for remaining jobs
    [[ ${#pids[@]} -gt 0 ]] && wait "${pids[@]}"
fi

# Stop progress bar
kill $progress_pid 2>/dev/null || true
wait $progress_pid 2>/dev/null || true

# Final progress update
completed=$(wc -l < "$PROGRESS_FILE")
progress_bar=$(printf "%50s" | tr ' ' '=')
echo "Progress: [$progress_bar] $completed/74 (100%)"

echo "Concatenating files..."
cat ${TGT_DIR}/high_res_wham.zip.* > "${TGT_DIR}/${FINAL_ZIP}"

echo "Verifying MD5..."
actual_md5=$(md5sum "$FINAL_ZIP" | cut -d' ' -f1)
if [[ "$actual_md5" == "$EXPECTED_MD5" ]]; then
    echo "✓ Download verified successfully!"
    echo "Cleaning up split files..."
    rm -f high_res_wham.zip.??
    echo "✓ Complete! File: $FINAL_ZIP (68.1GB)"
else
    echo "✗ MD5 verification failed!"
    echo "Expected: $EXPECTED_MD5"
    echo "Actual:   $actual_md5"
    exit 1
fi

unzip "${TGT_DIR}/$FINAL_ZIP"

# Cleanup progress file
rm -f "$PROGRESS_FILE"