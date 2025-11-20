#!/bin/bash

# Script to compare file sizes and contents between Megatron and Curator tokenization pipelines
# Tests 4 different tokenizers with append_eod flag on and off

# Color codes for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Base directories
BASE_DIR=$1
MEGATRON_DIR="${BASE_DIR}/megatron-tokens"
CURATOR_DIR="${BASE_DIR}/curator-tokens"

# Tokenizers to test
TOKENIZERS=(
    "openai-community_gpt2"
    "unsloth_Meta-Llama-3_1-70B-Instruct-bnb-4bit"
    "unsloth_Mistral-Nemo-Instruct-2407-bnb-4bit"
    "unsloth_Qwen3-8B-bnb-4bit"
)

# Function to format bytes to human-readable format
format_bytes() {
    local bytes=$1
    if [ $bytes -ge 1073741824 ]; then
        echo "$(awk "BEGIN {printf \"%.2f GB\", $bytes/1073741824}")"
    elif [ $bytes -ge 1048576 ]; then
        echo "$(awk "BEGIN {printf \"%.2f MB\", $bytes/1048576}")"
    elif [ $bytes -ge 1024 ]; then
        echo "$(awk "BEGIN {printf \"%.2f KB\", $bytes/1024}")"
    else
        echo "${bytes} B"
    fi
}

# Function to calculate percentage difference
calc_diff_percent() {
    local size1=$1
    local size2=$2
    if [ $size1 -eq 0 ]; then
        echo "N/A"
    else
        echo "$(awk "BEGIN {printf \"%.2f%%\", (($size2-$size1)/$size1)*100}")"
    fi
}

# Function to compare files for a specific tokenizer configuration
compare_files() {
    local tokenizer=$1
    local append_eod=$2

    # Construct the directory names
    if [ "$append_eod" = "true" ]; then
        local dir_name="${tokenizer}_append_eod"
        local flag_status="WITH append_eod"
    else
        local dir_name="${tokenizer}"
        local flag_status="WITHOUT append_eod"
    fi

    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${YELLOW}Tokenizer: ${tokenizer}${NC}"
    echo -e "${YELLOW}Status: ${flag_status}${NC}"
    echo -e "${BLUE}========================================${NC}"

    # File paths
    local megatron_bin="${MEGATRON_DIR}/${dir_name}/text_text_document.bin"
    local megatron_idx="${MEGATRON_DIR}/${dir_name}/text_text_document.idx"

    # Dynamically get curator file ID from the directory
    local curator_file_id=$(ls "${CURATOR_DIR}/${dir_name}"/*.bin 2>/dev/null | head -1 | xargs basename | sed 's/\.bin$//')
    local curator_bin="${CURATOR_DIR}/${dir_name}/${curator_file_id}.bin"
    local curator_idx="${CURATOR_DIR}/${dir_name}/${curator_file_id}.idx"

    # Check if files exist
    if [ ! -f "$megatron_bin" ] || [ ! -f "$megatron_idx" ] || \
       [ ! -f "$curator_bin" ] || [ ! -f "$curator_idx" ]; then
        echo -e "${RED}ERROR: Some files are missing!${NC}"
        [ ! -f "$megatron_bin" ] && echo -e "${RED}  Missing: $megatron_bin${NC}"
        [ ! -f "$megatron_idx" ] && echo -e "${RED}  Missing: $megatron_idx${NC}"
        [ ! -f "$curator_bin" ] && echo -e "${RED}  Missing: $curator_bin${NC}"
        [ ! -f "$curator_idx" ] && echo -e "${RED}  Missing: $curator_idx${NC}"
        return 1
    fi

    # Get file sizes
    local meg_bin_size=$(stat -c%s "$megatron_bin")
    local meg_idx_size=$(stat -c%s "$megatron_idx")
    local cur_bin_size=$(stat -c%s "$curator_bin")
    local cur_idx_size=$(stat -c%s "$curator_idx")

    # Calculate differences
    local bin_diff=$((cur_bin_size - meg_bin_size))
    local idx_diff=$((cur_idx_size - meg_idx_size))
    local bin_diff_percent=$(calc_diff_percent $meg_bin_size $cur_bin_size)
    local idx_diff_percent=$(calc_diff_percent $meg_idx_size $cur_idx_size)

    # Display results
    echo ""
    echo -e "${GREEN}BIN File Comparison:${NC}"
    printf "  %-20s %15s (%s)\n" "Megatron:" "$meg_bin_size bytes" "$(format_bytes $meg_bin_size)"
    printf "  %-20s %15s (%s)\n" "Curator:" "$cur_bin_size bytes" "$(format_bytes $cur_bin_size)"
    printf "  %-20s %15s (%s)\n" "Difference:" "$bin_diff bytes" "$(format_bytes ${bin_diff#-})"
    printf "  %-20s %15s\n" "Diff %:" "$bin_diff_percent"

    if [ $meg_bin_size -eq $cur_bin_size ]; then
        echo -e "  ${GREEN}✓ Sizes are IDENTICAL${NC}"
    elif [ $meg_bin_size -gt $cur_bin_size ]; then
        echo -e "  ${YELLOW}⚠ Curator is SMALLER${NC}"
    else
        echo -e "  ${YELLOW}⚠ Curator is LARGER${NC}"
    fi

    echo ""
    echo -e "${GREEN}IDX File Comparison:${NC}"
    printf "  %-20s %15s (%s)\n" "Megatron:" "$meg_idx_size bytes" "$(format_bytes $meg_idx_size)"
    printf "  %-20s %15s (%s)\n" "Curator:" "$cur_idx_size bytes" "$(format_bytes $cur_idx_size)"
    printf "  %-20s %15s (%s)\n" "Difference:" "$idx_diff bytes" "$(format_bytes ${idx_diff#-})"
    printf "  %-20s %15s\n" "Diff %:" "$idx_diff_percent"

    if [ $meg_idx_size -eq $cur_idx_size ]; then
        echo -e "  ${GREEN}✓ Sizes are IDENTICAL${NC}"
    elif [ $meg_idx_size -gt $cur_idx_size ]; then
        echo -e "  ${YELLOW}⚠ Curator is SMALLER${NC}"
    else
        echo -e "  ${YELLOW}⚠ Curator is LARGER${NC}"
    fi

    # Binary content comparison
    echo ""
    echo -e "${GREEN}Binary Content Comparison:${NC}"

    # Compare BIN files
    echo -n "  BIN files: "
    if cmp -s "$megatron_bin" "$curator_bin"; then
        echo -e "${GREEN}✓ Contents are IDENTICAL${NC}"
    else
        echo -e "${RED}✗ Contents DIFFER${NC}"
        # Show where the first difference occurs
        local first_diff=$(cmp -l "$megatron_bin" "$curator_bin" 2>/dev/null | head -1)
        if [ -n "$first_diff" ]; then
            local byte_pos=$(echo "$first_diff" | awk '{print $1}')
            echo -e "    ${RED}First difference at byte: $byte_pos${NC}"
        fi
    fi

    # Compare IDX files
    echo -n "  IDX files: "
    if cmp -s "$megatron_idx" "$curator_idx"; then
        echo -e "${GREEN}✓ Contents are IDENTICAL${NC}"
    else
        echo -e "${RED}✗ Contents DIFFER${NC}"
        # Show where the first difference occurs
        local first_diff=$(cmp -l "$megatron_idx" "$curator_idx" 2>/dev/null | head -1)
        if [ -n "$first_diff" ]; then
            local byte_pos=$(echo "$first_diff" | awk '{print $1}')
            echo -e "    ${RED}First difference at byte: $byte_pos${NC}"
        fi
    fi
}

# Main execution
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Tokenization Pipeline Comparison (Size & Content)         ║${NC}"
echo -e "${BLUE}║  Megatron vs Curator                                       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"

# Iterate through each tokenizer with and without append_eod
for tokenizer in "${TOKENIZERS[@]}"; do
    # Test without append_eod
    compare_files "$tokenizer" "false"

    # Test with append_eod
    compare_files "$tokenizer" "true"
done

echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}Comparison complete!${NC}"
echo -e "${BLUE}========================================${NC}"
