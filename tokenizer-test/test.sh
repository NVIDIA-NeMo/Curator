DATA_ROOT=/localhome/local-asolergibert/datasets/tokenizer-test

TOKENIZERS=("unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit" "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit" "unsloth/Qwen3-8B-bnb-4bit" "openai-community/gpt2")
APPEND_EOD_CONFIGS=("True" "False")

TOKENIZE_MEGATRON=True
TOKENIZE_CURATOR=True

# Clone Megatron-LM
if [ ! -d "tokenizer-test/Megatron-LM" ]; then
    git clone https://github.com/NVIDIA/Megatron-LM.git tokenizer-test/Megatron-LM
fi

export PYTHONPATH=$(realpath tokenizer-test/Megatron-LM):$PYTHONPATH

# Create a sample text with all special tokens included
SPECIAL_TOKENS=("<s>" "</s>" "<unk>" "<pad>" "<|begin_of_text|>" "<|eot_id|>" "<|finetune_right_pad_id|>" "<|im_end|>" "<|vision_pad|>" "<|im_start|>" "<|im_end|>" "<|object_ref_start|>" "<|object_ref_end|>" "<|box_start|>" "<|box_end|>" "<|quad_start|>" "<|quad_end|>" "<|vision_start|>" "<|vision_end|>" "<|vision_pad|>" "<|image_pad|>" "<|video_pad|> <|endoftext|>")
# Shuffle the SPECIAL_TOKENS array and interleave into the lorem words at random positions
generate_lorem_with_special_tokens() {
    LOREM_WORDS=("lorem" "ipsum" "dolor" "sit" "amet" "consectetur" "adipiscing" "elit" "sed" "do" "eiusmod" "tempor" "incididunt" "ut" "labore" "et" "dolore" "magna" "aliqua" "ut" "enim" "ad" "minim" "veniam" "quis" "nostrud" "exercitation" "ullamco" "laboris" "nisi" "ut" "aliquip" "ex" "ea" "commodo" "consequat" "duis" "aute" "irure" "dolor" "in" "reprehenderit" "in" "voluptate" "velit" "esse" "cillum" "dolore" "eu" "fugiat" "nulla" "pariatur")
    local word_count=0
    local min_chars=100
    local result=()
    local special_tokens_shuffled=($(printf "%s\n" "${SPECIAL_TOKENS[@]}" | shuf))
    local st_idx=0
    local lorem_idx=0

    while :; do
    # Add a lorem word
    result+=("${LOREM_WORDS[$((lorem_idx % ${#LOREM_WORDS[@]}))]}")
    lorem_idx=$((lorem_idx+1))

    # With probability 0.2 or if not all special tokens are used, insert a special token
    if [ $st_idx -lt ${#special_tokens_shuffled[@]} ]; then
        if [ $(( RANDOM % 5 )) -eq 0 ] || [ $(( ${#result[@]} )) -eq 1 ]; then
        result+=("${special_tokens_shuffled[$st_idx]}")
        st_idx=$((st_idx+1))
        fi
    fi

    word_count=${#result[@]}
    if [ $(echo "${result[*]}" | wc -c) -ge $min_chars ] && [ $st_idx -eq ${#special_tokens_shuffled[@]} ]; then
        break
    fi
    done

    echo "${result[*]}"
}

export RAY_ADDRESS=10.57.203.156:6379
export DASHBOARD_METRIC_PORT=44227
export AUTOSCALER_METRIC_PORT=44217
export XENNA_RAY_METRICS_PORT=8080
export XENNA_RESPECT_CUDA_VISIBLE_DEVICES=1

if ! ray status &>/dev/null; then
  echo "Ray is not running. Starting Ray..."
  RAY_MAX_LIMIT_FROM_API_SERVER=40000 RAY_MAX_LIMIT_FROM_DATA_SOURCE=40000 ray start --head --node-ip-address 10.57.203.156 --port 6379 --metrics-export-port 8080 --dashboard-port 8265 --dashboard-host 127.0.0.1 --ray-client-server-port 10001 --temp-dir /tmp/ray --disable-usage-stats --include-dashboard=True
fi

INPUT_JSONL_DIR=$DATA_ROOT/curated
INPUT_JSONL_FILE=$(find $INPUT_JSONL_DIR -type f | head -n 1)

# Download the dataset if it doesn't exist
if [ ! -f "$INPUT_JSONL_FILE" ]; then
  python3 tutorials/text/tinystories/main.py --data_root $DATA_ROOT
  INPUT_JSONL_FILE=$(find $INPUT_JSONL_DIR -type f | head -n 1)
  SAMPLE_TEXT=$(generate_lorem_with_special_tokens)
  echo '{"text": "'"$SAMPLE_TEXT"'", "file_name": "TinyStories-valid.txt"}' >> "$INPUT_JSONL_FILE"
  rm -rf $DATA_ROOT/raw
fi

mkdir -p $DATA_ROOT/megatron-tokens
mkdir -p $DATA_ROOT/curator-tokens

for TOKENIZER in "${TOKENIZERS[@]}"; do
  for APPEND_EOD in "${APPEND_EOD_CONFIGS[@]}"; do
    # Normalize folder names: remove slashes and special chars from tokenizer name
    FOLDER_NAME=$(echo "$TOKENIZER" | tr '/.' '__')
    if [ "$APPEND_EOD" = "True" ]; then
      FOLDER_NAME="${FOLDER_NAME}_append_eod"
      TOKENIZATION_KWARGS="--append-eod"
    else
      TOKENIZATION_KWARGS=""
    fi

    if [ "$TOKENIZE_MEGATRON" = "True" ]; then
    mkdir -p $DATA_ROOT/megatron-tokens/$FOLDER_NAME
    python3 tokenizer-test/Megatron-LM/tools/preprocess_data.py \
      --input $INPUT_JSONL_FILE \
      --output-prefix $DATA_ROOT/megatron-tokens/$FOLDER_NAME/text \
      --tokenizer-type HuggingFaceTokenizer \
      --tokenizer-model $TOKENIZER \
      --log-interval 100 \
      --workers 8 \
      $TOKENIZATION_KWARGS
    fi

    if [ "$TOKENIZE_CURATOR" = "True" ]; then
    mkdir -p $DATA_ROOT/curator-tokens/$FOLDER_NAME
    python3 tokenizer-test/preprocess_data_curator.py \
      --input $INPUT_JSONL_DIR \
      --output $DATA_ROOT/curator-tokens/$FOLDER_NAME \
      --tokenizer-model $TOKENIZER \
      $TOKENIZATION_KWARGS
    fi
  done
done

# Compare the tokenization pipelines
./tokenizer-test/compare-tokenization-pipelines.sh $DATA_ROOT
