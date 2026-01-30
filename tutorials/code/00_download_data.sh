#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Download code dataset for tutorials
#
# Downloads the bigcode/the-stack-smol-xs dataset from Hugging Face.
#
# Dataset: https://huggingface.co/datasets/bigcode/the-stack-smol-xs
# - 100 samples per language across 87 programming languages
# - ~32MB total size
# - Permissively licensed source code
#
# Usage:
#     ./00_download_data.sh [output_dir]

set -e

OUTPUT_DIR="${1:-tutorials/code/output/input_data}"

echo "============================================================"
echo "NeMo Curator Code Tutorial - Dataset Download"
echo "============================================================"
echo "Dataset: bigcode/the-stack-smol-xs (Hugging Face)"
echo "Output: $OUTPUT_DIR"
echo

huggingface-cli download bigcode/the-stack-smol-xs \
    --repo-type dataset \
    --local-dir "$OUTPUT_DIR"

echo
echo "============================================================"
echo "Download complete!"
echo "============================================================"
echo "Next steps - run the tutorial scripts:"
echo "  python tutorials/code/01_language_detection.py"
echo "  python tutorials/code/02_code_annotation.py"
echo "  python tutorials/code/03_code_filtering.py"
echo "  python tutorials/code/04_license_detection.py"
