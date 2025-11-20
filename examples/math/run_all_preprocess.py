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

import subprocess
import sys
from pathlib import Path

from loguru import logger

# Define dataset configurations: Input paths (local or HF) and column mappings
# Inputs can be:
# 1. Local filesystem glob: "/path/to/data/*.parquet"
# 2. Hugging Face Dataset: "hf://HuggingFaceFW/fineweb-math" (supported by NeMo Curator ParquetReader)
DATASETS = {
    "FINEMATH_4": {
        "input": "/lustre/fsw/portfolios/llmservice/users/rkarimimahab/data/finemath/original-data/finemath-4plus/*.parquet",
        "output": "finemath_4_processed",
        "fetch_cc": True,
        "columns": {
            "warc_filename": "warc_filename",
            "offset": "warc_record_offset",
            "length": "warc_record_length",
        },
    },
    "FINEMATH_3": {
        "input": "/lustre/fsw/portfolios/llmservice/users/rkarimimahab/data/finemath/original-data/finemath-3plus/*.parquet",
        "output": "finemath_3_processed",
        "fetch_cc": True,
        "columns": {
            "warc_filename": "warc_filename",
            "offset": "warc_record_offset",
            "length": "warc_record_length",
        },
    },
    "OWM": {
        # Can also use Huggingface input hf://open-web-math/open-web-math
        "input": "/home/sasatheesh/data/20t/jsonls/nv-math/open-web-math/*.parquet",  # Local path alternative
        "output": "owm_processed",
        "fetch_cc": True,
        "columns": {
            "warc_filename": "warc_filename",
            "offset": "warc_record_offset",
            "length": "warc_record_length",
        },
    },
    "INFIWEBMATH_4": {
        "input": "/lustre/fsw/portfolios/llmservice/users/rkarimimahab/data/finemath/original-data/infiwebmath-4plus/*.parquet",
        "output": "infiwebmath_4_processed",
        "fetch_cc": True,
        "columns": {
            "warc_filename": "warc_filename",
            "offset": "warc_record_offset",
            "length": "warc_record_length",
        },
    },
    "INFIWEBMATH_3": {
        "input": "/lustre/fsw/portfolios/llmservice/users/rkarimimahab/data/finemath/original-data/infiwebmath-3plus/*.parquet",
        "output": "infiwebmath_3_processed",
        "fetch_cc": True,
        "columns": {
            "warc_filename": "warc_filename",
            "offset": "warc_record_offset",
            "length": "warc_record_length",
        },
    },
    "MEGAMATH_PRO": {
        "input": "/lustre/fsw/portfolios/llmservice/projects/llmservice_fm_text/adlr-stem/megamath_dataset/megamath-web-pro/*.parquet",
        "output": "megamath_pro_processed",
        "fetch_cc": True,
        "columns": {
            "warc_filename": "warc_filename",
            "offset": "warc_record_offset",
            "length": "warc_record_length",
        },
    },
    "MEGAMATH_WEB": {
        "input": "/lustre/fsw/portfolios/llmservice/projects/llmservice_fm_text/adlr-stem/megamath_dataset/megamath-web/**/*.parquet",
        "output": "megamath_web_processed",
        "fetch_cc": True,
        "columns": {
            "warc_filename": "warc_filename",
            "offset": "warc_record_offset",
            "length": "warc_record_length",
        },
    },
    "FINEWEB_MATH": {
        # Can also use Huggingface input hf://HuggingFaceFW/fineweb-math
        "input": "/home/sasatheesh/data/20t/jsonls/nv-math/fineweb-math/*.parquet",
        "output": "fineweb_math_processed",
        "fetch_cc": True,
        "columns": {
            "warc_filename": "file_path", # FineWeb typically uses file_path or warc_filename
            "offset": "offset",
            "length": "length",
        },
     }
}

def run_dataset(
    name: str,
    config: dict,
    base_output_dir: str,
    continue_on_error: bool = False,
) -> None:
    script_path = Path(__file__).parent / "run_text_preprocess.py"
    output_path = Path(base_output_dir) / config["output"]

    logger.info(f"Processing dataset: {name}")
    logger.info(f"Input: {config['input']}")
    logger.info(f"Output: {output_path}")

    cmd = [
        sys.executable,
        str(script_path),
        "--input", config["input"],
        "--output", str(output_path),
        "--report-stats",
    ]

    if config.get("fetch_cc"):
        cmd.append("--fetch-cc")
        cols = config.get("columns", {})
        if "warc_filename" in cols:
            cmd.extend(["--warc-filename-col", cols["warc_filename"]])
        if "offset" in cols:
            cmd.extend(["--offset-col", cols["offset"]])
        if "length" in cols:
            cmd.extend(["--length-col", cols["length"]])

    logger.info(f"Running command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        logger.success(f"Successfully processed {name}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to process {name}: {e}")
        if not continue_on_error:
            raise


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run preprocessing for all configured math datasets")
    parser.add_argument("--output-base", required=True, help="Base directory for all outputs")
    parser.add_argument("--datasets", nargs="+", help="Specific datasets to run (default: all)", choices=DATASETS.keys())
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing datasets even if one fails (default: fail fast).",
    )

    args = parser.parse_args()

    targets = args.datasets if args.datasets else DATASETS.keys()

    for name in targets:
        if name in DATASETS:
            run_dataset(name, DATASETS[name], args.output_base, continue_on_error=args.continue_on_error)
        else:
            logger.warning(f"Dataset {name} not found in configuration. Skipping.")

if __name__ == "__main__":
    main()
