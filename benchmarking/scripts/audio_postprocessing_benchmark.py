# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Granary v2 audio postprocessing pipeline benchmark.

Runs the text-filtering pipeline (InitializeFields -> WhisperHallucination ->
FastTextLID -> RegexSubstitution -> FinalizeFields) through the full
Pipeline/Executor stack and collects per-stage timing, throughput, and
filtering metrics for regression tracking.

Can process a single JSONL manifest or a directory of manifests.
"""

import argparse
import time
import traceback
from pathlib import Path
from typing import Any

from loguru import logger
from utils import setup_executor, write_benchmark_results

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.alm import ALMManifestReader, ALMManifestWriterStage
from nemo_curator.stages.audio.text_filtering import (
    FastTextLIDStage,
    FinalizeFieldsStage,
    InitializeFieldsStage,
    RegexSubstitutionStage,
    WhisperHallucinationStage,
)

_DEFAULT_REGEX_YAML = str(
    Path(__file__).resolve().parent.parent.parent
    / "tutorials/audio/granary_v2_postprocessing/common.yaml"
)
_DEFAULT_HALL_PHRASES = str(
    Path(__file__).resolve().parent.parent.parent
    / "tutorials/audio/granary_v2_postprocessing/en.txt"
)


def run_postprocessing_benchmark(  # noqa: PLR0913, PLR0915
    benchmark_results_path: str,
    input_manifest: str,
    output_path: str,
    executor: str,
    fasttext_model: str,
    target_lang: str,
    min_lang_prob: float,
    regex_yaml: str,
    hall_phrases: str,
    unique_words_threshold: float,
    long_word_threshold: int,
    long_word_rel_threshold: float,
    char_rate_threshold: float,
    max_char_rate: float,
) -> dict[str, Any]:
    """Run the postprocessing benchmark and collect metrics."""
    benchmark_results_path = Path(benchmark_results_path)
    benchmark_results_path.mkdir(parents=True, exist_ok=True)

    logger.info("Starting audio postprocessing benchmark")
    logger.info(f"Input manifest: {input_manifest}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Executor: {executor}")
    logger.info(f"FastText model: {fasttext_model}")
    logger.info(f"Target lang: {target_lang}, min_lang_prob: {min_lang_prob}")

    pipeline = Pipeline(
        name="postprocessing_benchmark",
        description="Granary v2 postprocessing benchmark pipeline",
    )
    pipeline.add_stage(ALMManifestReader(manifest_path=input_manifest))
    pipeline.add_stage(InitializeFieldsStage())
    pipeline.add_stage(
        WhisperHallucinationStage(
            common_hall_file=hall_phrases,
            unique_words_threshold=unique_words_threshold,
            long_word_threshold=long_word_threshold,
            long_word_rel_threshold=long_word_rel_threshold,
            char_rate_threshold=char_rate_threshold,
            max_char_rate=max_char_rate,
        )
    )
    pipeline.add_stage(
        FastTextLIDStage(
            model_path=fasttext_model,
            target_lang=target_lang,
            min_lang_prob=min_lang_prob,
        )
    )
    pipeline.add_stage(RegexSubstitutionStage(regex_params_yaml=regex_yaml))
    pipeline.add_stage(FinalizeFieldsStage())

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pipeline.add_stage(ALMManifestWriterStage(output_path=output_path))

    exc = setup_executor(executor)

    run_start_time = time.perf_counter()

    try:
        logger.info("Running postprocessing pipeline...")
        logger.info(f"Pipeline description:\n{pipeline.describe()}")

        output_tasks = pipeline.run(exc)
        run_time_taken = time.perf_counter() - run_start_time

        num_output = len(output_tasks) if output_tasks else 0

        n_flagged = 0
        n_clean = 0
        flag_reasons: dict[str, int] = {}
        if output_tasks:
            for task in output_tasks:
                skip = task.data.get("skip_me", "") or task.data.get("_skip_me", "")
                if skip:
                    n_flagged += 1
                    reason = skip.split(":")[0] if ":" in skip else skip
                    flag_reasons[reason] = flag_reasons.get(reason, 0) + 1
                else:
                    n_clean += 1

        logger.success(f"Benchmark completed in {run_time_taken:.2f}s")
        logger.success(f"Output: {num_output} entries ({n_clean} clean, {n_flagged} flagged)")
        logger.success(f"Throughput: {num_output / run_time_taken:.1f} entries/sec")
        if flag_reasons:
            logger.success(f"Flag reasons: {flag_reasons}")
        success = True

    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Benchmark failed: {e}")
        logger.debug(f"Full traceback:\n{error_traceback}")
        output_tasks = []
        run_time_taken = time.perf_counter() - run_start_time
        num_output = 0
        n_clean = 0
        n_flagged = 0
        flag_reasons = {}
        success = False

    return {
        "params": {
            "executor": executor,
            "input_manifest": input_manifest,
            "output_path": output_path,
            "fasttext_model": fasttext_model,
            "target_lang": target_lang,
            "min_lang_prob": min_lang_prob,
            "unique_words_threshold": unique_words_threshold,
            "long_word_threshold": long_word_threshold,
        },
        "metrics": {
            "is_success": success,
            "time_taken_s": run_time_taken,
            "num_output_entries": num_output,
            "num_clean": n_clean,
            "num_flagged": n_flagged,
            "pass_through_pct": 100.0 * n_clean / num_output if num_output > 0 else 0,
            "throughput_entries_per_sec": num_output / run_time_taken if run_time_taken > 0 else 0,
            "flag_reasons": flag_reasons,
        },
        "tasks": output_tasks or [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Granary v2 audio postprocessing benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--benchmark-results-path", type=str, required=True, help="Path for benchmark results")
    parser.add_argument("--input-manifest", type=str, required=True, help="Input JSONL manifest path")
    parser.add_argument("--output-path", type=str, required=True, help="Output JSONL manifest path")
    parser.add_argument("--executor", default="xenna", choices=["xenna", "ray_data"], help="Executor backend")
    parser.add_argument("--fasttext-model", default="lid.176.ftz", help="FastText LID model path")
    parser.add_argument("--target-lang", default="en", help="Target language code")
    parser.add_argument("--min-lang-prob", type=float, default=0.8, help="Minimum language probability")
    parser.add_argument("--regex-yaml", default=_DEFAULT_REGEX_YAML, help="Regex rules YAML path")
    parser.add_argument("--hall-phrases", default=_DEFAULT_HALL_PHRASES, help="Hallucination phrases file")
    parser.add_argument("--unique-words-threshold", type=float, default=0.4, help="Unique words ratio threshold")
    parser.add_argument("--long-word-threshold", type=int, default=25, help="Long word character threshold")
    parser.add_argument("--long-word-rel-threshold", type=float, default=3.0, help="Long word relative threshold")
    parser.add_argument("--char-rate-threshold", type=float, default=4.0, help="Min char rate (chars/sec)")
    parser.add_argument("--max-char-rate", type=float, default=40.0, help="Max char rate (chars/sec)")

    args = parser.parse_args()

    logger.info("=== Audio Postprocessing Benchmark Starting ===")
    logger.info(f"Arguments: {vars(args)}")

    run_args = {k.replace("-", "_"): v for k, v in vars(args).items()}

    results = {
        "params": run_args,
        "metrics": {"is_success": False},
        "tasks": [],
    }
    try:
        results.update(run_postprocessing_benchmark(**run_args))
    finally:
        write_benchmark_results(results, args.benchmark_results_path)

    return 0 if results["metrics"]["is_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
