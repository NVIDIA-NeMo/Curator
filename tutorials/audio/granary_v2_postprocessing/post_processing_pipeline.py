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

"""Granary v2 ASR postprocessing pipeline.

Recursively finds all *.jsonl manifests under an input directory, applies text
cleaning and filtering, and writes output manifests mirroring the same
subdirectory structure under output_dir.

Pipeline stages (per manifest):
  1. ALMManifestReader         — read JSONL manifest → one AudioTask per line
  2. InitializeFieldsStage     — text → granary_v1_prediction; pred_text → cleaned_text; skip_me = ""
  3. WhisperHallucinationStage — flag Whisper hallucination patterns (skip_me = "Hallucination")
  4. FastTextLIDStage          — flag non-target-language / low-confidence (skip_me = reason string)
  5. RegexSubstitutionStage    — apply regex normalization rules to cleaned_text
  6. FinalizeFieldsStage       — drop pnc/itn/timestamp; keep cleaned_text
  7. ALMManifestWriterStage    — write all entries (including flagged) to mirrored output path

Usage::

    python tutorials/audio/granary_v2_postprocessing/pipeline.py \\
        --input_dir /path/to/results_dir \\
        --output_dir /path/to/output_root \\
        --fasttext_model lid.176.ftz
"""

import argparse
import os
import sys
from pathlib import Path

from loguru import logger

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.alm.alm_manifest_reader import ALMManifestReader
from nemo_curator.stages.audio.alm.alm_manifest_writer import ALMManifestWriterStage
from nemo_curator.stages.audio.text_filtering import (
    FastTextLIDStage,
    FinalizeFieldsStage,
    InitializeFieldsStage,
    RegexSubstitutionStage,
    WhisperHallucinationStage,
)

_TUTORIAL_DIR = Path(__file__).parent
_DEFAULT_REGEX_YAML = str(_TUTORIAL_DIR / "common.yaml")
_DEFAULT_HALL_PHRASES = str(_TUTORIAL_DIR / "en.txt")


def _find_manifests(input_dir: str) -> list[str]:
    """Return all *.jsonl files found recursively under input_dir, sorted."""
    return sorted(str(p) for p in Path(input_dir).rglob("*.jsonl"))


def _compute_output_paths(manifest_paths: list[str], input_dir: str, output_dir: str) -> dict[str, str]:
    """Mirror each manifest path from input_dir into output_dir, preserving relative structure.

    Example::

        input_dir:  /data/results_large_scale_6
        input:      /data/results_large_scale_6/corpus_a/manifest_0.jsonl
        output_dir: /out
        →           /out/corpus_a/manifest_0.jsonl
    """
    root = Path(input_dir)
    return {str(p): str(Path(output_dir) / Path(p).relative_to(root)) for p in manifest_paths}


def _create_pipeline(manifest_path: str, output_path: str, args: argparse.Namespace) -> Pipeline:
    pipeline = Pipeline(
        name="Granary_v2_postprocessing",
        description=(
            "Text cleaning, hallucination detection, and language ID filtering for Granary v2 ASR manifests."
        ),
    )
    pipeline.add_stage(ALMManifestReader(manifest_path=manifest_path))
    pipeline.add_stage(InitializeFieldsStage())

    pipeline.add_stage(
        WhisperHallucinationStage(
            common_hall_file=args.hall_phrases,
            unique_words_threshold=args.unique_words_threshold,
            long_word_threshold=args.long_word_threshold,
            long_word_rel_threshold=args.long_word_rel_threshold,
            char_rate_threshold=args.char_rate_threshold,
            max_char_rate=args.max_char_rate,
        )
    )
    pipeline.add_stage(
        FastTextLIDStage(
            model_path=args.fasttext_model,
            target_lang=args.target_lang,
            min_lang_prob=args.min_lang_prob,
        )
    )
    pipeline.add_stage(RegexSubstitutionStage(regex_params_yaml=args.regex_yaml))
    pipeline.add_stage(FinalizeFieldsStage())
    pipeline.add_stage(ALMManifestWriterStage(output_path=output_path))
    return pipeline


def main(args: argparse.Namespace) -> None:
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "INFO")

    if args.manifests:
        manifest_paths = args.manifests
        logger.info(f"Processing {len(manifest_paths)} specified manifest(s)")
    else:
        manifest_paths = _find_manifests(args.input_dir)
        if not manifest_paths:
            logger.error(f"No *.jsonl files found under {args.input_dir}")
            sys.exit(1)
        logger.info(f"Found {len(manifest_paths)} manifest(s) under {args.input_dir}")

    output_map = _compute_output_paths(manifest_paths, args.input_dir, args.output_dir)
    for src, dst in output_map.items():
        logger.info(f"  {src}")
        logger.info(f"  → {dst}")

    executor = XennaExecutor()

    n_done = n_skipped = 0
    for i, (manifest_path, output_path) in enumerate(output_map.items(), 1):
        logger.info(f"\n[{i}/{len(output_map)}] {manifest_path}")

        # Skip manifests whose output already exists and is non-empty.
        # This makes reruns safe: preempted or partially-run jobs can be
        # resubmitted and only the missing manifests will be processed.
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"  Already done, skipping → {output_path}")
            n_skipped += 1
            continue

        # Write to a .tmp file first, then rename atomically on success.
        # A preempted run leaves only the .tmp file, which is ignored on
        # the next run (not a valid .jsonl), so the manifest is reprocessed.
        tmp_path = output_path + ".tmp"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        pipeline = _create_pipeline(manifest_path, tmp_path, args)
        if args.verbose:
            logger.debug(pipeline.describe())
        pipeline.run(executor)
        os.rename(tmp_path, output_path)
        logger.info(f"  Written → {output_path}")
        n_done += 1

    logger.info(f"\nDone. processed={n_done}, skipped={n_skipped} (total={len(output_map)}) → {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Granary v2 ASR postprocessing pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root input directory used to compute mirrored output paths.",
    )
    parser.add_argument(
        "--manifests",
        type=str,
        nargs="+",
        default=None,
        help="Process specific manifests instead of scanning all of input_dir. "
        "All paths must be under input_dir so output paths can be computed correctly.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root output directory. Input manifest paths are mirrored here.",
    )
    parser.add_argument(
        "--fasttext_model",
        type=str,
        default="lid.176.ftz",
        help="FastText LID model: local path or known name (lid.176.bin / lid.176.ftz).",
    )
    parser.add_argument(
        "--regex_yaml",
        type=str,
        default=_DEFAULT_REGEX_YAML,
        help="Path to regex substitution rules YAML.",
    )
    parser.add_argument(
        "--hall_phrases",
        type=str,
        default=_DEFAULT_HALL_PHRASES,
        help="Path to hallucination phrases text file.",
    )
    parser.add_argument(
        "--target_lang",
        type=str,
        default="en",
        help="Expected language code for LID filtering.",
    )
    parser.add_argument(
        "--min_lang_prob",
        type=float,
        default=0.8,
        help="Minimum FastText language probability to keep an entry.",
    )
    parser.add_argument(
        "--unique_words_threshold",
        type=float,
        default=0.4,
        help="Unique-word ratio threshold for repeated n-gram hallucination detection.",
    )
    parser.add_argument(
        "--long_word_threshold",
        type=int,
        default=25,
        help="Absolute character length above which a word is flagged as abnormally long.",
    )
    parser.add_argument(
        "--long_word_rel_threshold",
        type=float,
        default=3.0,
        help="Relative length ratio (longest/second-longest) for long-word hallucination detection.",
    )
    parser.add_argument(
        "--char_rate_threshold",
        type=float,
        default=4.0,
        help="Max chars/s below which text is considered too sparse (low char-rate hallucination).",
    )
    parser.add_argument(
        "--max_char_rate",
        type=float,
        default=40.0,
        help="Min chars/s above which text is considered impossibly dense (high char-rate hallucination).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    main(parser.parse_args())
