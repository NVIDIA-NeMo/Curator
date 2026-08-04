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

"""Curation pipeline for clinical/medical text reports.

Stages, in order:
    1. PII anonymization (GlinerPiiRedactor), run FIRST as a privacy-by-design
       principle: patient identifiers should be stripped before any other
       processing touches the data. The GLiNER PII label set already
       includes healthcare-specific identifiers (medical_record_number,
       health_plan_beneficiary_number), making it directly applicable here.
    2. Cheap, generic heuristic pre-filters (word count, alphanumeric ratio),
       with thresholds re-tuned for medical text: dosages, lab values, and
       clinical abbreviations make the library defaults too strict.
    3. Score-only pass with ClinicalSectionFilter, to inspect the
       section-count distribution on the real corpus before committing to
       a cutoff.
    4. Actual filtering with ClinicalSectionFilter, keeping only documents
       with a minimum number of distinct clinical sections.
    5. Write the curated, anonymized dataset back to JSONL.
"""

import argparse
import os
import time

# NOTE: GlinerPiiRedactor lives in tutorials/text/gliner-pii-redaction/ as an
# example stage, not (yet) a core library class. Import it from there, or
# copy it into your own stages module, depending on how your project is set
# up. This tutorial ships a self-contained copy of that module beside main.py
# so the example is runnable in isolation.
from gliner_pii_redactor import GlinerPiiRedactor

from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.text.filters import Score, ScoreFilter
from nemo_curator.stages.text.filters.clinical import ClinicalSectionFilter
from nemo_curator.stages.text.filters.heuristic import (
    NonAlphaNumericFilter,
    WordCountFilter,
)
from nemo_curator.stages.text.io.reader.jsonl import JsonlReaderStage
from nemo_curator.stages.text.io.writer import JsonlWriter


def main(args: argparse.Namespace) -> None:
    """Run the clinical report curation pipeline."""
    ray_client = RayClient()
    ray_client.start()

    curated_dir = os.path.join(args.data_root, "curated", args.language)
    os.makedirs(curated_dir, exist_ok=True)

    print("Running the clinical report curation pipeline")
    print(f"    Input reports: '{args.input_dir}'")
    print(f"    Curated output: '{curated_dir}'")

    stages = [
        # 0. Partition and read the raw JSONL reports.
        FilePartitioningStage(
            file_paths=args.input_dir,
            file_extensions=[".jsonl"],
        ),
        JsonlReaderStage(fields=["text"]),
        # 1. PII anonymization, run first (privacy-by-design).
        GlinerPiiRedactor(
            text_field="text",
            use_gpu=args.use_gpu,
        ),
        # 2. Cheap generic heuristic pre-filters, re-tuned for medical text.
        #    - min 30 words: valid clinical reports are rarely shorter.
        #    - max 30% non-alphanumeric characters: the common 0.25 default
        #      is often too strict for text full of dosages and lab values.
        ScoreFilter(
            filter_obj=WordCountFilter(min_words=30),
            text_field="text",
            score_field="word_count",
        ),
        ScoreFilter(
            filter_obj=NonAlphaNumericFilter(max_non_alpha_numeric_to_text_ratio=0.30),
            text_field="text",
            score_field="non_alpha_ratio",
        ),
        # 3. Score-only pass (no filtering yet): records the section count
        #    per document, useful for tuning min_sections against the real
        #    corpus distribution before committing to a cutoff.
        Score(
            score_fn=ClinicalSectionFilter(language=args.language),
            text_field="text",
            score_field="clinical_section_count",
        ),
        # 4. Actual filtering with the custom clinical structure filter.
        ScoreFilter(
            filter_obj=ClinicalSectionFilter(
                language=args.language,
                min_sections=2,
            ),
            text_field="text",
            score_field="kept_by_clinical_filter",
        ),
        # 5. Write the curated, anonymized dataset back to JSONL.
        JsonlWriter(curated_dir),
    ]

    pipeline = Pipeline(
        name="clinical_report_curation",
        description=(
            "Anonymizes and curates clinical/medical text reports: PII "
            "redaction, generic quality pre-filters, and domain-specific "
            "structural filtering via ClinicalSectionFilter."
        ),
        stages=stages,
    )

    print("Starting the curation pipeline")
    start_time = time.time()
    results = pipeline.run()
    execution_time = time.time() - start_time
    print(f"\n\nCuration pipeline finished (took {execution_time:.1f} seconds)")
    print(f"The results were written to '{[result.data for result in results]}'")

    ray_client.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clinical report curation pipeline.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)) + "/data",
        help="Directory where the curated output will be written.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing the raw clinical report JSONL files.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "it", "es", "fr", "de"],
        help="Language of the clinical reports, for section-keyword matching.",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        default=False,
        help="Use GPU acceleration for the GLiNER PII redaction model.",
    )
    args = parser.parse_args()
    main(args)
