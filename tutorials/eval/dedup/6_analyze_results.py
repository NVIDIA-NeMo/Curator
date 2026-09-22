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

"""
Summarize the fuzzy-dedup-eval LLM judge output (step 6): bucket the judge's
`relation_type` verdict into duplicate/not_duplicate/unresolved and compare
it against what each pair's `pair_type` implied fuzzy dedup decided
(`expected_duplicate`, written by 3_build_pair_dataset.py).

See README.md's "Output shape" section for the bucketing rationale and
disagreement-rate definitions.

Example:
    python tutorials/eval/dedup/6_analyze_results.py \
        --judge-output-path output/dedup_eval/judged_pairs \
        --disagreements-output output/dedup_eval/disagreements.jsonl
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd
from loguru import logger

JUDGE_COLUMN = "pair_semantic_judgment"
SCORE_NAME = "relation_type"
DUPLICATE_RELATIONS = {"exact", "canonical_exact", "near_surface", "containment"}
NOT_DUPLICATE_RELATIONS = {"version_related", "related_non_duplicate", "unrelated"}


def _load_judge_output(judge_output_path: str) -> pd.DataFrame:
    jsonl_files = sorted(glob.glob(str(Path(judge_output_path) / "*.jsonl")))
    parquet_files = sorted(glob.glob(str(Path(judge_output_path) / "*.parquet")))
    if jsonl_files:
        frames = [pd.read_json(f, lines=True) for f in jsonl_files]
    elif parquet_files:
        frames = [pd.read_parquet(f) for f in parquet_files]
    else:
        msg = f"No .jsonl or .parquet part files found under {judge_output_path}"
        raise FileNotFoundError(msg)
    return pd.concat(frames, ignore_index=True)


def _extract_relation(row: dict | None) -> str | None:
    if not isinstance(row, dict):
        return None
    return row.get(SCORE_NAME, {}).get("score")


def _bucket_relation(relation: str | None) -> str:
    if relation in DUPLICATE_RELATIONS:
        return "duplicate"
    if relation in NOT_DUPLICATE_RELATIONS:
        return "not_duplicate"
    return "unresolved"


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """Add `relation_type`, `verdict`, and `is_disagreement` columns to the raw judge output."""
    if JUDGE_COLUMN not in df.columns:
        msg = f"Expected judge output column {JUDGE_COLUMN!r} not found; columns present: {list(df.columns)}"
        raise KeyError(msg)
    if "pair_type" not in df.columns or "expected_duplicate" not in df.columns:
        msg = (
            "Expected passthrough columns 'pair_type'/'expected_duplicate' not found in judge output. "
            "These come from 3_build_pair_dataset.py's pairs part files and must survive into the judge's output "
            "(Data Designer preserves original input columns alongside judge columns)."
        )
        raise KeyError(msg)

    df = df.copy()
    df["relation_type"] = df[JUDGE_COLUMN].map(_extract_relation)
    df["verdict"] = df["relation_type"].map(_bucket_relation)
    duplicate_expected = df["expected_duplicate"].astype(bool)
    df["is_disagreement"] = (duplicate_expected & (df["verdict"] != "duplicate")) | (
        ~duplicate_expected & (df["verdict"] == "duplicate")
    )
    # Older pairs files predate build_semantic_diff()/the `truncated` column -- treat those as
    # not truncated rather than dropping them, since we have no truncation info for them either way.
    df["truncated"] = df["truncated"].fillna(False).astype(bool) if "truncated" in df.columns else False
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate an already-`annotate`d dataframe into one row per `pair_type`.

    Truncated pairs are counted but excluded from the disagreement rate: a judge verdict on a
    pair where the judge never saw the full document isn't conclusive about the full documents
    (see README.md's "Span alignment and truncation" section), so pooling them with untruncated
    verdicts would understate or overstate the rate depending on how the cut-off half breaks.
    """
    rows = []
    for pair_type, group in df.groupby("pair_type"):
        total = len(group)
        num_truncated = int(group["truncated"].sum())
        conclusive = group[~group["truncated"]]
        relation_counts = conclusive["relation_type"].value_counts().to_dict()
        expected = bool(group["expected_duplicate"].iloc[0])
        disagreement_label = "judge_says_not_duplicate_rate" if expected else "judge_says_duplicate_rate"
        num_conclusive = len(conclusive)
        rows.append(
            {
                "pair_type": pair_type,
                "num_pairs": total,
                "num_truncated": num_truncated,
                "expected_duplicate": expected,
                **{f"relation_{k}": v for k, v in relation_counts.items()},
                disagreement_label: conclusive["is_disagreement"].sum() / num_conclusive
                if num_conclusive
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


_DISAGREEMENT_COLUMNS = [
    "pair_id",
    "pair_type",
    "expected_duplicate",
    "doc_id_a",
    "doc_id_b",
    "truncated",
    "relation_type",
    "material_difference",
    "primary_material_difference",
    "confidence_tier",
    "primary_risk_factor",
    "dominant_overlap_source",
    JUDGE_COLUMN,
]


def write_disagreements(df: pd.DataFrame, output_path: str) -> int:
    """Write every disagreeing pair, with the full rubric output, to a JSONL file for manual review.

    Includes truncated pairs -- excluded from the headline rate in `summarize()`, but still worth a
    human look, so they're kept here with `truncated` set for the reviewer to see.
    """
    disagreements = df[df["is_disagreement"]]
    columns = [c for c in _DISAGREEMENT_COLUMNS if c in disagreements.columns]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    disagreements[columns].to_json(output_path, orient="records", lines=True, force_ascii=False)
    return len(disagreements)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--judge-output-path",
        required=True,
        help="Directory of JSONL or Parquet part files written by 5_run_llm_judge.py (step 5).",
    )
    parser.add_argument(
        "--disagreements-output",
        default=None,
        help=(
            "Optional JSONL path to write every disagreeing pair to, with its full rubric output, for manual "
            "review. Without this, the aggregate rate alone doesn't tell you which pairs to go read."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = _load_judge_output(args.judge_output_path)
    logger.info(f"Loaded {len(df)} judged pairs from {args.judge_output_path}")

    df = annotate(df)
    summary = summarize(df)
    logger.warning(
        "These rates are diagnostics from one unvalidated LLM judge on a minimal example config -- "
        "not calibrated fuzzy-dedup accuracy. Read a sample of disagreements before drawing conclusions."
    )
    num_truncated = int(df["truncated"].sum())
    if num_truncated:
        logger.warning(
            f"{num_truncated}/{len(df)} pairs had a truncated document (see 'num_truncated' per pair_type "
            "below) and were excluded from the disagreement rate -- the judge never saw the full text for "
            "those pairs, so its verdict isn't conclusive about them."
        )
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary.to_string(index=False))

    if args.disagreements_output:
        num_written = write_disagreements(df, args.disagreements_output)
        logger.info(f"Wrote {num_written} disagreeing pairs to {args.disagreements_output}")


if __name__ == "__main__":
    main()
