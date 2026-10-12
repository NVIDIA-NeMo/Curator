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

"""Summaries of a scored manifest: accuracy with confidence intervals, slices, and paired comparison.

These are plain functions over the output rows of :class:`~.score.NumericEqualityStage`, applied
after the pipeline has run.
"""

import json
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

DEFAULT_MATCH_KEY = "numeric_match_corrected"
DEFAULT_DIGIT_BINS: tuple[tuple[int, int], ...] = ((1, 6), (7, 10), (11, 15), (16, 20), (21, 25), (26, 99))
_Z95 = 1.959964


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def wilson_interval(successes: int, total: int, z: float = _Z95) -> tuple[float, float]:
    """95% Wilson score interval for a proportion, as (low, high) in [0, 1]."""
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def accuracy(rows: Iterable[dict], key: str = DEFAULT_MATCH_KEY) -> dict:
    """Accuracy of ``key`` over ``rows`` with its Wilson interval."""
    rows = list(rows)
    n = len(rows)
    k = sum(1 for r in rows if r.get(key))
    low, high = wilson_interval(k, n)
    return {
        "n": n,
        "correct": k,
        "accuracy": 100.0 * k / n if n else 0.0,
        "ci_low": 100.0 * low,
        "ci_high": 100.0 * high,
    }


def accuracy_by(rows: Sequence[dict], field: str, key: str = DEFAULT_MATCH_KEY) -> dict[str, dict]:
    """Accuracy per distinct value of ``field`` (for example ``kind`` or ``style``)."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(str(r.get(field, "?")), []).append(r)
    return {g: accuracy(members, key) for g, members in sorted(groups.items())}


def digit_bin(num_digits: int, bins: Sequence[tuple[int, int]] = DEFAULT_DIGIT_BINS) -> str:
    for low, high in bins:
        if low <= num_digits <= high:
            return f"{low}-{high}" if high < 99 else f"{low}+"  # noqa: PLR2004
    return "?"


def accuracy_by_length(
    rows: Sequence[dict], key: str = DEFAULT_MATCH_KEY, bins: Sequence[tuple[int, int]] = DEFAULT_DIGIT_BINS
) -> dict[str, dict]:
    """Accuracy per digit-length bin of the spoken value (``num_digits``)."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(digit_bin(int(r.get("num_digits", 0)), bins), []).append(r)
    order = [f"{lo}-{hi}" if hi < 99 else f"{lo}+" for lo, hi in bins]  # noqa: PLR2004
    return {g: accuracy(groups[g], key) for g in order if g in groups}


def compare_systems(rows_a: Sequence[dict], rows_b: Sequence[dict], key: str = DEFAULT_MATCH_KEY) -> dict:
    """Paired comparison on identical clips: how many clips only B gets right, and only A.

    Rows are matched on ``id``. A large imbalance between ``gained`` and ``lost`` means the
    difference is not chance.
    """
    a = {r["id"]: bool(r.get(key)) for r in rows_a}
    b = {r["id"]: bool(r.get(key)) for r in rows_b}
    common = a.keys() & b.keys()
    gained = sum(1 for i in common if b[i] and not a[i])
    lost = sum(1 for i in common if a[i] and not b[i])
    return {"n_common": len(common), "gained": gained, "lost": lost, "net": gained - lost}


def summarize(rows: Sequence[dict], key: str = DEFAULT_MATCH_KEY) -> dict:
    """Overall accuracy plus slices by kind, reading style and digit length."""
    return {
        "metric": key,
        "overall": accuracy(rows, key),
        "by_kind": accuracy_by(rows, "kind", key),
        "by_style": accuracy_by(rows, "style", key),
        "by_length": accuracy_by_length(rows, key),
        "judge": dict(Counter(str(r.get("judge_faithful", "n/a")) for r in rows)),
    }


def _row(label: str, stats: dict) -> str:
    return f"| {label} | {stats['n']} | {stats['accuracy']:.1f} | [{stats['ci_low']:.1f}, {stats['ci_high']:.1f}] |"


def render_markdown(summary: dict, title: str = "Number transcription accuracy") -> str:
    """Render :func:`summarize` output as Markdown tables."""
    lines = [f"## {title}", "", f"Metric: `{summary['metric']}` (value match, digit-word homophones tolerated)", ""]
    lines += ["| slice | n | accuracy % | 95% interval |", "|---|---|---|---|", _row("all", summary["overall"])]
    for section, table in (
        ("kind", summary["by_kind"]),
        ("style", summary["by_style"]),
        ("digits", summary["by_length"]),
    ):
        lines += [f"| **by {section}** | | | |"]
        lines += [_row(label, stats) for label, stats in table.items()]
    return "\n".join(lines) + "\n"
