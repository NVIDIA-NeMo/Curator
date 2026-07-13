#!/usr/bin/env python3
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
"""Consolidate a modality PR review corpus pulled by a corpus-pull script.

Reads per-PR JSON in the corpus dir and writes one markdown file with reviewer
comments grouped by PR plus a recurring-themes keyword tally.

Usage: build_corpus.py [--outdir DIR] [--today YYYY-MM-DD]
                        [--title TITLE] [--intro INTRO] [--output-prefix PREFIX]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

BOT_LOGINS = {"greptile-apps[bot]", "copy-pr-bot[bot]", "github-actions[bot]"}

THEMES = [
    ("setup/teardown lifecycle", r"setup_on_node|\bsetup\(|teardown|_setup_done"),
    ("optional/lazy imports", r"top[- ]level import|lazy import|import .* fails|optional (dep|extra)"),
    ("dependency declaration/pins", r"pyproject|optional[- ]?group|==|version pin|requirement"),
    ("stage contract inputs/outputs", r"inputs\(\)|outputs\(\)|validate_input|NotImplementedError"),
    ("batch_size / process_batch", r"batch_size|process_batch"),
    ("memory / serialization", r"ndarray|json\.dumps|serializ|waveform|tensor|OOM|memory"),
    ("fsspec / cloud I/O", r"fsspec|url_to_fs|s3|gcs|http"),
    ("secrets / logging", r"token|secret|credential|password|redact"),
    ("tests / coverage", r"\btest|coverage|pytest|fixture"),
    ("copyright / lint", r"copyright|header|ruff|lint"),
    ("naming / convention", r"naming|rename|convention|AudioTask|AudioBatch"),
    ("trust_remote_code", r"trust_remote_code"),
]


def load(p: Path) -> object:
    return json.loads(p.read_text()) if p.exists() else []


def shorten(s: str, n: int = 1200) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + " […]"


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".curator-pr-review/audio-corpus")
    ap.add_argument("--today", default=None)
    ap.add_argument("--title", default="PR review corpus")
    ap.add_argument(
        "--intro",
        default=(
            "Consolidated reviewer feedback on recent PRs (open + closed/merged). "
            "Read-only pre-review context: recognise patterns reviewers repeatedly "
            "raise, and check the PR in front of you against them."
        ),
    )
    ap.add_argument("--output-prefix", default="pr_corpus")
    args = ap.parse_args()

    today = args.today or dt.datetime.now(dt.UTC).date().isoformat()
    outdir = Path(args.outdir)
    nums_file = outdir / "_audio_pr_numbers.txt"
    if not nums_file.exists():
        msg = f"no {nums_file}; run the corpus pull script first"
        raise SystemExit(msg)
    numbers = [int(x) for x in nums_file.read_text().split() if x.strip()]
    numbers.sort(reverse=True)

    date_us = today.replace("-", "_")
    theme_counts = {label: 0 for label, _ in THEMES}
    theme_rx = [(label, re.compile(rx, re.IGNORECASE)) for label, rx in THEMES]

    out: list[str] = []
    out.append(f"# {args.title} - {today}\n")
    out.append(f"{args.intro} Bot reviewers are marked `[bot]`.\n")
    out.append(f"PRs in corpus: **{len(numbers)}** ({', '.join('#' + str(n) for n in numbers)})\n")

    per_pr_sections: list[str] = []
    total_comments = 0
    for n in numbers:
        gh = load(outdir / f"pr{n}_gh.json")
        if isinstance(gh, list):
            gh = gh[0] if gh else {}
        reviews = load(outdir / f"pr{n}_reviews.json")
        rcomments = load(outdir / f"pr{n}_review_comments.json")
        icomments = load(outdir / f"pr{n}_issue_comments.json")

        author = (gh.get("author") or {}).get("login", "?")
        state = gh.get("state", "?")
        title = gh.get("title", "")
        url = gh.get("url", f"https://github.com/NVIDIA-NeMo/Curator/pull/{n}")

        sec: list[str] = []
        sec.append(f"## PR #{n} - {title}\n")
        sec.append(f"- state: **{state}**  author: @{author}  created: {gh.get('createdAt', '?')}  link: {url}\n")

        rev_bodies = [r for r in reviews if (r.get("body") or "").strip()]
        if rev_bodies:
            sec.append("### Review summaries\n")
            for r in rev_bodies:
                login = (r.get("user") or {}).get("login", "?")
                bot = " `[bot]`" if login in BOT_LOGINS else ""
                sec.append(f"- **@{login}{bot}** [{r.get('state', '')}] {r.get('submitted_at', '')}:\n")
                sec.append(f"  > {shorten(r.get('body'))}\n")
                for label, rx in theme_rx:
                    if rx.search(r.get("body") or ""):
                        theme_counts[label] += 1

        by_file: dict[str, list] = {}
        for c in rcomments:
            by_file.setdefault(c.get("path", "?"), []).append(c)
        if by_file:
            sec.append("### Inline review comments\n")
            for path in sorted(by_file):
                sec.append(f"#### `{path}`\n")
                for c in sorted(by_file[path], key=lambda x: (x.get("line") or x.get("original_line") or 0)):
                    login = (c.get("user") or {}).get("login", "?")
                    bot = " `[bot]`" if login in BOT_LOGINS else ""
                    line = c.get("line") or c.get("original_line") or "?"
                    body = c.get("body") or ""
                    total_comments += 1
                    sec.append(f"- **@{login}{bot}** line {line} ([link]({c.get('html_url', '')})):\n")
                    sec.append(f"  > {shorten(body)}\n")
                    for label, rx in theme_rx:
                        if rx.search(body):
                            theme_counts[label] += 1

        human_ic = [
            c
            for c in icomments
            if (c.get("user") or {}).get("login") not in BOT_LOGINS and (c.get("body") or "").strip()
        ]
        if human_ic:
            sec.append("### Discussion (top-level)\n")
            for c in human_ic:
                login = (c.get("user") or {}).get("login", "?")
                sec.append(f"- **@{login}** {c.get('created_at', '')}: {shorten(c.get('body'), 600)}\n")

        per_pr_sections.append("\n".join(sec))

    out.append("## Recurring themes (comment hits across the corpus)\n")
    out.append("| Theme | Comments mentioning it |")
    out.append("|-------|------------------------|")
    for label, _ in THEMES:
        out.append(f"| {label} | {theme_counts[label]} |")
    out.append(f"\nTotal inline review comments scanned: **{total_comments}**\n")
    out.append("---\n")
    out.extend(per_pr_sections)

    outpath = outdir / f"{args.output_prefix}_{date_us}.md"
    outpath.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {outpath}  ({outpath.stat().st_size} bytes; {len(numbers)} PRs)")


if __name__ == "__main__":
    main()
