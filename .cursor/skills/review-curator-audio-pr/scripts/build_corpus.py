#!/usr/bin/env python3
"""Consolidate the audio PR corpus pulled by pull_audio_pr_corpus.sh.

Reads the per-PR JSON in the corpus dir and writes one markdown file with every
reviewer comment (verbatim, anchored to path:line) grouped by PR, plus a
recurring-themes keyword tally. Read-only context for pre-review; posts nothing.

Usage: build_corpus.py [--outdir DIR] [--today YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

BOT_LOGINS = {"greptile-apps[bot]", "copy-pr-bot[bot]", "github-actions[bot]"}

# (label, regex) recurring-theme buckets, mirroring knowledge-sources.md lenses.
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


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else []


def shorten(s: str, n: int = 1200) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + " […]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".curator-pr-review/audio-corpus")
    ap.add_argument("--today", default=dt.date.today().isoformat())
    args = ap.parse_args()

    outdir = Path(args.outdir)
    nums_file = outdir / "_audio_pr_numbers.txt"
    if not nums_file.exists():
        raise SystemExit(f"no {nums_file}; run pull_audio_pr_corpus.sh first")
    numbers = [int(x) for x in nums_file.read_text().split() if x.strip()]
    numbers.sort(reverse=True)

    date_us = args.today.replace("-", "_")
    theme_counts = {label: 0 for label, _ in THEMES}
    theme_rx = [(label, re.compile(rx, re.I)) for label, rx in THEMES]

    out: list[str] = []
    out.append(f"# Audio PR review corpus (post-#1608) - {args.today}\n")
    out.append(
        "Consolidated reviewer feedback on audio PRs opened after #1608 "
        "(open + closed/merged). Read-only pre-review context: recognise the "
        "patterns reviewers repeatedly raise, and check the PR in front of you "
        "against them. Bot reviewers are marked `[bot]`.\n"
    )
    out.append(f"Audio PRs in corpus: **{len(numbers)}** "
               f"({', '.join('#' + str(n) for n in numbers)})\n")

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
        sec.append(f"- state: **{state}**  author: @{author}  "
                   f"created: {gh.get('createdAt','?')}  link: {url}\n")

        # Review summaries (non-empty bodies only).
        rev_bodies = [r for r in reviews if (r.get("body") or "").strip()]
        if rev_bodies:
            sec.append("### Review summaries\n")
            for r in rev_bodies:
                login = (r.get("user") or {}).get("login", "?")
                bot = " `[bot]`" if login in BOT_LOGINS else ""
                sec.append(f"- **@{login}{bot}** [{r.get('state','')}] "
                           f"{r.get('submitted_at','')}:\n")
                sec.append(f"  > {shorten(r.get('body'))}\n")
                for label, rx in theme_rx:
                    if rx.search(r.get("body") or ""):
                        theme_counts[label] += 1

        # Inline comments grouped by file.
        by_file: dict[str, list] = {}
        for c in rcomments:
            by_file.setdefault(c.get("path", "?"), []).append(c)
        if by_file:
            sec.append("### Inline review comments\n")
            for path in sorted(by_file):
                sec.append(f"#### `{path}`\n")
                for c in sorted(by_file[path],
                                key=lambda x: (x.get("line") or x.get("original_line") or 0)):
                    login = (c.get("user") or {}).get("login", "?")
                    bot = " `[bot]`" if login in BOT_LOGINS else ""
                    line = c.get("line") or c.get("original_line") or "?"
                    body = c.get("body") or ""
                    total_comments += 1
                    sec.append(f"- **@{login}{bot}** line {line} "
                               f"([link]({c.get('html_url','')})):\n")
                    sec.append(f"  > {shorten(body)}\n")
                    for label, rx in theme_rx:
                        if rx.search(body):
                            theme_counts[label] += 1

        # Top-level issue comments (skip pure CI noise).
        human_ic = [c for c in icomments
                    if (c.get("user") or {}).get("login") not in BOT_LOGINS
                    and (c.get("body") or "").strip()]
        if human_ic:
            sec.append("### Discussion (top-level)\n")
            for c in human_ic:
                login = (c.get("user") or {}).get("login", "?")
                sec.append(f"- **@{login}** {c.get('created_at','')}: "
                           f"{shorten(c.get('body'), 600)}\n")

        per_pr_sections.append("\n".join(sec))

    # Theme tally up top.
    out.append("## Recurring themes (comment hits across the corpus)\n")
    out.append("| Theme | Comments mentioning it |")
    out.append("|-------|------------------------|")
    for label, _ in THEMES:
        out.append(f"| {label} | {theme_counts[label]} |")
    out.append(f"\nTotal inline review comments scanned: **{total_comments}**\n")
    out.append("---\n")
    out.extend(per_pr_sections)

    outpath = outdir / f"audio_pr_corpus_{date_us}.md"
    outpath.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {outpath}  ({outpath.stat().st_size} bytes; {len(numbers)} PRs)")


if __name__ == "__main__":
    main()
