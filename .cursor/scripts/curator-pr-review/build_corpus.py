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
comments grouped by pull request and file.

Usage: build_corpus.py [--outdir DIR] [--numbers-file FILE]
                        [--repo OWNER/REPO] [--today YYYY-MM-DD]
                        [--title TITLE] [--intro INTRO] [--output-prefix PREFIX]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

UTC = dt.timezone.utc  # noqa: UP017 - Python 3.10 compatibility

BOT_LOGINS = {"greptile-apps[bot]", "copy-pr-bot[bot]", "github-actions[bot]"}


def load(p: Path) -> object:
    return json.loads(p.read_text()) if p.exists() else []


def blockquote(s: str | None) -> str:
    """Render a complete comment body as an indented Markdown blockquote."""
    lines = (s or "").strip().splitlines()
    return "\n".join(f"  > {line}" for line in lines)


def main() -> None:  # noqa: C901, PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".curator-pr-review/corpus")
    ap.add_argument("--numbers-file", default="_pr_numbers.txt")
    ap.add_argument("--repo", default="NVIDIA-NeMo/Curator")
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

    today = args.today or dt.datetime.now(UTC).date().isoformat()
    outdir = Path(args.outdir)
    nums_file = Path(args.numbers_file)
    if not nums_file.is_absolute():
        nums_file = outdir / nums_file
    if not nums_file.exists():
        msg = f"no {nums_file}; run the corpus pull script first"
        raise SystemExit(msg)
    numbers = [int(x) for x in nums_file.read_text().split() if x.strip()]
    numbers.sort(reverse=True)

    date_us = today.replace("-", "_")
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
        url = gh.get("url", f"https://github.com/{args.repo}/pull/{n}")

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
                sec.append(f"{blockquote(r.get('body'))}\n")

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
                    sec.append(f"{blockquote(body)}\n")

        human_ic = [
            c
            for c in icomments
            if (c.get("user") or {}).get("login") not in BOT_LOGINS and (c.get("body") or "").strip()
        ]
        if human_ic:
            sec.append("### Discussion (top-level)\n")
            for c in human_ic:
                login = (c.get("user") or {}).get("login", "?")
                sec.append(f"- **@{login}** {c.get('created_at', '')}:\n")
                sec.append(f"{blockquote(c.get('body'))}\n")

        per_pr_sections.append("\n".join(sec))

    out.append(f"Total inline review comments included: **{total_comments}**\n")
    out.append("---\n")
    out.extend(per_pr_sections)

    outpath = outdir / f"{args.output_prefix}_{date_us}.md"
    outpath.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {outpath}  ({outpath.stat().st_size} bytes; {len(numbers)} PRs)")


if __name__ == "__main__":
    main()
