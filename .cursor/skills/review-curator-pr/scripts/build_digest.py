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
"""Build a Curator PR fresh-review digest + open-comment queue.

Usage: build_digest.py <PR_NUMBER> [--outdir DIR] [--today YYYY-MM-DD]
                        [--prev-head SHA] [--baseline-ts TS]
                        [--path-regex REGEX] [--modality-label LABEL]

Reads the pr<N>_*_latest.json files written by pr_review_pull.sh from --outdir
(default .curator-pr-review) and writes into the same directory:
  curator_pr<N>_fresh_review_<date>.md          full digest
  curator_pr<N>_github_comment_queue_<date>.md  open-thread paste queue
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

BODY_KEY_LEN = 120
AUDIO_AREA_PARTS = 4
TUTORIAL_AREA_PARTS = 3


def load(path: Path) -> object:
    return json.loads(path.read_text())


def load_baseline_ids(baseline: Path | None, key: str) -> set[Any]:
    if baseline is None or not baseline.exists():
        return set()
    return {entry[key] for entry in load(baseline) if key in entry}


def build_thread_index(threads_payload: dict[str, Any]) -> dict[str, Any]:
    idx = {"by_dbid": {}, "by_pathbody": {}, "nthreads": 0}
    try:
        nodes = threads_payload["data"]["repository"]["pullRequest"]["reviewThreads"]["nodes"]
    except (KeyError, TypeError):
        return idx
    idx["nthreads"] = len(nodes)
    for thread in nodes:
        cnodes = thread["comments"]["nodes"]
        path = thread.get("path")
        for pos, c in enumerate(cnodes):
            meta = {
                "thread_id": thread.get("id"),
                "is_resolved": thread.get("isResolved", False),
                "is_outdated": thread.get("isOutdated", False),
                "thread_comment_count": len(cnodes),
                "is_first_in_thread": pos == 0,
            }
            db = c.get("databaseId")
            if db is not None:
                idx["by_dbid"][int(db)] = meta
            body = (c.get("body") or "").strip()
            if body:
                idx["by_pathbody"].setdefault((path, body[:BODY_KEY_LEN]), meta)
    return idx


def thread_meta(idx: dict[str, Any], c: dict[str, Any]) -> dict[str, Any] | None:
    m = idx["by_dbid"].get(c["id"])
    if m is not None:
        return m
    body = (c.get("body") or "").strip()
    return idx["by_pathbody"].get((c.get("path"), body[:BODY_KEY_LEN]))


def shorten(body: str | None, n: int = 600) -> str:
    body = (body or "").strip()
    if len(body) <= n:
        return body
    return body[:n].rstrip() + "\n[...truncated...]"


def status_of(meta: dict[str, Any] | None) -> str:
    if meta is None:
        return "ORPHAN"
    if meta["is_resolved"]:
        return "RESOLVED"
    if meta["is_outdated"]:
        return "OUTDATED"
    return "OPEN"


def p(parts: list[str], *lines: str) -> None:
    parts.extend(lines)


def area_of(path: str) -> str:  # noqa: PLR0911
    parts = path.split("/")
    if path.startswith("nemo_curator/stages/audio/") and len(parts) >= AUDIO_AREA_PARTS:
        return f"stages/audio/{parts[3]}"
    if path.startswith("nemo_curator/tasks/"):
        return "tasks"
    if path.startswith("tutorials/audio/") and len(parts) >= TUTORIAL_AREA_PARTS:
        return f"tutorials/audio/{parts[2]}"
    if path.startswith("tests/"):
        return "tests"
    if path.startswith("benchmarking/"):
        return "benchmarking"
    if path.startswith("fern/"):
        return "docs (fern)"
    return parts[0] if parts else path


def login_of(obj: dict[str, Any]) -> str:
    return (obj.get("user") or {}).get("login", "?")


def build(  # noqa: C901, PLR0912, PLR0913, PLR0915
    pr: int,
    outdir: Path,
    today: str,
    baseline_ts: str | None,
    prev_head: str | None,
    path_regex: str | None,
    modality_label: str | None,
) -> None:
    def latest(kind: str) -> Path:
        return outdir / f"pr{pr}_{kind}_latest.json"

    gh = load(latest("gh"))
    reviews = load(latest("reviews"))
    review_comments = load(latest("review_comments"))
    issue_comments = load(latest("issue_comments"))
    files = load(latest("files"))
    commits = load(latest("commits"))

    if path_regex is not None:
        rx = re.compile(path_regex)
        if files and not any(rx.search(f.get("filename", "")) for f in files):
            label = modality_label or "matching"
            msg = f"PR {pr} touches no {label} path; aborting."
            raise SystemExit(msg)

    threads_path = latest("review_threads")
    idx = build_thread_index(load(threads_path)) if threads_path.exists() else build_thread_index({})

    def base(kind: str) -> Path | None:
        return outdir / f"pr{pr}_{kind}_{baseline_ts}.json" if baseline_ts else None

    base_comment_ids = load_baseline_ids(base("review_comments"), "id")
    base_review_ids = load_baseline_ids(base("reviews"), "id")
    base_issue_ids = load_baseline_ids(base("issue_comments"), "id")
    base_commit_shas = set()
    bc = base("commits")
    if bc and bc.exists():
        base_commit_shas = {c["sha"] for c in load(bc) if "sha" in c}

    head_oid = gh.get("headRefOid", "")
    base_oid = gh.get("baseRefOid", "")
    prev8 = (prev_head or "")[:8] or "(none)"
    date_us = today.replace("-", "_")

    d = []
    p(d, f"# Curator PR {pr} Fresh Review - {today}", "")
    p(d, f"Review target: https://github.com/NVIDIA-NeMo/Curator/pull/{pr}", "")
    p(d, f"Current PR head reviewed: `{head_oid}`", "")
    p(d, f"Base recorded by GitHub metadata: `{base_oid}`", "")
    if prev_head:
        p(d, f"Previous reviewed head: `{prev_head}`", "")
    p(d, "")

    author = (gh.get("author") or {}).get("login", "?")
    p(d, "## What this PR does (overview)", "")
    p(
        d,
        f"**{gh.get('title', '')}** - by @{author}  "
        f"({gh.get('state')}, +{gh.get('additions')}/-{gh.get('deletions')} "
        f"across {gh.get('changedFiles')} files, {len(commits)} commits)",
        "",
    )

    p(d, "### Author's description", "")
    pr_body = (gh.get("body") or "").strip()
    p(d, pr_body if pr_body else "_(no PR description provided)_", "")

    p(d, "### Areas touched", "")
    areas = {}
    for f in files:
        e = areas.setdefault(area_of(f["filename"]), {"n": 0, "add": 0, "dele": 0})
        e["n"] += 1
        e["add"] += f.get("additions", 0)
        e["dele"] += f.get("deletions", 0)
    p(d, "| Area | files | +/- |", "|---|---|---|")
    for a in sorted(areas):
        e = areas[a]
        p(d, f"| {a} | {e['n']} | +{e['add']} -{e['dele']} |")
    p(d, "")

    p(d, "### Plain-language summary (reviewer writes this BEFORE any findings)", "")
    p(
        d,
        "_After reading the knowledge sources, explain in detail what this PR "
        "changes and why before moving to findings._",
        "",
    )

    p(d, "## PR state at review time", "")
    p(d, "| Field | Value |", "|---|---|")
    p(d, f"| state | {gh.get('state')} |")
    p(d, f"| isDraft | {gh.get('isDraft')} |")
    p(d, f"| reviewDecision | {gh.get('reviewDecision') or '(none)'} |")
    p(d, f"| mergeable / mergeStateStatus | {gh.get('mergeable')} / {gh.get('mergeStateStatus')} |")
    p(d, f"| changedFiles | {gh.get('changedFiles')} |")
    p(d, f"| additions / deletions | +{gh.get('additions')} / -{gh.get('deletions')} |")
    p(d, f"| commits | {len(commits)} |")
    p(d, f"| inline review comments total | {len(review_comments)} |")
    p(d, f"| reviews submitted | {len(reviews)} |")
    p(d, f"| top-level issue comments | {len(issue_comments)} |")
    p(d, f"| updatedAt | {gh.get('updatedAt')} |", "")

    new_commits = [c for c in commits if c["sha"] not in base_commit_shas] if base_commit_shas else commits
    p(d, f"## Commits since prior reviewed head `{prev8}` ({len(new_commits)} new)", "")
    p(d, "```")
    for c in new_commits:
        sha = c["sha"][:8]
        msg = (c["commit"]["message"] or "").split("\n", 1)[0]
        cdate = (c["commit"]["author"] or {}).get("date", "")
        p(d, f"{sha}  {msg}  ({cdate[:10]})")
    p(d, "```", "")

    p(d, "Files changed in current PR head (`git diff <base>..<head>`):", "")
    p(d, "| File | +/- | status |", "|---|---|---|")
    for f in sorted(files, key=lambda x: x["filename"]):
        p(d, f"| `{f['filename']}` | +{f['additions']} -{f['deletions']} | {f['status']} |")
    p(d, "")

    p(d, "## Existing reviews (by other reviewers)", "")
    review_by_id = {r["id"]: r for r in reviews}
    for r in sorted(reviews, key=lambda x: x.get("submitted_at") or ""):
        marker = " (NEW)" if base_review_ids and r["id"] not in base_review_ids else ""
        commit = (r.get("commit_id") or "")[:8] or "n/a"
        p(
            d,
            f"### #{r['id']} by @{login_of(r)}  state={r['state']}  "
            f"commit={commit}  submitted={r.get('submitted_at')}{marker}",
        )
        body = (r.get("body") or "").strip()
        if body:
            p(d, "", shorten(body, 1500))
        p(d, "")

    p(d, "## Existing inline comments (by other reviewers)", "")
    p(d, f"Total: {len(review_comments)} comments across {idx['nthreads'] or '?'} threads.", "")
    by_status = {"OPEN": [], "OUTDATED": [], "RESOLVED": [], "ORPHAN": []}
    for c in review_comments:
        by_status[status_of(thread_meta(idx, c))].append(c)
    for s in ("OPEN", "OUTDATED", "RESOLVED", "ORPHAN"):
        p(d, f"- {s}: {len(by_status[s])}")
    p(d, "")

    by_path = {}
    for c in review_comments:
        by_path.setdefault(c["path"], []).append(c)
    for path in sorted(by_path):
        p(d, f"### `{path}`", "")
        for c in sorted(by_path[path], key=lambda x: (x.get("line") or x.get("original_line") or 0, x["created_at"])):
            meta = thread_meta(idx, c)
            status = status_of(meta)
            new = " NEW" if base_comment_ids and c["id"] not in base_comment_ids else ""
            line = c.get("line") or c.get("original_line")
            review = review_by_id.get(c.get("pull_request_review_id"), {})
            commit = (c.get("commit_id") or "")[:8]
            in_reply = c.get("in_reply_to_id")
            reply_marker = f"  reply->{in_reply}" if in_reply else ""
            tcount = meta["thread_comment_count"] if meta else 1
            tpos = " (root)" if meta and meta["is_first_in_thread"] else (" (reply)" if meta else "")
            p(
                d,
                f"- **#{c['id']}** @{login_of(c)} {c['created_at']}  "
                f"line={line} commit={commit} status=**{status}**{new}  "
                f"thread_comments={tcount}{tpos} review_state={review.get('state', '?')}{reply_marker}",
            )
            p(d, f"  url: {c['html_url']}")
            body = shorten(c.get("body") or "", 700).replace("\n", "\n  > ")
            p(d, f"  > {body}", "")
    p(d, "")

    p(d, "## Existing issue comments (by other reviewers)", "")
    for c in sorted(issue_comments, key=lambda x: x["created_at"]):
        new = " NEW" if base_issue_ids and c["id"] not in base_issue_ids else ""
        p(d, f"### #{c['id']} by @{login_of(c)}  {c['created_at']}{new}", "")
        p(d, shorten(c.get("body") or "", 1500), "")

    p(d, "## My findings (your review)", "")
    p(
        d,
        "_Add your findings here as you review. Classify each P0-P3, cite "
        "path:line on the current head, and propose a concrete fix._",
        "",
    )
    p(d, "## Verdict", "")
    p(d, "_APPROVE / COMMENT / REQUEST CHANGES + blockers._", "")

    digest_path = outdir / f"curator_pr{pr}_fresh_review_{date_us}.md"
    digest_path.write_text("\n".join(d) + "\n", encoding="utf-8")
    print(f"wrote {digest_path}  ({digest_path.stat().st_size} bytes)")

    q = []
    p(q, f"# Curator PR {pr} Open Review Threads - {today}", "")
    p(q, f"Review target: https://github.com/NVIDIA-NeMo/Curator/pull/{pr}", "")
    p(q, f"Current PR head: `{head_oid}`", "")
    p(
        q,
        "Threads other reviewers left that are still unresolved on the current "
        "head. Scan these before adding your own comments.",
        "",
    )

    def is_root(c: dict[str, Any]) -> bool:
        return (thread_meta(idx, c) or {}).get("is_first_in_thread", True)

    open_root = [c for c in review_comments if status_of(thread_meta(idx, c)) == "OPEN" and is_root(c)]
    open_root.sort(key=lambda c: (c["path"], c.get("line") or c.get("original_line") or 0, c["created_at"]))
    for i, c in enumerate(open_root, 1):
        meta = thread_meta(idx, c) or {}
        line = c.get("line") or c.get("original_line")
        new = " NEW" if base_comment_ids and c["id"] not in base_comment_ids else ""
        p(q, f"## Thread {i} - {c['path']}:{line} by @{login_of(c)}{new}", "")
        p(q, f"Thread: {c['html_url']}", "")
        if meta.get("thread_comment_count", 1) > 1:
            p(q, f"Thread has {meta['thread_comment_count']} comments - see the digest for the full thread.", "")
        p(q, f"File: `{c['path']}`", "")
        p(q, f"Line: `{line}`", "")
        p(q, "Reviewer text:", "")
        p(q, "```text", (c.get("body") or "").strip(), "```", "")

    p(q, "## Stale (outdated/resolved) comments", "")
    stale = [c for c in review_comments if status_of(thread_meta(idx, c)) in ("OUTDATED", "RESOLVED") and is_root(c)]
    stale.sort(key=lambda c: (status_of(thread_meta(idx, c)), c["path"], c.get("original_line") or 0))
    for c in stale:
        line = c.get("line") or c.get("original_line")
        body = shorten((c.get("body") or "").strip(), 200).replace("\n", " ")
        p(q, f"- [{status_of(thread_meta(idx, c))}] `{c['path']}:{line}` @{login_of(c)} ({c['html_url']}): {body}")
    p(q, "")

    queue_path = outdir / f"curator_pr{pr}_github_comment_queue_{date_us}.md"
    queue_path.write_text("\n".join(q) + "\n", encoding="utf-8")
    print(f"wrote {queue_path}  ({queue_path.stat().st_size} bytes)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pr", type=int, help="PR number")
    ap.add_argument("--outdir", default=".curator-pr-review")
    ap.add_argument("--today", default=None, help="ISO date for filenames; default today UTC")
    ap.add_argument("--baseline-ts", default=None)
    ap.add_argument("--prev-head", default=None)
    ap.add_argument("--path-regex", default=None)
    ap.add_argument("--modality-label", default=None)
    args = ap.parse_args()
    today = args.today or dt.datetime.now(dt.UTC).date().isoformat()
    build(args.pr, Path(args.outdir), today, args.baseline_ts, args.prev_head, args.path_regex, args.modality_label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
