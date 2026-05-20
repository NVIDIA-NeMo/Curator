"""Run-level lineage for ``run_warc_pipeline.py``.

Writes a ``_run.json`` sidecar in each output directory recording:

* When the run started / finished (UTC, ISO-8601)
* Which Curator git revision produced the output
* The full resolved CLI args (preset overrides folded in)
* A per-stage funnel parsed back out of the run log (records → docs → rows →
  tokens through every filter)

Kept in a separate module from the pipeline script so the script stays focused
on building and launching the pipeline; lineage bookkeeping lives here.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path


# ---------------------------------------------------------------------------
# Time + git helpers (used to stamp the manifest)
# ---------------------------------------------------------------------------


def utc_iso() -> str:
    """Current UTC time as ``"YYYY-MM-DDTHH:MM:SSZ"`` (ISO-8601, Z = UTC)."""
    return _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_sha() -> str | None:
    """``git rev-parse HEAD`` of the Curator checkout this module lives in.

    Returns ``None`` if git isn't available or this isn't a checkout — the
    manifest still gets written, just without a sha.
    """
    repo = Path(__file__).resolve().parent
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo), capture_output=True, text=True, timeout=3,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return None


# ---------------------------------------------------------------------------
# Funnel — parses the run log emitted by LoggingInterleavedFilterStage
# ---------------------------------------------------------------------------

# Two regexes — one for the extractor's baseline line, one for every
# LoggingInterleavedFilterStage line.  Both formats are intentionally stable:
# any change here means the parser must change too.
_FUNNEL_FILTER_RE = re.compile(
    r"\[(?P<stage>[\w]+)\]\s+"
    r"docs\s+(?P<docs_in>\d+)\s+→\s+(?P<docs_out>\d+).*?"
    r"rows\s+(?P<rows_in>\d+)\s+→\s+(?P<rows_out>\d+).*?"
    r"tokens\s+(?P<tokens_in>\d+)\s+→\s+(?P<tokens_out>\d+)"
)
_FUNNEL_EXTRACTOR_RE = re.compile(
    r"\[(?P<stage>warc_to_interleaved_extract)/\w+\]\s+"
    r"records\s+(?P<records>\d+)\s+→\s+docs\s+(?P<docs>\d+),\s+"
    r"rows\s+(?P<rows>\d+),\s+tokens\s+(?P<tokens>\d+)"
)


def _parse_funnel(log_path: Path) -> list[dict]:
    """Read the run log and aggregate per-stage counts across all batches.

    Returns one dict per stage in the order each stage first appears.  Filter
    stages report
        {stage, docs_in, docs_out, rows_in, rows_out, tokens_in, tokens_out}
    The extractor reports
        {stage, records, docs, rows, tokens}
    """
    if not log_path.is_file():
        return []
    extractor_totals: dict[str, dict[str, int]] = OrderedDict()
    filter_totals: dict[str, dict[str, int]] = OrderedDict()
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = _FUNNEL_EXTRACTOR_RE.search(line)
                if m:
                    s = m.group("stage")
                    bucket = extractor_totals.setdefault(s, {
                        "stage": s, "records": 0, "docs": 0,
                        "rows": 0, "tokens": 0,
                    })
                    bucket["records"] += int(m.group("records"))
                    bucket["docs"]    += int(m.group("docs"))
                    bucket["rows"]    += int(m.group("rows"))
                    bucket["tokens"]  += int(m.group("tokens"))
                    continue
                m = _FUNNEL_FILTER_RE.search(line)
                if m:
                    s = m.group("stage")
                    bucket = filter_totals.setdefault(s, {
                        "stage": s,
                        "docs_in": 0, "docs_out": 0,
                        "rows_in": 0, "rows_out": 0,
                        "tokens_in": 0, "tokens_out": 0,
                    })
                    for k in ("docs_in", "docs_out", "rows_in", "rows_out",
                              "tokens_in", "tokens_out"):
                        bucket[k] += int(m.group(k))
    except OSError:
        return []
    return list(extractor_totals.values()) + list(filter_totals.values())


# ---------------------------------------------------------------------------
# Manifest writer
# ---------------------------------------------------------------------------


def write_run_manifest(
    args: argparse.Namespace, started: str, finished: str
) -> None:
    """Write ``_run.json`` sidecar in ``args.output_path``.

    Best-effort: any I/O failure logs a warning to stderr and otherwise
    leaves the pipeline untouched.
    """
    funnel: list[dict] = []
    log_path = getattr(args, "log_path", None)
    if log_path:
        funnel = _parse_funnel(Path(log_path))

    sha = _git_sha()
    manifest = {
        "run_id": f"{started}-{(sha or 'nogit')[:7]}",
        "snapshot": None,  # filled in if any output Parquet exists & has lineage
        "preset": args.preset,
        "extractor": args.extractor,
        "git_sha": sha,
        "input_path": args.input_path,
        "output_path": args.output_path,
        "started_at": started,
        "finished_at": finished,
        "funnel": funnel,
        "cli_args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "host": os.uname().nodename,
        "user": os.environ.get("USER", "?"),
    }
    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "_run.json"
    try:
        out.write_text(json.dumps(manifest, indent=2, default=str))
        print(f"[run-manifest] wrote {out}", file=sys.stderr)
    except OSError as e:
        print(f"[run-manifest] WARNING — could not write {out}: {e}", file=sys.stderr)
