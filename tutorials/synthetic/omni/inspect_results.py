#!/usr/bin/env python3
"""Inspect OCR pipeline output files and print a summary of results.

Usage:
    python inspect_results.py /path/to/output_dir_or_file [--show-errors] [--show-sample N]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def iter_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  WARN: bad JSON line in {path}: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="JSONL file or directory of JSONL files")
    parser.add_argument("--show-errors", action="store_true")
    parser.add_argument("--show-sample", type=int, default=0, metavar="N",
                        help="Print N sample records (one per route)")
    args = parser.parse_args()

    p = Path(args.path)
    files = sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    if not files:
        print(f"No JSONL files found at {p}", file=sys.stderr)
        sys.exit(1)

    total = 0
    routes = Counter()
    ocr_status = Counter()  # None / empty_list / has_words
    verif_status = Counter()  # not_run / passed / failed / no_parse
    errors = []
    samples: dict[str, dict] = {}  # route -> first record

    for fpath in files:
        for d in iter_jsonl(fpath):
            total += 1
            route = d.get("ocr_language_route") or "none"
            routes[route] += 1

            # OCR status — distinguish None from []
            ocr_dense = d.get("ocr_dense")
            if ocr_dense is None:
                ocr_status["no_ocr"] += 1
            elif len(ocr_dense) == 0:
                ocr_status["ocr_empty"] += 1
            else:
                ocr_status["ocr_words"] += 1

            # Verification status
            verif_answers = d.get("ocr_verification_answers")
            verif_raw = d.get("ocr_verification_response_raw")
            if verif_raw is None and verif_answers is None:
                verif_status["not_run"] += 1
            elif verif_answers is None:
                verif_status["parse_failed"] += 1
            else:
                is_valid = d.get("is_valid", True)
                if is_valid:
                    verif_status["passed"] += 1
                else:
                    verif_status["failed"] += 1

            if d.get("error"):
                errors.append(d["error"])

            if route not in samples and args.show_sample > 0:
                samples[route] = d

    print(f"Files: {len(files)}, Records: {total}")
    print(f"\nRoutes:        {dict(routes)}")
    print(f"OCR status:    {dict(ocr_status)}")
    print(f"Verif status:  {dict(verif_status)}")
    if errors:
        print(f"\nErrors ({len(errors)} total):")
        seen = set()
        for e in errors:
            key = e[:80]
            if key not in seen:
                seen.add(key)
                print(f"  {e[:120]}")
                if args.show_errors:
                    pass  # already printed

    if args.show_sample and samples:
        print(f"\nSamples (one per route):")
        for route, d in samples.items():
            print(f"\n--- route={route} ---")
            # Print key fields only
            for k in [
                "image_id", "is_valid", "error",
                "ocr_language_route",
                "ocr_has_text", "ocr_has_english", "ocr_has_chinese",
                "ocr_verification_model",
                "ocr_verification_answers",
                "ocr_verification_response_raw",
            ]:
                v = d.get(k)
                if v is not None:
                    if isinstance(v, str) and len(v) > 120:
                        v = v[:120] + "..."
                    print(f"  {k}: {v!r}")
            ocr = d.get("ocr_dense")
            if ocr is not None:
                print(f"  ocr_dense: [{len(ocr)} words]")
                if ocr and args.show_sample >= 2:
                    for w in ocr[:3]:
                        print(f"    {w}")


if __name__ == "__main__":
    main()
