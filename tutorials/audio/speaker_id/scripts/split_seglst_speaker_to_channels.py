#!/usr/bin/env python3
"""
Split a mixed seglst (speaker1/speaker2) into two files with channel labels c1/c2.

Output: en_<id>_c1.seglst.json and en_<id>_c2.seglst.json per input en_<id>.seglst.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SPEAKER_TO_CH = {"speaker1": "c1", "speaker2": "c2"}


def split_file(in_path: Path, out_dir: Path) -> None:
    m = re.match(r"^(en_[^.]+)\.seglst\.json$", in_path.name)
    if not m:
        return
    base = m.group(1)
    with in_path.open() as f:
        segments = json.load(f)
    by_ch: dict[str, list[dict]] = {"c1": [], "c2": []}
    for seg in segments:
        sp = seg.get("speaker")
        ch = SPEAKER_TO_CH.get(sp)
        if ch is None:
            raise ValueError(f"{in_path}: unknown speaker {sp!r}")
        row = dict(seg)
        row["speaker"] = ch
        row["session_id"] = f"{base}_{ch}"
        by_ch[ch].append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ch in ("c1", "c2"):
        out_path = out_dir / f"{base}_{ch}.seglst.json"
        with out_path.open("w") as f:
            json.dump(by_ch[ch], f, indent=4)
            f.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing en_*.seglst.json",
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Destination for en_*_c1.seglst.json and en_*_c2.seglst.json",
    )
    args = ap.parse_args()
    ind: Path = args.input_dir
    if not ind.is_dir():
        print(f"error: not a directory: {ind}", file=sys.stderr)
        return 1
    paths = sorted(ind.glob("en_*.seglst.json"))
    if not paths:
        print(f"error: no en_*.seglst.json under {ind}", file=sys.stderr)
        return 1
    for p in paths:
        split_file(p, args.output_dir)
        print(p.name)
    print(f"done: {len(paths)} sessions -> {2 * len(paths)} files in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
