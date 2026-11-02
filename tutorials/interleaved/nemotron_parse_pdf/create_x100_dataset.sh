#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
set -euo pipefail

"${PYTHON:-python3}" - "$@" <<'PY'
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

if len(sys.argv) != 3:
    raise SystemExit("usage: create_x100_dataset.sh SOURCE DESTINATION")
source, destination = (Path(value).resolve() for value in sys.argv[1:])
if destination.exists():
    raise SystemExit(f"Destination already exists: {destination}")
if destination.is_relative_to(source):
    raise SystemExit("Destination must be outside SOURCE")
pdf_root = source / "pdfs"
records = []
with (source / "manifest.jsonl").open() as stream:
    for line_number, line in enumerate(stream, 1):
        record = json.loads(line)
        name = record.get("file_name")
        if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
            raise SystemExit(f"Invalid file_name at line {line_number}")
        path = (pdf_root / name).resolve()
        if name.startswith("pdfs/"):
            path = (source / name).resolve()
        if not path.is_relative_to(pdf_root) or not path.is_file():
            raise SystemExit(f"Missing source PDF at line {line_number}: {name}")
        records.append((record, path.relative_to(pdf_root)))
source_pdfs = [path for path in pdf_root.rglob("*") if path.suffix.lower() == ".pdf"]
if not records or not source_pdfs or any(path.is_symlink() or not path.is_file() for path in source_pdfs):
    raise SystemExit("SOURCE needs nonempty manifest and regular PDF files")
destination.mkdir(parents=True)


def copy(replica):
    target = destination / f"copy-{replica:02d}"
    target.mkdir()
    subprocess.run(["cp", "-a", "--reflink=never", str(source) + "/.", str(target)], check=True)


with ThreadPoolExecutor(max_workers=8) as executor:
    list(executor.map(copy, range(100)))
with (destination / "manifest.jsonl").open("x") as stream:
    for replica in range(100):
        for record, relative in records:
            copied_name = f"copy-{replica:02d}/pdfs/{relative.as_posix()}"
            copied_path = destination / copied_name
            if not copied_path.is_file():
                raise SystemExit(f"Missing copied PDF: {copied_path}")
            stream.write(json.dumps({**record, "file_name": copied_name}) + "\n")
copied_pdfs = [path for path in destination.rglob("*") if path.suffix.lower() == ".pdf"]
if len(copied_pdfs) != len(source_pdfs) * 100:
    raise SystemExit("Copied PDF count does not match source count times 100")
source_inodes = {(path.stat().st_dev, path.stat().st_ino) for path in source_pdfs}
copied_inodes = {(path.stat().st_dev, path.stat().st_ino) for path in copied_pdfs}
if source_inodes & copied_inodes or len(copied_inodes) != len(copied_pdfs):
    raise SystemExit("PDF copies must use independent physical files")
print(f"Validated {len(copied_pdfs)} PDFs and {len(records) * 100} manifest rows")
PY
