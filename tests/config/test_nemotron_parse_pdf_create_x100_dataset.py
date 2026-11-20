# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Check repeated manifest paths and physical copies on a tiny corpus."""

import json
import os
import subprocess
import sys
from pathlib import Path

TUTORIAL = Path(__file__).parents[2] / "tutorials/interleaved/nemotron_parse_pdf"


def test_compact_json_paths_and_metadata_survive_copying(tmp_path: Path):
    source = tmp_path / "source"
    (source / "pdfs/nested").mkdir(parents=True)
    (source / "pdfs/nested/a.pdf").write_bytes(b"tiny fixture")
    (source / "manifest.jsonl").write_text('{"file_name":"nested/a.pdf","selected_page_indices":[2]}\n')
    destination = tmp_path / "copies"
    subprocess.run(  # noqa: S603
        ["bash", str(TUTORIAL / "create_x100_dataset.sh"), str(source), str(destination)],  # noqa: S607
        check=True,
        env={**os.environ, "PYTHON": sys.executable},
        capture_output=True,
        timeout=30,
    )
    records = [json.loads(line) for line in (destination / "manifest.jsonl").read_text().splitlines()]
    assert len(records) == 100
    assert all(record["selected_page_indices"] == [2] for record in records)
    assert all((destination / record["file_name"]).is_file() for record in records)
    assert records[0]["file_name"] == "copy-00/pdfs/nested/a.pdf"
    assert len({(destination / record["file_name"]).stat().st_ino for record in records}) == 100
