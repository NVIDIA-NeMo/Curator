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

"""Download a small set of public arxiv PDFs and write a manifest.jsonl
suitable for the ``--pdf-dir`` mode of the Nemotron-Parse PDF tutorial.

Usage::

    python download_data.py --output-dir ./data
    # then run main.py with the command printed at the end

Produces::

    <output-dir>/
    ├── pdfs/
    │   ├── attention_is_all_you_need.pdf
    │   ├── gpt3.pdf
    │   └── rag.pdf
    └── manifest.jsonl   # one line per PDF: {"file_name": "...", "url": "..."}

Only stdlib + loguru are required (loguru is already a runtime dep of main.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

DEFAULT_PDFS: list[tuple[str, str]] = [
    ("attention_is_all_you_need.pdf", "https://arxiv.org/pdf/1706.03762.pdf"),
    ("gpt3.pdf", "https://arxiv.org/pdf/2005.14165.pdf"),
    ("rag.pdf", "https://arxiv.org/pdf/2005.11401.pdf"),
    ("bert.pdf", "https://arxiv.org/pdf/1810.04805.pdf"),
    ("resnet.pdf", "https://arxiv.org/pdf/1512.03385.pdf"),
    ("batch_norm.pdf", "https://arxiv.org/pdf/1502.03167.pdf"),
    ("adam.pdf", "https://arxiv.org/pdf/1412.6980.pdf"),
    ("gan.pdf", "https://arxiv.org/pdf/1406.2661.pdf"),
    ("vit.pdf", "https://arxiv.org/pdf/2010.11929.pdf"),
    ("clip.pdf", "https://arxiv.org/pdf/2103.00020.pdf"),
    ("palm.pdf", "https://arxiv.org/pdf/2204.02311.pdf"),
    ("instructgpt.pdf", "https://arxiv.org/pdf/2203.02155.pdf"),
    ("codex.pdf", "https://arxiv.org/pdf/2107.03374.pdf"),
    ("roberta.pdf", "https://arxiv.org/pdf/1907.11692.pdf"),
    ("t5.pdf", "https://arxiv.org/pdf/1910.10683.pdf"),
    ("roformer.pdf", "https://arxiv.org/pdf/2104.09864.pdf"),
    ("lora.pdf", "https://arxiv.org/pdf/2106.09685.pdf"),
    ("llama.pdf", "https://arxiv.org/pdf/2302.13971.pdf"),
    ("bahdanau_attention.pdf", "https://arxiv.org/pdf/1409.0473.pdf"),
    ("dropout.pdf", "https://arxiv.org/pdf/1207.0580.pdf"),
]

_USER_AGENT = "nemo-curator-tutorial-downloader/1.0 (+https://github.com/NVIDIA-NeMo/Curator)"


@dataclass
class DownloadConfig:
    output_dir: Path
    num_pdfs: int
    force: bool
    workers: int


def download_one(
    filename: str,
    url: str,
    pdfs_dir: Path,
    force: bool,
) -> tuple[str, str, Path | None, str | None]:
    """Download a single PDF. Returns (filename, url, local_path or None, error or None)."""
    local_path = pdfs_dir / filename

    if local_path.exists() and not force:
        return (filename, url, local_path, None)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as resp, open(local_path, "wb") as f:
            while chunk := resp.read(64 * 1024):
                f.write(chunk)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        if local_path.exists():
            local_path.unlink()
        return (filename, url, None, str(e))

    return (filename, url, local_path, None)


def download_all(cfg: DownloadConfig) -> list[tuple[str, str, Path]]:
    """Download up to ``cfg.num_pdfs`` from DEFAULT_PDFS in parallel."""
    targets = DEFAULT_PDFS[: cfg.num_pdfs]
    pdfs_dir = cfg.output_dir / "pdfs"
    pdfs_dir.mkdir(parents=True, exist_ok=True)

    total = len(targets)
    logger.info(f"Downloading {total} PDFs into {pdfs_dir} (workers={cfg.workers})")

    successes: list[tuple[str, str, Path]] = []
    failures: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        futures = {ex.submit(download_one, fn, url, pdfs_dir, cfg.force): (fn, url) for fn, url in targets}
        for i, future in enumerate(as_completed(futures), 1):
            fn, url, local_path, error = future.result()
            if error:
                logger.warning(f"[{i}/{total}] FAILED {fn}: {error}")
                failures.append((fn, error))
            else:
                size_mb = local_path.stat().st_size / (1024 * 1024)
                logger.info(f"[{i}/{total}] {fn} ({size_mb:.1f} MB)")
                successes.append((fn, url, local_path))

    if failures:
        logger.warning(f"{len(failures)}/{total} downloads failed")
    return successes


def write_manifest(successes: list[tuple[str, str, Path]], manifest_path: Path) -> None:
    with open(manifest_path, "w") as f:
        for filename, url, _ in successes:
            f.write(json.dumps({"file_name": filename, "url": url}) + "\n")
    logger.info(f"Wrote manifest with {len(successes)} entries to {manifest_path}")


def print_next_command(output_dir: Path, n: int) -> None:
    pdfs_dir = (output_dir / "pdfs").resolve()
    manifest = (output_dir / "manifest.jsonl").resolve()
    out_dir = (output_dir / "output").resolve()
    main_py = (Path(__file__).parent / "main.py").resolve()

    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  Downloaded {n} PDFs to {pdfs_dir}")
    print(f"  Manifest:    {manifest}")
    print(bar)
    print("\n  Next: run the pipeline with:\n")
    print(f"    python {main_py} \\")
    print(f"        --pdf-dir {pdfs_dir} \\")
    print(f"        --manifest {manifest} \\")
    print(f"        --output-dir {out_dir} \\")
    print(f"        --backend vllm --enforce-eager --max-pdfs {n}\n")
    print(f"{bar}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download sample PDFs for the Nemotron-Parse tutorial")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data"),
        help="Output directory; pdfs/ and manifest.jsonl are written underneath (default: ./data)",
    )
    parser.add_argument(
        "--num-pdfs",
        type=int,
        default=3,
        help=f"Number of PDFs to download (1-{len(DEFAULT_PDFS)}, default: 3)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if a PDF already exists")
    parser.add_argument("--workers", type=int, default=3, help="Parallel download workers (default: 3)")
    args = parser.parse_args()

    if not 1 <= args.num_pdfs <= len(DEFAULT_PDFS):
        parser.error(f"--num-pdfs must be between 1 and {len(DEFAULT_PDFS)}")

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = DownloadConfig(
        output_dir=args.output_dir,
        num_pdfs=args.num_pdfs,
        force=args.force,
        workers=args.workers,
    )

    successes = download_all(cfg)
    if not successes:
        logger.error("0 PDFs downloaded successfully — check network access and try again")
        sys.exit(1)

    write_manifest(successes, args.output_dir / "manifest.jsonl")
    print_next_command(args.output_dir, len(successes))


if __name__ == "__main__":
    main()
