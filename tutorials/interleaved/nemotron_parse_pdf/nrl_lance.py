# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""CLI for the NRL Nemotron Parse to NeMo Curator Lance handoff."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

import nrl_lance_contract as contract  # noqa: E402
from nrl_lance_runtime import run_consume, run_ingest  # noqa: E402


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser(
        "ingest",
        help="Deduplicate PDFs, run NRL Nemotron Parse extraction, and write a validated Lance handoff",
    )
    source = ingest.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-dir", help="Directory searched recursively for PDFs")
    source.add_argument("--manifest", help="JSONL with a path and optional url and valid_blank_pages per PDF")
    ingest.add_argument("--output-root", required=True, help="Parent directory for fresh run directories")
    ingest.add_argument("--run-id", help="Name of the new run directory; defaults to a timestamp")
    ingest.add_argument(
        "--projection-workers",
        type=int,
        default=contract.MAX_PROJECTION_WORKERS,
        help=f"Maximum CPU projection workers (1-{contract.MAX_PROJECTION_WORKERS})",
    )
    ingest.add_argument(
        "--projection-block-rows",
        type=int,
        help="Target page rows per block before CPU projection; omitted preserves existing blocks",
    )
    ingest.add_argument(
        "--parse-batch-size",
        type=int,
        default=contract.DEFAULT_PARSE_BATCH_SIZE,
        help="Pages requested per Parse Ray batch (>=2)",
    )
    ingest.add_argument(
        "--parse-cpus",
        type=int,
        default=contract.DEFAULT_PARSE_CPUS,
        help="CPU reservation for the single Parse actor (>=1)",
    )

    consume = commands.add_parser(
        "consume",
        help="Run the pinned Lance table through native Curator stages and publish completion",
    )
    consume.add_argument("--handoff-manifest", required=True)
    consume.add_argument("--output-dir", required=True, help="Fresh directory for the Parquet export")
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    if args.command == "consume":
        print(run_consume(args))
        return
    try:
        contract.validate_projection_workers(args.projection_workers)
        contract.validate_parse_scheduling(args.parse_batch_size, args.parse_cpus)
        contract.validate_projection_block_rows(args.projection_block_rows)
        if args.run_id is not None:
            contract.validate_run_id(args.run_id)
    except ValueError as error:
        parser.error(str(error))
    print(run_ingest(args))


if __name__ == "__main__":
    main()
