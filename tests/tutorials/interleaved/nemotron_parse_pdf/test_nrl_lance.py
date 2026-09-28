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

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

TUTORIAL_DIR = Path(__file__).resolve().parents[4] / "tutorials" / "interleaved" / "nemotron_parse_pdf"
sys.path.insert(0, str(TUTORIAL_DIR))

import nrl_lance as cli  # noqa: E402

INGEST = ["nrl_lance", "ingest", "--input-dir", "/data/pdfs", "--output-root", "/data/runs"]


def test_cli_contains_only_ingest_and_consume() -> None:
    parser = cli.create_parser()
    subparsers = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    assert set(subparsers.choices) == {"ingest", "consume"}


def test_ingest_requires_an_explicit_output_root() -> None:
    with pytest.raises(SystemExit) as error:
        cli.create_parser().parse_args(["ingest", "--input-dir", "/data/pdfs"])
    assert error.value.code == 2


@pytest.mark.parametrize(("flags", "expected"), [([], None), (["--projection-block-rows", "16"], 16)])
def test_cli_passes_optional_projection_block_rows(
    monkeypatch: pytest.MonkeyPatch, flags: list[str], expected: int | None
) -> None:
    calls = []
    monkeypatch.setattr(cli, "run_ingest", lambda args: calls.append(args.projection_block_rows))
    monkeypatch.setattr(sys, "argv", [*INGEST, *flags])
    cli.main()
    assert calls == [expected]


@pytest.mark.parametrize(
    ("flags", "expected"), [([], (64, 1)), (["--parse-cpus", "4"], (64, 4)), (["--parse-batch-size", "128"], (128, 1))]
)
def test_cli_passes_parse_scheduling_to_ingest(
    monkeypatch: pytest.MonkeyPatch, flags: list[str], expected: tuple[int, int]
) -> None:
    calls = []
    monkeypatch.setattr(cli, "run_ingest", lambda args: calls.append((args.parse_batch_size, args.parse_cpus)))
    monkeypatch.setattr(sys, "argv", [*INGEST, *flags])
    cli.main()
    assert calls == [expected]


@pytest.mark.parametrize(
    "flags",
    [
        ["--parse-cpus", "0"],
        ["--parse-cpus", "-1"],
        ["--parse-cpus", "1.5"],
        ["--parse-batch-size", "0"],
        ["--parse-batch-size", "1"],
        ["--parse-batch-size", "1.5"],
        ["--projection-block-rows", "0"],
        ["--projection-block-rows", "-1"],
        ["--projection-block-rows", "1.5"],
        ["--projection-workers", "0"],
        ["--projection-workers", "9"],
        ["--run-id", ".."],
        ["--run-id", "nested/run"],
    ],
)
def test_cli_rejects_invalid_options_before_ingest(monkeypatch: pytest.MonkeyPatch, flags: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", [*INGEST, *flags])
    monkeypatch.setattr(cli, "run_ingest", lambda _args: pytest.fail("Invalid options reached ingestion"))
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
