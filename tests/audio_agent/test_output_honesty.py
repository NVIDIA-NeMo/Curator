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

"""A run says how much of its output carries a value, not merely how many rows exist.

The run this pins down wrote a 4-row manifest holding 21 ALM windows, reported "4 rows", and
was announced as complete -- while three of those rows had an empty ``filtered_windows`` and 20
of the 21 windows held a single speaker. The per-field fill counts that contradict the claim
were already being computed, but only for a run that declared acceptance criteria, so the runs
nobody was checking were the only ones reporting nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar

from nemo_curator.audio_agent import verbs
from nemo_curator.audio_agent.recipe import Recipe, StageRef
from nemo_curator.audio_agent.report import rows_written_in, sparse_fields_in

if TYPE_CHECKING:
    import pytest

_Scan = tuple[list[dict[str, Any]], dict[str, Any]]


def _manifest(tmp_path: Path, rows: list[dict[str, Any]], name: str = "out.jsonl") -> str:
    path = tmp_path / name
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return str(path)


def _count_readbacks(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Record which locations get read, while still performing the real read."""
    calls: list[list[str]] = []
    real = verbs._scan_terminal_output

    def counted(outputs: list[str], *, limit: int = 0) -> _Scan:
        calls.append(list(outputs))
        return real(outputs, limit=limit)

    monkeypatch.setattr(verbs, "_scan_terminal_output", counted)
    return calls


# The shape of the run that prompted this: one row carried windows, three did not.
_ALM_ROWS: list[dict[str, Any]] = [
    {"audio_filepath": "a.wav", "filtered_windows": [{"start": 0.0}, {"start": 30.0}]},
    {"audio_filepath": "b.wav", "filtered_windows": []},
    {"audio_filepath": "c.wav", "filtered_windows": []},
    {"audio_filepath": "d.wav"},
]


class TestSparseFieldsNamesWhatIsMissing:
    def test_a_field_blank_in_most_rows_is_reported(self, tmp_path: Path) -> None:
        _rows, scan = verbs._scan_terminal_output([_manifest(tmp_path, _ALM_ROWS)], limit=0)

        sparse = {entry["field"]: entry for entry in sparse_fields_in(scan)}

        assert sparse["filtered_windows"] == {
            "field": "filtered_windows",
            "rows": 4,
            "non_empty": 1,
            "empty": 3,
        }

    def test_a_field_absent_from_a_row_counts_as_missing_a_value(self, tmp_path: Path) -> None:
        """Present-but-empty and absent-entirely are the same fact to whoever reads the row."""
        _rows, scan = verbs._scan_terminal_output([_manifest(tmp_path, _ALM_ROWS)], limit=0)

        # filtered_windows is absent from the fourth row and empty in two others: 3 rows short.
        assert next(e for e in sparse_fields_in(scan) if e["field"] == "filtered_windows")["empty"] == 3

    def test_a_fully_populated_field_is_not_reported(self, tmp_path: Path) -> None:
        _rows, scan = verbs._scan_terminal_output([_manifest(tmp_path, _ALM_ROWS)], limit=0)

        assert "audio_filepath" not in {entry["field"] for entry in sparse_fields_in(scan)}

    def test_a_complete_output_reports_nothing(self, tmp_path: Path) -> None:
        rows = [{"audio_filepath": f"{i}.wav", "text": "hello"} for i in range(3)]
        _rows, scan = verbs._scan_terminal_output([_manifest(tmp_path, rows)], limit=0)

        assert sparse_fields_in(scan) == []

    def test_the_worst_field_comes_first(self, tmp_path: Path) -> None:
        rows = [
            {"a": 1, "b": 1, "c": 1},
            {"a": 1, "b": None},
            {"a": 1, "b": None},
        ]
        _rows, scan = verbs._scan_terminal_output([_manifest(tmp_path, rows)], limit=0)

        assert [entry["field"] for entry in sparse_fields_in(scan)] == ["b", "c"]

    def test_an_unreadable_output_reports_nothing_rather_than_guessing(self) -> None:
        _rows, scan = verbs._scan_terminal_output(["/nonexistent/out.jsonl"], limit=0)

        assert sparse_fields_in(scan) == []


class TestRowsWrittenSeparatesEmptyFromUnread:
    def test_it_counts_the_rows_on_disk(self, tmp_path: Path) -> None:
        _rows, scan = verbs._scan_terminal_output([_manifest(tmp_path, _ALM_ROWS)], limit=0)

        assert rows_written_in(scan) == 4

    def test_an_output_that_could_not_be_read_is_unknown_not_zero(self) -> None:
        """Zero claims the output exists and holds nothing; that is a different statement."""
        _rows, scan = verbs._scan_terminal_output(["/nonexistent/out.jsonl"], limit=0)

        assert rows_written_in(scan) is None

    def test_an_empty_file_really_is_zero(self, tmp_path: Path) -> None:
        _rows, scan = verbs._scan_terminal_output([_manifest(tmp_path, [])], limit=0)

        assert rows_written_in(scan) == 0

    def test_no_declared_output_is_unknown(self) -> None:
        _rows, scan = verbs._scan_terminal_output([], limit=0)

        assert rows_written_in(scan) is None


class TestTheReadbackHappensOnce:
    """Reading the output for the report must not add a second pass for acceptance."""

    def _recipe(self, output: str) -> Recipe:
        source = str(Path(output).parent / "audio")
        return Recipe(
            stages=[
                StageRef(ref="CreateInitialManifestAudioFolderStage", params={"data_dir": source}),
                StageRef(ref="ManifestWriterStage", params={"output_path": output}),
            ],
            acceptance_criteria=[
                {
                    "id": "windows",
                    "type": "output_completeness",
                    "severity": "must",
                    "check": {"field": "filtered_windows", "op": "non_empty"},
                }
            ],
        )

    def test_a_covering_scan_is_reused_instead_of_re_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output = _manifest(tmp_path, _ALM_ROWS)
        rec = self._recipe(output)
        _rows, scan = verbs._scan_terminal_output([output], limit=0)

        calls = _count_readbacks(monkeypatch)
        verbs._acceptance_result(rec, _Report(), ["audio_filepath"], ["audio_filepath"], [output], output_scan=scan)

        assert calls == []

    def test_a_scan_of_a_different_output_is_not_trusted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output = _manifest(tmp_path, _ALM_ROWS)
        elsewhere = _manifest(tmp_path, [{"filtered_windows": [1]}], name="other.jsonl")
        rec = self._recipe(output)
        _rows, stale = verbs._scan_terminal_output([elsewhere], limit=0)

        calls = _count_readbacks(monkeypatch)
        result = verbs._acceptance_result(
            rec, _Report(), ["audio_filepath"], ["audio_filepath"], [output], output_scan=stale
        )

        assert calls == [[output]], "a scan naming another location must be re-read, not believed"
        # The re-read is what keeps the verdict honest: the real output is 1-of-4, not 1-of-1.
        assert result.get("overall") != "met"

    def test_the_contract_still_fails_on_the_shared_scan(self, tmp_path: Path) -> None:
        """Sharing the read must not weaken what the contract concludes from it."""
        output = _manifest(tmp_path, _ALM_ROWS)
        rec = self._recipe(output)
        _rows, scan = verbs._scan_terminal_output([output], limit=0)

        shared = verbs._acceptance_result(
            rec, _Report(), ["audio_filepath"], ["audio_filepath"], [output], output_scan=scan
        )
        unshared = verbs._acceptance_result(rec, _Report(), ["audio_filepath"], ["audio_filepath"], [output])

        assert shared == unshared
        assert shared.get("overall") != "met"


class _Report:
    """The two numbers ``_acceptance_result`` reads off a RunReport."""

    accepted = 4
    input_count = 4
    output_paths: ClassVar[list[str]] = []


class TestRunReportsItWithoutBeingAsked:
    """The counts have to reach the caller of ``run``, and without a success contract."""

    def _drive(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recipe: dict[str, Any]) -> dict[str, Any]:
        from nemo_curator.audio_agent import run_store

        env = SimpleNamespace(has_gpu=False, gpu_count=0, to_dict=dict)
        plan = SimpleNamespace(
            feasible=True,
            mode="batch",
            machine_fingerprint="machine",
            escalations=[],
            to_dict=lambda: {"mode": "batch"},
        )
        source = tmp_path / "source.jsonl"
        source.write_text('{"audio_filepath": "/tmp/a.wav"}\n', encoding="utf-8")

        monkeypatch.delenv("AUDIO_AGENT_REQUIRE_SMOKE", raising=False)
        monkeypatch.setattr(verbs, "probe_env", lambda: env)
        monkeypatch.setattr(verbs, "build_stages", lambda _rec: ([object()], []))
        monkeypatch.setattr(verbs, "_plan_resources", lambda *_a, **_k: plan)
        # Four rows come back in memory; the file on disk is what the report must read.
        monkeypatch.setattr(
            verbs,
            "_run_pipeline_autofallback",
            lambda *_a, **_k: ([SimpleNamespace(num_items=4)], "batch"),
        )
        monkeypatch.setattr(verbs, "_publish_artifacts", lambda *_a, **_k: [])
        monkeypatch.setattr(verbs, "_record_run", lambda *_a, **_k: "run-test")
        monkeypatch.setattr(run_store, "new_run_id", lambda _config_hash: "run-test")

        stages = [
            {"ref": "ManifestReader", "params": {"manifest_path": str(source)}},
            {"ref": "GetAudioDurationStage", "params": {}},
            *recipe["stages"],
        ]
        return verbs.run({**recipe, "stages": stages}, confirm=True)

    def test_a_run_with_no_success_contract_still_reports_the_blank_rows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output = _manifest(tmp_path, _ALM_ROWS)
        result = self._drive(
            tmp_path,
            monkeypatch,
            {"stages": [{"ref": "ManifestWriterStage", "params": {"output_path": output}}]},
        )

        report = result["report"]
        assert report["output_rows_written"] == 4
        sparse = {entry["field"]: entry["non_empty"] for entry in report["sparse_fields"]}
        assert sparse["filtered_windows"] == 1, "the run must say 1 of 4 rows carries the deliverable"

    def test_a_fully_populated_output_adds_no_noise(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        rows = [{"audio_filepath": f"{i}.wav", "duration": 1.0} for i in range(4)]
        output = _manifest(tmp_path, rows)
        result = self._drive(
            tmp_path,
            monkeypatch,
            {"stages": [{"ref": "ManifestWriterStage", "params": {"output_path": output}}]},
        )

        assert result["report"]["sparse_fields"] == []
        assert result["report"]["output_rows_written"] == 4
