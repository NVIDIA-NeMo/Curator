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

"""Unit tests for ManifestGroupExportStage (per-group txt/json/csv export)."""

import csv
import json
import os
from pathlib import Path

import pytest

from nemo_curator.stages.audio.io.group_export import ManifestGroupExportStage
from nemo_curator.tasks import AudioTask

_ROWS = [
    {"speaker_id": "spk 0", "text": "hello there", "start": 0.0, "end": 1.5},
    {"speaker_id": "spk 1", "text": "general kenobi", "start": 1.5, "end": 3.0},
    {"speaker_id": "spk 0", "text": "you are a bold one", "start": 3.0, "end": 5.0},
]


def _run(stage: ManifestGroupExportStage, rows: list[dict]) -> None:
    stage.setup()
    for row in rows:
        stage.process(AudioTask(dataset_name="t", data=dict(row)))
    stage.teardown()


class TestGroupExport:
    def test_txt_one_file_per_group_with_timestamps(self, tmp_path) -> None:  # noqa: ANN001
        out = str(tmp_path / "by_speaker")
        _run(ManifestGroupExportStage(output_dir=out), _ROWS)
        assert sorted(os.listdir(out)) == ["spk_0.txt", "spk_1.txt"]  # unsafe chars are made filename-safe
        lines = (tmp_path / "by_speaker" / "spk_0.txt").read_text().splitlines()
        assert lines == ["[0.00 - 1.50] hello there", "[3.00 - 5.00] you are a bold one"]

    def test_rows_pass_through_unchanged(self, tmp_path) -> None:  # noqa: ANN001
        # It is a tee, not a sink: a writer can follow it.
        stage = ManifestGroupExportStage(output_dir=str(tmp_path / "g"))
        stage.setup()
        task = AudioTask(dataset_name="t", data=dict(_ROWS[0]))
        assert stage.process(task).data == _ROWS[0]

    def test_json_format_selects_columns(self, tmp_path) -> None:  # noqa: ANN001
        out = str(tmp_path / "g")
        _run(ManifestGroupExportStage(output_dir=out, format="json", columns=["text", "start"]), _ROWS)
        rows = [json.loads(line) for line in (tmp_path / "g" / "spk_0.jsonl").read_text().splitlines()]
        assert rows == [{"text": "hello there", "start": 0.0}, {"text": "you are a bold one", "start": 3.0}]

    def test_csv_writes_one_header(self, tmp_path) -> None:  # noqa: ANN001
        out = str(tmp_path / "g")
        _run(ManifestGroupExportStage(output_dir=out, format="csv", columns=["text"]), _ROWS)
        lines = (tmp_path / "g" / "spk_0.csv").read_text().splitlines()
        assert lines[0] == "text"
        assert len([line for line in lines if line == "text"]) == 1

    def test_a_zero_group_value_is_its_own_group_not_the_missing_one(self, tmp_path: Path) -> None:
        """Speaker ids are commonly zero-indexed, and 0 is a real group, not an absent one.

        Selecting the group with ``row.get(key) or missing_group`` filed every ``speaker_id == 0``
        row under ``unknown``, silently merging a real speaker with the rows that genuinely had
        no value.
        """
        out = str(tmp_path / "g")
        rows = [
            {"speaker_id": 0, "text": "zero is a speaker"},
            {"speaker_id": 1, "text": "so is one"},
            {"text": "this one really has none"},
        ]
        _run(ManifestGroupExportStage(output_dir=out, include_timestamps=False), rows)
        assert sorted(os.listdir(out)) == ["0.txt", "1.txt", "unknown.txt"]
        assert (tmp_path / "g" / "0.txt").read_text().strip() == "zero is a speaker"
        assert (tmp_path / "g" / "unknown.txt").read_text().strip() == "this one really has none"

    def test_csv_columns_stay_under_the_header_they_were_written_for(self, tmp_path: Path) -> None:
        """A heterogeneous manifest must not shift values into the wrong csv columns.

        The header is written once, from the first row. Rows after it can legitimately differ
        in shape -- a missing transcript, a column dropped for being non-serializable, or
        merely a different key insertion order because two code paths built the dicts. Taking
        each row's own keys as the fieldnames wrote those rows under a header they did not
        match, so ``{"speaker_id", "duration"}`` landed duration under ``text``, and a
        reordered row transposed every value at once.
        """
        out = str(tmp_path / "g")
        rows = [
            {"speaker_id": "spk 0", "text": "hello", "duration": 1.0},
            {"speaker_id": "spk 0", "duration": 2.0},  # no text
            {"duration": 3.0, "text": "reordered", "speaker_id": "spk 0"},  # different key order
        ]
        _run(ManifestGroupExportStage(output_dir=out, format="csv"), rows)
        parsed = list(csv.DictReader((tmp_path / "g" / "spk_0.csv").read_text().splitlines()))
        assert [r["duration"] for r in parsed] == ["1.0", "2.0", "3.0"]
        assert [r["text"] for r in parsed] == ["hello", "", "reordered"]
        assert {r["speaker_id"] for r in parsed} == {"spk 0"}

    def test_csv_says_so_when_a_late_column_cannot_be_written(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A column absent from the header is dropped -- rewriting the file is worse -- but said."""
        out = str(tmp_path / "g")
        rows = [
            {"speaker_id": "spk 0", "text": "hello"},
            {"speaker_id": "spk 0", "text": "world", "lang": "en"},  # 'lang' has no header slot
        ]
        with caplog.at_level("WARNING"):
            _run(ManifestGroupExportStage(output_dir=out, format="csv"), rows)
        parsed = list(csv.DictReader((tmp_path / "g" / "spk_0.csv").read_text().splitlines()))
        assert [r["text"] for r in parsed] == ["hello", "world"]
        assert "lang" not in parsed[0]

    def test_timeline_is_ordered_across_groups(self, tmp_path) -> None:  # noqa: ANN001
        out = str(tmp_path / "g")
        _run(ManifestGroupExportStage(output_dir=out, write_timeline=True), list(reversed(_ROWS)))
        lines = (tmp_path / "g" / "timeline.txt").read_text().splitlines()
        assert [line.split("] ")[1].split(":")[0] for line in lines] == ["spk_0", "spk_1", "spk_0"]

    def test_missing_group_and_unserializable_values(self, tmp_path) -> None:  # noqa: ANN001
        out = str(tmp_path / "g")
        _run(
            ManifestGroupExportStage(output_dir=out, format="json"),
            [{"text": "no speaker", "waveform": object()}],
        )
        assert os.path.isfile(os.path.join(out, "unknown.jsonl"))
        # A resident tensor/object must be dropped, not crash the export.
        assert json.loads((tmp_path / "g" / "unknown.jsonl").read_text()) == {"text": "no speaker"}

    def test_rerun_replaces_its_own_output(self, tmp_path) -> None:  # noqa: ANN001
        out = str(tmp_path / "g")
        stage = ManifestGroupExportStage(output_dir=out)
        _run(stage, _ROWS)
        _run(stage, _ROWS)
        assert len((tmp_path / "g" / "spk_0.txt").read_text().splitlines()) == 2  # not 4

    def test_output_dir_and_format_are_validated(self, tmp_path) -> None:  # noqa: ANN001
        with pytest.raises(ValueError, match="output_dir is required"):
            ManifestGroupExportStage(output_dir="")
        with pytest.raises(ValueError, match="format must be one of"):
            ManifestGroupExportStage(output_dir=str(tmp_path), format="parquet")

    def test_declares_its_disk_gate_without_instantiation(self) -> None:
        # output_dir is required, so discovery uses the static contract; it must still show
        # that this stage writes to disk.
        gates = ManifestGroupExportStage.describe_static().gates
        assert gates.writes_to_disk and gates.lifecycle_side_effects  # noqa: PT018
