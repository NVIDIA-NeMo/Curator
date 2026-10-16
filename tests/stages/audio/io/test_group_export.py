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
import sys
from pathlib import Path

import pytest

from nemo_curator.backends.ray_data.adapter import RayDataStageAdapter
from nemo_curator.stages.audio._agent._agent_registry import build_contract
from nemo_curator.stages.audio._agent._planning import validate_pipeline
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


def _unsafe_group_file(directory: Path, prefix: str, extension: str) -> Path:
    matches = list(directory.glob(f"{prefix}~*.{extension}"))
    assert len(matches) == 1
    return matches[0]


class TestGroupExport:
    def test_unencodable_nested_value_is_dropped_without_changing_the_input(self, tmp_path: Path) -> None:
        nested = []
        for _ in range(sys.getrecursionlimit() + 20):
            nested = [nested]
        task = AudioTask(dataset_name="test", data={"speaker_id": "a", "text": "kept", "custom": nested})
        stage = ManifestGroupExportStage(output_dir=str(tmp_path), format="json")
        stage.setup()
        assert stage.process_batch([task]) == [task]
        assert json.loads((tmp_path / "a.jsonl").read_text()) == {"speaker_id": "a", "text": "kept"}
        assert task.data["custom"] is nested

    def test_ray_data_batches_publish_final_timeline_without_teardown(self, tmp_path: Path) -> None:
        import pickle

        stage = ManifestGroupExportStage(output_dir=str(tmp_path), write_timeline=True, timeline_flush_rows=100)
        stage.prepare_on_driver()
        worker = pickle.loads(pickle.dumps(stage))  # noqa: S301 - local stage fixture
        worker.setup()
        adapter = RayDataStageAdapter(worker)
        later = AudioTask(dataset_name="test", data={"speaker_id": "b", "text": "later", "start": 2, "end": 3})
        earlier = AudioTask(dataset_name="test", data={"speaker_id": "a", "text": "earlier", "start": 0, "end": 1})
        assert adapter._process_batch_internal({"item": [later]})["item"] == [later]
        timeline = tmp_path / "timeline.txt"
        assert not timeline.exists()
        assert adapter._process_batch_internal({"item": [earlier]})["item"] == [earlier]
        stage.finalize()
        assert timeline.read_text().splitlines() == ["[0.00 - 1.00] a: earlier", "[2.00 - 3.00] b: later"]

    def test_txt_one_file_per_group_with_timestamps(self, tmp_path) -> None:  # noqa: ANN001
        out = str(tmp_path / "by_speaker")
        _run(ManifestGroupExportStage(output_dir=out), _ROWS)
        files = sorted(os.listdir(out))
        assert len(files) == 2
        assert files[0].startswith("spk_0~")
        assert files[1].startswith("spk_1~")
        lines = _unsafe_group_file(tmp_path / "by_speaker", "spk_0", "txt").read_text().splitlines()
        assert lines == ["[0.00 - 1.50] hello there", "[3.00 - 5.00] you are a bold one"]

    def test_distinct_groups_that_sanitize_alike_get_distinct_files(self, tmp_path: Path) -> None:
        out = tmp_path / "g"
        rows = [
            {"speaker_id": "speaker 1/A", "text": "unsafe spelling"},
            {"speaker_id": "speaker_1_A", "text": "already safe"},
        ]

        _run(ManifestGroupExportStage(output_dir=str(out), include_timestamps=False), rows)

        assert (out / "speaker_1_A.txt").read_text().strip() == "already safe"
        assert _unsafe_group_file(out, "speaker_1_A", "txt").read_text().strip() == "unsafe spelling"

    def test_typed_group_values_with_the_same_text_get_distinct_files(self, tmp_path: Path) -> None:
        out = tmp_path / "g"
        rows = [
            {"speaker_id": 1, "text": "integer group"},
            {"speaker_id": "1", "text": "string group"},
        ]

        _run(ManifestGroupExportStage(output_dir=str(out), include_timestamps=False), rows)

        outputs = {path.read_text().strip() for path in out.glob("1*.txt")}
        assert outputs == {"integer group", "string group"}

    def test_colliding_group_names_are_stable_across_order_and_partial_reruns(self, tmp_path: Path) -> None:
        rows = [
            {"speaker_id": 1, "text": "integer group"},
            {"speaker_id": "1", "text": "string group"},
        ]

        mappings = []
        for directory, ordered_rows in ((tmp_path / "a", rows), (tmp_path / "b", list(reversed(rows)))):
            _run(ManifestGroupExportStage(output_dir=str(directory), include_timestamps=False), ordered_rows)
            mappings.append({path.name: path.read_text() for path in directory.glob("*.txt")})

        assert mappings[0] == mappings[1]

        isolated = tmp_path / "isolated"
        _run(ManifestGroupExportStage(output_dir=str(isolated), include_timestamps=False), [rows[0]])
        integer_name = next(name for name, content in mappings[0].items() if content.strip() == "integer group")
        assert (isolated / integer_name).read_text().strip() == "integer group"

    @pytest.mark.parametrize("format_name", ["txt", "json", "csv"])
    def test_long_group_names_fit_filesystem_component_limit(self, tmp_path: Path, format_name: str) -> None:
        value = "speaker_" + "x" * 300
        rows = [{"speaker_id": value, "text": "kept"}]
        out = tmp_path / format_name

        _run(ManifestGroupExportStage(output_dir=str(out), format=format_name), rows)

        files = list(out.iterdir())
        assert len(files) == 1
        assert len(files[0].name.encode("utf-8")) <= 255
        assert "~" in files[0].stem

    def test_long_group_names_are_stable_across_arrival_order(self, tmp_path: Path) -> None:
        groups = ["speaker_" + "a" * 300, "speaker_" + "b" * 300]
        mappings = []
        for directory, order in ((tmp_path / "a", groups), (tmp_path / "b", list(reversed(groups)))):
            _run(
                ManifestGroupExportStage(output_dir=str(directory), include_timestamps=False),
                [{"speaker_id": group, "text": group[-1]} for group in order],
            )
            mappings.append({path.name: path.read_text() for path in directory.glob("*.txt")})
        assert mappings[0] == mappings[1]

    def test_colliding_timeline_labels_do_not_depend_on_arrival_order(self, tmp_path: Path) -> None:
        rows = [
            {"speaker_id": 1, "text": "integer", "start": 0.0, "end": 1.0},
            {"speaker_id": "1", "text": "string", "start": 1.0, "end": 2.0},
        ]

        timelines = []
        for directory, ordered_rows in ((tmp_path / "a", rows), (tmp_path / "b", list(reversed(rows)))):
            _run(ManifestGroupExportStage(output_dir=str(directory), write_timeline=True), ordered_rows)
            timelines.append((directory / "timeline.txt").read_text())

        assert timelines[0] == timelines[1]

    def test_missing_group_and_real_fallback_value_get_distinct_files(self, tmp_path: Path) -> None:
        out = tmp_path / "g"
        rows = [
            {"text": "missing group"},
            {"speaker_id": "unknown", "text": "real unknown group"},
        ]

        _run(ManifestGroupExportStage(output_dir=str(out), include_timestamps=False), rows)

        assert (out / "unknown.txt").read_text().strip() == "missing group"
        colliding = list(out.glob("unknown~*.txt"))
        assert len(colliding) == 1
        assert colliding[0].read_text().strip() == "real unknown group"

    def test_unsafe_missing_group_and_real_fallback_value_get_distinct_files(self, tmp_path: Path) -> None:
        out = tmp_path / "g"
        rows = [
            {"text": "missing group"},
            {"speaker_id": "not known", "text": "real fallback group"},
        ]

        _run(
            ManifestGroupExportStage(
                output_dir=str(out),
                include_timestamps=False,
                missing_group="not known",
            ),
            rows,
        )

        outputs = {path.read_text().strip() for path in out.glob("not_known~*.txt")}
        assert outputs == {"missing group", "real fallback group"}

    def test_timeline_group_cannot_replace_the_combined_timeline(self, tmp_path: Path) -> None:
        out = tmp_path / "g"
        stage = ManifestGroupExportStage(
            output_dir=str(out),
            include_timestamps=False,
            write_timeline=True,
        )

        _run(stage, [{"speaker_id": "timeline", "text": "kept in both outputs"}])

        assert (out / "timeline.txt").read_text().strip() == "timeline: kept in both outputs"
        assert _unsafe_group_file(out, "timeline", "txt").read_text().strip() == "kept in both outputs"

    def test_timeline_labels_distinguish_groups_that_sanitize_alike(self, tmp_path: Path) -> None:
        out = tmp_path / "g"
        rows = [
            {"speaker_id": "speaker 1/A", "text": "first", "start": 0.0, "end": 1.0},
            {"speaker_id": "speaker_1_A", "text": "second", "start": 1.0, "end": 2.0},
        ]

        _run(ManifestGroupExportStage(output_dir=str(out), write_timeline=True), rows)

        lines = (out / "timeline.txt").read_text().splitlines()
        labels = [line.split("] ", 1)[1].split(": ", 1)[0] for line in lines]
        assert labels[0] != labels[1]
        assert [line.rsplit(": ", 1)[1] for line in lines] == ["first", "second"]

    def test_rows_pass_through_unchanged(self, tmp_path) -> None:  # noqa: ANN001
        # It is a tee, not a sink: a writer can follow it.
        stage = ManifestGroupExportStage(output_dir=str(tmp_path / "g"))
        stage.setup()
        task = AudioTask(dataset_name="t", data=dict(_ROWS[0]))
        assert stage.process(task).data == _ROWS[0]

    def test_json_format_selects_columns(self, tmp_path) -> None:  # noqa: ANN001
        out = str(tmp_path / "g")
        _run(ManifestGroupExportStage(output_dir=out, format="json", columns=["text", "start"]), _ROWS)
        rows = [
            json.loads(line) for line in _unsafe_group_file(tmp_path / "g", "spk_0", "jsonl").read_text().splitlines()
        ]
        assert rows == [{"text": "hello there", "start": 0.0}, {"text": "you are a bold one", "start": 3.0}]

    def test_csv_writes_one_header(self, tmp_path) -> None:  # noqa: ANN001
        out = str(tmp_path / "g")
        _run(ManifestGroupExportStage(output_dir=out, format="csv", columns=["text"]), _ROWS)
        lines = _unsafe_group_file(tmp_path / "g", "spk_0", "csv").read_text().splitlines()
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
        assert len(os.listdir(out)) == 3
        assert next(path for path in (tmp_path / "g").glob("0~*.txt")).read_text().strip() == "zero is a speaker"
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
        parsed = list(csv.DictReader(_unsafe_group_file(tmp_path / "g", "spk_0", "csv").read_text().splitlines()))
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
        parsed = list(csv.DictReader(_unsafe_group_file(tmp_path / "g", "spk_0", "csv").read_text().splitlines()))
        assert [r["text"] for r in parsed] == ["hello", "world"]
        assert "lang" not in parsed[0]

    def test_csv_schema_is_rebuilt_for_a_new_run(self, tmp_path: Path) -> None:
        out = tmp_path / "g"
        stage = ManifestGroupExportStage(output_dir=str(out), format="csv")
        _run(stage, [{"speaker_id": "spk", "text": "first run"}])

        _run(stage, [{"speaker_id": "spk", "language": "en"}])

        rows = list(csv.DictReader((out / "spk.csv").read_text().splitlines()))
        assert rows == [{"speaker_id": "spk", "language": "en"}]

    def test_timeline_is_ordered_across_groups(self, tmp_path) -> None:  # noqa: ANN001
        out = str(tmp_path / "g")
        _run(ManifestGroupExportStage(output_dir=out, write_timeline=True), list(reversed(_ROWS)))
        lines = (tmp_path / "g" / "timeline.txt").read_text().splitlines()
        labels = [line.split("] ")[1].split(":")[0] for line in lines]
        assert labels[0] == labels[2]
        assert labels[0].startswith("spk_0~")
        assert labels[1].startswith("spk_1~")

    def test_missing_group_and_unserializable_values(self, tmp_path) -> None:  # noqa: ANN001
        out = str(tmp_path / "g")
        _run(
            ManifestGroupExportStage(output_dir=out, format="json"),
            [{"text": "no speaker", "waveform": object()}],
        )
        assert os.path.isfile(os.path.join(out, "unknown.jsonl"))
        # A resident tensor/object must be dropped, not crash the export.
        assert json.loads((tmp_path / "g" / "unknown.jsonl").read_text()) == {"text": "no speaker"}

    def test_missing_group_fallback_survives_framework_validation(self, tmp_path: Path) -> None:
        stage = ManifestGroupExportStage(output_dir=str(tmp_path / "g"), include_timestamps=False)
        stage.setup()
        task = AudioTask(dataset_name="t", data={"text": "no speaker"})

        assert stage.process_batch([task]) == [task]
        assert (tmp_path / "g" / "unknown.txt").read_text().strip() == "no speaker"

        contract = build_contract(stage)
        assert contract.reads.data_keys == ["text"]
        assert contract.optional_reads.data_keys == ["speaker_id"]
        assert validate_pipeline([stage], initial_keys={"text"}).ok

    def test_configured_text_and_selected_columns_are_required_reads(self, tmp_path: Path) -> None:
        txt = ManifestGroupExportStage(
            output_dir=str(tmp_path / "txt"),
            text_key="transcript",
            include_timestamps=False,
        )
        timeline = ManifestGroupExportStage(
            output_dir=str(tmp_path / "timeline"),
            format="json",
            text_key="transcript",
            write_timeline=True,
        )
        selected = ManifestGroupExportStage(
            output_dir=str(tmp_path / "json"),
            format="json",
            columns=["transcript", "lang"],
        )

        assert build_contract(txt).reads.data_keys == ["transcript"]
        assert build_contract(timeline).reads.data_keys == ["transcript"]
        assert build_contract(selected).reads.data_keys == ["transcript", "lang"]
        assert not validate_pipeline([txt], initial_keys={"speaker_id"}).ok

        txt.setup()
        task = AudioTask(dataset_name="t", data={"speaker_id": "s", "transcript": "nonblank"})
        assert txt.process_batch([task]) == [task]
        assert (tmp_path / "txt" / "s.txt").read_text().strip() == "nonblank"

    def test_rerun_replaces_its_own_output(self, tmp_path) -> None:  # noqa: ANN001
        out = str(tmp_path / "g")
        stage = ManifestGroupExportStage(output_dir=out)
        _run(stage, _ROWS)
        _run(stage, _ROWS)
        assert len(_unsafe_group_file(tmp_path / "g", "spk_0", "txt").read_text().splitlines()) == 2  # not 4

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
        assert ManifestGroupExportStage.describe_static().error_policy == "fail"


def test_configured_csv_columns_survive_an_unserializable_first_value(tmp_path: Path) -> None:
    stage = ManifestGroupExportStage(output_dir=str(tmp_path), format="csv", columns=["text", "score"])
    _run(
        stage,
        [
            {"speaker_id": "speaker", "text": "first", "score": object()},
            {"speaker_id": "speaker", "text": "second", "score": 4.0},
        ],
    )

    with (tmp_path / "speaker.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["text", "score"]
        assert list(reader) == [{"text": "first", "score": ""}, {"text": "second", "score": "4.0"}]


@pytest.mark.parametrize("format_name", ["txt", "json", "csv"])
@pytest.mark.parametrize("write_timeline", [False, True])
def test_replacement_worker_restores_committed_rows_and_headers(
    tmp_path: Path, format_name: str, write_timeline: bool
) -> None:
    import pickle

    driver = ManifestGroupExportStage(
        output_dir=str(tmp_path), format=format_name, columns=["text"], write_timeline=write_timeline
    )
    driver.prepare_on_driver()
    serialized = pickle.dumps(driver)
    first = pickle.loads(serialized)  # noqa: S301
    first.setup()
    first.process(AudioTask(data={"speaker_id": "speaker", "text": "first é", "start": 0, "end": 1}))
    extension = "jsonl" if format_name == "json" else format_name
    output = tmp_path / f"speaker.{extension}"
    committed = output.read_bytes()
    with output.open("ab") as stream:
        stream.write(b"uncommitted partial write")

    replacement = pickle.loads(serialized)  # noqa: S301
    replacement.setup()
    assert output.read_bytes() == committed
    replacement.process(AudioTask(data={"speaker_id": "speaker", "text": "second", "start": 2, "end": 3}))
    replacement.teardown()
    driver.finalize()

    if format_name == "csv":
        with output.open(newline="") as stream:
            assert list(csv.DictReader(stream)) == [{"text": "first é"}, {"text": "second"}]
        assert output.read_text().count("text\n") == 1
    elif format_name == "json":
        assert [json.loads(line) for line in output.read_text().splitlines()] == [
            {"text": "first é"},
            {"text": "second"},
        ]
    else:
        assert len(output.read_text().splitlines()) == 2
        assert "first é" in output.read_text()
        assert "second" in output.read_text()
    if write_timeline:
        timeline = (tmp_path / "timeline.txt").read_text()
        assert "first é" in timeline
        assert "second" in timeline
    assert not list(tmp_path.glob(".*"))

    driver.prepare_on_driver()
    fresh_worker = pickle.loads(pickle.dumps(driver))  # noqa: S301
    fresh_worker.setup()
    fresh_worker.process(AudioTask(data={"speaker_id": "speaker", "text": "fresh", "start": 0, "end": 1}))
    driver.finalize()
    assert "fresh" in output.read_text()
    assert "first é" not in output.read_text()
    assert "second" not in output.read_text()


def test_group_export_refuses_concurrent_driver_ownership(tmp_path: Path) -> None:
    first = ManifestGroupExportStage(output_dir=str(tmp_path))
    second = ManifestGroupExportStage(output_dir=str(tmp_path))
    first.prepare_on_driver()
    try:
        with pytest.raises(FileExistsError):
            second.prepare_on_driver()
    finally:
        first.abort_on_driver()


def test_pipeline_uses_caller_prepared_group_export(tmp_path: Path) -> None:
    import pickle
    from types import SimpleNamespace

    from nemo_curator.pipeline import Pipeline

    stage = ManifestGroupExportStage(output_dir=str(tmp_path), write_timeline=True)
    tasks = [AudioTask(data={"speaker_id": "speaker", "text": text}) for text in ("first", "second")]

    def execute(stages: list[ManifestGroupExportStage], initial_tasks: list[AudioTask]) -> list[AudioTask]:
        serialized = pickle.dumps(stages[0])
        for task in initial_tasks:
            worker = pickle.loads(serialized)  # noqa: S301
            worker.setup()
            worker.process(task)
            worker.teardown()
        return initial_tasks

    pipeline = Pipeline(name="export", stages=[stage])
    stage.prepare_on_driver()
    try:
        result = pipeline.run(SimpleNamespace(execute=execute), initial_tasks=tasks)
        assert not (tmp_path / "timeline.txt").exists()
        stage.finalize()
    finally:
        stage.abort_on_driver()

    assert result == tasks
    assert (tmp_path / "speaker.txt").read_text().splitlines() == ["first", "second"]
    assert (tmp_path / "timeline.txt").read_text().splitlines() == ["speaker: first", "speaker: second"]
    assert not list(tmp_path.glob(".*"))


def test_unprepared_export_cannot_replace_an_active_drivers_rows(tmp_path: Path) -> None:
    import pickle

    driver = ManifestGroupExportStage(output_dir=str(tmp_path))
    driver.prepare_on_driver()
    worker = pickle.loads(pickle.dumps(driver))  # noqa: S301
    worker.setup()
    worker.process(AudioTask(data={"speaker_id": "speaker", "text": "committed"}))
    output = tmp_path / "speaker.txt"
    original = output.read_bytes()
    unprepared = ManifestGroupExportStage(output_dir=str(tmp_path))
    try:
        with pytest.raises(RuntimeError, match="active pipeline run"):
            unprepared.setup()
        with pytest.raises(RuntimeError, match="active pipeline run"):
            unprepared.process(AudioTask(data={"speaker_id": "speaker", "text": "unowned"}))
        assert output.read_bytes() == original
    finally:
        driver.abort_on_driver()
    unprepared.setup()
    unprepared.process(AudioTask(data={"speaker_id": "speaker", "text": "fresh"}))
    assert output.read_text() == "fresh\n"
    with pytest.raises(RuntimeError, match="driver-prepared shared output"):
        worker.process(AudioTask(data={"speaker_id": "speaker", "text": "stale worker"}))
    assert output.read_text() == "fresh\n"


@pytest.mark.parametrize("release", ["finalize", "abort_on_driver"])
def test_retired_worker_teardown_preserves_the_next_runs_timeline(tmp_path: Path, release: str) -> None:
    import pickle

    first = ManifestGroupExportStage(output_dir=str(tmp_path), write_timeline=True, timeline_flush_rows=1)
    first.prepare_on_driver()
    worker = pickle.loads(pickle.dumps(first))  # noqa: S301
    worker.setup()
    worker.process(AudioTask(data={"speaker_id": "speaker", "text": "old run"}))
    getattr(first, release)()

    current = ManifestGroupExportStage(output_dir=str(tmp_path), write_timeline=True, timeline_flush_rows=1)
    current.prepare_on_driver()
    try:
        current.setup()
        current.process(AudioTask(data={"speaker_id": "speaker", "text": "new run"}))
        current.finalize()
        timeline = tmp_path / "timeline.txt"
        committed = timeline.read_bytes()
        worker.teardown()
        assert timeline.read_bytes() == committed
        assert "new run" in timeline.read_text()
    finally:
        current.abort_on_driver()


def test_failed_repeated_preparation_retains_the_original_reservation(tmp_path: Path) -> None:
    stage = ManifestGroupExportStage(output_dir=str(tmp_path))
    stage.prepare_on_driver()
    with pytest.raises(FileExistsError):
        stage.prepare_on_driver()
    stage.abort_on_driver()
    assert not (tmp_path / ".nemo_curator_group_export.jsonl._RUN").exists()
    stage.prepare_on_driver()
    stage.abort_on_driver()


@pytest.mark.parametrize("prepared", [False, True])
def test_group_name_cannot_overwrite_the_run_journal(tmp_path: Path, prepared: bool) -> None:
    stage = ManifestGroupExportStage(output_dir=str(tmp_path), format="json")
    if prepared:
        stage.prepare_on_driver()
    stage.setup()
    row = {"speaker_id": ".nemo_curator_group_export", "text": "kept"}
    try:
        stage.process_batch([AudioTask(data=row)])
        output = _unsafe_group_file(tmp_path, ".nemo_curator_group_export", "jsonl")
        assert json.loads(output.read_text()) == row
        stage.finalize()
        assert not (tmp_path / ".nemo_curator_group_export.jsonl").exists()
        assert not (tmp_path / ".nemo_curator_group_export.jsonl._RUN").exists()
    finally:
        stage.abort_on_driver()


def test_prepared_timeline_is_published_once_after_all_batches(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    import builtins
    import pickle

    driver = ManifestGroupExportStage(output_dir=str(tmp_path), write_timeline=True, timeline_flush_rows=3)
    driver.prepare_on_driver()
    worker = pickle.loads(pickle.dumps(driver))  # noqa: S301 - local stage fixture
    worker.setup()
    real_open = builtins.open
    rewrites = []

    def count_timeline_writes(path, mode="r", *args, **kwargs):  # noqa: ANN001, ANN202
        if str(path) == str(tmp_path / "timeline.txt") and mode == "w":
            rewrites.append(path)
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", count_timeline_writes)
    try:
        for i in reversed(range(20)):
            worker.process_batch(
                [AudioTask(data={"speaker_id": "speaker", "text": f"row {i}", "start": i, "end": i + 1})]
            )
        worker.teardown()
        assert rewrites == []
        driver.finalize()
        assert len(rewrites) == 1
        assert (tmp_path / "timeline.txt").read_text().splitlines() == [
            f"[{i:.2f} - {i + 1:.2f}] speaker: row {i}" for i in range(20)
        ]
    finally:
        driver.abort_on_driver()
