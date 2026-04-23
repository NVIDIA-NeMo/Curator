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

import json
from pathlib import Path

import pandas as pd
import pytest

from nemo_curator.stages.deduplication.id_generator import (
    CURATOR_DEDUP_ID_STR,
)
from nemo_curator.stages.text.io.reader.jsonl import JsonlAudioReaderStage, JsonlReader, JsonlReaderStage
from nemo_curator.tasks import AudioTask, FileGroupTask, _EmptyTask


@pytest.fixture
def sample_jsonl_files(tmp_path: Path) -> list[str]:
    """Create multiple JSONL files for testing."""
    files = []
    for i in range(3):
        data = pd.DataFrame({"text": [f"Doc {i}-1", f"Doc {i}-2"]})
        file_path = tmp_path / f"test_{i}.jsonl"
        data.to_json(file_path, orient="records", lines=True)
        files.append(str(file_path))
    return files


@pytest.fixture
def file_group_tasks(sample_jsonl_files: list[str]) -> list[FileGroupTask]:
    """Create multiple FileGroupTasks."""
    return [
        FileGroupTask(task_id=f"task_{i}", dataset_name="test_dataset", data=[file_path], _metadata={})
        for i, file_path in enumerate(sample_jsonl_files)
    ]


class TestJsonlReaderWithoutIdGenerator:
    """Test JSONL reader without ID generation."""

    def test_processing_without_ids(self, file_group_tasks: list[FileGroupTask]) -> None:
        """Test processing without ID generation."""
        for task in file_group_tasks:
            stage = JsonlReaderStage()
            result = stage.process(task)
            df = result.to_pandas()
            assert CURATOR_DEDUP_ID_STR not in df.columns
            assert len(df) == 2  # Each file has 2 rows

    def test_columns_selection(self, file_group_tasks: list[FileGroupTask]) -> None:
        """When columns are provided, only those are returned (existing ones)."""
        for task in file_group_tasks:
            stage = JsonlReaderStage(fields=["text"])  # select single column
            result = stage.process(task)
            df = result.to_pandas()
            assert list(df.columns) == ["text"]
            assert len(df) == 2

    def test_storage_options_via_read_kwargs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reader should use storage options from reader.read_kwargs."""
        # Create a file
        file_path = tmp_path / "one.jsonl"
        pd.DataFrame({"a": [1]}).to_json(file_path, orient="records", lines=True)

        # Reader uses read_kwargs storage options
        task = FileGroupTask(task_id="t1", dataset_name="ds", data=[str(file_path)], _metadata={})
        stage = JsonlReaderStage(read_kwargs={"storage_options": {"auto_mkdir": True}})

        seen: dict[str, object] = {}

        def fake_read_json(_path: object, *_args: object, **kwargs: object) -> pd.DataFrame:
            seen["storage_options"] = kwargs.get("storage_options") if isinstance(kwargs, dict) else None
            return pd.DataFrame({"a": [1]})

        monkeypatch.setattr(pd, "read_json", fake_read_json)

        out = stage.process(task)
        assert seen["storage_options"] == {"auto_mkdir": True}
        df = out.to_pandas()
        assert len(df) == 1

    def test_composite_reader_propagates_storage_options(self, tmp_path: Path) -> None:
        """Composite JsonlReader should pass storage options to partitioning stage and underlying stage."""
        f = tmp_path / "a.jsonl"
        pd.DataFrame({"text": ["x"]}).to_json(f, orient="records", lines=True)
        reader = JsonlReader(
            file_paths=str(tmp_path), read_kwargs={"storage_options": {"anon": True}}, fields=["text"]
        )
        stages = reader.decompose()
        # First stage is file partitioning, ensure storage options are set
        first = stages[0]
        assert getattr(first, "storage_options", None) == {"anon": True}

    def test_reader_uses_storage_options_from_read_kwargs_when_task_has_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        f = tmp_path / "b.jsonl"
        pd.DataFrame({"x": [1, 2]}).to_json(f, orient="records", lines=True)

        seen: dict[str, object] = {}

        def fake_read_json(_path: object, *_args: object, **kwargs: object) -> pd.DataFrame:
            seen["storage_options"] = kwargs.get("storage_options") if isinstance(kwargs, dict) else None
            return pd.DataFrame({"x": [1, 2]})

        monkeypatch.setattr(pd, "read_json", fake_read_json)
        task = FileGroupTask(task_id="t2", dataset_name="ds", data=[str(f)], _metadata={})
        stage = JsonlReaderStage(read_kwargs={"storage_options": {"auto_mkdir": True}})
        out = stage.process(task)
        assert seen["storage_options"] == {"auto_mkdir": True}
        df = out.to_pandas()
        assert len(df) == 2


class TestJsonlAudioReader:
    """Tests for the audio-task JSONL reader path."""

    def test_audio_stage_reads_audio_tasks(self, tmp_path: Path) -> None:
        manifest = tmp_path / "audio.jsonl"
        entries = [
            {"audio_filepath": "a.wav", "text": "alpha", "segments": [{"start": 0.0, "end": 1.0}]},
            {"audio_filepath": "b.wav", "text": "beta", "segments": [{"start": 1.0, "end": 2.0}]},
        ]
        manifest.write_text("\n".join(json.dumps(entry) for entry in entries))

        stage = JsonlAudioReaderStage()
        task = FileGroupTask(
            task_id="audio_task",
            dataset_name="audio_dataset",
            data=[str(manifest)],
            _metadata={"source": "unit-test"},
        )
        result = stage.process(task)

        assert len(result) == 2
        assert all(isinstance(item, AudioTask) for item in result)
        assert [item.data["audio_filepath"] for item in result] == ["a.wav", "b.wav"]
        assert [item.task_id for item in result] == ["audio_task_0", "audio_task_1"]
        assert all(item.dataset_name == "audio_dataset" for item in result)
        assert all(item._metadata == {"source": "unit-test"} for item in result)
        assert all(item.sample_key for item in result)
        assert result[0].sample_key != result[1].sample_key
        assert result[0].data["segments"][0]["end"] == 1.0

    def test_audio_stage_filters_fields_and_skips_blank_lines(self, tmp_path: Path) -> None:
        manifest = tmp_path / "audio_fields.jsonl"
        manifest.write_text(
            json.dumps({"audio_filepath": "a.wav", "text": "alpha", "speaker_id": "spk1"})
            + "\n\n  \n"
            + json.dumps({"audio_filepath": "b.wav", "text": "beta", "speaker_id": "spk2"})
            + "\n"
        )

        stage = JsonlAudioReaderStage(fields=["audio_filepath", "text"])
        task = FileGroupTask(task_id="audio_task", dataset_name="audio_dataset", data=[str(manifest)], _metadata={})
        result = stage.process(task)

        assert len(result) == 2
        assert result[0].data == {"audio_filepath": "a.wav", "text": "alpha"}
        assert result[1].data == {"audio_filepath": "b.wav", "text": "beta"}

    def test_audio_stage_preserves_explicit_sample_key_from_manifest(self, tmp_path: Path) -> None:
        manifest = tmp_path / "audio_sample_keys.jsonl"
        manifest.write_text(
            json.dumps({"audio_filepath": "a.wav", "text": "alpha", "sample_key": "explicit-a"}) + "\n"
        )

        stage = JsonlAudioReaderStage(fields=["audio_filepath", "text"])
        task = FileGroupTask(task_id="audio_task", dataset_name="audio_dataset", data=[str(manifest)], _metadata={})
        [result] = stage.process(task)

        assert result.sample_key == "explicit-a"

    def test_audio_composite_decomposes_to_audio_stage(self, tmp_path: Path) -> None:
        manifest = tmp_path / "audio_manifest.jsonl"
        manifest.write_text(json.dumps({"audio_filepath": "a.wav", "text": "alpha"}) + "\n")

        reader = JsonlReader(
            file_paths=str(tmp_path),
            task_type="audio",
            fields=["audio_filepath", "text"],
            read_kwargs={"storage_options": {"anon": True}},
        )
        stages = reader.decompose()

        assert len(stages) == 2
        assert getattr(stages[0], "storage_options", None) == {"anon": True}
        assert isinstance(stages[1], JsonlAudioReaderStage)
        assert stages[1].fields == ["audio_filepath", "text"]

    def test_audio_composite_propagates_id_generation_flags(self, tmp_path: Path) -> None:
        manifest = tmp_path / "audio_manifest.jsonl"
        manifest.write_text(json.dumps({"audio_filepath": "a.wav", "text": "alpha"}) + "\n")

        reader = JsonlReader(file_paths=str(tmp_path), task_type="audio", _generate_ids=True)
        stages = reader.decompose()

        assert len(stages) == 2
        assert isinstance(stages[1], JsonlAudioReaderStage)
        assert stages[1]._generate_ids is True
        assert stages[1]._assign_ids is False

    def test_audio_pipeline_outputs_audio_tasks(self, tmp_path: Path) -> None:
        from nemo_curator.backends.xenna import XennaExecutor
        from nemo_curator.pipeline import Pipeline

        input_dir = tmp_path / "audio_inputs"
        input_dir.mkdir()
        for file_idx in range(2):
            manifest = input_dir / f"audio_{file_idx}.jsonl"
            entries = [
                {"audio_filepath": f"{file_idx}_0.wav", "text": f"doc-{file_idx}-0"},
                {"audio_filepath": f"{file_idx}_1.wav", "text": f"doc-{file_idx}-1"},
            ]
            manifest.write_text("\n".join(json.dumps(entry) for entry in entries))

        pipeline = Pipeline(name="audio_reader_test")
        pipeline.add_stage(JsonlReader(file_paths=str(input_dir), files_per_partition=1, task_type="audio"))

        results = pipeline.run(XennaExecutor(config={"execution_mode": "streaming"}))

        assert results is not None
        assert len(results) == 4
        assert all(isinstance(task, AudioTask) for task in results)
        assert sorted(task.data["audio_filepath"] for task in results) == [
            "0_0.wav",
            "0_1.wav",
            "1_0.wav",
            "1_1.wav",
        ]

    @pytest.mark.usefixtures("ray_client_with_id_generator")
    def test_audio_stage_generates_and_assigns_stable_ids(self, tmp_path: Path) -> None:
        manifest = tmp_path / "audio_ids.jsonl"
        manifest.write_text(
            "\n".join(
                [
                    json.dumps({"audio_filepath": "a.wav", "text": "alpha"}),
                    json.dumps({"audio_filepath": "b.wav", "text": "beta"}),
                ]
            )
        )
        task = FileGroupTask(task_id="audio_task", dataset_name="audio_dataset", data=[str(manifest)], _metadata={})

        generation_stage = JsonlAudioReaderStage(_generate_ids=True)
        generation_stage.setup()

        generated = generation_stage.process(task)
        generated_ids = [audio_task.data[CURATOR_DEDUP_ID_STR] for audio_task in generated]
        assert generated_ids == [0, 1]

        repeated = generation_stage.process(task)
        repeated_ids = [audio_task.data[CURATOR_DEDUP_ID_STR] for audio_task in repeated]
        assert repeated_ids == [0, 1]

        assign_stage = JsonlAudioReaderStage(_assign_ids=True)
        assign_stage.setup()
        assigned = assign_stage.process(task)
        assigned_ids = [audio_task.data[CURATOR_DEDUP_ID_STR] for audio_task in assigned]
        assert assigned_ids == [0, 1]

    @pytest.mark.usefixtures("ray_client_with_id_generator")
    def test_audio_stage_assign_ids_raises_on_task_count_mismatch(self, tmp_path: Path) -> None:
        manifest = tmp_path / "audio_ids_mismatch.jsonl"
        manifest.write_text(
            "\n".join(
                [
                    json.dumps({"audio_filepath": "a.wav", "text": "alpha"}),
                    json.dumps({"audio_filepath": "b.wav", "text": "beta"}),
                ]
            )
        )
        task = FileGroupTask(task_id="audio_task", dataset_name="audio_dataset", data=[str(manifest)], _metadata={})

        generation_stage = JsonlAudioReaderStage(_generate_ids=True)
        generation_stage.setup()
        generation_stage.process(task)

        assign_stage = JsonlAudioReaderStage(_assign_ids=True, fields=["audio_filepath"])
        assign_stage.setup()

        with pytest.raises(RuntimeError, match=r"Assigned-ID range \[0, 1\] \(2 ids\) does not match 1 tasks"):
            assign_stage.process(task)

    def test_audio_stage_generate_ids_no_actor_error(self) -> None:
        stage = JsonlAudioReaderStage(_generate_ids=True)

        with pytest.raises(RuntimeError, match="actor 'id_generator' does not exist"):
            stage.setup()

        stage = JsonlAudioReaderStage(_assign_ids=True)

        with pytest.raises(RuntimeError, match="actor 'id_generator' does not exist"):
            stage.setup()


class TestJsonlReaderWithIdGenerator:
    """Test JSONL reader with ID generation."""

    @pytest.mark.usefixtures("ray_client_with_id_generator")
    def test_sequential_id_generation_and_assignment(self, file_group_tasks: list[FileGroupTask]) -> None:
        """Test sequential ID generation across multiple batches."""
        generation_stage = JsonlReaderStage(_generate_ids=True)
        generation_stage.setup()

        all_ids = []
        for task in file_group_tasks:
            result = generation_stage.process(task)
            ids = result.to_pandas()[CURATOR_DEDUP_ID_STR].tolist()
            all_ids.extend(ids)

        # IDs should be monotonically increasing: [0,1,2,3,4,5]
        assert all_ids == list(range(6))

        """If the same batch is processed again (when generate_id=True), the IDs should be the same."""
        repeated_ids = []
        for task in file_group_tasks:
            result = generation_stage.process(task)
            ids = result.to_pandas()[CURATOR_DEDUP_ID_STR].tolist()
            repeated_ids.extend(ids)

        # IDs should be the same as the first time: [0,1,2,3,4,5]
        assert repeated_ids == list(range(6))

        """ If we now create a new stage with _assign_ids=True, the IDs should be the same as the previous batch."""
        all_ids = []
        assign_stage = JsonlReaderStage(_assign_ids=True)
        assign_stage.setup()
        for i, task in enumerate(file_group_tasks):
            result = assign_stage.process(task)
            df = result.to_pandas()
            expected_ids = [i * 2, i * 2 + 1]  # Task 0: [0,1], Task 1: [2,3], Task 2: [4,5]
            assert (
                df[CURATOR_DEDUP_ID_STR].tolist() == expected_ids
            )  # These ids should be the same as the previous batch
            all_ids.extend(df[CURATOR_DEDUP_ID_STR].tolist())

        assert all_ids == list(range(6))

    def test_generate_ids_no_actor_error(self) -> None:
        """Test error when actor doesn't exist and ID generation is requested."""
        stage = JsonlReaderStage(_generate_ids=True)

        with pytest.raises(RuntimeError, match="actor 'id_generator' does not exist"):
            stage.setup()

        stage = JsonlReaderStage(_assign_ids=True)

        with pytest.raises(RuntimeError, match="actor 'id_generator' does not exist"):
            stage.setup()


def test_jsonl_reader_with_blocksize_limit(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    # Storage size is larger than 10 million bytes
    # In-memory size is also larger than 10 million bytes
    size = 1000
    df = pd.DataFrame({"id": list(range(size)), "text": ["a" * 4000] * size, "other_field": ["b" * 10_000] * size})
    df.to_json(tmp_path / "test.jsonl", orient="records", lines=True)

    stage = JsonlReader(file_paths=str(tmp_path), blocksize=10_000_000)
    assert len(stage.decompose()) == 2

    # Since the storage size is larger than 10 million bytes, the FilePartitioningStage should warn
    file_partitioning_stage = stage.decompose()[0]
    with caplog.at_level("WARNING"):
        file_partitioning_stage.process(_EmptyTask)
    assert "File group task has exceeded the storage limit per partition" in caplog.text
