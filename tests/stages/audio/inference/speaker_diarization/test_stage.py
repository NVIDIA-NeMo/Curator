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

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from nemo_curator.config.run import _instantiate_stage
from nemo_curator.models.audio.speaker_diarization.base import DiarizationResult
from nemo_curator.stages.audio.inference.base import AdapterInferenceStage
from nemo_curator.stages.audio.inference.speaker_diarization.stage import (
    InferenceSortformerStage,
    _write_rttm,
)
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

_PIPELINE_YAML = Path(__file__).parents[5] / "tutorials" / "audio" / "sortformer" / "pipeline.yaml"


@dataclass
class _RecordingAdapter:
    results: list[DiarizationResult] | None = None
    items: list[dict] = field(default_factory=list)

    def diarize_batch(self, items: list[dict]) -> list[DiarizationResult]:
        self.items = items
        if self.results is not None:
            return self.results
        return [
            DiarizationResult(
                segments=[{"start": 0.0, "end": float(item["audio_seconds"]), "speaker": item["task_id"]}]
            )
            for item in items
        ]


def _waveform_task(task_id: str, samples: int, *, sample_rate: int = 16000) -> AudioTask:
    return AudioTask(
        task_id=task_id,
        data={"waveform": np.zeros(samples, dtype=np.float32), "sample_rate": sample_rate},
    )


def test_stage_uses_shared_adapter_lifecycle() -> None:
    assert issubclass(InferenceSortformerStage, AdapterInferenceStage)


def test_current_main_positional_constructor_order_is_preserved() -> None:
    model = object()
    resources = Resources(cpus=2.0)
    stage = InferenceSortformerStage(
        "provider/model",
        "/models/model.nemo",
        "/cache",
        model,
        "path",
        "turns",
        "/rttm",
        100,
        2,
        8,
        16,
        30,
        64,
        3,
        "LegacySortformer",
        4,
        resources,
    )

    assert stage.model_name == "provider/model"
    assert stage.model_id == "provider/model"
    assert stage.model_path == "/models/model.nemo"
    assert stage.cache_dir == "/cache"
    assert stage.diar_model is model
    assert stage.filepath_key == "path"
    assert stage.audio_filepath_key == "path"
    assert stage.diar_segments_key == "turns"
    assert stage.rttm_out_dir == "/rttm"
    assert stage.chunk_len == 100
    assert stage.chunk_left_context == 2
    assert stage.chunk_right_context == 8
    assert stage.fifo_len == 16
    assert stage.spkcache_update_period == 30
    assert stage.spkcache_len == 64
    assert stage.inference_batch_size == 3
    assert stage.name == "LegacySortformer"
    assert stage.batch_size == 4
    assert stage.resources is resources


def test_current_main_dataclass_defaults_are_preserved() -> None:
    fields = InferenceSortformerStage.__dataclass_fields__

    assert fields["model_name"].default == "nvidia/diar_streaming_sortformer_4spk-v2.1"
    assert fields["filepath_key"].default == "audio_filepath"
    assert fields["chunk_len"].default == 340
    assert fields["chunk_left_context"].default == 1
    assert fields["chunk_right_context"].default == 40
    assert fields["fifo_len"].default == 40
    assert fields["spkcache_update_period"].default == 300
    assert fields["spkcache_len"].default == 188
    assert fields["inference_batch_size"].default == 1


def test_process_batch_sorts_model_work_and_restores_task_order() -> None:
    stage = InferenceSortformerStage(waveform_key="waveform")
    adapter = _RecordingAdapter()
    stage._adapter = adapter
    tasks = [_waveform_task("long", 32000), _waveform_task("short", 8000)]

    result = stage.process_batch(tasks)

    assert result == tasks
    assert [item["task_id"] for item in adapter.items] == ["short", "long"]
    assert result[0].data["diar_segments"][0]["speaker"] == "long"
    assert result[1].data["diar_segments"][0]["speaker"] == "short"
    assert result[0].data["num_speakers"] == 1
    assert [task.task_id for task in result] == ["long", "short"]


def test_file_mode_delegates_path_without_eager_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    stage = InferenceSortformerStage()
    adapter = _RecordingAdapter(results=[DiarizationResult(segments=[])])
    stage._adapter = adapter
    monkeypatch.setattr("soundfile.info", lambda _path: SimpleNamespace(frames=160, samplerate=16000))

    task = AudioTask(task_id="file", data={"audio_filepath": "/audio/file.wav"})
    result = stage.process_batch([task])

    assert result[0].data["diar_segments"] == []
    assert result[0].data["num_speakers"] == 0
    assert adapter.items[0]["audio_filepath"] == "/audio/file.wav"
    assert adapter.items[0]["audio_seconds"] == 0.01


def test_file_mode_leaves_unprobeable_path_for_provider_decoder(monkeypatch: pytest.MonkeyPatch) -> None:
    stage = InferenceSortformerStage()
    adapter = _RecordingAdapter(results=[DiarizationResult(segments=[])])
    stage._adapter = adapter
    monkeypatch.setattr("soundfile.info", lambda _path: (_ for _ in ()).throw(RuntimeError("codec")))

    task = AudioTask(task_id="provider", data={"audio_filepath": "/provider/native.codec"})
    result = stage.process_batch([task])

    assert result == [task]
    assert adapter.items[0]["audio_filepath"] == "/provider/native.codec"
    assert adapter.items[0]["audio_seconds"] == float("inf")


def test_legacy_diarize_method_delegates_paths_to_initialized_adapter() -> None:
    stage = InferenceSortformerStage()
    adapter = _RecordingAdapter(
        results=[DiarizationResult(segments=[{"start": 0.0, "end": 1.0, "speaker": "speaker_0"}])]
    )
    stage._adapter = adapter

    assert stage.diarize(["/audio/file.wav"]) == [
        [{"start": 0.0, "end": 1.0, "speaker": "speaker_0"}],
    ]
    assert adapter.items == [{"audio_filepath": "/audio/file.wav"}]


def test_legacy_diarize_method_requires_initialized_adapter_or_model() -> None:
    with pytest.raises(RuntimeError, match=r"call setup\(\) before diarize"):
        InferenceSortformerStage().diarize(["/audio/file.wav"])


def test_resume_uses_key_presence_for_empty_result() -> None:
    stage = InferenceSortformerStage(waveform_key="waveform", skip_if_output_exists=True)
    adapter = _RecordingAdapter()
    stage._adapter = adapter
    task = _waveform_task("done", 160)
    task.data.update({"diar_segments": [], "num_speakers": 0})

    assert stage.process_batch([task]) == [task]
    assert adapter.items == []


def test_read_error_passes_through_without_adapter_call() -> None:
    stage = InferenceSortformerStage(waveform_key="waveform")
    adapter = _RecordingAdapter()
    stage._adapter = adapter
    task = AudioTask(task_id="bad", data={"read_error": True})

    assert stage.process_batch([task]) == [task]
    assert adapter.items == []


def test_audio_error_is_task_local_by_default() -> None:
    stage = InferenceSortformerStage(waveform_key="waveform")
    stage._adapter = _RecordingAdapter()
    task = AudioTask(task_id="bad", data={"waveform": np.array([np.nan]), "sample_rate": 16000})

    result = stage.process_batch([task])[0]

    assert result.data["diar_segments"] == []
    assert result.data["num_speakers"] == 0
    assert result.data["additional_notes"]["Sortformer_inference"] == "audio_load_error"


def test_fail_on_audio_error_raises() -> None:
    stage = InferenceSortformerStage(waveform_key="waveform", fail_on_audio_error=True)
    stage._adapter = _RecordingAdapter()

    with pytest.raises(RuntimeError, match="Failed to prepare diarization audio"):
        stage.process_batch([AudioTask(task_id="bad", data={"waveform": np.array([np.nan]), "sample_rate": 16000})])


def test_adapter_result_count_must_match() -> None:
    stage = InferenceSortformerStage(waveform_key="waveform")
    stage._adapter = _RecordingAdapter(results=[])

    with pytest.raises(RuntimeError, match="must match 1:1"):
        stage.process_batch([_waveform_task("one", 160)])


def test_rttm_uses_safe_sharded_path_and_relative_manifest_path(tmp_path) -> None:  # noqa: ANN001
    stage = InferenceSortformerStage(waveform_key="waveform", rttm_out_dir=str(tmp_path))
    stage._adapter = _RecordingAdapter(
        results=[
            DiarizationResult(
                segments=[
                    {"start": 0.0, "end": 1.25, "speaker": "speaker_0"},
                    {"start": 2.0, "end": 2.0, "speaker": "speaker_1"},
                ]
            )
        ]
    )
    task = _waveform_task("catalog/locale/recording", 16000)
    task._metadata["_shard_key"] = "catalog/locale"

    result = stage.process_batch([task])[0]

    assert result.data["rttm_filepath"] == "catalog/locale/catalog_locale_recording.rttm"
    rttm = tmp_path / result.data["rttm_filepath"]
    assert rttm.read_text() == ("SPEAKER catalog/locale/recording 1 0.000 1.250 <NA> <NA> speaker_0 <NA> <NA>\n")


@pytest.mark.parametrize(
    "shard_key",
    ["/absolute", "../traversal", "catalog/../../traversal", r"C:\absolute", r"..\traversal"],
)
def test_write_rttm_rejects_absolute_and_traversal_shard_keys(tmp_path: Path, shard_key: str) -> None:
    with pytest.raises(ValueError, match="RTTM shard key"):
        _write_rttm([], "session", str(tmp_path), shard_key=shard_key)


def test_write_rttm_rejects_shard_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "rttm"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes rttm_out_dir"):
        _write_rttm([], "session", str(root), shard_key="escape")


def test_resume_requires_referenced_rttm_file_to_exist(tmp_path: Path) -> None:
    stage = InferenceSortformerStage(
        waveform_key="waveform",
        rttm_out_dir=str(tmp_path),
        skip_if_output_exists=True,
    )
    adapter = _RecordingAdapter()
    stage._adapter = adapter
    task = _waveform_task("done", 160)
    task.data.update({"diar_segments": [], "num_speakers": 0, "rttm_filepath": "missing.rttm"})

    result = stage.process_batch([task])[0]

    assert [item["task_id"] for item in adapter.items] == ["done"]
    assert result.data["rttm_filepath"] == "done.rttm"
    assert (tmp_path / "done.rttm").is_file()


@pytest.mark.parametrize("rttm_filepath", ["../outside.rttm", r"..\outside.rttm"])
def test_resume_rejects_traversal_rttm_path(tmp_path: Path, rttm_filepath: str) -> None:
    stage = InferenceSortformerStage(
        waveform_key="waveform",
        rttm_out_dir=str(tmp_path),
        skip_if_output_exists=True,
    )
    adapter = _RecordingAdapter()
    stage._adapter = adapter
    task = _waveform_task("done", 160)
    task.data.update({"diar_segments": [], "num_speakers": 0, "rttm_filepath": rttm_filepath})

    result = stage.process_batch([task])[0]

    assert [item["task_id"] for item in adapter.items] == ["done"]
    assert result.data["rttm_filepath"] == "done.rttm"


def test_resume_rejects_absolute_rttm_path(tmp_path: Path) -> None:
    stage = InferenceSortformerStage(
        waveform_key="waveform",
        rttm_out_dir=str(tmp_path),
        skip_if_output_exists=True,
    )
    adapter = _RecordingAdapter()
    stage._adapter = adapter
    task = _waveform_task("done", 160)
    task.data.update(
        {
            "diar_segments": [],
            "num_speakers": 0,
            "rttm_filepath": str(tmp_path.parent / "outside.rttm"),
        }
    )

    result = stage.process_batch([task])[0]

    assert [item["task_id"] for item in adapter.items] == ["done"]
    assert result.data["rttm_filepath"] == "done.rttm"


def test_resume_accepts_existing_rttm_file_below_root(tmp_path: Path) -> None:
    (tmp_path / "done.rttm").write_text("", encoding="utf-8")
    stage = InferenceSortformerStage(
        waveform_key="waveform",
        rttm_out_dir=str(tmp_path),
        skip_if_output_exists=True,
    )
    adapter = _RecordingAdapter()
    stage._adapter = adapter
    task = _waveform_task("done", 160)
    task.data.update({"diar_segments": [], "num_speakers": 0, "rttm_filepath": "done.rttm"})

    assert stage.process_batch([task]) == [task]
    assert adapter.items == []


def test_write_rttm_rejects_no_valid_segments_by_writing_empty_file(tmp_path) -> None:  # noqa: ANN001
    path = _write_rttm(
        [{"start": 1.0, "end": 1.0, "speaker": "speaker_0"}],
        "session",
        str(tmp_path),
    )
    assert Path(path).read_text() == ""


def test_contract_and_validation(tmp_path: Path) -> None:
    stage = InferenceSortformerStage(waveform_key="audio", store_segments=False, rttm_out_dir=str(tmp_path))
    assert stage.inputs() == ([], ["audio", "sample_rate"])
    assert stage.outputs() == ([], ["num_speakers", "additional_notes", "rttm_filepath"])
    stage._adapter = _RecordingAdapter()
    task = AudioTask(task_id="one", data={"audio": np.zeros(1, dtype=np.float32), "sample_rate": 16000})
    assert stage.process(task).data["num_speakers"] == 1
    with pytest.raises(ValueError, match="rttm_out_dir"):
        InferenceSortformerStage(store_segments=False)


def test_legacy_constructor_aliases_are_forwarded_to_default_adapter() -> None:
    model = object()
    stage = InferenceSortformerStage(
        model_name="local/model",
        model_path="/models/model.nemo",
        cache_dir="/cache",
        diar_model=model,
        filepath_key="path",
        chunk_len=124,
        chunk_left_context=2,
        inference_batch_size=3,
    )

    assert stage.model_id == "local/model"
    assert stage.audio_filepath_key == "path"
    assert stage.adapter_kwargs["model_path"] == "/models/model.nemo"
    assert stage.adapter_kwargs["cache_dir"] == "/cache"
    assert stage.adapter_kwargs["preloaded_model"] is model
    assert stage.adapter_kwargs["chunk_len"] == 124
    assert stage.adapter_kwargs["chunk_left_context"] == 2
    assert stage.adapter_kwargs["inference_batch_size"] == 3


def test_legacy_alias_conflicts_are_rejected() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        InferenceSortformerStage(chunk_len=10, adapter_kwargs={"chunk_len": 20})


def test_empty_batch_and_setup_requirement() -> None:
    stage = InferenceSortformerStage(waveform_key="waveform")
    assert stage.process_batch([]) == []
    with pytest.raises(RuntimeError, match="not initialized"):
        stage.process_batch([_waveform_task("one", 1)])


def test_example_yaml_instantiates_the_stage_adapter_contract() -> None:
    cfg = OmegaConf.load(_PIPELINE_YAML)
    cfg.manifest_path = "/input.jsonl"

    stage = _instantiate_stage(cfg.stages[1])

    assert isinstance(stage, InferenceSortformerStage)
    assert stage.model_id == "nvidia/diar_streaming_sortformer_4spk-v2.1"
    assert stage.adapter_target.endswith(".NeMoSortformerAdapter")
    assert stage.adapter_kwargs["precision"] == "fp32"
    assert stage.adapter_kwargs["inference_batch_size"] == 1
    assert stage.batch_size == 4
    assert stage.resources.gpus == 1
