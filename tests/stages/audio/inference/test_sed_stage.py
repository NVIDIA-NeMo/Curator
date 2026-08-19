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

"""Tests for the generic SED stage using a stand-in adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile
import torch
from omegaconf import OmegaConf

from nemo_curator.config.run import _instantiate_stage
from nemo_curator.models.sed.base import SEDResult
from nemo_curator.stages.audio.inference.sed.stage import SEDInferenceStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

_SR = 16000
_HOP = 320
_CLASSES = 527
_ADAPTER_TARGET = "nemo_curator.models.sed.panns.PANNsSEDAdapter"
_CHECKPOINT = "/weights/Cnn14.pth"
_PIPELINE_YAML = Path(__file__).parents[4] / "tutorials" / "audio" / "sed" / "pipeline.yaml"


class _FakeSEDAdapter:
    """Record stage-normalized items and emit deterministic canonical results."""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []
        self.unloaded = False

    def infer_batch(self, items: list[dict[str, Any]]) -> list[SEDResult]:
        self.calls.append(items)
        max_frames = max(int(np.ceil(len(item["waveform"]) / _HOP)) for item in items)
        return [
            SEDResult(
                framewise_output=np.full((max_frames, _CLASSES), 0.5, dtype=np.float32),
                fps=_SR / _HOP,
                valid_frames=int(np.ceil(len(item["waveform"]) / _HOP)),
                original_num_samples=len(item["waveform"]),
            )
            for item in items
        ]

    def unload_model(self) -> None:
        self.unloaded = True


def _stage(**kwargs: object) -> tuple[SEDInferenceStage, _FakeSEDAdapter]:
    stage = SEDInferenceStage(
        adapter_target=_ADAPTER_TARGET,
        checkpoint_path=_CHECKPOINT,
        sample_rate=_SR,
        waveform_key="waveform",
        **kwargs,
    )
    adapter = _FakeSEDAdapter()
    stage._adapter = adapter
    return stage, adapter


def _task(seconds: float = 1.0, sample_rate: int = _SR, **extra: object) -> AudioTask:
    return AudioTask(
        data={"waveform": np.zeros(int(seconds * sample_rate), dtype=np.float32), "sample_rate": sample_rate, **extra}
    )


def test_an_empty_batch_returns_an_empty_list() -> None:
    stage, _ = _stage()
    assert stage.process_batch([]) == []


def test_a_ray_data_numpy_batch_is_processed() -> None:
    stage, adapter = _stage()
    tasks: Any = np.asarray([_task(), _task()], dtype=object)

    results = stage.process_batch(tasks)

    assert len(results) == 2
    assert len(adapter.calls[0]) == 2
    assert all("_sed_framewise" in task.data for task in results)


def test_every_task_gets_the_canonical_sed_outputs() -> None:
    stage, _ = _stage()
    results = stage.process_batch([_task(), _task()])
    expected = {"_sed_framewise", "sed_valid_frames", "sed_fps"}
    assert all(expected <= set(task.data) for task in results)


def test_a_ragged_batch_becomes_one_adapter_call() -> None:
    stage, adapter = _stage()
    stage.process_batch([_task(1.0), _task(3.0)])
    assert len(adapter.calls) == 1
    assert [len(item["waveform"]) for item in adapter.calls[0]] == [_SR, 3 * _SR]


def test_valid_frames_belong_to_each_real_waveform() -> None:
    stage, _ = _stage()
    short, long = stage.process_batch([_task(1.0), _task(3.0)])
    assert short.data["sed_valid_frames"] == _SR / _HOP
    assert long.data["sed_valid_frames"] == 3 * _SR / _HOP
    assert short.data["_sed_framewise"].shape == long.data["_sed_framewise"].shape


def test_process_rejects_non_batch_execution() -> None:
    stage, adapter = _stage()
    with pytest.raises(NotImplementedError, match="only supports process_batch"):
        stage.process(_task())
    assert adapter.calls == []


def test_a_flagged_task_is_passed_through_untouched() -> None:
    stage, adapter = _stage()
    (task,) = stage.process_batch([_task(_skipme="bad audio")])
    assert "_sed_framewise" not in task.data
    assert adapter.calls == []


def test_a_partially_skipped_batch_emits_a_warning() -> None:
    stage, _ = _stage()
    with patch("nemo_curator.stages.audio.inference.sed.stage.logger.warning") as warning:
        stage.process_batch([_task(), _task(_skipme="bad audio")])

    warning.assert_called_once_with("SED batch: skipped {}/{} tasks", 1, 2)


def test_a_missing_waveform_marks_only_that_task() -> None:
    stage, adapter = _stage()
    good, bad = stage.process_batch([_task(), AudioTask(data={"sample_rate": _SR})])
    assert good.data["_sed_framewise"] is not None
    assert "_sed_framewise" not in bad.data
    assert bad.data["_skipme"] == "audio_load_error"
    assert len(adapter.calls[0]) == 1


def test_a_missing_audio_filepath_is_marked_without_adapter_inference() -> None:
    stage = SEDInferenceStage(
        adapter_target=_ADAPTER_TARGET,
        checkpoint_path=_CHECKPOINT,
        sample_rate=_SR,
        waveform_key=None,
    )
    adapter = _FakeSEDAdapter()
    stage._adapter = adapter

    (task,) = stage.process_batch([AudioTask(data={})])

    assert task.data["_skipme"] == "audio_load_error"
    assert adapter.calls == []


def test_audio_preparation_failure_clears_partial_outputs_and_is_resumable() -> None:
    stage, adapter = _stage(save_npz=True, skip_if_output_exists=True)
    task = AudioTask(
        data={
            "waveform": np.zeros(_SR, dtype=np.float32),
            "sample_rate": 0,
            "_sed_framewise": np.zeros((5, _CLASSES), dtype=np.float32),
            "sed_valid_frames": 5,
            "sed_fps": 50.0,
        }
    )

    (failed,) = stage.process_batch([task])

    assert failed.data["_skipme"] == "audio_load_error"
    assert "_sed_framewise" not in failed.data
    assert "sed_valid_frames" not in failed.data
    assert "sed_fps" not in failed.data
    assert "npz_filepath" not in failed.data
    assert adapter.calls == []

    resumed_stage, resumed_adapter = _stage(save_npz=True, skip_if_output_exists=True)
    resumed_stage.process_batch([failed])
    assert resumed_adapter.calls == []


def test_resume_sends_only_unfinished_tasks_to_the_adapter() -> None:
    stage, adapter = _stage(skip_if_output_exists=True)
    done = _task()
    done.data.update(_sed_framewise=np.zeros((5, _CLASSES)), sed_valid_frames=5, sed_fps=50.0)
    stage.process_batch([done, _task()])
    assert len(adapter.calls[0]) == 1


def test_resume_returns_before_the_adapter_when_all_tasks_are_done() -> None:
    stage, adapter = _stage(skip_if_output_exists=True)
    done = _task()
    done.data.update(_sed_framewise=np.zeros((5, _CLASSES)), sed_valid_frames=5, sed_fps=50.0)
    stage.process_batch([done])
    assert adapter.calls == []


def test_resume_with_sidecars_requires_the_npz_path_field(tmp_path: Path) -> None:
    stage, adapter = _stage(skip_if_output_exists=True, save_npz=True, output_dir=str(tmp_path))
    partial = _task()
    partial._set_task_id("manifest", 0)
    partial.data.update(_sed_framewise=np.zeros((5, _CLASSES)), sed_valid_frames=5, sed_fps=50.0)
    stage.process_batch([partial])
    assert adapter.calls
    assert Path(partial.data["npz_filepath"]).is_file()


@pytest.mark.parametrize(
    "waveform",
    [
        np.zeros(_SR, dtype=np.float32),
        torch.zeros(_SR, dtype=torch.float32),
        np.zeros((2, _SR), dtype=np.float32),
        torch.zeros((2, _SR), dtype=torch.float32),
    ],
)
def test_waveforms_are_normalized_to_contiguous_mono(waveform: object) -> None:
    stage, adapter = _stage()
    stage.process_batch([AudioTask(data={"waveform": waveform, "sample_rate": _SR})])
    prepared = adapter.calls[0][0]["waveform"]
    assert prepared.shape == (_SR,)
    assert prepared.dtype == np.float32
    assert prepared.flags.c_contiguous


def test_a_mismatched_sample_rate_is_resampled_before_the_adapter() -> None:
    librosa = MagicMock()
    librosa.resample.side_effect = lambda waveform, *, orig_sr, target_sr: np.repeat(waveform, target_sr // orig_sr)
    stage, adapter = _stage()
    with patch.dict("sys.modules", {"librosa": librosa}):
        stage.process_batch([_task(seconds=1.0, sample_rate=8000)])
    assert len(adapter.calls[0][0]["waveform"]) == pytest.approx(_SR, rel=0.01)
    librosa.resample.assert_called_once()


def test_a_matching_sample_rate_avoids_librosa() -> None:
    librosa = MagicMock()
    stage, _ = _stage()
    with patch.dict("sys.modules", {"librosa": librosa}):
        stage.process_batch([_task()])
    librosa.resample.assert_not_called()


def test_file_mode_loads_the_yaml_manifest_contract(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    soundfile.write(audio_path, np.zeros(_SR, dtype=np.float32), _SR)
    stage = SEDInferenceStage(
        adapter_target=_ADAPTER_TARGET,
        checkpoint_path=_CHECKPOINT,
        sample_rate=_SR,
        waveform_key=None,
    )
    adapter = _FakeSEDAdapter()
    stage._adapter = adapter
    (task,) = stage.process_batch([AudioTask(data={"audio_filepath": str(audio_path)})])
    assert len(adapter.calls[0][0]["waveform"]) == _SR
    assert task.data["sed_fps"] == _SR / _HOP


def test_framewise_output_is_float16_by_default() -> None:
    stage, _ = _stage()
    (task,) = stage.process_batch([_task()])
    assert task.data["_sed_framewise"].dtype == np.float16


def test_float32_storage_can_be_requested() -> None:
    stage, _ = _stage(framewise_dtype="float32")
    (task,) = stage.process_batch([_task()])
    assert task.data["_sed_framewise"].dtype == np.float32


def test_invalid_storage_dtype_is_rejected() -> None:
    with pytest.raises(ValueError, match="framewise_dtype"):
        _stage(framewise_dtype="float64")


def test_no_sidecar_is_written_by_default() -> None:
    stage, _ = _stage()
    (task,) = stage.process_batch([_task(audio_filepath="/audio/clip.wav")])
    assert "npz_filepath" not in task.data


def test_a_sidecar_round_trips_the_adapter_result(tmp_path: Path) -> None:
    stage, _ = _stage(save_npz=True, output_dir=str(tmp_path))
    (task,) = stage.process_batch([_task(audio_filepath="/audio/clip.wav")])
    with np.load(task.data["npz_filepath"]) as npz:
        assert npz["framewise"].shape[1] == _CLASSES
        assert float(npz["fps"]) == _SR / _HOP
        assert int(npz["valid_frames"]) == task.data["sed_valid_frames"]


def test_waveform_mode_writes_a_sidecar_without_an_audio_filepath(tmp_path: Path) -> None:
    stage, adapter = _stage(save_npz=True, output_dir=str(tmp_path))
    task = _task()
    task._set_task_id("manifest", 0)

    (result,) = stage.process_batch([task])

    assert len(adapter.calls) == 1
    assert {"_sed_framewise", "sed_valid_frames", "sed_fps", "npz_filepath"} <= set(result.data)
    assert "_skipme" not in result.data
    npz_path = Path(result.data["npz_filepath"])
    assert npz_path.is_file()
    assert npz_path.parent == tmp_path / "framewise"
    with np.load(npz_path) as npz:
        np.testing.assert_array_equal(npz["framewise"], result.data["_sed_framewise"])
        assert float(npz["fps"]) == _SR / _HOP
        assert npz["audio_filepath"].item() == ""
        assert int(npz["original_num_samples"]) == _SR
        assert int(npz["valid_frames"]) == result.data["sed_valid_frames"]


def test_waveform_sidecar_makes_the_task_resumable(tmp_path: Path) -> None:
    first_stage, first_adapter = _stage(save_npz=True, output_dir=str(tmp_path))
    task = _task()
    task._set_task_id("manifest", 0)
    (completed,) = first_stage.process_batch([task])
    original_path = completed.data["npz_filepath"]
    assert len(first_adapter.calls) == 1

    resume_stage, resume_adapter = _stage(
        save_npz=True,
        skip_if_output_exists=True,
        output_dir=str(tmp_path),
    )
    (resumed,) = resume_stage.process_batch([completed])

    assert resume_adapter.calls == []
    assert resumed.data["npz_filepath"] == original_path


def test_waveform_sidecar_paths_use_the_task_identity(tmp_path: Path) -> None:
    stage, _ = _stage(save_npz=True, output_dir=str(tmp_path))
    first = _task()
    first._set_task_id("manifest", 0)
    second = _task()
    second._set_task_id("manifest", 1)
    first, second = stage.process_batch([first, second])
    assert first.data["npz_filepath"] != second.data["npz_filepath"]

    rerun, _ = _stage(save_npz=True, output_dir=str(tmp_path))
    same_identity = _task()
    same_identity._set_task_id("manifest", 0)
    (same_identity,) = rerun.process_batch([same_identity])
    assert same_identity.data["npz_filepath"] == first.data["npz_filepath"]


def test_same_basenames_get_distinct_stable_sidecar_paths(tmp_path: Path) -> None:
    stage, _ = _stage(save_npz=True, output_dir=str(tmp_path))
    first, second = stage.process_batch([_task(audio_filepath="/a/clip.wav"), _task(audio_filepath="/b/clip.wav")])
    assert first.data["npz_filepath"] != second.data["npz_filepath"]

    rerun, _ = _stage(save_npz=True, output_dir=str(tmp_path))
    (again,) = rerun.process_batch([_task(audio_filepath="/a/clip.wav")])
    assert again.data["npz_filepath"] == first.data["npz_filepath"]


class _LifecycleAdapter:
    """Adapter used to verify setup constructor and lifecycle calls."""

    instance: _LifecycleAdapter | None = None

    def __init__(self, checkpoint_path: str | None, sample_rate: int, marker: str) -> None:
        self.checkpoint_path = checkpoint_path
        self.sample_rate = sample_rate
        self.marker = marker
        self.download_calls = 0
        self.load_calls: list[int] = []
        self.unloaded = False
        type(self).instance = self

    def download_weights_on_node(self) -> None:
        self.download_calls += 1

    def load_model(self, *, num_gpus: int) -> None:
        self.load_calls.append(num_gpus)

    def unload_model(self) -> None:
        self.unloaded = True


def test_setup_constructs_and_loads_the_selected_adapter() -> None:
    stage = SEDInferenceStage(
        adapter_target="package.Adapter",
        checkpoint_path=_CHECKPOINT,
        adapter_kwargs={"marker": "from-yaml"},
        resources=Resources(gpus=0),
    )
    with patch("hydra.utils.get_class", return_value=_LifecycleAdapter):
        stage.setup()
    adapter = _LifecycleAdapter.instance
    assert adapter is not None
    assert adapter.checkpoint_path == _CHECKPOINT
    assert adapter.sample_rate == 32000
    assert adapter.marker == "from-yaml"
    assert adapter.load_calls == [0]


def test_setup_on_node_prefetches_without_loading_the_model() -> None:
    stage = SEDInferenceStage(
        adapter_target="package.Adapter",
        checkpoint_path=None,
        adapter_kwargs={"marker": "auto"},
    )
    with patch("hydra.utils.get_class", return_value=_LifecycleAdapter):
        stage.setup_on_node()
    adapter = _LifecycleAdapter.instance
    assert adapter is not None
    assert adapter.checkpoint_path is None
    assert adapter.download_calls == 1
    assert adapter.load_calls == []


def test_setup_on_node_raises_on_prefetch_failure_by_default() -> None:
    stage = SEDInferenceStage(
        adapter_target="package.Adapter",
        adapter_kwargs={"marker": "auto"},
    )
    with (
        patch("hydra.utils.get_class", return_value=_LifecycleAdapter),
        patch.object(_LifecycleAdapter, "download_weights_on_node", side_effect=RuntimeError("offline")),
        pytest.raises(RuntimeError, match="download_weights_on_node failed"),
    ):
        stage.setup_on_node()


def test_setup_on_node_can_warn_and_retry_during_worker_setup() -> None:
    stage = SEDInferenceStage(
        adapter_target="package.Adapter",
        adapter_kwargs={"marker": "auto"},
        prefetch_fail_on_error=False,
    )
    with (
        patch("hydra.utils.get_class", return_value=_LifecycleAdapter),
        patch.object(_LifecycleAdapter, "download_weights_on_node", side_effect=RuntimeError("offline")),
    ):
        stage.setup_on_node()


def test_teardown_delegates_to_the_adapter() -> None:
    stage, adapter = _stage()
    stage.teardown()
    assert adapter.unloaded
    assert stage._adapter is None


def test_adapter_result_count_must_match_input_count() -> None:
    stage, adapter = _stage()
    adapter.infer_batch = lambda _items: []  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="must match 1:1"):
        stage.process_batch([_task()])


def test_declared_inputs_switch_between_waveform_and_file_modes() -> None:
    waveform_stage, _ = _stage()
    file_stage = SEDInferenceStage(adapter_target=_ADAPTER_TARGET, checkpoint_path=_CHECKPOINT)
    assert waveform_stage.inputs()[1] == ["waveform", "sample_rate"]
    assert file_stage.inputs()[1] == ["audio_filepath"]


def test_sidecar_output_is_declared_only_when_enabled() -> None:
    plain, _ = _stage()
    sidecars, _ = _stage(save_npz=True)
    assert "_skipme" in plain.outputs()[1]
    assert "npz_filepath" not in plain.outputs()[1]
    assert "npz_filepath" in sidecars.outputs()[1]


def test_worker_count_is_backend_selected_or_explicitly_pinned() -> None:
    default, _ = _stage()
    pinned = default.with_(num_workers=4)
    assert default.num_workers() is None
    assert pinned.num_workers() == 4


def test_valid_frames_and_fps_recover_real_duration() -> None:
    stage, _ = _stage()
    short, _long = stage.process_batch([_task(2.0), _task(5.0)])
    assert short.data["sed_valid_frames"] / short.data["sed_fps"] == pytest.approx(2.0, abs=0.05)
    assert short.data["sed_valid_frames"] < short.data["_sed_framewise"].shape[0]


def test_example_yaml_instantiates_the_stage_adapter_contract() -> None:
    cfg = OmegaConf.load(_PIPELINE_YAML)
    cfg.manifest_path = "/input.jsonl"
    cfg.output_dir = "/output"
    stage = _instantiate_stage(cfg.stages[1])
    assert isinstance(stage, SEDInferenceStage)
    assert stage.adapter_target == _ADAPTER_TARGET
    assert stage.checkpoint_path is None
    assert stage.sample_rate == 32000
    assert stage.adapter_kwargs["model_type"] == "Cnn14_DecisionLevelMax"
    assert stage.save_npz is True


def test_example_yaml_accepts_a_checkpoint_file_path() -> None:
    cfg = OmegaConf.load(_PIPELINE_YAML)
    cfg.manifest_path = "/input.jsonl"
    cfg.checkpoint_path = _CHECKPOINT
    cfg.output_dir = "/output"
    stage = _instantiate_stage(cfg.stages[1])
    assert stage.checkpoint_path == _CHECKPOINT


def test_module_docstring_contains_the_exact_yaml_command() -> None:
    from nemo_curator.stages.audio.inference.sed import stage

    assert "tutorials/audio/sed" in stage.__doc__
    assert "--extra audio_cuda12" in stage.__doc__
    assert "--config-name pipeline" in stage.__doc__
    default_command = stage.__doc__.split("To choose the exact download file", maxsplit=1)[0]
    assert "checkpoint_path=" not in default_command
    assert "checkpoint_path=/absolute/path/to/Cnn14_DecisionLevelMax_mAP\\=0.385.pth" in stage.__doc__
    assert "downloaded to that exact path" in stage.__doc__
    assert "loaded without network access" in stage.__doc__
    assert "Cnn14_mAP=0.431.pth" in stage.__doc__
    assert "clip-level audio tagging" in stage.__doc__
