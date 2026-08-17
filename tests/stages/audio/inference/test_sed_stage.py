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

"""Tests for SEDInferenceStage, driven by a stand-in model so no weights are needed."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nemo_curator.stages.audio.inference.sed import SEDInferenceStage  # noqa: E402
from nemo_curator.tasks import AudioTask  # noqa: E402

_SR = 16000
_HOP = 320
_CLASSES = 527


class _FakeSEDModel:
    """Stands in for CNN14: records its input and returns correctly shaped output."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def __call__(self, x: "torch.Tensor", _mixup: object = None) -> dict[str, "torch.Tensor"]:
        batch, samples = x.shape
        self.calls.append((batch, samples))
        frames = samples // _HOP
        return {
            "framewise_output": torch.full((batch, frames, _CLASSES), 0.5),
            "clipwise_output": torch.full((batch, _CLASSES), 0.5),
        }


def _stage(**kwargs: object) -> tuple[SEDInferenceStage, _FakeSEDModel]:
    stage = SEDInferenceStage(sample_rate=_SR, hop_size=_HOP, classes_num=_CLASSES, **kwargs)
    model = _FakeSEDModel()
    stage._model = model
    stage._device = torch.device("cpu")
    return stage, model


def _task(seconds: float = 1.0, sample_rate: int = _SR, **extra: object) -> AudioTask:
    return AudioTask(
        data={"waveform": np.zeros(int(seconds * sample_rate), dtype=np.float32), "sample_rate": sample_rate, **extra}
    )


# ----------------------------------------------------------------------
# Batch contract
# ----------------------------------------------------------------------


def test_an_empty_batch_returns_an_empty_list() -> None:
    stage, _ = _stage()
    assert stage.process_batch([]) == []


def test_every_task_comes_back_with_framewise_output() -> None:
    stage, _ = _stage()
    results = stage.process_batch([_task(), _task()])
    assert all(task.data["_sed_framewise"] is not None for task in results)


def test_a_ragged_batch_becomes_one_padded_model_call() -> None:
    stage, model = _stage()
    stage.process_batch([_task(1.0), _task(3.0)])
    assert len(model.calls) == 1
    batch, samples = model.calls[0]
    assert batch == 2
    assert samples == 3 * _SR


def test_valid_frames_reflect_each_row_not_the_padded_length() -> None:
    """The short row must not claim the frames that only exist because of padding."""
    stage, _ = _stage()
    short, long = stage.process_batch([_task(1.0), _task(3.0)])
    assert short.data["sed_valid_frames"] == pytest.approx(_SR / _HOP, abs=1)
    assert long.data["sed_valid_frames"] == pytest.approx(3 * _SR / _HOP, abs=1)


def test_fps_is_derived_from_the_sample_rate_and_hop() -> None:
    stage, _ = _stage()
    (task,) = stage.process_batch([_task()])
    assert task.data["sed_fps"] == pytest.approx(_SR / _HOP)


def test_process_delegates_to_process_batch() -> None:
    stage, model = _stage()
    task = stage.process(_task())
    assert task.data["_sed_framewise"] is not None
    assert model.calls[0][0] == 1


# ----------------------------------------------------------------------
# Skipping
# ----------------------------------------------------------------------


def test_a_flagged_task_is_passed_through_untouched() -> None:
    stage, model = _stage()
    (task,) = stage.process_batch([_task(_skipme="Hallucination")])
    assert "_sed_framewise" not in task.data
    assert model.calls == []


def test_a_task_missing_its_waveform_is_skipped_without_failing_the_batch() -> None:
    stage, _ = _stage()
    good, bad = stage.process_batch([_task(), AudioTask(data={"sample_rate": _SR})])
    assert good.data["_sed_framewise"] is not None
    assert "_sed_framewise" not in bad.data


def test_a_batch_of_only_skipped_tasks_never_reaches_the_model() -> None:
    stage, model = _stage()
    stage.process_batch([_task(_skipme="x"), _task(_skipme="y")])
    assert model.calls == []


def test_resume_skips_tasks_that_already_have_output() -> None:
    stage, model = _stage(skip_if_output_exists=True)
    done = _task()
    done.data.update(_sed_framewise=np.zeros((5, _CLASSES), dtype=np.float16), sed_valid_frames=5, sed_fps=50.0)

    stage.process_batch([done, _task()])

    assert model.calls[0][0] == 1, "only the unfinished task should be sent to the model"


def test_resume_returns_early_when_the_whole_batch_is_done() -> None:
    stage, model = _stage(skip_if_output_exists=True)
    done = _task()
    done.data.update(_sed_framewise=np.zeros((5, _CLASSES), dtype=np.float16), sed_valid_frames=5, sed_fps=50.0)

    stage.process_batch([done])

    assert model.calls == []


def test_resume_reprocesses_a_task_whose_npz_is_missing() -> None:
    """With save_npz on, in-memory output alone is not a complete result."""
    stage, model = _stage(skip_if_output_exists=True, save_npz=True)
    partial = _task()
    partial.data.update(_sed_framewise=np.zeros((5, _CLASSES), dtype=np.float16), sed_valid_frames=5, sed_fps=50.0)

    stage.process_batch([partial])

    assert model.calls, "task without an npz_filepath should be reprocessed"


# ----------------------------------------------------------------------
# Waveform preparation
# ----------------------------------------------------------------------


def test_a_clip_shorter_than_the_model_minimum_is_padded_up() -> None:
    stage, model = _stage(window_size=1024)
    stage.process_batch([_task(seconds=0.001)])
    _batch, samples = model.calls[0]
    assert samples >= max(1024, _HOP * 32)


def test_padding_short_clips_can_be_disabled() -> None:
    stage, model = _stage(window_size=1024, pad_short_segments=False)
    stage.process_batch([_task(seconds=0.001)])
    _batch, samples = model.calls[0]
    assert samples == 16


def test_a_stereo_waveform_is_mixed_down_to_mono() -> None:
    stage, model = _stage()
    task = AudioTask(data={"waveform": np.zeros((_SR, 2), dtype=np.float32), "sample_rate": _SR})
    stage.process_batch([task])
    _batch, samples = model.calls[0]
    assert samples == _SR


def test_a_mismatched_sample_rate_is_resampled_to_the_model_rate() -> None:
    pytest.importorskip("librosa")
    stage, model = _stage()
    stage.process_batch([_task(seconds=1.0, sample_rate=8000)])
    _batch, samples = model.calls[0]
    assert samples == pytest.approx(_SR, rel=0.01)


def test_a_matching_sample_rate_skips_resampling() -> None:
    pytest.importorskip("librosa")
    stage, model = _stage()
    with patch("librosa.resample") as resample:
        stage.process_batch([_task(sample_rate=_SR)])
    resample.assert_not_called()
    assert model.calls[0][1] == _SR


# ----------------------------------------------------------------------
# Output storage
# ----------------------------------------------------------------------


def test_framewise_output_is_stored_as_float16_by_default() -> None:
    stage, _ = _stage()
    (task,) = stage.process_batch([_task()])
    assert task.data["_sed_framewise"].dtype == np.float16


def test_float32_storage_can_be_requested() -> None:
    stage, _ = _stage(framewise_dtype="float32")
    (task,) = stage.process_batch([_task()])
    assert task.data["_sed_framewise"].dtype == np.float32


def test_no_sidecar_is_written_by_default() -> None:
    stage, _ = _stage()
    (task,) = stage.process_batch([_task(audio_filepath="/audio/clip.wav")])
    assert "npz_filepath" not in task.data


def test_a_sidecar_round_trips_the_framewise_array(tmp_path: Path) -> None:
    stage, _ = _stage(save_npz=True, output_dir=str(tmp_path))
    (task,) = stage.process_batch([_task(audio_filepath="/audio/clip.wav")])

    with np.load(task.data["npz_filepath"]) as npz:
        assert npz["framewise"].shape[1] == _CLASSES
        assert float(npz["fps"]) == pytest.approx(_SR / _HOP)
        assert int(npz["valid_frames"]) == task.data["sed_valid_frames"]


def test_same_named_files_from_different_directories_get_distinct_sidecars(tmp_path: Path) -> None:
    stage, _ = _stage(save_npz=True, output_dir=str(tmp_path))
    first, second = stage.process_batch([_task(audio_filepath="/a/clip.wav"), _task(audio_filepath="/b/clip.wav")])
    assert first.data["npz_filepath"] != second.data["npz_filepath"]


def test_the_sidecar_path_is_stable_across_runs(tmp_path: Path) -> None:
    """Resume depends on the same input mapping to the same output path."""
    paths = []
    for _ in range(2):
        stage, _ = _stage(save_npz=True, output_dir=str(tmp_path))
        (task,) = stage.process_batch([_task(audio_filepath="/audio/clip.wav")])
        paths.append(task.data["npz_filepath"])
    assert paths[0] == paths[1]


def test_no_sidecar_is_written_when_the_audio_path_is_unknown(tmp_path: Path) -> None:
    stage, _ = _stage(save_npz=True, output_dir=str(tmp_path))
    (task,) = stage.process_batch([_task()])
    assert "npz_filepath" not in task.data


# ----------------------------------------------------------------------
# Setup and configuration
# ----------------------------------------------------------------------


def test_setup_requires_a_checkpoint_path() -> None:
    with pytest.raises(ValueError, match="checkpoint_path is required"):
        SEDInferenceStage().setup()


def test_setup_rejects_an_unknown_model_type() -> None:
    stage = SEDInferenceStage(checkpoint_path="/weights/model.pth", model_type="NotAModel")
    with pytest.raises(ValueError, match="Unknown SED model_type"):
        stage.setup()


def test_setup_loads_the_checkpoint_onto_cpu_first() -> None:
    """Loading straight to GPU can collide with another engine already holding memory."""
    stage = SEDInferenceStage(checkpoint_path="/weights/model.pth")
    with (
        patch("nemo_curator.stages.audio.inference.sed_models.get_model_class"),
        patch("torch.load", return_value={"model": {}}) as torch_load,
        patch("torch.cuda.is_available", return_value=False),
    ):
        stage.setup()
    assert torch_load.call_args.kwargs["map_location"] == "cpu"


def test_setup_loads_weights_without_executing_pickled_code() -> None:
    stage = SEDInferenceStage(checkpoint_path="/weights/model.pth")
    with (
        patch("nemo_curator.stages.audio.inference.sed_models.get_model_class"),
        patch("torch.load", return_value={"model": {}}) as torch_load,
        patch("torch.cuda.is_available", return_value=False),
    ):
        stage.setup()
    assert torch_load.call_args.kwargs["weights_only"] is True


def test_teardown_releases_the_model() -> None:
    stage, _ = _stage()
    stage.teardown()
    assert stage._model is None


def test_the_npz_output_key_is_declared_only_when_sidecars_are_enabled() -> None:
    assert "npz_filepath" not in SEDInferenceStage().outputs()[1]
    assert "npz_filepath" in SEDInferenceStage(save_npz=True).outputs()[1]


def test_declared_outputs_match_what_process_batch_writes() -> None:
    stage, _ = _stage()
    (task,) = stage.process_batch([_task()])
    declared = set(stage.outputs()[1])
    assert {"_sed_framewise", "sed_valid_frames", "sed_fps"} <= declared
    assert {"_sed_framewise", "sed_valid_frames", "sed_fps"} <= set(task.data)


def test_worker_count_is_left_to_the_backend_by_default() -> None:
    assert "num_workers" not in SEDInferenceStage().xenna_stage_spec()


def test_worker_count_can_be_pinned() -> None:
    assert SEDInferenceStage(num_workers_override=4).xenna_stage_spec()["num_workers"] == 4


# ----------------------------------------------------------------------
# Contract offered to consumers
# ----------------------------------------------------------------------


def test_framewise_output_is_a_two_dimensional_probability_track_per_class() -> None:
    """Consumers index the matrix as [frame, class], so pin that layout."""
    stage, _ = _stage()
    (task,) = stage.process_batch([_task(seconds=2.0)])

    framewise = task.data["_sed_framewise"]
    assert framewise.ndim == 2
    assert framewise.shape[1] == _CLASSES
    assert framewise.min() >= 0.0
    assert framewise.max() <= 1.0


def test_valid_frames_and_fps_recover_the_real_audio_duration() -> None:
    """These two keys are all a consumer gets to convert frame indices into times."""
    stage, _ = _stage()
    short, _long = stage.process_batch([_task(seconds=2.0), _task(seconds=5.0)])

    assert short.data["sed_valid_frames"] / short.data["sed_fps"] == pytest.approx(2.0, abs=0.05)
    # Padding up to the longest row in the batch must never read as real audio.
    assert short.data["sed_valid_frames"] < short.data["_sed_framewise"].shape[0]
