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

"""Tests for the generic ``ASRStage`` exercised against a mock ``ASRAdapter`` (no real model load)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.models.asr.base import ASRResult
from nemo_curator.stages.audio.inference.asr.stage import ASRStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

_QWEN_ADAPTER_TARGET = "nemo_curator.models.asr.qwen_omni.QwenOmniASRAdapter"
_FASTER_WHISPER_ADAPTER_TARGET = "nemo_curator.models.faster_whisper_asr.FasterWhisperASR"
_SR = 16000
_RESAMPLED_AUDIO_PATH = "/data/resampled.wav"


def _make_stage(  # noqa: PLR0913
    *,
    default_language: str | None = None,
    batch_size: int = 32,
    supported_language_codes: list[str] | None = None,
    skip_if_output_exists: bool = False,
    waveform_key: str | None = None,
    keep_waveform: bool = False,
    extras_key: str | None = None,
    fail_on_audio_error: bool = False,
) -> ASRStage:
    """Build an ASRStage wired to a mock adapter (no real model load)."""
    stage = ASRStage(
        adapter_target=_QWEN_ADAPTER_TARGET,
        model_id="mock/qwen-omni",
        pred_text_key="pred_text",
        default_language=default_language,
        batch_size=batch_size,
        supported_language_codes=supported_language_codes,
        skip_if_output_exists=skip_if_output_exists,
        waveform_key=waveform_key,
        keep_waveform=keep_waveform,
        extras_key=extras_key,
        fail_on_audio_error=fail_on_audio_error,
    )
    mock_adapter = MagicMock()
    stage._adapter = mock_adapter
    stage._load_audio = MagicMock(  # type: ignore[method-assign]
        return_value=(np.zeros(_SR, dtype=np.float32), _SR)
    )
    return stage


def _make_task(source_lang: str | None = "en") -> AudioTask:
    data: dict[str, object] = {"resampled_audio_filepath": _RESAMPLED_AUDIO_PATH}
    if source_lang is not None:
        data["source_lang"] = source_lang
    return AudioTask(data=data)


def _make_waveform_task(
    *,
    waveform: np.ndarray | None = None,
    sample_rate: int = _SR,
    source_lang: str | None = "en",
) -> AudioTask:
    data: dict[str, object] = {
        "waveform": np.zeros(_SR, dtype=np.float32) if waveform is None else waveform,
        "sampling_rate": sample_rate,
    }
    if source_lang is not None:
        data["source_lang"] = source_lang
    return AudioTask(data=data)


def test_process_raises_not_implemented() -> None:
    stage = _make_stage()
    with pytest.raises(NotImplementedError):
        stage.process(_make_task())


def test_empty_batch_does_not_create_an_unparented_sentinel() -> None:
    stage = _make_stage()
    assert stage.process_batch([]) == []


def test_basic_inference() -> None:
    stage = _make_stage()
    stage._adapter.transcribe_batch.return_value = [
        ASRResult(text="hello world"),
    ]

    results = stage.process_batch([_make_task()])

    assert results[0].data["pred_text"] == "hello world"
    assert results[0].data == {
        "resampled_audio_filepath": _RESAMPLED_AUDIO_PATH,
        "source_lang": "en",
        "pred_text": "hello world",
    }
    inferred_item = stage._adapter.transcribe_batch.call_args.args[0][0]
    assert set(inferred_item) == {
        "waveform",
        "sample_rate",
        "language",
        "language_code",
        "task_id",
    }
    assert inferred_item["waveform"].shape == (_SR,)
    assert inferred_item["sample_rate"] == _SR


def test_adapter_not_initialized_raises() -> None:
    stage = ASRStage(adapter_target=_QWEN_ADAPTER_TARGET, model_id="mock/model")
    with pytest.raises(RuntimeError, match="setup"):
        stage.process_batch([_make_task()])


def test_multi_task_batch_preserves_order() -> None:
    stage = _make_stage()
    stage._adapter.transcribe_batch.return_value = [
        ASRResult(text="text1"),
        ASRResult(text="text2"),
    ]
    results = stage.process_batch([_make_task(), _make_task()])

    assert results[0].data["pred_text"] == "text1"
    assert results[1].data["pred_text"] == "text2"


def test_audio_load_failure_skips_only_failed_item_and_preserves_order() -> None:
    stage = _make_stage()
    waveform = np.zeros(_SR, dtype=np.float32)
    stage._load_audio.side_effect = [
        (waveform, _SR),
        RuntimeError("corrupt audio"),
        (waveform, _SR),
    ]
    stage._adapter.transcribe_batch.return_value = [
        ASRResult(text="text1"),
        ASRResult(text="text3"),
    ]
    tasks = [_make_task(), _make_task(), _make_task()]
    for index, task in enumerate(tasks, start=1):
        task.task_id = f"task-{index}"
        task.data["resampled_audio_filepath"] = f"/data/resampled-{index}.wav"

    results = stage.process_batch(tasks)

    assert [task.data["pred_text"] for task in results] == ["text1", "", "text3"]
    assert "_skipme" not in results[0].data
    assert results[1].data["_skipme"] == "audio_load_error"
    assert "_skipme" not in results[2].data
    inferred_items = stage._adapter.transcribe_batch.call_args.args[0]
    assert [item["task_id"] for item in inferred_items] == [tasks[0].task_id, tasks[2].task_id]


def test_audio_load_failure_can_fail_strict_benchmarks() -> None:
    stage = _make_stage(fail_on_audio_error=True)
    stage._load_audio.side_effect = RuntimeError("corrupt audio")

    with pytest.raises(RuntimeError, match="failed to prepare audio"):
        stage.process_batch([_make_task()])

    stage._adapter.transcribe_batch.assert_not_called()


def test_skip_if_output_exists_reuses_prediction_and_only_infers_missing_rows() -> None:
    stage = _make_stage(skip_if_output_exists=True)
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="new prediction")]
    existing = _make_task()
    existing.data["pred_text"] = "existing prediction"
    missing = _make_task()

    results = stage.process_batch([existing, missing])

    assert results == [existing, missing]
    assert existing.data["pred_text"] == "existing prediction"
    assert missing.data["pred_text"] == "new prediction"
    inferred_items = stage._adapter.transcribe_batch.call_args.args[0]
    assert len(inferred_items) == 1
    stage._load_audio.assert_called_once_with(_RESAMPLED_AUDIO_PATH)


def test_skip_if_output_exists_skips_entire_prefilled_batch() -> None:
    stage = _make_stage(skip_if_output_exists=True)
    tasks = [_make_task(), _make_task()]
    tasks[0].data["pred_text"] = "first"
    tasks[1].data["pred_text"] = "second"

    results = stage.process_batch(tasks)

    assert [task.data["pred_text"] for task in results] == ["first", "second"]
    stage._adapter.transcribe_batch.assert_not_called()


def test_skip_if_output_exists_does_not_skip_empty_prediction() -> None:
    stage = _make_stage(skip_if_output_exists=True)
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="filled")]
    task = _make_task()
    task.data["pred_text"] = ""

    result = stage.process_batch([task])

    assert result[0].data["pred_text"] == "filled"
    stage._adapter.transcribe_batch.assert_called_once()


def test_adapter_result_length_mismatch_raises() -> None:
    stage = _make_stage()
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="x")]  # 1 result
    with pytest.raises(RuntimeError, match=r"returned 1 results for 2 supported items"):
        stage.process_batch([_make_task(), _make_task()])


def test_language_resolution_from_task() -> None:
    stage = _make_stage()
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="hola")]

    task = AudioTask(
        data={
            "resampled_audio_filepath": "/data/spanish.wav",
            "source_lang": "es",
        }
    )
    stage.process_batch([task])

    items = stage._adapter.transcribe_batch.call_args[0][0]
    assert items[0]["language"] == "Spanish"


def test_default_language_used_when_task_language_missing() -> None:
    stage = _make_stage(default_language="en")
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="hello")]

    task = AudioTask(data={"resampled_audio_filepath": _RESAMPLED_AUDIO_PATH})
    stage.process_batch([task])

    items = stage._adapter.transcribe_batch.call_args[0][0]
    assert items[0]["language"] == "English"


def test_supported_language_filter_skips_before_adapter_call() -> None:
    stage = _make_stage(supported_language_codes=["en"])

    results = stage.process_batch([_make_task(source_lang="pl")])

    stage._adapter.transcribe_batch.assert_not_called()
    stage._load_audio.assert_not_called()
    assert results[0].data["pred_text"] == ""
    assert results[0].data["_skipme"] == "language_not_supported"
    assert results[0].data["additional_notes"]["ASR_inference"] == "skipped (unsupported language: pl)"
    assert results[0].data["additional_notes"]["pred_text"] == "lang_not_supported:pl"


def test_supported_language_filter_annotates_missing_language() -> None:
    stage = _make_stage(supported_language_codes=["en"])

    results = stage.process_batch([_make_task(source_lang=None)])

    stage._adapter.transcribe_batch.assert_not_called()
    stage._load_audio.assert_not_called()
    assert results[0].data["pred_text"] == ""
    assert results[0].data["_skipme"] == "language_missing"
    assert results[0].data["additional_notes"]["ASR_inference"] == "skipped (missing language)"
    assert results[0].data["additional_notes"]["pred_text"] == "language_missing"


def test_resumability_preserves_unsupported_task_lineage() -> None:
    stage = _make_stage(supported_language_codes=["en"])
    task = _make_task(source_lang="pl")
    task.task_id = "source_0"
    task._source_id = "source"
    captured: list[tuple[str, str, int]] = []

    with (
        patch("nemo_curator.backends.base.is_resumability_actor_active", return_value=True),
        patch("nemo_curator.backends.base.flush_resumability_deltas", side_effect=captured.extend),
    ):
        results = BaseStageAdapter(stage).process_batch([task])

    assert results == [task]
    assert task.task_id == "source_0_0"
    assert task._source_id == "source"
    assert captured == [("source_0_0", "source", 0)]
    stage._adapter.transcribe_batch.assert_not_called()


def test_inputs_and_exact_output_contract() -> None:
    stage = ASRStage(
        adapter_target=_QWEN_ADAPTER_TARGET,
        model_id="mock/model",
        pred_text_key="custom_prediction",
    )
    _required, required_inputs = stage.inputs()
    assert required_inputs == ["resampled_audio_filepath"]

    _required, optional_outputs = stage.outputs()
    assert optional_outputs == ["custom_prediction", "_skipme", "additional_notes"]


def test_adapter_extras_are_copied_to_one_nested_manifest_field() -> None:
    stage = _make_stage(extras_key="asr_extras")
    adapter_extras = {
        "detected_language": "English",
        "confidence": 0.98,
        "segments": [{"start": 0.0, "end": 1.0}],
    }
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="hello", extras=adapter_extras)]

    result = stage.process_batch([_make_task()])[0]

    assert result.data["asr_extras"] == adapter_extras
    assert result.data["asr_extras"] is not adapter_extras


def test_empty_adapter_extras_remove_stale_manifest_metadata() -> None:
    stage = _make_stage(extras_key="asr_extras")
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="hello")]
    task = _make_task()
    task.data["asr_extras"] = {"stale": True}

    result = stage.process_batch([task])[0]

    assert "asr_extras" not in result.data


def test_adapter_extras_output_can_be_disabled() -> None:
    stage = _make_stage(extras_key=None)
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="hello", extras={"detected_language": "English"})]

    result = stage.process_batch([_make_task()])[0]

    assert "asr_extras" not in result.data
    assert stage.outputs() == ([], ["pred_text", "_skipme", "additional_notes"])


def test_in_memory_input_contract_requires_waveform_and_sample_rate() -> None:
    stage = ASRStage(
        adapter_target=_QWEN_ADAPTER_TARGET,
        model_id="mock/model",
        waveform_key="waveform",
        sample_rate_key="sampling_rate",
    )

    _required, required_inputs = stage.inputs()

    assert required_inputs == ["waveform", "sampling_rate"]


def test_stage_loads_resampled_audio_with_torchaudio_and_preserves_sample_rate(tmp_path: Path) -> None:
    decoded_sample_rate = 8000
    audio_path = tmp_path / "resampled.wav"
    sf.write(audio_path, np.ones(_SR, dtype=np.float32), decoded_sample_rate, subtype="FLOAT")

    waveform, sample_rate = ASRStage._load_audio(str(audio_path))
    assert sample_rate == decoded_sample_rate
    assert waveform.shape == (_SR,)
    assert waveform.dtype == np.float32
    np.testing.assert_array_equal(waveform, np.ones(_SR, dtype=np.float32))


def test_stage_load_audio_preserves_stereo_channel_first(tmp_path: Path) -> None:
    decoded = np.ones((_SR, 2), dtype=np.float32)
    audio_path = tmp_path / "stereo.wav"
    sf.write(audio_path, decoded, _SR, subtype="FLOAT")

    waveform, sample_rate = ASRStage._load_audio(str(audio_path))

    assert sample_rate == _SR
    assert waveform.shape == (2, _SR)
    assert waveform.flags.c_contiguous


def test_in_memory_waveform_is_normalized_once_and_removed_after_inference() -> None:
    stage = _make_stage(waveform_key="waveform")
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="hello")]
    stereo_8khz = np.ones((2, 8000), dtype=np.float64)
    task = _make_waveform_task(waveform=stereo_8khz, sample_rate=8000)

    results = stage.process_batch([task])

    assert results[0].data["pred_text"] == "hello"
    assert "waveform" not in results[0].data
    inferred_item = stage._adapter.transcribe_batch.call_args.args[0][0]
    assert inferred_item["sample_rate"] == _SR
    assert inferred_item["waveform"].shape == (_SR,)
    assert inferred_item["waveform"].dtype == np.float32
    stage._load_audio.assert_not_called()


def test_in_memory_waveform_can_be_retained_for_recovery_inference() -> None:
    stage = _make_stage(waveform_key="waveform", keep_waveform=True)
    stage._adapter.transcribe_batch.return_value = [ASRResult(text="hello")]
    waveform = np.ones(_SR, dtype=np.float32)
    task = _make_waveform_task(waveform=waveform)

    results = stage.process_batch([task])

    assert results[0].data["waveform"] is waveform


def test_invalid_target_sample_rate_is_rejected() -> None:
    with pytest.raises(ValueError, match="target_sample_rate must be > 0"):
        ASRStage(
            adapter_target=_QWEN_ADAPTER_TARGET,
            model_id="mock/model",
            target_sample_rate=0,
        )


def test_stage_requires_resampled_path_and_does_not_fallback_to_original_audio() -> None:
    stage = _make_stage()
    task = AudioTask(data={"audio_filepath": "/data/original.wav", "source_lang": "en"})

    with pytest.raises(ValueError, match="missing required columns"):
        stage.process_batch([task])

    stage._load_audio.assert_not_called()
    stage._adapter.transcribe_batch.assert_not_called()


def test_empty_prediction_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="pred_text_key must be non-empty"):
        ASRStage(
            adapter_target=_QWEN_ADAPTER_TARGET,
            model_id="mock/model",
            pred_text_key="",
        )


def test_empty_extras_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="extras_key must be non-empty or None"):
        ASRStage(
            adapter_target=_QWEN_ADAPTER_TARGET,
            model_id="mock/model",
            extras_key=" ",
        )


@pytest.mark.parametrize("extras_key", ["pred_text", "_skipme", "additional_notes"])
def test_extras_key_cannot_collide_with_another_output(extras_key: str) -> None:
    with pytest.raises(ValueError, match="extras_key cannot collide"):
        ASRStage(
            adapter_target=_QWEN_ADAPTER_TARGET,
            model_id="mock/model",
            extras_key=extras_key,
        )


@pytest.mark.parametrize("pred_text_key", ["_skipme", "additional_notes"])
def test_control_columns_cannot_be_used_as_prediction_key(pred_text_key: str) -> None:
    with pytest.raises(ValueError, match="reserved control column"):
        ASRStage(
            adapter_target=_QWEN_ADAPTER_TARGET,
            model_id="mock/model",
            pred_text_key=pred_text_key,
        )


@pytest.mark.parametrize(
    ("result", "expected_reason"),
    [
        (ASRResult(text="", skipped=True), "empty_audio"),
        (ASRResult(text="", skipped=True, skip_reason="decode_failed"), "decode_failed"),
        (ASRResult(text="", skipped=True, extras={"skip_reason": "ignored"}), "empty_audio"),
    ],
)
def test_skipped_result_sets_typed_skip_reason(result: ASRResult, expected_reason: str) -> None:
    stage = _make_stage()
    stage._adapter.transcribe_batch.return_value = [result]
    results = stage.process_batch([_make_task()])
    assert results[0].data["_skipme"] == expected_reason


@patch("nemo_curator.models.asr.qwen_omni.snapshot_download")
def test_setup_on_node_downloads_weights(mock_download: MagicMock) -> None:
    stage = ASRStage(
        adapter_target=_QWEN_ADAPTER_TARGET,
        model_id="mock/model",
        adapter_kwargs={"revision": "abc123"},
    )
    stage.setup_on_node()
    mock_download.assert_called_once_with("mock/model", revision="abc123")


@patch("nemo_curator.models.faster_whisper_asr._download_whisper_model")
def test_setup_on_node_downloads_faster_whisper_weights(mock_download: MagicMock) -> None:
    stage = ASRStage(
        adapter_target=_FASTER_WHISPER_ADAPTER_TARGET,
        model_id="large-v3",
        adapter_kwargs={"revision": "abc123"},
    )
    stage.setup_on_node()
    mock_download.assert_called_once_with("large-v3", "abc123")


@patch(
    "nemo_curator.models.asr.qwen_omni.snapshot_download",
    side_effect=RuntimeError("missing auth"),
)
def test_setup_on_node_raises_by_default(mock_download: MagicMock) -> None:
    stage = ASRStage(adapter_target=_QWEN_ADAPTER_TARGET, model_id="mock/model")
    with pytest.raises(RuntimeError, match="download_weights_on_node failed"):
        stage.setup_on_node()
    mock_download.assert_called_once_with("mock/model")


@patch(
    "nemo_curator.models.asr.qwen_omni.snapshot_download",
    side_effect=RuntimeError("offline"),
)
def test_setup_on_node_can_warn_and_retry_later(mock_download: MagicMock) -> None:
    stage = ASRStage(
        adapter_target=_QWEN_ADAPTER_TARGET,
        model_id="mock/model",
        prefetch_fail_on_error=False,
    )
    stage.setup_on_node()
    mock_download.assert_called_once_with("mock/model")


def test_adapter_target_required() -> None:
    with pytest.raises(TypeError):
        ASRStage(model_id="mock/model")


def test_model_id_required() -> None:
    with pytest.raises(TypeError):
        ASRStage(adapter_target=_QWEN_ADAPTER_TARGET)


def test_stage_rejects_model_specific_revision_field() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument 'revision'"):
        ASRStage(
            adapter_target=_QWEN_ADAPTER_TARGET,
            model_id="mock/model",
            revision="abc123",  # type: ignore[call-arg]
        )


def test_setup_uses_adapter_target_and_kwargs() -> None:
    """``setup()`` resolves adapter_target via hydra.utils.get_class and
    constructs the adapter with model_id plus its explicit adapter_kwargs."""
    stage = ASRStage(
        adapter_target=_QWEN_ADAPTER_TARGET,
        model_id="mock/model",
        adapter_kwargs={
            "revision": "abc123",
            "vllm_kwargs": {
                "max_model_len": 8192,
                "enable_prefix_caching": False,
            },
        },
        resources=Resources(gpus=2),
    )

    fake_adapter = MagicMock()
    fake_cls = MagicMock(return_value=fake_adapter)
    with patch("hydra.utils.get_class", return_value=fake_cls) as get_class:
        stage.setup()

    get_class.assert_called_with(_QWEN_ADAPTER_TARGET)
    fake_cls.assert_called_once_with(
        model_id="mock/model",
        revision="abc123",
        vllm_kwargs={
            "max_model_len": 8192,
            "enable_prefix_caching": False,
        },
    )
    fake_adapter.load_model.assert_called_once_with(num_gpus=2)
    assert stage._adapter is fake_adapter


@pytest.mark.parametrize(
    ("requested_gpus", "expected_num_gpus"),
    [(0.0, 0), (0.25, 1), (1.0, 1), (1.5, 2), (2.0, 2)],
)
def test_setup_derives_adapter_gpu_count_from_stage_resources(
    requested_gpus: float,
    expected_num_gpus: int,
) -> None:
    stage = ASRStage(
        adapter_target="tests.fake.Adapter",
        model_id="mock/model",
        resources=Resources(gpus=requested_gpus),
    )
    fake_adapter = MagicMock()

    with patch("hydra.utils.get_class", return_value=MagicMock(return_value=fake_adapter)):
        stage.setup()

    fake_adapter.load_model.assert_called_once_with(num_gpus=expected_num_gpus)


@pytest.mark.parametrize("requested_gpus", [-1.0, float("inf"), float("nan")])
def test_setup_rejects_invalid_stage_gpu_resource(requested_gpus: float) -> None:
    stage = ASRStage(
        adapter_target="tests.fake.Adapter",
        model_id="mock/model",
        resources=Resources(gpus=requested_gpus),
    )
    fake_adapter = MagicMock()

    with (
        patch("hydra.utils.get_class", return_value=MagicMock(return_value=fake_adapter)),
        pytest.raises(ValueError, match=r"resources\.gpus must be a finite non-negative value"),
    ):
        stage.setup()

    fake_adapter.unload_model.assert_called_once_with()


def test_setup_failure_cleans_partial_adapter_and_allows_retry() -> None:
    stage = ASRStage(adapter_target=_QWEN_ADAPTER_TARGET, model_id="mock/model")
    failed_adapter = MagicMock()
    failed_adapter.load_model.side_effect = RuntimeError("engine init failed")
    working_adapter = MagicMock()
    fake_cls = MagicMock(side_effect=[failed_adapter, working_adapter])

    with patch("hydra.utils.get_class", return_value=fake_cls):
        with pytest.raises(RuntimeError, match="engine init failed"):
            stage.setup()

        assert stage._adapter is None
        failed_adapter.unload_model.assert_called_once_with()

        stage.setup()

    assert stage._adapter is working_adapter
    working_adapter.load_model.assert_called_once_with(num_gpus=1)


def test_teardown_delegates_to_adapter_unload_model_once() -> None:
    stage = _make_stage()
    adapter = stage._adapter

    stage.teardown()
    stage.teardown()

    adapter.unload_model.assert_called_once_with()
    assert stage._adapter is None
