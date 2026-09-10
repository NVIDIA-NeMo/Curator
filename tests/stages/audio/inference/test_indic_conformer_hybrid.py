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

"""CPU-only contract tests for the IndicConformer compatibility stage."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from nemo_curator.models.asr.base import ASRResult
from nemo_curator.models.indic_conformer_hybrid import IndicConformerHybridASR
from nemo_curator.stages.audio.inference.indic_conformer_hybrid import (
    INDIC_CONFORMER_HYBRID_LANGS,
    InferenceIndicConformerHybridStage,
)
from nemo_curator.tasks import AudioTask

_SAMPLE_RATE = 16_000


def _task(language: str | None = "hi") -> AudioTask:
    data: dict[str, object] = {
        "waveform": np.zeros(800, dtype=np.float32),
        "sampling_rate": _SAMPLE_RATE,
    }
    if language is not None:
        data["source_lang"] = language
    return AudioTask(data=data)


def test_stage_exposes_integration_pipeline_contract() -> None:
    stage = InferenceIndicConformerHybridStage(
        model_id="ai4bharat/test-checkpoint",
        num_workers_override=3,
    )

    assert stage.inputs() == ([], ["waveform", "sampling_rate"])
    assert stage.outputs() == ([], ["asr_prediction", "asr_language"])
    assert stage.name == "IndicConformerHybrid_inference"
    assert stage.batch_size == 128
    assert stage.num_workers() == 3
    assert set(stage.supported_language_codes) == set(INDIC_CONFORMER_HYBRID_LANGS)


def test_stage_builds_adapter_with_all_model_options() -> None:
    stage = InferenceIndicConformerHybridStage(
        model_id="ai4bharat/test-checkpoint",
        decode_mode="ctc",
        rnnt_precision="bf16",
        batch_size=5,
    )

    adapter = stage._create_adapter()

    assert isinstance(adapter, IndicConformerHybridASR)
    assert adapter.model_id == "ai4bharat/test-checkpoint"
    assert adapter.decode_mode == "ctc"
    assert adapter.rnnt_precision == "bf16"
    assert adapter.empty_audio_marks_skip is False
    assert stage.batch_size == 5


def test_supported_batch_preserves_order_and_writes_language() -> None:
    stage = InferenceIndicConformerHybridStage(model_id="ai4bharat/test-checkpoint")
    adapter = MagicMock()
    adapter.transcribe_batch.return_value = [
        ASRResult(text="namaste", extras={"language_code": "hi"}),
        ASRResult(text="vanakkam", extras={"language_code": "ta"}),
    ]
    stage._adapter = adapter
    tasks = [_task("hi"), _task("ta")]

    results = stage.process_batch(tasks)

    assert [task.data["asr_prediction"] for task in results] == ["namaste", "vanakkam"]
    assert [task.data["asr_language"] for task in results] == ["hi", "ta"]
    assert all("waveform" not in task.data for task in results)
    inferred = adapter.transcribe_batch.call_args.args[0]
    assert [item["language_code"] for item in inferred] == ["hi", "ta"]
    assert all(item["sample_rate"] == _SAMPLE_RATE for item in inferred)


def test_unsupported_language_is_not_marked_for_global_skip() -> None:
    stage = InferenceIndicConformerHybridStage(model_id="ai4bharat/test-checkpoint")
    stage._adapter = MagicMock()

    result = stage.process_batch([_task("zz")])[0]

    assert result.data["asr_prediction"] == ""
    assert result.data["asr_language"] == ""
    assert "_skipme" not in result.data
    assert result.data["additional_notes"] == {
        stage.name: "skipped (unsupported language: zz)",
        "asr_prediction": "lang_not_supported:zz",
    }
    stage._adapter.transcribe_batch.assert_not_called()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"backend": "tensorrt"}, "backend='nemo' only"),
        ({"tensorrt_engine_dir": "/engine"}, "only valid with backend='tensorrt'"),
    ],
)
def test_stage_rejects_unsupported_runtime_configuration(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        InferenceIndicConformerHybridStage(**kwargs)  # type: ignore[arg-type]
