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

"""CPU-only contract tests for the Parakeet compatibility stage."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from nemo_curator.models.asr.base import ASRResult
from nemo_curator.models.asr.nemo_asr import NeMoASRAdapter
from nemo_curator.stages.audio.inference.parakeet import (
    PARAKEET_TDT_0_6B_V3_LANGS,
    InferenceParakeetStage,
)
from nemo_curator.tasks import AudioTask

_SAMPLE_RATE = 16_000


def _task(language: str) -> AudioTask:
    return AudioTask(
        data={
            "waveform": np.zeros(800, dtype=np.float32),
            "sampling_rate": _SAMPLE_RATE,
            "source_lang": language,
        }
    )


def test_stage_exposes_parakeet_v3_defaults() -> None:
    stage = InferenceParakeetStage(num_workers_override=2)

    assert stage.inputs() == ([], ["waveform", "sampling_rate"])
    assert stage.outputs() == ([], ["asr_prediction", "asr_language"])
    assert stage.model_id == "nvidia/parakeet-tdt-0.6b-v3"
    assert stage.batch_size == 128
    assert set(stage.supported_language_codes) == set(PARAKEET_TDT_0_6B_V3_LANGS)
    assert stage.num_workers() == 2


def test_stage_builds_nemo_adapter_with_model_options() -> None:
    stage = InferenceParakeetStage(
        model_id="/models/indic-parakeet.nemo",
        supported_langs={"hi", "ta"},
        batch_size=7,
    )

    adapter = stage._create_adapter()

    assert isinstance(adapter, NeMoASRAdapter)
    assert adapter.model_id == "/models/indic-parakeet.nemo"
    assert adapter.empty_audio_marks_skip is False
    assert adapter.use_cuda_graph_decoder is False
    assert stage.batch_size == 7
    assert stage.supported_language_codes == ["hi", "ta"]


def test_language_filter_only_sends_supported_rows_to_adapter() -> None:
    stage = InferenceParakeetStage(supported_langs={"en"}, keep_waveform=True)
    adapter = MagicMock()
    adapter.transcribe_batch.return_value = [ASRResult(text="hello", extras={"language_code": "en"})]
    stage._adapter = adapter

    results = stage.process_batch([_task("hi"), _task("en")])

    assert [task.data["asr_prediction"] for task in results] == ["", "hello"]
    assert [task.data["asr_language"] for task in results] == ["", "en"]
    assert "_skipme" not in results[0].data
    assert results[0].data["additional_notes"]["asr_prediction"] == "lang_not_supported:hi"
    inferred = adapter.transcribe_batch.call_args.args[0]
    assert len(inferred) == 1
    assert inferred[0]["language_code"] == "en"
    assert "waveform" in results[0].data


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"backend": "tensorrt"}, "backend='nemo' only"),
        ({"tensorrt_engine_dir": "/engine"}, "only valid with backend='tensorrt'"),
        ({"chunking_mode": "invalid"}, "Unsupported Parakeet chunking mode"),
    ],
)
def test_stage_rejects_unsupported_runtime_configuration(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        InferenceParakeetStage(**kwargs)  # type: ignore[arg-type]
