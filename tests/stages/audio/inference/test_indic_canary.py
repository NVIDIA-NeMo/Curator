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

"""CPU-only contract tests for the Indic Canary compatibility stage."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from nemo_curator.models.asr.base import ASRResult
from nemo_curator.models.asr.indic_canary import IndicCanaryTRTLLMASR
from nemo_curator.stages.audio.inference.indic_canary import InferenceIndicCanaryStage
from nemo_curator.tasks import AudioTask

_SAMPLE_RATE = 16_000
_REQUIRED_ENGINE_FILES = (
    "encoder/encoder.plan",
    "encoder/config.json",
    "decoder/config.json",
    "decoder/rank0.engine",
    "decoder/vocab.json",
    "preprocessor/config.json",
    "preprocessor/mel_basis.pt",
)


def _task(language: str, *, samples: int = 800, existing_skip: str | None = None) -> AudioTask:
    data: dict[str, object] = {
        "waveform": np.zeros(samples, dtype=np.float32),
        "sampling_rate": _SAMPLE_RATE,
        "source_lang": language,
    }
    if existing_skip is not None:
        data["_skipme"] = existing_skip
    return AudioTask(data=data)


def _write_engine_files(engine_dir: Path) -> None:
    for relative_path in _REQUIRED_ENGINE_FILES:
        path = engine_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")


def test_stage_exposes_integration_pipeline_contract() -> None:
    stage = InferenceIndicCanaryStage(engine_dir="/models/indic-canary", num_workers_override=4)

    assert stage.inputs() == ([], ["waveform", "sampling_rate"])
    assert stage.outputs() == ([], ["asr_prediction", "asr_language"])
    assert stage.model_id == "/models/indic-canary"
    assert stage.name == "IndicCanary_inference"
    assert stage.num_workers() == 4


def test_stage_builds_adapter_with_engine_options() -> None:
    stage = InferenceIndicCanaryStage(
        engine_dir="/models/indic-canary",
        num_beams=2,
        max_new_tokens=99,
        pnc=True,
        max_duration_sec=30.0,
        min_duration_sec=1.0,
        kv_cache_free_gpu_memory_fraction=0.15,
        cross_kv_cache_fraction=0.25,
    )

    adapter = stage._create_adapter()

    assert isinstance(adapter, IndicCanaryTRTLLMASR)
    assert adapter.engine_dir == "/models/indic-canary"
    assert adapter.num_beams == 2
    assert adapter.max_new_tokens == 99
    assert adapter.pnc is True
    assert adapter.max_duration_sec == 30.0
    assert adapter.min_duration_sec == 1.0
    assert adapter.kv_cache_free_gpu_memory_fraction == 0.15
    assert adapter.cross_kv_cache_fraction == 0.25


def test_stage_writes_canary_unsupported_language_contract() -> None:
    stage = InferenceIndicCanaryStage(engine_dir="/models/indic-canary")
    adapter = MagicMock()
    adapter.transcribe_batch.return_value = [
        ASRResult(text="namaste", extras={"language_code": "hi", "audio_duration_sec": 0.05}),
        ASRResult(
            text="",
            skipped=True,
            skip_reason="language_not_supported",
            unsupported_language="zz",
            extras={"language_code": "zz", "language_unsupported": True},
        ),
    ]
    stage._adapter = adapter

    supported, unsupported = stage.process_batch([_task("hi"), _task("zz")])

    assert supported.data["asr_prediction"] == "namaste"
    assert supported.data["asr_language"] == "hi"
    assert unsupported.data["asr_prediction"] == ""
    assert unsupported.data["asr_language"] == ""
    assert unsupported.data["_skipme"] == f"lang_not_supported:{stage.name}"
    assert unsupported.data["additional_notes"] == {
        stage.name: "skipped (unsupported language: zz)",
        "asr_prediction": "lang_not_supported:zz",
    }


def test_stage_preserves_existing_skip_for_unsupported_language() -> None:
    stage = InferenceIndicCanaryStage(engine_dir="/models/indic-canary", keep_waveform=True)
    adapter = MagicMock()
    adapter.transcribe_batch.return_value = [
        ASRResult(
            text="",
            skipped=True,
            skip_reason="language_not_supported",
            unsupported_language="zz",
            extras={"language_code": "zz", "language_unsupported": True},
        )
    ]
    stage._adapter = adapter

    result = stage.process_batch([_task("zz", existing_skip="upstream_reason")])[0]

    assert result.data["_skipme"] == "upstream_reason"
    assert "waveform" in result.data


def test_stage_adds_long_audio_truncation_note() -> None:
    stage = InferenceIndicCanaryStage(engine_dir="/models/indic-canary", max_duration_sec=40.0)
    adapter = MagicMock()
    adapter.transcribe_batch.return_value = [
        ASRResult(
            text="partial",
            extras={"language_code": "hi", "truncated": True, "audio_duration_sec": 45.0},
        )
    ]
    stage._adapter = adapter

    result = stage.process_batch([_task("hi")])[0]

    assert "45.00s exceeds 40s encoder window" in result.data["additional_notes"][stage.name]


def test_setup_on_node_only_validates_engine_artifacts(tmp_path: Path) -> None:
    _write_engine_files(tmp_path)
    stage = InferenceIndicCanaryStage(engine_dir=str(tmp_path))

    stage.setup_on_node()

    assert stage._adapter is None


def test_missing_engine_directory_is_rejected_during_prefetch() -> None:
    stage = InferenceIndicCanaryStage(engine_dir="")

    with pytest.raises(RuntimeError, match="download_weights_on_node failed") as exc_info:
        stage.setup_on_node()

    assert isinstance(exc_info.value.__cause__, ValueError)
