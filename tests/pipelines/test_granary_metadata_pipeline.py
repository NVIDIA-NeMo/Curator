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

"""Exercise the tutorial graph with real audio I/O and mocked model results."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf
from omegaconf import OmegaConf

from nemo_curator.backends.base import BaseStageAdapter
from nemo_curator.config.run import create_pipeline_from_yaml
from nemo_curator.models.audio.lid.base import AudioLIDResult
from nemo_curator.models.audio.sed.base import SEDResult
from nemo_curator.models.audio.speaker_diarization.base import DiarizationResult
from nemo_curator.models.audio.vad.base import VADResult, VADSegment
from nemo_curator.stages.audio.inference.base import AdapterInferenceStage
from nemo_curator.stages.audio.inference.lid.stage import AudioLIDInferenceStage
from nemo_curator.stages.audio.inference.sed.stage import SEDInferenceStage
from nemo_curator.stages.audio.inference.speaker_diarization.stage import InferenceSortformerStage
from nemo_curator.stages.audio.segmentation.vad_segmentation import VADSegmentationStage
from nemo_curator.tasks import EmptyTask


def test_full_graph_preserves_fanout_silence_and_language_rejection(  # noqa: C901, PLR0915
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    sample_rate = 24000
    samples = np.arange(5 * sample_rate)
    signal = (0.1 * np.sin(2 * np.pi * 440 * samples / sample_rate)).astype(np.float32)
    speech_path, silence_path = tmp_path / "speech.wav", tmp_path / "silence.wav"
    sf.write(speech_path, np.column_stack((signal, signal)), sample_rate)
    sf.write(silence_path, np.zeros(2 * sample_rate), sample_rate)
    manifest = tmp_path / "input.jsonl"
    manifest.write_text(
        "".join(json.dumps({"audio_filepath": str(path)}) + "\n" for path in (speech_path, silence_path))
    )

    config_path = Path(__file__).resolve().parents[2] / "tutorials/audio/granary_v2/metadata/pipeline.yaml"
    cfg = OmegaConf.load(config_path)
    cfg.manifest_path = str(manifest)
    cfg.output_dir = str(tmp_path / "result")
    cfg.model_dir = str(tmp_path / "models")
    monkeypatch.setattr("nemo_curator.stages.resources._get_gpu_memory_gb", lambda: 80.0)
    pipeline = create_pipeline_from_yaml(cfg)
    pipeline.build()
    model_calls = []

    def diarize(items: list[dict]) -> list[DiarizationResult]:
        return [
            DiarizationResult(
                [
                    {"start": 0.0, "end": 3.0, "speaker": "speaker_0"},
                    {"start": 3.0, "end": 5.0, "speaker": "speaker_1"},
                ]
            )
            for _ in items
        ]

    def detect(items: list[dict]) -> list[VADResult]:
        return [
            VADResult([VADSegment(0.2, 2.2), VADSegment(3.0, 4.5)] if np.any(item["waveform"]) else [])
            for item in items
        ]

    def sed(items: list[dict]) -> list[SEDResult]:
        results = []
        for item in items:
            frames = int(len(item["waveform"]) / 16000 * 50)
            probabilities = np.zeros((frames, 527), dtype=np.float32)
            probabilities[:, 0] = 0.9
            results.append(SEDResult(probabilities, 50.0, frames, len(item["waveform"])))
        return results

    def identify(items: list[dict], *, whisper: bool) -> list[AudioLIDResult]:
        return [AudioLIDResult("ta" if whisper and len(item["waveform"]) == 24000 else "hi", 0.9) for item in items]

    tasks = [EmptyTask()]
    for stage in pipeline.stages:
        if isinstance(stage, AdapterInferenceStage):
            stage._create_adapter()
        if isinstance(stage, InferenceSortformerStage):
            stage._adapter = SimpleNamespace(diarize_batch=Mock(side_effect=diarize))
        elif isinstance(stage, VADSegmentationStage):
            stage._adapter = SimpleNamespace(detect_batch=Mock(side_effect=detect))
        elif isinstance(stage, SEDInferenceStage):
            call = Mock(side_effect=sed)
            model_calls.append(call)
            stage._adapter = SimpleNamespace(infer_batch=call)
        elif isinstance(stage, AudioLIDInferenceStage):
            call = Mock(
                side_effect=lambda items, is_whisper=stage.model_id == "WhisperLangID": identify(
                    items, whisper=is_whisper
                )
            )
            model_calls.append(call)
            stage._adapter = SimpleNamespace(identify_batch=call)
        else:
            stage.setup_on_node()
            stage.setup()
        adapter = BaseStageAdapter(stage)
        tasks = [
            result
            for start in range(0, len(tasks), stage.batch_size)
            for result in adapter.process_batch(tasks[start : start + stage.batch_size])
        ]
        if not isinstance(stage, AdapterInferenceStage):
            stage.teardown()

    rows = [json.loads(line) for line in (Path(cfg.output_dir) / "metadata.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    segments = [row for row in rows if not row.get("vad_empty")]
    assert [row["original_file"] for row in segments] == [str(speech_path)] * 2
    assert [(row["start_ms"], row["end_ms"]) for row in segments] == [(200, 2200), (3000, 4500)]
    assert [row["source_lang"] for row in segments] == ["hi", "skipped"]
    assert [row["num_speakers"] for row in segments] == [1, 1]
    for row in segments:
        assert row["sed_events"]
        assert row["sample_rate"] == 16000
        assert sf.info(row["audio_filepath"]).duration == pytest.approx(row["duration"], abs=0.002)
        assert (Path(cfg.output_dir) / "rttm" / row["rttm_filepath"]).is_file()
        assert "diar_segments" not in row
        assert "resampled_audio_filepath" not in row
    empty = rows[-1]
    assert empty["vad_empty"] is True
    assert empty["_skipme"] == "vad_empty"
    assert empty["audio_filepath"] == str(silence_path)
    for call in model_calls:
        assert sum(len(args.args[0]) for args in call.call_args_list) == 2
