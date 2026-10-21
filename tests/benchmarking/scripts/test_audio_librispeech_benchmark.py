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

"""LibriSpeech pipeline construction without starting an executor or ASR model."""

import inspect
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarking" / "scripts"))

import audio_librispeech_benchmark as benchmark


def test_asr_constructor_arguments_survive_stage_metaclass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert "max_audio_sec_per_actor" not in inspect.signature(benchmark.ASRStage).parameters
    assert "max_audio_sec_per_actor" in inspect.signature(benchmark.ASRStage.__init__).parameters
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"audio_filepath": "/audio.wav"}) + "\n")
    results_dir = tmp_path / "benchmark"
    stages = []

    def run_pipeline(pipeline: benchmark.Pipeline, _executor: object) -> None:
        stages.extend(pipeline.stages)
        output = results_dir / "scratch" / "librispeech_writer_output"
        output.mkdir(parents=True, exist_ok=True)
        (output / "part.jsonl").write_text(
            json.dumps({"audio_filepath": "/audio.wav", "duration": 1.0, "wer_pct": 0.0}) + "\n"
        )

    monkeypatch.setattr(benchmark, "setup_executor", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(benchmark.Pipeline, "run", run_pipeline)

    result = benchmark.run_audio_librispeech_benchmark(str(results_dir), str(manifest), "test-model", 1000.0)

    assert result["metrics"]["is_success"]
    assert result["metrics"]["num_tasks_processed"] == 1
    asr = next(stage for stage in stages if isinstance(stage, benchmark.ASRStage))
    assert asr.max_audio_sec_per_actor == 240.0
    assert asr.max_inference_duration_s == 120.0
    assert asr.batch_size == 16
    assert asr.local_bucketing
    assert asr.adapter_kwargs == {"use_cuda_graph_decoder": False}
