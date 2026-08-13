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

import importlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from pytest import MonkeyPatch

pytest.importorskip("nemo.collections.asr")


@pytest.fixture(scope="module")
def benchmark_module():
    scripts_dir = Path(__file__).parents[2] / "benchmarking" / "scripts"
    import sys

    sys.path.insert(0, str(scripts_dir))
    try:
        yield importlib.import_module("audio_sortformer_benchmark")
    finally:
        sys.path.remove(str(scripts_dir))
        sys.modules.pop("audio_sortformer_benchmark", None)


def _manifest_rows(benchmark_module: ModuleType) -> list[dict]:
    return [
        {
            "audio_filepath": f"audio/{filename}",
            "audio_item_id": filename.removesuffix(".wav"),
            "duration": 2000.0,
        }
        for filename in benchmark_module.EXPECTED_AUDIO_FILENAMES
    ]


def _stage_perf(benchmark_module: ModuleType) -> list[SimpleNamespace]:
    return [SimpleNamespace(stage_name=benchmark_module.SORTFORMER_STAGE_NAME, num_items_processed=1)]


def test_write_staged_manifest_uses_local_audio_paths(tmp_path: Path, benchmark_module: ModuleType) -> None:
    data_dir = tmp_path / "input"
    audio_dir = data_dir / "audio"
    audio_dir.mkdir(parents=True)
    rows = _manifest_rows(benchmark_module)
    for row in rows:
        (audio_dir / Path(row["audio_filepath"]).name).touch()
    source_manifest = data_dir / "manifest.jsonl"
    source_manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))

    located_manifest, located_audio = benchmark_module._locate_prestaged_data(data_dir)
    target_manifest = tmp_path / "scratch" / "manifest.jsonl"
    num_rows = benchmark_module._write_staged_manifest(located_manifest, target_manifest, located_audio)
    rewritten = [json.loads(line) for line in target_manifest.read_text().splitlines()]

    assert num_rows == 34
    assert all(Path(row["audio_filepath"]).parent == audio_dir.resolve() for row in rewritten)


def test_validate_outputs_matches_tagging_methodology(benchmark_module: ModuleType) -> None:
    tasks = [
        SimpleNamespace(
            data={
                "duration": 10.0,
                "diar_segments": [{"start": 0.0, "end": 10.02, "speaker": "speaker_0"}],
            },
            _stage_perf=_stage_perf(benchmark_module),
        ),
        SimpleNamespace(
            data={
                "duration": 10.0,
                "diar_segments": [{"start": 1.0, "end": 2.0, "speaker": "speaker_1"}],
            },
            _stage_perf=_stage_perf(benchmark_module),
        ),
    ]

    metrics = benchmark_module._validate_outputs(tasks, num_input_rows=2)

    assert metrics["input_output_row_count_match"] is True
    assert metrics["num_tasks_with_segments"] == 2
    assert metrics["num_segments_processed"] == 2
    assert metrics["stage_execution_coverage_ratio"] == 1.0


@pytest.mark.parametrize(
    ("tasks", "match"),
    [
        ([], "returned 0 rows"),
        (
            [SimpleNamespace(data={"duration": 10.0, "diar_segments": []}, _stage_perf=[])],
            "no diarization segments",
        ),
        (
            [
                SimpleNamespace(
                    data={
                        "duration": 10.0,
                        "diar_segments": [{"start": 2.0, "end": 1.0, "speaker": "speaker_0"}],
                    },
                    _stage_perf=[],
                )
            ],
            "invalid timestamps",
        ),
    ],
)
def test_validate_outputs_rejects_incomplete_or_malformed_output(
    tasks: list[SimpleNamespace], match: str, benchmark_module: ModuleType
) -> None:
    with pytest.raises(RuntimeError, match=match):
        benchmark_module._validate_outputs(tasks, num_input_rows=1)


def test_run_uses_eight_one_gpu_workers(
    tmp_path: Path, monkeypatch: MonkeyPatch, benchmark_module: ModuleType
) -> None:
    data_dir = tmp_path / "input"
    audio_dir = data_dir / "audio"
    audio_dir.mkdir(parents=True)
    rows = _manifest_rows(benchmark_module)
    for row in rows:
        (audio_dir / Path(row["audio_filepath"]).name).touch()
    (data_dir / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    model_path = tmp_path / "model.nemo"
    model_path.touch()

    output_tasks = [
        SimpleNamespace(
            data={
                **row,
                "diar_segments": [{"start": 0.0, "end": 1.0, "speaker": "speaker_0"}],
            },
            _stage_perf=_stage_perf(benchmark_module),
        )
        for row in rows
    ]
    captured: dict = {}

    def fake_run(pipeline: object, _executor: object) -> list[SimpleNamespace]:
        captured["pipeline"] = pipeline
        return output_tasks

    monkeypatch.setattr(benchmark_module, "setup_executor", lambda _name: object())
    monkeypatch.setattr(benchmark_module.Pipeline, "run", fake_run)

    result = benchmark_module.run_audio_sortformer_benchmark(
        benchmark_results_path=str(tmp_path / "results"),
        scratch_output_path=str(tmp_path / "scratch"),
        raw_data_dir=str(data_dir),
        model_path=str(model_path),
        gpu_stage_num_workers=8,
    )

    _, inference = captured["pipeline"].stages
    assert inference.resources.gpus == 1
    assert inference.num_workers() == 8
    assert inference.chunk_len == 6
    assert inference.chunk_right_context == 7
    assert result["metrics"]["num_input_rows"] == 34
    assert result["metrics"]["num_output_rows"] == 34
