# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Exercise the bundled FastMSS, child processes and manifest I/O."""

import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio._agent._conformance import assert_agent_ready
from nemo_curator.stages.audio.common import ManifestReaderStage
from nemo_curator.stages.audio.simulation import FastMSS, FastMSSSimulationStage
from nemo_curator.tasks import EmptyTask, FileGroupTask

_ENTRY_POINT = """
import json
import os
import sys
import time
import wave
from pathlib import Path

from hydra import compose, initialize_config_dir

args = dict(arg.split("=", 1) for arg in sys.argv[1:])
name = args["--config-name"]
with initialize_config_dir(config_dir=args["--config-path"], version_base=None):
    cfg = compose(config_name=name, overrides=[arg for arg in sys.argv[1:] if not arg.startswith("--")])
assert cfg.n_jobs >= 1
assert cfg.save_rttm and cfg.save_nemo_manifest
output = Path(json.loads(args["output_dir"]))
assert str(output) == cfg.output_dir
(output / "arguments.json").write_text(json.dumps(args))
assert os.environ["OMP_NUM_THREADS"] == "1"
assert os.environ["MKL_NUM_THREADS"] == "1"
assert str(Path.cwd()) in os.environ["PYTHONPATH"].split(os.pathsep)
if name == "failure":
    print("intentional child error", flush=True)
    sys.exit(9)
if name == "timeout":
    time.sleep(30)
if name == "missing":
    sys.exit(0)
if name == "empty":
    (output / "nemo_manifest.json").touch()
    sys.exit(0)
audio = output / "meeting.wav"
with wave.open(str(audio), "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(16000)
    wav.writeframes(b"\\x00\\x00" * 1600)
rttm = output / "meeting.rttm"
rttm.write_text("SPEAKER meeting 1 0.000 0.100 <NA> <NA> speaker-1 <NA> <NA>\\n")
row = {"audio_filepath": str(audio), "duration": 0.1, "text": "hello", "rttm_filepath": str(rttm), "num_speakers": 1, "custom_provenance": name}
(output / "nemo_manifest.json").write_text(json.dumps(row) + "\\n")
"""


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    root = tmp_path / "FastMSS checkout"
    script = root / "recipes" / "sim.py"
    script.parent.mkdir(parents=True)
    script.write_text(_ENTRY_POINT, encoding="utf-8")
    return root


def _config(tmp_path: Path, name: str = "meetings") -> Path:
    config = tmp_path / "configs with spaces" / f"{name}.yaml"
    config.parent.mkdir(exist_ok=True)
    config.write_text("n_meetings: 1\noutput_dir: null\nn_jobs: 1\nstage: 0\nduration: 1\n", encoding="utf-8")
    return config


def test_subprocess_to_audio_tasks_preserves_annotations_and_lineage(checkout: Path, tmp_path: Path) -> None:
    config = _config(tmp_path)
    task = FileGroupTask(dataset_name="simulation", data=[str(config)], _metadata={"source": "reference"})
    stage = FastMSSSimulationStage(
        fastmss_root=str(checkout),
        output_dir=str(tmp_path / "generated meetings"),
        jobs_per_config=2,
        overrides=["duration=0.1"],
    )
    groups = stage.process(task)
    rows = ManifestReaderStage().process(groups[0])
    assert len(rows) == 1
    row = rows[0]
    assert row.dataset_name == "simulation"
    assert row._metadata == task._metadata
    assert groups[0]._stage_perf == task._stage_perf
    assert groups[0]._stage_perf is not task._stage_perf
    assert row.data["text"] == "hello"
    assert row.data["custom_provenance"] == "meetings"
    assert Path(row.data["audio_filepath"]).is_file()
    assert Path(row.data["rttm_filepath"]).read_text().startswith("SPEAKER meeting")
    args = json.loads((Path(groups[0].data[0]).parent / "arguments.json").read_text())
    assert args["--config-path"] == str(config.parent)
    assert args["n_jobs"] == "2"
    assert args["duration"] == "0.1"
    assert args["++save_rttm"] == "true"
    assert args["++save_nemo_manifest"] == "true"
    assert stage.resources.cpus == 2


def test_same_named_configs_and_retries_use_separate_output_directories(checkout: Path, tmp_path: Path) -> None:
    first = _config(tmp_path)
    other = tmp_path / "other" / "meetings.yaml"
    other.parent.mkdir()
    other.write_text(first.read_text())
    stage = FastMSSSimulationStage(fastmss_root=str(checkout), output_dir=str(tmp_path / "output"))
    groups = stage.process(FileGroupTask(dataset_name="simulation", data=[str(first), str(other)]))
    repeated = stage.process(FileGroupTask(dataset_name="simulation", data=[str(first)]))
    paths = [group.data[0] for group in [*groups, *repeated]]
    assert len(set(paths)) == 3
    assert all(Path(path).is_file() for path in paths)


def test_custom_entry_point(checkout: Path, tmp_path: Path) -> None:
    (checkout / "sim_im_masr.py").write_text(_ENTRY_POINT)
    stage = FastMSSSimulationStage(
        fastmss_root=str(checkout), output_dir=str(tmp_path / "output"), script_path="sim_im_masr.py"
    )
    groups = stage.process(FileGroupTask(dataset_name="simulation", data=[str(_config(tmp_path))]))
    assert Path(groups[0].data[0]).is_file()


def test_child_failure_keeps_diagnostic_log_and_emits_no_manifest(checkout: Path, tmp_path: Path) -> None:
    stage = FastMSSSimulationStage(fastmss_root=str(checkout), output_dir=str(tmp_path / "output"))
    with pytest.raises(RuntimeError, match="exit code 9"):
        stage.process(FileGroupTask(dataset_name="simulation", data=[str(_config(tmp_path, "failure"))]))
    logs = list((tmp_path / "output").glob("*/fastmss.log"))
    assert len(logs) == 1
    assert "intentional child error" in logs[0].read_text()
    assert not list((tmp_path / "output").glob("*/nemo_manifest.json"))


@pytest.mark.parametrize("name", ["missing", "empty"])
def test_successful_exit_requires_nonempty_manifest(checkout: Path, tmp_path: Path, name: str) -> None:
    stage = FastMSSSimulationStage(fastmss_root=str(checkout), output_dir=str(tmp_path / "output"))
    with pytest.raises(RuntimeError, match="nonempty recording manifest"):
        stage.process(FileGroupTask(dataset_name="simulation", data=[str(_config(tmp_path, name))]))


def test_timeout_stops_child(checkout: Path, tmp_path: Path) -> None:
    stage = FastMSSSimulationStage(
        fastmss_root=str(checkout), output_dir=str(tmp_path / "output"), timeout_seconds=0.2
    )
    with pytest.raises(subprocess.TimeoutExpired):
        stage.process(FileGroupTask(dataset_name="simulation", data=[str(_config(tmp_path, "timeout"))]))
    assert not list((tmp_path / "output").glob("*/nemo_manifest.json"))


@pytest.mark.parametrize("override", ["output_dir=/other", "+n_jobs=12", "~save_rttm", "stage=4", "--help"])
def test_rejects_overrides_that_break_output_or_resource_contract(
    checkout: Path, tmp_path: Path, override: str
) -> None:
    with pytest.raises(ValueError, match="override"):
        FastMSSSimulationStage(fastmss_root=str(checkout), output_dir=str(tmp_path / "output"), overrides=[override])


def test_rejects_missing_script_and_config(checkout: Path, tmp_path: Path) -> None:
    stage = FastMSSSimulationStage(fastmss_root=str(checkout), output_dir=str(tmp_path / "output"))
    with pytest.raises(ValueError, match="YAML config"):
        stage.process(FileGroupTask(dataset_name="simulation", data=[str(tmp_path / "missing.yaml")]))
    stage.script_path = "missing.py"
    with pytest.raises(FileNotFoundError, match="entry point"):
        stage.process(FileGroupTask(dataset_name="simulation", data=[]))


@pytest.mark.parametrize("kwargs", [{"jobs_per_config": 0}, {"timeout_seconds": 0}, {"script_path": "../sim.py"}])
def test_rejects_invalid_configuration(checkout: Path, tmp_path: Path, kwargs: dict) -> None:
    with pytest.raises(ValueError, match=r"jobs_per_config|timeout_seconds|script_path"):
        FastMSSSimulationStage(fastmss_root=str(checkout), output_dir=str(tmp_path / "output"), **kwargs)


def test_composite_discovers_yaml_and_runs_each_shard(checkout: Path, tmp_path: Path) -> None:
    config = _config(tmp_path)
    composite = FastMSS(
        config_paths=str(config),
        fastmss_root=str(checkout),
        output_dir=str(tmp_path / "output"),
        python_executable=sys.executable,
    )
    partition, simulate, reader = composite.decompose_and_apply_with()
    groups = partition.process(EmptyTask())
    rows = [row for group in groups for output in simulate.process(group) for row in reader.process(output)]
    assert len(rows) == 1
    assert rows[0].data["duration"] == 0.1
    assert_agent_ready(composite, run=False, expected_cardinality="1:N fan-out")


def test_simulation_contract_matches_real_process_boundary(checkout: Path, tmp_path: Path) -> None:
    stage = FastMSSSimulationStage(fastmss_root=str(checkout), output_dir=str(tmp_path / "output"))
    config = _config(tmp_path)
    assert_agent_ready(
        stage,
        fixture_factory=lambda: FileGroupTask(dataset_name="simulation", data=[str(config)]),
        expected_cardinality="1:N fan-out",
    )


def test_pipeline_executes_composite_on_actor_pool(checkout: Path, tmp_path: Path) -> None:
    pipeline = Pipeline(
        name="fastmss_cli_boundary",
        stages=[
            FastMSS(
                config_paths=str(_config(tmp_path)),
                fastmss_root=str(checkout),
                output_dir=str(tmp_path / "pipeline-output"),
            )
        ],
    )
    results = pipeline.run(RayActorPoolExecutor(show_progress=False))
    assert len(results) == 1
    assert results[0].data["text"] == "hello"
    assert Path(results[0].data["audio_filepath"]).is_file()


def test_documented_yaml_resolves_external_defaults(
    checkout: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hydra import compose, initialize_config_dir

    (checkout / "recipes" / "default.yaml").write_text("output_dir: null\nn_jobs: 1\nstage: 0\n")
    monkeypatch.setenv("FASTMSS_ROOT", str(checkout))
    monkeypatch.setenv("FASTMSS_MANIFEST_DIR", str(tmp_path / "source"))
    example_dir = Path(__file__).resolve().parents[4] / "tutorials" / "audio" / "fastmss"
    with initialize_config_dir(config_dir=str(example_dir), version_base=None):
        config = compose(config_name="meetings")
    assert config.manifest_dir == str(tmp_path / "source")
    assert config.min_max_spk == [2, 2]
    assert config.n_jobs == 1


def test_bundled_upstream_files_are_unmodified(tmp_path: Path) -> None:
    root = Path(FastMSSSimulationStage(output_dir=str(tmp_path / "output")).fastmss_root)
    provenance = json.loads((root / "UPSTREAM.json").read_text())
    assert provenance["revision"] == "81fc2c3e4b938cf73343f58df9e013bc106cb77a"
    assert provenance["license"] == "Apache-2.0"
    for name, checksum in provenance["files_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == checksum, name
    assert not (root / "fastmss" / "simulator_masr.py").exists()


def test_real_upstream_generates_audio_and_diarization_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("lhotse")
    pytest.importorskip("pyroomacoustics")
    import numpy as np
    import soundfile as sf
    from lhotse import CutSet, MonoCut, Recording
    from lhotse.supervision import AlignmentItem, SupervisionSegment

    sources = tmp_path / "sources"
    sources.mkdir()
    cuts = []
    sample_rate = 16000
    duration = 0.6
    for index, frequency in enumerate((220, 330)):
        audio = sources / f"speaker-{index}.wav"
        samples = 0.05 * np.sin(2 * np.pi * frequency * np.arange(int(sample_rate * duration)) / sample_rate)
        sf.write(audio, samples, sample_rate)
        recording = Recording.from_file(audio)
        supervision = SupervisionSegment(
            id=f"utterance-{index}",
            recording_id=recording.id,
            start=0,
            duration=duration,
            channel=0,
            text="hello world",
            speaker=f"speaker-{index}",
            alignment={"word": [AlignmentItem("hello", 0, 0.3), AlignmentItem("world", 0.3, 0.3)]},
        )
        cuts.append(
            MonoCut(
                id=f"cut-{index}",
                start=0,
                duration=duration,
                channel=0,
                recording=recording,
                supervisions=[supervision],
            )
        )
    CutSet.from_cuts(cuts).to_file(sources / "source_train.jsonl.gz")
    monkeypatch.setenv("FASTMSS_MANIFEST_DIR", str(sources))
    config = Path(__file__).resolve().parents[4] / "tutorials" / "audio" / "fastmss" / "meetings.yaml"
    stage = FastMSSSimulationStage(
        output_dir=str(tmp_path / "generated"),
        overrides=["n_meetings=1", "duration=2", "hmm_params.p_ind=[0,1,0,0]"],
        timeout_seconds=60,
    )
    output = stage.process(FileGroupTask(dataset_name="upstream-smoke", data=[str(config)]))
    rows = ManifestReaderStage().process(output[0])
    assert len(rows) == 1
    row = rows[0].data
    waveform, actual_rate = sf.read(row["audio_filepath"])
    assert actual_rate == sample_rate
    assert np.isfinite(waveform).all()
    assert np.max(np.abs(waveform)) > 0
    assert row["duration"] == pytest.approx(len(waveform) / sample_rate, abs=1e-5)
    annotations = Path(row["rttm_filepath"]).read_text().splitlines()
    assert {line.split()[7] for line in annotations} == {"speaker-0", "speaker-1"}
    assert row["num_speakers"] == 2
    with gzip.open(
        next((Path(output[0].data[0]).parent / "manifests").glob("*-supervisions.jsonl.gz")), "rt"
    ) as source:
        supervisions = sorted(
            (json.loads(line) for line in source), key=lambda segment: (segment["start"], segment["id"])
        )
    assert row["text"] == " ".join(segment["text"].strip() for segment in supervisions)
    assert "hello world" in row["text"]


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ("min_max_spk: [1, 1]\n", "single-speaker recipes are unsupported"),
        ("save_spk: true\nsave_anechoic: true\nreverberate: true\nreverb_prob: 0.5\n", "reverb_prob=1"),
    ],
)
def test_resolved_config_is_rejected_before_engine_starts(
    checkout: Path, tmp_path: Path, settings: str, message: str
) -> None:
    config = _config(tmp_path)
    with config.open("a") as output:
        output.write(settings)
    stage = FastMSSSimulationStage(fastmss_root=str(checkout), output_dir=str(tmp_path / "output"))
    with pytest.raises(RuntimeError, match="exit code"):
        stage.process(FileGroupTask(dataset_name="simulation", data=[str(config)]))
    assert message in next((tmp_path / "output").glob("*/fastmss.log")).read_text()
    assert not list((tmp_path / "output").glob("*/arguments.json"))
    assert not list((tmp_path / "output").glob("*/nemo_manifest.json"))


def test_preflight_uses_cli_overrides_over_yaml_settings(checkout: Path, tmp_path: Path) -> None:
    config = _config(tmp_path)
    with config.open("a") as output:
        output.write("min_max_spk: [1, 1]\nsave_anechoic: true\nreverberate: true\nreverb_prob: 0.5\n")
    stage = FastMSSSimulationStage(
        fastmss_root=str(checkout),
        output_dir=str(tmp_path / "output"),
        overrides=["min_max_spk=[2,2]", "reverb_prob=1"],
    )
    groups = stage.process(FileGroupTask(dataset_name="simulation", data=[str(config)]))
    assert ManifestReaderStage().process(groups[0])[0].data["text"] == "hello"
