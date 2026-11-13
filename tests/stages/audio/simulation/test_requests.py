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

"""Native-job compatibility against unchanged agent protocols and real consumers."""

import gzip
import json
import wave
from importlib.metadata import distribution
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_curator.backends.base import BaseExecutor, BaseStageAdapter
from nemo_curator.stages.audio.audio_agent import smoke, validate
from nemo_curator.stages.audio.audio_agent.artifacts import model_version
from nemo_curator.stages.audio.audio_agent.recipe import Recipe, StageRef, build_stages
from nemo_curator.stages.audio.simulation import FastMSSRequestStage, write_fastmss_requests
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import EmptyTask, Task


class LocalExecutor(BaseExecutor):
    def execute(self, stages: list[ProcessingStage], initial_tasks: list[Task] | None = None) -> list[Task]:
        tasks = initial_tasks or [EmptyTask()]
        for stage in stages:
            stage.setup()
            tasks = BaseStageAdapter(stage).process_batch(tasks)
        return tasks


@pytest.fixture
def native_job(tmp_path: Path) -> SimpleNamespace:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    audio = inputs / "speaker.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 1600)
    cut = {"type": "MonoCut", "recording": {"sources": [{"type": "file", "source": str(audio)}]}}
    with gzip.open(inputs / "source_train.jsonl.gz", "wt") as stream:
        stream.write(json.dumps(cut) + "\n")
    toolkit = tmp_path / "toolkit"
    engine = toolkit / "fastmss"
    engine.mkdir(parents=True)
    (engine / "__init__.py").touch()
    recipes = toolkit / "recipes"
    recipes.mkdir()
    (recipes / "default.yaml").write_text("min_max_spk: [2, 2]\nn_jobs: 1\nstage: 0\noutput_dir: null\n")
    (recipes / "sim.py").write_text("""import json, sys, wave
from pathlib import Path
from hydra import compose, initialize_config_dir
args = dict(x.split('=', 1) for x in sys.argv[1:] if x.startswith('--'))
with initialize_config_dir(config_dir=args['--config-path'], version_base=None):
    cfg = compose(config_name=args['--config-name'], overrides=[x for x in sys.argv[1:] if not x.startswith('--')])
output = Path(cfg.output_dir)
(output / 'native_count').write_text(str(cfg.n_meetings))
rows = []
for i in range(cfg.n_meetings):
    audio = output / f'{i}.wav'
    with wave.open(str(audio), 'wb') as stream:
        stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(16000)
        stream.writeframes(b'\\0\\0' * 1600)
    rttm = output / f'{i}.rttm'
    rttm.write_text('SPEAKER meeting 1 0 0.1 <NA> <NA> speaker <NA> <NA>\\n')
    rows.append({'audio_filepath': str(audio), 'duration': 0.1, 'text': 'reference', 'rttm_filepath': str(rttm), 'custom': 17})
(output / 'nemo_manifest.json').write_text(''.join(json.dumps(row) + '\\n' for row in rows))
""")
    config = inputs / "meeting.yaml"
    config.write_text(
        "defaults: [default, _self_]\nhydra:\n  searchpath: ['file://${oc.env:FASTMSS_ROOT}/recipes']\n"
        f"manifest_dir: {json.dumps(str(inputs))}\nmanifest_prefix: source\ndset_splits: [train]\n"
        "n_meetings: 2\nduration: 2\nseed: 42\ndata_dir: /protected/data\nrir_dir: /protected/rir\n"
    )
    return SimpleNamespace(inputs=inputs, audio=audio, config=config, engine=engine, recipes=recipes, root=tmp_path)


def request_recipe(job: SimpleNamespace, renamed: bool = False) -> Recipe:
    roots = {"native_inputs_path": str(job.inputs), "engine_path": str(job.engine), "recipe_path": str(job.recipes)}
    manifest = job.root / "requests.jsonl"
    request_key = "native_plan" if renamed else "fastmss_request"
    write_fastmss_requests([str(job.config), str(job.config)], str(manifest), request_key=request_key, **roots)
    keys = (
        {
            "audio_filepath_key": "mixture",
            "duration_key": "seconds",
            "text_key": "reference",
            "rttm_filepath_key": "rttm",
        }
        if renamed
        else {}
    )
    return Recipe(
        stages=[
            StageRef("FastMSSRequests", {"manifest_path": str(manifest), "request_key": request_key, **roots}),
            StageRef(
                "FastMSSRequestStage",
                {"output_dir": str(job.root / "production"), "request_key": request_key, **roots, **keys},
            ),
            StageRef(
                "GetAudioDurationStage",
                {
                    "audio_filepath_key": keys.get("audio_filepath_key", "audio_filepath"),
                    "duration_key": "measured_duration",
                },
            ),
        ]
    )


@pytest.mark.parametrize("renamed", [False, True])
def test_unchanged_agent_validation_smoke_and_downstream_consumer(native_job: SimpleNamespace, renamed: bool) -> None:
    recipe = request_recipe(native_job, renamed)
    expected = (
        ["mixture", "seconds", "reference", "rttm"]
        if renamed
        else ["audio_filepath", "duration", "text", "rttm_filepath"]
    )
    verdict = validate(recipe, expected_outputs=expected)
    assert verdict["runnable"], verdict
    before = native_job.config.read_bytes()
    report = smoke(recipe, sample=1, executor=LocalExecutor())
    assert report["status"] == "completed", report
    assert not report["errors"], report
    assert report["input_count"] == 1  # Complete native jobs, not meetings.
    assert report["retained"] == 2  # n_meetings is preserved, never silently capped.
    assert report["rejected"] is None
    assert native_job.config.read_bytes() == before
    assert not (native_job.root / "production").exists()
    stages, errors = build_stages(recipe)
    assert not errors
    from nemo_curator.pipeline import Pipeline

    rows = Pipeline(name="native_requests", stages=stages).run(LocalExecutor())
    assert len(rows) == 4
    for row in rows:
        assert getattr(row.data, expected[0]) == row.data[expected[0]]
        assert row.data.measured_duration == pytest.approx(row.data[expected[1]])
        assert row.data.custom == 17


@pytest.mark.parametrize("changed", ["audio", "config", "default", "engine", "cutset", "noise", "fit"])
@pytest.mark.parametrize("stage_index", [0, 1])
def test_native_dependencies_change_existing_step_identity(
    native_job: SimpleNamespace, changed: str, stage_index: int
) -> None:
    noise = native_job.inputs / "noise.wav"
    noise.write_bytes(native_job.audio.read_bytes())
    fit = native_job.inputs / "fit.jsonl.gz"
    fit.write_bytes((native_job.inputs / "source_train.jsonl.gz").read_bytes())
    if changed == "noise":
        native_job.config.write_text(
            native_job.config.read_text()
            + f"\nadd_noise: true\nnoise_folders: [{json.dumps(str(native_job.inputs))}]\n"
        )
    if changed == "fit":
        native_job.config.write_text(
            native_job.config.read_text() + f"\nhmm_fit_transitions_to: {json.dumps(str(fit))}\n"
        )
    recipe = request_recipe(native_job)
    params = recipe.stages[stage_index].params
    ref = recipe.stages[stage_index].ref
    before = model_version(params, stage_ref=ref)
    path = {
        "audio": native_job.audio,
        "config": native_job.config,
        "default": native_job.recipes / "default.yaml",
        "engine": native_job.engine / "__init__.py",
        "cutset": native_job.inputs / "source_train.jsonl.gz",
        "noise": noise,
        "fit": fit,
    }[changed]
    path.write_bytes(path.read_bytes() + b"\n")
    assert model_version(params, stage_ref=ref) != before


def test_stale_bounds_and_external_inputs_fail_before_generation(native_job: SimpleNamespace) -> None:
    recipe = request_recipe(native_job)
    source, generation, _ = build_stages(recipe)[0]
    task = source.decompose()[1].process(source.decompose()[0].process(EmptyTask())[0])[0]
    task.data.fastmss_request["n_meetings"] = 1
    with pytest.raises(ValueError, match="planning metadata"):
        generation.process(task)
    assert not (native_job.root / "production").exists()
    native_job.config.write_text(
        native_job.config.read_text().replace("manifest_dir:", "unused_manifest_dir:") + "\nmanifest_dir: /outside\n"
    )
    with pytest.raises(ValueError, match="declared input tree"):
        generation.process(task)


@pytest.mark.parametrize("renamed", [False, True])
def test_real_native_generation_then_consumer(native_job: SimpleNamespace, renamed: bool) -> None:
    pytest.importorskip("lhotse")
    pytest.importorskip("pyroomacoustics")
    import numpy as np
    import soundfile as sf
    from lhotse import CutSet, MonoCut, Recording
    from lhotse.supervision import AlignmentItem, SupervisionSegment

    cuts = []
    for index in range(2):
        audio = native_job.inputs / f"real-{index}.wav"
        sf.write(audio, 0.05 * np.sin(2 * np.pi * (220 + index * 110) * np.arange(9600) / 16000), 16000)
        recording = Recording.from_file(audio)
        supervision = SupervisionSegment(
            id=f"u{index}",
            recording_id=recording.id,
            start=0,
            duration=0.6,
            channel=0,
            text="hello world",
            speaker=f"speaker-{index}",
            alignment={"word": [AlignmentItem("hello", 0, 0.3), AlignmentItem("world", 0.3, 0.3)]},
        )
        cuts.append(
            MonoCut(id=f"c{index}", start=0, duration=0.6, channel=0, recording=recording, supervisions=[supervision])
        )
    CutSet.from_cuts(cuts).to_file(native_job.inputs / "source_train.jsonl.gz")
    package = distribution("fastmss")
    native_job.engine = Path(package.locate_file("fastmss"))
    native_job.recipes = Path(package.locate_file("recipes"))
    native_job.config.write_text(
        native_job.config.read_text()
        + "\nmin_spk_utt: 1\nsplit_fa_factor: null\nmin_max_spk: [2,2]\nmax_concurrent_speakers: 2\n"
    )
    recipe = request_recipe(native_job, renamed)
    before = native_job.config.read_bytes()
    verdict = validate(recipe)
    assert verdict["runnable"], verdict
    result = smoke(recipe, sample=1, executor=LocalExecutor())
    assert result["status"] == "completed", result
    assert result["retained"] == 2, result
    assert result["input_count"] == 1
    assert result["rejected"] is None
    assert native_job.config.read_bytes() == before


def test_invalid_native_job_fails_without_hidden_generation(native_job: SimpleNamespace) -> None:
    native_job.config.write_text(native_job.config.read_text() + "\nmin_max_spk: [1,1]\n")
    recipe = request_recipe(native_job)
    stages, errors = build_stages(recipe)
    assert not errors
    from nemo_curator.pipeline import Pipeline

    with pytest.raises(RuntimeError, match="exit code"):
        Pipeline(name="native_failure", stages=stages).run(LocalExecutor())
    assert "single-speaker" in next((native_job.root / "production").glob("*/fastmss.log")).read_text()


def test_request_metadata_cannot_be_consumed_as_audio(native_job: SimpleNamespace) -> None:
    recipe = request_recipe(native_job)
    recipe.stages.pop(1)
    assert not validate(recipe)["runnable"]


def test_unsupported_native_prefix_is_refused_before_writes(native_job: SimpleNamespace) -> None:
    native_job.config.write_text(
        native_job.config.read_text().replace("manifest_prefix: source", "manifest_prefix: ../outside")
    )
    with pytest.raises(ValueError, match="plain source manifest prefix"):
        request_recipe(native_job)
    assert not (native_job.root / "production").exists()


def test_output_cannot_overlap_declared_dependencies(native_job: SimpleNamespace) -> None:
    params = request_recipe(native_job).stages[1].params
    params["output_dir"] = str(native_job.inputs / "outputs")
    with pytest.raises(ValueError, match="outside all declared native input trees"):
        FastMSSRequestStage(**params)


@pytest.mark.parametrize("unsupported", ["command", "external_audio", "external_noise"])
def test_untracked_native_sources_are_explicitly_unsupported(native_job: SimpleNamespace, unsupported: str) -> None:
    if unsupported == "external_noise":
        native_job.config.write_text(native_job.config.read_text() + "\nadd_noise: true\nnoise_folders: [/outside]\n")
    else:
        recording = {
            "sources": [{"type": "command" if unsupported == "command" else "file", "source": "/outside.wav"}]
        }
        with gzip.open(native_job.inputs / "source_train.jsonl.gz", "wt") as stream:
            stream.write(json.dumps({"type": "MonoCut", "recording": recording}) + "\n")
    with pytest.raises(ValueError, match=r"unsupported|declared input tree"):
        request_recipe(native_job)
    assert not (native_job.root / "requests.jsonl").exists()


@pytest.mark.parametrize("value", ["0", "true", "2.5"])
def test_nonfinite_job_units_are_not_advertised(native_job: SimpleNamespace, value: str) -> None:
    native_job.config.write_text(native_job.config.read_text().replace("n_meetings: 2", f"n_meetings: {value}"))
    with pytest.raises(ValueError, match="positive integer n_meetings"):
        request_recipe(native_job)
    assert not (native_job.root / "requests.jsonl").exists()


def test_dynamic_environment_recipe_identity_is_explicitly_unsupported(native_job: SimpleNamespace) -> None:
    native_job.config.write_text(native_job.config.read_text().replace("seed: 42", "seed: ${oc.env:USER_SEED}"))
    with pytest.raises(ValueError, match="Dynamic resolvers"):
        request_recipe(native_job)


def test_native_request_stage_conforms_at_runtime(native_job: SimpleNamespace) -> None:
    from nemo_curator.stages.audio._agent._conformance import assert_agent_ready

    recipe = request_recipe(native_job)
    source, generation, _ = build_stages(recipe)[0]

    def fixture() -> Task:
        partition, reader = source.decompose()
        return reader.process(partition.process(EmptyTask())[0])[0]

    assert_agent_ready(generation, fixture_factory=fixture, expected_cardinality="1:N fan-out")
