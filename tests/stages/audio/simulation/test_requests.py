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

"""One-request/one-meeting integration through unchanged Curator protocols."""

import gzip
import json
import sys
import wave
from importlib.metadata import distribution
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_curator.backends.base import BaseExecutor, BaseStageAdapter
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.audio_agent import smoke, validate
from nemo_curator.stages.audio.audio_agent.artifacts import model_version
from nemo_curator.stages.audio.audio_agent.recipe import Recipe, StageRef, build_stages
from nemo_curator.stages.audio.simulation import FastMSSRequestStage, FastMSSSimulationStage, write_fastmss_requests
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask, EmptyTask, FileGroupTask, Task


class LocalExecutor(BaseExecutor):
    def execute(self, stages: list[ProcessingStage], initial_tasks: list[Task] | None = None) -> list[Task]:
        tasks = initial_tasks or [EmptyTask()]
        for stage in stages:
            stage.setup()
            try:
                tasks = BaseStageAdapter(stage).process_batch(tasks)
            finally:
                stage.teardown()
        return tasks


@pytest.fixture
def native_job(tmp_path: Path) -> SimpleNamespace:
    from lhotse import CutSet, MonoCut, Recording
    from lhotse.supervision import AlignmentItem, SupervisionSegment

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    audio = inputs / "speaker.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 1600)
    recording = Recording.from_file(audio)
    cuts = []
    for speaker in ("alice", "bob"):
        supervision = SupervisionSegment(
            id=speaker,
            recording_id=recording.id,
            start=0,
            duration=0.1,
            channel=0,
            speaker=speaker,
            text="reference",
            alignment={"word": [AlignmentItem("reference", 0, 0.1)]},
        )
        cuts.append(
            MonoCut(id=speaker, start=0, duration=0.1, channel=0, recording=recording, supervisions=[supervision])
        )
    CutSet.from_cuts(cuts).to_file(inputs / "source_train.jsonl.gz")
    toolkit = tmp_path / "toolkit"
    engine = toolkit / "fastmss"
    engine.mkdir(parents=True)
    (engine / "__init__.py").touch()
    recipes = toolkit / "recipes"
    recipes.mkdir()
    (recipes / "default.yaml").write_text(
        "min_max_spk: [2, 2]\nn_jobs: 1\nstage: 0\noutput_dir: null\nmin_spk_utt: 1\nsamplerate: 16000\n"
    )
    (recipes / "sim.py").write_text(
        """import json, sys, wave
from pathlib import Path
from hydra import compose, initialize_config_dir
args = dict(x.split('=', 1) for x in sys.argv[1:] if x.startswith('--'))
with initialize_config_dir(config_dir=args['--config-path'], version_base=None):
    cfg = compose(config_name=args['--config-name'], overrides=[x for x in sys.argv[1:] if not x.startswith('--')])
output = Path(cfg.output_dir)
(output / 'effective_config.json').write_text(json.dumps(dict(cfg)))
(output / 'worker_observed_config.json').write_text(json.dumps(dict(cfg)))
rows = []
for i in range(cfg.n_meetings):
    audio = output / f'simulation_{i}.wav'
    seconds = 0.1 * (i + 1)
    with wave.open(str(audio), 'wb') as stream:
        stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(16000)
        stream.writeframes(b'\\0\\0' * int(16000 * seconds))
    rttm = output / f'simulation_{i}.rttm'
    rttm.write_text('SPEAKER meeting 1 0 0.1 <NA> <NA> speaker <NA> <NA>\\n')
    rows.append({'audio_filepath': str(audio), 'duration': seconds, 'text': 'reference', 'rttm_filepath': str(rttm), 'custom': 17})
(output / 'nemo_manifest.json').write_text(''.join(json.dumps(row) + '\\n' for row in rows))
""".replace("json.dumps(dict(cfg))", "json.dumps(__import__('omegaconf').OmegaConf.to_container(cfg, resolve=True))")
    )
    config = inputs / "meeting.yaml"
    config.write_text(
        "defaults: [default, _self_]\nhydra:\n  searchpath: ['file://${oc.env:FASTMSS_ROOT}/recipes']\n"
        f"manifest_dir: {json.dumps(str(inputs))}\nmanifest_prefix: source\ndset_splits: [train]\n"
        "n_meetings: 4\nduration: 2\nseed: 42\ndata_dir: /protected/data\nrir_dir: /protected/rir\n"
    )
    return SimpleNamespace(inputs=inputs, audio=audio, config=config, engine=engine, recipes=recipes, root=tmp_path)


def roots(job: SimpleNamespace) -> dict:
    return {
        "native_inputs_path": str(job.inputs),
        "engine_path": str(job.engine),
        "recipe_path": str(job.recipes),
        "worker_lock_path": str(job.root / "worker-lock.json"),
    }


def request_recipe(job: SimpleNamespace, renamed: bool = False, *, filter_rows: bool = False) -> Recipe:
    manifest = job.root / "requests.jsonl"
    request_key = "native_plan" if renamed else "fastmss_request"
    write_fastmss_requests([str(job.config)], str(manifest), request_key=request_key, **roots(job))
    keys = (
        {
            "audio_filepath_key": "mixture",
            "duration_key": "seconds",
            "text_key": "reference",
            "rttm_filepath_key": "rttm",
            "request_id_key": "meeting_request",
            "native_annotations_key": "native",
        }
        if renamed
        else {}
    )
    stages = [
        StageRef("FastMSSRequests", {"manifest_path": str(manifest), "request_key": request_key, **roots(job)}),
        StageRef(
            "FastMSSRequestStage",
            {"output_dir": str(job.root / "production"), "request_key": request_key, **roots(job), **keys},
        ),
        StageRef(
            "GetAudioDurationStage",
            {
                "audio_filepath_key": keys.get("audio_filepath_key", "audio_filepath"),
                "duration_key": "measured_duration",
            },
        ),
    ]
    if filter_rows:
        stages.append(
            StageRef(
                "PreserveByValueStage",
                {"input_value_key": "measured_duration", "target_value": 0.15, "operator": "ge"},
            )
        )
    stages.append(StageRef("ManifestWriterStage", {"output_path": str(job.root / "meetings.jsonl")}))
    return Recipe(stages=stages)


def source_rows(recipe: Recipe) -> list[AudioTask]:
    source = build_stages(recipe)[0][0]
    return source.process(EmptyTask())


@pytest.mark.parametrize("renamed", [False, True])
def test_public_smoke_bounds_meetings_then_standard_filter_and_writer(
    native_job: SimpleNamespace, renamed: bool
) -> None:
    recipe = request_recipe(native_job, renamed, filter_rows=True)
    verdict = validate(recipe)
    assert verdict["runnable"], verdict
    before = {p: p.read_bytes() for p in native_job.inputs.iterdir()}
    # A one-row smoke creates one meeting; the filter may reject that meeting.
    report = smoke(recipe, sample=1, executor=LocalExecutor())
    assert report["status"] == "completed", report
    assert report["input_count"] == 1
    assert report["retained"] == 0
    assert report["rejected"] == 1
    assert not (native_job.root / "production").exists()
    assert not (native_job.root / "meetings.jsonl").exists()
    stages, errors = build_stages(recipe)
    assert not errors
    rows = Pipeline(name="meeting_requests", stages=stages).run(LocalExecutor())
    assert len(rows) == 3
    saved = [json.loads(line) for line in (native_job.root / "meetings.jsonl").read_text().splitlines()]
    assert len(saved) == 3
    path_key = "mixture" if renamed else "audio_filepath"
    duration_key = "seconds" if renamed else "duration"
    identity_key = "meeting_request" if renamed else "fastmss_request_id"
    assert len({row.data[identity_key] for row in rows}) == 3
    for row in rows:
        assert getattr(row.data, path_key) == row.data[path_key]
        assert row.data.measured_duration == pytest.approx(row.data[duration_key])
        assert row.data["native" if renamed else "fastmss_annotations"]["custom"] == 17
        assert "fastmss_request" not in row.data
        assert "native_plan" not in row.data
    invocations = list((native_job.root / "production").glob("*/effective_config.json"))
    assert len(invocations) == 1  # Bulk native path, rather than one process per meeting.
    assert json.loads(invocations[0].read_text())["n_meetings"] == 4
    assert {p: p.read_bytes() for p in native_job.inputs.iterdir()} == before


def test_hundred_meeting_yaml_is_one_hundred_requests_without_generation(native_job: SimpleNamespace) -> None:
    native_job.config.write_text(native_job.config.read_text().replace("n_meetings: 4", "n_meetings: 100"))
    recipe = request_recipe(native_job)
    tasks = source_rows(recipe)
    assert len(tasks) == 100
    assert len({t.data.fastmss_request["request_id"] for t in tasks}) == 100
    assert not (native_job.root / "production").exists()
    report = smoke(recipe, sample=1, executor=LocalExecutor())
    assert report["status"] == "completed", report
    assert (report["input_count"], report["retained"], report["rejected"]) == (1, 1, 0)


@pytest.mark.parametrize("target", ["audio", "config", "default", "engine", "cutset"])
@pytest.mark.parametrize("stage_index", [0, 1])
def test_dependency_mutation_invalidates_reuse_and_stale_requests(
    native_job: SimpleNamespace, target: str, stage_index: int
) -> None:
    recipe = request_recipe(native_job)
    task = source_rows(recipe)[0]
    ref = recipe.stages[stage_index]
    before = model_version(ref.params, stage_ref=ref.ref)
    path = {
        "audio": native_job.audio,
        "config": native_job.config,
        "default": native_job.recipes / "default.yaml",
        "engine": native_job.engine / "__init__.py",
        "cutset": native_job.inputs / "source_train.jsonl.gz",
    }[target]
    path.write_bytes(path.read_bytes() + b"\n")
    assert model_version(ref.params, stage_ref=ref.ref) != before
    generator = build_stages(recipe)[0][1]
    with pytest.raises(ValueError, match="Stale"):
        generator.process(task)
    assert not (native_job.root / "production").exists()


@pytest.mark.parametrize("corrupt", ["missing", "duplicate", "index", "config", "count"])
def test_invalid_requests_fail_before_any_generation_writes(native_job: SimpleNamespace, corrupt: str) -> None:
    recipe = request_recipe(native_job)
    tasks = source_rows(recipe)
    generator = build_stages(recipe)[0][1]
    if corrupt == "missing":
        tasks[0].data.clear()
    elif corrupt == "duplicate":
        tasks[1].data.fastmss_request = tasks[0].data.fastmss_request
    elif corrupt == "index":
        tasks[1].data.fastmss_request["meeting_index"] = 999
    elif corrupt == "config":
        tasks[1].data.fastmss_request["native_config"]["duration"] = 1
    else:
        tasks[1].data.fastmss_request["native_config"]["n_meetings"] = 2
    with pytest.raises((TypeError, ValueError), match=r"Missing|Duplicate|Stale|capacity"):
        generator.process_batch(tasks)
    assert not (native_job.root / "production").exists()


@pytest.mark.parametrize(
    "unsupported",
    [
        "add_noise: true",
        "reverberate: true",
        "split_fa_factor: 0.2",
        "hmm_fit_transitions_to: /somewhere",
        "min_max_spk: [1,1]",
        "min_max_spk: [2,3]",
        "duration: 301",
        "n_meetings: 0",
    ],
)
def test_unsupported_modes_or_capacity_fail_during_planning(native_job: SimpleNamespace, unsupported: str) -> None:
    key = unsupported.split(":")[0]
    lines = [line for line in native_job.config.read_text().splitlines() if not line.startswith(key + ":")]
    native_job.config.write_text("\n".join(lines) + "\n" + unsupported + "\n")
    with pytest.raises(ValueError, match=r"unsupported|requires|Insufficient|must be"):
        request_recipe(native_job)
    assert not (native_job.root / "requests.jsonl").exists()
    assert not (native_job.root / "production").exists()


def test_request_fields_cannot_be_consumed_as_generated_audio_or_after_generation(native_job: SimpleNamespace) -> None:
    recipe = request_recipe(native_job)
    recipe.stages.pop(1)
    assert not validate(recipe)["runnable"]
    manifest = native_job.root / "requests.jsonl"
    manifest.unlink()
    recipe = request_recipe(native_job)
    recipe.stages.insert(
        2,
        StageRef("PreserveByValueStage", {"input_value_key": "fastmss_request", "target_value": 0, "operator": "ge"}),
    )
    assert not validate(recipe)["runnable"]


def test_partial_generation_is_a_failure_not_a_filter(native_job: SimpleNamespace) -> None:
    script = native_job.recipes / "sim.py"
    script.write_text(script.read_text().replace("range(cfg.n_meetings)", "range(cfg.n_meetings - 1)"))
    recipe = request_recipe(native_job)
    generator = build_stages(recipe)[0][1]
    with pytest.raises(RuntimeError, match="requested 4, generated 3"):
        generator.process_batch(source_rows(recipe))


def test_lineage_metadata_and_native_annotations_are_preserved(native_job: SimpleNamespace) -> None:
    recipe = request_recipe(native_job)
    tasks = source_rows(recipe)
    for i, task in enumerate(tasks):
        task._metadata = {"origin": str(i)}
        task._source_id = f"source-{i}"
    generator = build_stages(recipe)[0][1]
    rows = BaseStageAdapter(generator).process_batch(tasks)
    assert [row._metadata["origin"] for row in rows] == [str(i) for i in range(4)]
    assert [row._source_id for row in rows] == [f"source-{i}" for i in range(4)]
    assert all(row.data.fastmss_annotations["custom"] == 17 and row.task_id for row in rows)


def test_meeting_adapter_runtime_contract(native_job: SimpleNamespace) -> None:
    from nemo_curator.stages.audio._agent._conformance import assert_agent_ready

    recipe = request_recipe(native_job)
    source = build_stages(recipe)[0][0]
    assert_agent_ready(source, fixture_factory=EmptyTask, expected_cardinality="1:N fan-out")
    generator = build_stages(recipe)[0][1]
    assert_agent_ready(generator, fixture_factory=lambda: source_rows(recipe)[0], expected_cardinality="1:1")


@pytest.fixture
def real_native_job(native_job: SimpleNamespace) -> SimpleNamespace:
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
    assert package.version == "0.1.0"
    native_job.engine = Path(package.locate_file("fastmss"))
    native_job.recipes = Path(package.locate_file("recipes"))
    native_job.config.write_text(
        native_job.config.read_text().replace("n_meetings: 4", "n_meetings: 2")
        + "\nmin_spk_utt: 1\nsplit_fa_factor: null\nmin_max_spk: [2,2]\nmax_concurrent_speakers: 2\n"
    )
    return native_job


@pytest.mark.parametrize("renamed", [False, True])
def test_real_engine_public_smoke_and_standard_consumer(real_native_job: SimpleNamespace, renamed: bool) -> None:
    recipe = request_recipe(real_native_job, renamed)
    before = {p: p.read_bytes() for p in real_native_job.inputs.iterdir()}
    verdict = validate(recipe)
    assert verdict["runnable"], verdict
    result = smoke(recipe, sample=1, executor=LocalExecutor())
    assert result["status"] == "completed", result
    assert (result["input_count"], result["retained"], result["rejected"]) == (1, 1, 0)
    stages, errors = build_stages(recipe)
    assert not errors
    rows = Pipeline(name="real_meetings", stages=stages).run(LocalExecutor())
    assert len(rows) == 2
    assert len((real_native_job.root / "meetings.jsonl").read_text().splitlines()) == 2
    assert {p: p.read_bytes() for p in real_native_job.inputs.iterdir()} == before


def test_real_native_bulk_and_adapter_preserve_effective_configuration(real_native_job: SimpleNamespace) -> None:
    job = real_native_job
    recipe = request_recipe(job)
    native = FastMSSSimulationStage(
        fastmss_root=str(job.engine.parent),
        output_dir=str(job.root / "direct"),
        isolate_work_dirs=True,
        python_executable=sys.executable,
    )
    direct = native.process(FileGroupTask(dataset_name="native_equivalence", data=[str(job.config)]))[0]
    rows = build_stages(recipe)[0][1].process_batch(source_rows(recipe))
    assert len(rows) == len(Path(direct.data[0]).read_text().splitlines()) == 2
    # Compare the actual worker settings; native PID seeding precludes waveform-byte comparisons.
    direct_config = json.loads(next((job.root / "direct").glob("*/effective_config.json")).read_text())
    adapted_config = json.loads(next((job.root / "production").glob("*/effective_config.json")).read_text())
    for config in (direct_config, adapted_config):
        for key in ("output_dir", "data_dir", "rir_dir"):
            config.pop(key)
    assert direct_config == adapted_config
    assert direct_config["n_meetings"] == 2
    assert direct_config["seed"] == 42
    for row in rows:
        assert row.data.text.strip()
        assert Path(row.data.rttm_filepath).is_file()
        assert row.data.duration > 0


def test_native_defaults_overrides_and_explicit_lists_are_preserved(native_job: SimpleNamespace) -> None:
    second = native_job.inputs / "other.yaml"
    second.write_text(native_job.config.read_text().replace("seed: 42", "seed: 43"))
    destination = native_job.root / "requests.jsonl"
    count = write_fastmss_requests(
        [str(native_job.config), str(second)],
        str(destination),
        overrides=["duration=3", "n_meetings=2"],
        **roots(native_job),
    )
    requests = [json.loads(line)["fastmss_request"] for line in destination.read_text().splitlines()]
    assert count == len(requests) == 4
    assert {r["native_config"]["seed"] for r in requests} == {42, 43}
    assert all(r["native_config"]["min_max_spk"] == [2, 2] and r["native_config"]["duration"] == 3 for r in requests)
    stage = FastMSSRequestStage(output_dir=str(native_job.root / "production"), **roots(native_job))
    rows = stage.process_batch([AudioTask(data={"fastmss_request": request}) for request in requests])
    assert len(rows) == 4
    configs = [json.loads(p.read_text()) for p in (native_job.root / "production").glob("*/effective_config.json")]
    assert len(configs) == 2
    assert all(c["duration"] == 3 and c["n_meetings"] == 2 for c in configs)


@pytest.mark.parametrize("mode", ["command", "external", "prefix", "dynamic"])
def test_untracked_reads_are_refused(native_job: SimpleNamespace, mode: str) -> None:
    if mode in {"command", "external"}:
        path = native_job.inputs / "source_train.jsonl.gz"
        with gzip.open(path, "rt") as stream:
            cuts = [json.loads(line) for line in stream]
        cuts[0]["recording"]["sources"][0] = {
            "type": "command" if mode == "command" else "file",
            "source": "/outside.wav",
        }
        with gzip.open(path, "wt") as stream:
            stream.writelines(json.dumps(cut) + "\n" for cut in cuts)
    else:
        native_job.config.write_text(
            native_job.config.read_text().replace(
                "manifest_prefix: source" if mode == "prefix" else "seed: 42",
                "manifest_prefix: ../outside" if mode == "prefix" else "seed: ${oc.env:USER_SEED}",
            )
        )
    with pytest.raises(ValueError, match=r"unsupported|outside|plain|Dynamic"):
        request_recipe(native_job)
    assert not (native_job.root / "requests.jsonl").exists()


def test_real_engine_on_actor_pool(real_native_job: SimpleNamespace) -> None:
    from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor

    recipe = request_recipe(real_native_job)
    stages, errors = build_stages(recipe)
    assert not errors
    rows = Pipeline(name="real_native_workers", stages=stages).run(RayActorPoolExecutor(show_progress=False))
    assert len(rows) == 2
    assert all(Path(row.data.audio_filepath).is_file() for row in rows)


@pytest.mark.parametrize("field", ["request_id_key", "native_annotations_key"])
def test_output_remapping_cannot_overwrite_native_annotations(native_job: SimpleNamespace, field: str) -> None:
    recipe = request_recipe(native_job)
    recipe.stages[1].params[field] = "custom"
    generator = build_stages(recipe)[0][1]
    with pytest.raises(ValueError, match="conflicts with an existing native annotation"):
        generator.process(source_rows(recipe)[0])


@pytest.mark.parametrize("stage_index", [0, 1])
def test_changed_worker_environment_cannot_reuse_the_old_plan(native_job: SimpleNamespace, stage_index: int) -> None:
    recipe = request_recipe(native_job)
    ref = recipe.stages[stage_index]
    before = model_version(ref.params, stage_ref=ref.ref)
    lock = Path(ref.params["worker_lock_path"])
    recorded = json.loads(lock.read_text())
    recorded["packages"]["numpy"] = "different-worker-version"
    lock.write_text(json.dumps(recorded))
    assert model_version(ref.params, stage_ref=ref.ref) != before
    _, errors = build_stages(recipe)
    assert errors
    assert not validate(recipe)["runnable"]
    assert not (native_job.root / "production").exists()


def test_ray_data_numpy_task_batches_use_the_native_bulk_path(native_job: SimpleNamespace) -> None:
    import numpy as np

    recipe = request_recipe(native_job)
    generator = build_stages(recipe)[0][1]
    rows = generator.process_batch(np.asarray(source_rows(recipe), dtype=object))
    assert len(rows) == 4
    assert len(list((native_job.root / "production").glob("*/effective_config.json"))) == 1
    assert generator.process_batch(np.asarray([], dtype=object)) == []


def test_reference_text_role_is_declared_for_default_and_renamed_meeting_outputs(native_job: SimpleNamespace) -> None:
    from nemo_curator.stages.audio import agent
    from nemo_curator.stages.audio.simulation import FastMSSManifestReaderStage

    recipe = request_recipe(native_job)
    generator = build_stages(recipe)[0][1]
    assert agent.build_contract(generator).key_roles["text"] == "reference_text"
    assert (
        agent.build_contract(FastMSSManifestReaderStage(text_key="reference")).key_roles["reference"]
        == "reference_text"
    )


@pytest.mark.parametrize("inside_tree", [False, True])
def test_lazy_rir_dependencies_are_rejected_before_planning_writes(
    real_native_job: SimpleNamespace, inside_tree: bool
) -> None:
    import numpy as np
    import soundfile as sf
    from lhotse import CutSet, Recording

    job = real_native_job
    rir = (job.inputs if inside_tree else job.root) / "external-rir.wav"
    sf.write(rir, np.r_[1.0, np.zeros(1599)], 16000)
    manifest = job.inputs / "source_train.jsonl.gz"
    cuts = list(CutSet.from_file(manifest))
    for cut in cuts:
        cut.recording = cut.recording.reverb_rir(Recording.from_file(rir), normalize_output=True)
    CutSet.from_cuts(cuts).to_file(manifest)
    for samples in (np.r_[1.0, np.zeros(1599)], np.r_[0.0, 1.0, np.zeros(1598)]):
        sf.write(rir, samples, 16000)
        with pytest.raises(ValueError, match="transforms"):
            request_recipe(job)
        assert not (job.root / "requests.jsonl").exists()
        assert not (job.root / "worker-lock.json").exists()
        assert not (job.root / "production").exists()


@pytest.mark.parametrize("selected", [1, 3])
def test_bounded_requests_preserve_all_resolved_nonmanaged_settings(
    native_job: SimpleNamespace, selected: int
) -> None:
    job = native_job
    job.config.write_text(
        job.config.read_text().replace("seed: 42", "seed: ${n_meetings}")
        + "\nother_setting: ${data_dir}\nlinked:\n  seed: ${seed}\n  output: ${output_dir}\n"
        "  jobs: ${n_jobs}\nliteral: '\\${n_meetings}'\n" + "literal_backslash: '" + "\\" * 3 + "${missing}'\n"
    )
    recipe = request_recipe(job)
    tasks = source_rows(recipe)
    planned = tasks[0].data.fastmss_request["native_config"]
    assert planned["seed"] == 4
    assert planned["literal"] == "${n_meetings}"
    generator = build_stages(recipe)[0][1]
    generator.jobs_per_config = 2
    rows = generator.process_batch(tasks[:selected])
    assert len(rows) == selected
    actual = json.loads(next((job.root / "production").glob("*/worker_observed_config.json")).read_text())
    managed = {"n_meetings", "output_dir", "data_dir", "rir_dir", "n_jobs", "stage", "save_rttm", "save_nemo_manifest"}
    assert {k: v for k, v in actual.items() if k not in managed} == {
        k: v for k, v in planned.items() if k not in managed
    }
    assert actual["n_meetings"] == selected
    assert actual["n_jobs"] == 2


def test_real_engine_keeps_planned_seed_when_only_one_request_is_selected(real_native_job: SimpleNamespace) -> None:
    job = real_native_job
    job.config.write_text(job.config.read_text().replace("seed: 42", "seed: ${n_meetings}"))
    recipe = request_recipe(job)
    task = source_rows(recipe)[0]
    rows = build_stages(recipe)[0][1].process_batch([task])
    actual = json.loads(next((job.root / "production").glob("*/effective_config.json")).read_text())
    assert len(rows) == actual["n_meetings"] == 1
    assert actual["seed"] == task.data.fastmss_request["native_config"]["seed"] == 2


def test_legacy_request_with_a_changed_external_rir_is_rejected_before_generation(
    real_native_job: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np
    import soundfile as sf
    from lhotse import CutSet, Recording

    from nemo_curator.stages.audio.simulation.requests import _NativeInputs

    job = real_native_job
    rir = job.root / "external-rir.wav"
    sf.write(rir, np.r_[1.0, np.zeros(1599)], 16000)
    manifest = job.inputs / "source_train.jsonl.gz"
    cuts = list(CutSet.from_file(manifest))
    for cut in cuts:
        cut.recording = cut.recording.reverb_rir(Recording.from_file(rir))
    CutSet.from_cuts(cuts).to_file(manifest)
    # Recreate planning by an older adapter that admitted lazy transforms.
    with monkeypatch.context() as legacy:
        legacy.setattr(_NativeInputs, "_admit_speech", lambda _self, _config: None)
        recipe = request_recipe(job)
    generator = build_stages(recipe)[0][1]
    before = generator._identity()
    sf.write(rir, np.r_[0.0, 1.0, np.zeros(1598)], 16000)
    assert generator._identity() == before  # Outside the supported dependency trees.
    with pytest.raises(ValueError, match="transforms"):
        generator.process(source_rows(recipe)[0])
    assert not (job.root / "production").exists()


@pytest.mark.parametrize("case", ["sample_rate", "multiple_supervisions", "empty_supervision", "speaker_whitespace"])
@pytest.mark.parametrize("legacy_worker", [False, True])
def test_native_incompatible_speech_is_rejected_before_planning_writes(
    real_native_job: SimpleNamespace, case: str, legacy_worker: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    import copy

    import numpy as np
    import soundfile as sf
    from lhotse import CutSet
    from lhotse import validate as validate_lhotse

    job = real_native_job
    manifest = job.inputs / "source_train.jsonl.gz"
    with gzip.open(manifest, "rt") as stream:
        cuts = [json.loads(line) for line in stream]
    if case == "sample_rate":
        for index, cut in enumerate(cuts):
            recording = cut["recording"]
            sf.write(
                recording["sources"][0]["source"],
                0.05 * np.sin(2 * np.pi * (220 + index * 110) * np.arange(4800) / 8000),
                8000,
            )
            recording.update(sampling_rate=8000, num_samples=4800)
    elif case == "multiple_supervisions":
        for cut in cuts:
            first = cut["supervisions"][0]
            first.update(
                duration=0.3, text="FIRST", alignment={"word": [{"symbol": "FIRST", "start": 0, "duration": 0.3}]}
            )
            second = {
                **copy.deepcopy(first),
                "id": first["id"] + "-second",
                "start": 0.3,
                "text": "SECOND",
                "alignment": {"word": [{"symbol": "SECOND", "start": 0.3, "duration": 0.3}]},
            }
            cut["supervisions"] = [first, second]
    elif case == "empty_supervision":
        extra = copy.deepcopy(cuts[0])
        extra.update(id=extra["id"] + "-unannotated", supervisions=[])
        cuts.append(extra)
    else:
        for index, cut in enumerate(cuts):
            cut["supervisions"][0]["speaker"] = f"Speaker {index}"
    with gzip.open(manifest, "wt") as stream:
        stream.writelines(json.dumps(cut) + "\n" for cut in cuts)
    validate_lhotse(CutSet.from_file(manifest))
    if legacy_worker:
        from nemo_curator.stages.audio.simulation.requests import _NativeInputs

        with monkeypatch.context() as legacy:
            legacy.setattr(_NativeInputs, "_admit_speech", lambda _self, _config: None)
            recipe = request_recipe(job)
        generator = build_stages(recipe)[0][1]
        with pytest.raises(ValueError, match=r"sampling|samplerate|supervision|speaker"):
            generator.process(source_rows(recipe)[0])
        assert not (job.root / "production").exists()
        return
    with pytest.raises(ValueError, match=r"sampling|samplerate|supervision|speaker"):
        request_recipe(job)
    for name in ["requests.jsonl", "worker-lock.json", "production"]:
        assert not (job.root / name).exists()


@pytest.mark.parametrize("label", ["", "   ", "Speaker\t0", "Speaker\n0", "Speaker\u00a00", None, 7])
def test_invalid_speaker_on_an_additional_cut_is_not_hidden_by_capacity(
    real_native_job: SimpleNamespace, label: object
) -> None:
    import copy

    job = real_native_job
    manifest = job.inputs / "source_train.jsonl.gz"
    with gzip.open(manifest, "rt") as stream:
        cuts = [json.loads(line) for line in stream]
    extra = copy.deepcopy(cuts[0])
    extra["id"] += "-extra"
    extra["supervisions"][0].update(id="extra", speaker=label)
    cuts.append(extra)
    with gzip.open(manifest, "wt") as stream:
        stream.writelines(json.dumps(cut) + "\n" for cut in cuts)
    with pytest.raises(ValueError, match="speaker"):
        request_recipe(job)
    assert not (job.root / "requests.jsonl").exists()
    assert not (job.root / "worker-lock.json").exists()


@pytest.mark.parametrize("case", ["partial_supervision", "missing_alignment", "word_outside", "word_nan"])
def test_every_cut_requires_complete_usable_reference_metadata(real_native_job: SimpleNamespace, case: str) -> None:
    import copy

    job = real_native_job
    manifest = job.inputs / "source_train.jsonl.gz"
    with gzip.open(manifest, "rt") as stream:
        cuts = [json.loads(line) for line in stream]
    extra = copy.deepcopy(cuts[0])
    extra["id"] += "-extra"
    supervision = extra["supervisions"][0]
    supervision["id"] = "extra"
    if case == "partial_supervision":
        supervision.update(duration=0.3, text="hello")
        supervision["alignment"]["word"] = supervision["alignment"]["word"][:1]
    elif case == "missing_alignment":
        supervision["alignment"] = None
    else:
        word = supervision["alignment"]["word"][-1]
        word["duration" if isinstance(word, dict) else 2] = 2 if case == "word_outside" else float("nan")
    cuts.append(extra)
    with gzip.open(manifest, "wt") as stream:
        stream.writelines(json.dumps(cut) + "\n" for cut in cuts)
    with pytest.raises(ValueError, match=r"supervision|alignment"):
        request_recipe(job)
    assert not (job.root / "requests.jsonl").exists()
    assert not (job.root / "worker-lock.json").exists()


@pytest.mark.parametrize("samplerate", [8000, 16000])
def test_real_engine_preserves_matched_rate_pitch_full_turn_and_speaker_roundtrip(
    real_native_job: SimpleNamespace, samplerate: int
) -> None:
    import os
    import subprocess

    import numpy as np
    import soundfile as sf
    from lhotse import CutSet
    from lhotse import validate as validate_lhotse

    job = real_native_job
    manifest = job.inputs / "source_train.jsonl.gz"
    labels = ["Speaker_0", "Speaker-1"]
    frequencies = dict(zip(labels, [220, 330], strict=True))
    with gzip.open(manifest, "rt") as stream:
        cuts = [json.loads(line) for line in stream]
    for index, cut in enumerate(cuts):
        recording = cut["recording"]
        count = int(0.6 * samplerate)
        sf.write(
            recording["sources"][0]["source"],
            0.05 * np.sin(2 * np.pi * (220 + index * 110) * np.arange(count) / samplerate),
            samplerate,
        )
        recording.update(sampling_rate=samplerate, num_samples=count)
        cut["supervisions"][0]["speaker"] = labels[index]
    with gzip.open(manifest, "wt") as stream:
        stream.writelines(json.dumps(cut) + "\n" for cut in cuts)
    validate_lhotse(CutSet.from_file(manifest))
    job.config.write_text(job.config.read_text() + f"\nsamplerate: {samplerate}\nhmm_params:\n  p_ind: [0,1,0,0]\n")
    recipe = request_recipe(job)
    assert validate(recipe)["runnable"]
    rows = build_stages(recipe)[0][1].process_batch(source_rows(recipe)[:1])
    assert len(rows) == 1
    row = rows[0]
    audio, rate = sf.read(row.data.audio_filepath)
    assert rate == samplerate
    with gzip.open(next((job.root / "production").glob("*/manifests/*supervisions.jsonl.gz")), "rt") as stream:
        supervisions = sorted((json.loads(line) for line in stream), key=lambda item: (item["start"], item["id"]))
    assert row.data.text == " ".join(item["text"] for item in supervisions)
    assert {item["speaker"] for item in supervisions} == set(labels)
    last = max(supervisions, key=lambda item: item["start"])
    early = audio[int((last["start"] + 0.01) * rate) : int((last["start"] + 0.25) * rate)]
    late = audio[int((last["start"] + 0.35) * rate) : int((last["start"] + 0.55) * rate)]
    peak = np.fft.rfftfreq(len(early), 1 / rate)[np.argmax(np.abs(np.fft.rfft(early)))]
    assert abs(peak - frequencies[last["speaker"]]) < 6
    assert np.max(np.abs(late)) > 1e-4
    lines = Path(row.data.rttm_filepath).read_text().splitlines()
    assert all(len(line.split()) == 10 for line in lines)
    output_manifest = next((job.root / "production").glob("*/nemo_manifest.json"))
    program = (
        "import json,runpy\n"
        f"reader=runpy.run_path({str(job.recipes / 'sim.py')!r})['nemo_manifest_to_cutset']\n"
        f"cuts=reader({str(output_manifest)!r})\n"
        "print(json.dumps(sorted({item.speaker for cut in cuts for item in cut.supervisions})))\n"
    )
    observed = subprocess.check_output(  # noqa: S603 - explicit Python and trusted installed recipe; no shell
        [sys.executable, "-c", program],
        cwd=job.engine.parent,
        env={**os.environ, "PYTHONPATH": str(job.engine.parent) + os.pathsep + os.environ.get("PYTHONPATH", "")},
        text=True,
        timeout=30,
    )
    parsed_speakers = json.loads(observed.splitlines()[-1])
    assert set(parsed_speakers) == set(labels)
    (job.root / "compatibility.json").write_text(
        json.dumps(
            {
                "sample_rate": rate,
                "expected_frequency": frequencies[last["speaker"]],
                "observed_frequency": float(peak),
                "late_turn_peak_amplitude": float(np.max(np.abs(late))),
                "parsed_speakers": parsed_speakers,
                "rttm_fields": 10,
                "reference_text_preserved": True,
            },
            indent=2,
        )
    )


@pytest.mark.parametrize("across_splits", [False, True])
@pytest.mark.parametrize("legacy_worker", [False, True])
def test_duplicate_cut_ids_across_the_whole_job_are_rejected(
    real_native_job: SimpleNamespace, across_splits: bool, legacy_worker: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lhotse import CutSet
    from lhotse import validate as validate_lhotse

    from nemo_curator.stages.audio.simulation.requests import _NativeInputs

    job = real_native_job
    manifest = job.inputs / "source_train.jsonl.gz"
    with gzip.open(manifest, "rt") as stream:
        cuts = [json.loads(line) for line in stream]
    for cut in cuts:
        cut["id"] = "shared-cut-id"
    partitions = {"train": cuts[:1], "dev": cuts[1:]} if across_splits else {"train": cuts}
    for split, items in partitions.items():
        path = job.inputs / f"source_{split}.jsonl.gz"
        with gzip.open(path, "wt") as stream:
            stream.writelines(json.dumps(cut) + "\n" for cut in items)
        if across_splits:
            validate_lhotse(CutSet.from_file(path))
    if across_splits:
        job.config.write_text(job.config.read_text().replace("dset_splits: [train]", "dset_splits: [train, dev]"))
    if legacy_worker:
        with monkeypatch.context() as legacy:
            legacy.setattr(_NativeInputs, "_admit_speech", lambda _self, _config: None)
            recipe = request_recipe(job)
        with pytest.raises(ValueError, match="Duplicate cut ID"):
            build_stages(recipe)[0][1].process(source_rows(recipe)[0])
    else:
        with pytest.raises(ValueError, match="Duplicate cut ID"):
            request_recipe(job)
        assert not (job.root / "requests.jsonl").exists()
        assert not (job.root / "worker-lock.json").exists()
    assert not (job.root / "production").exists()


@pytest.mark.parametrize("speaker", ["group/speaker", "group\\speaker", "bad\x00speaker", "x" * 201])
def test_per_speaker_export_rejects_unusable_filename_labels(real_native_job: SimpleNamespace, speaker: str) -> None:
    job = real_native_job
    manifest = job.inputs / "source_train.jsonl.gz"
    with gzip.open(manifest, "rt") as stream:
        cuts = [json.loads(line) for line in stream]
    cuts[0]["supervisions"][0]["speaker"] = speaker
    with gzip.open(manifest, "wt") as stream:
        stream.writelines(json.dumps(cut) + "\n" for cut in cuts)
    job.config.write_text(job.config.read_text() + "\nsave_spk: true\n")
    with pytest.raises(ValueError, match="save_spk"):
        request_recipe(job)
    for name in ["requests.jsonl", "worker-lock.json", "production"]:
        assert not (job.root / name).exists()


def test_overlapping_real_requests_ignore_tmpdir_inside_inputs(
    real_native_job: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time
    from concurrent.futures import ThreadPoolExecutor

    job = real_native_job
    recipe = request_recipe(job)
    tasks = source_rows(recipe)
    generator = build_stages(recipe)[0][1]
    generator.timeout_seconds = 30
    baseline = generator._identity()
    monkeypatch.setenv("TMPDIR", str(job.inputs))
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(generator.process, tasks[0])
        deadline = time.monotonic() + 15
        observed = []
        while time.monotonic() < deadline and not first.done():
            observed = [
                *job.inputs.glob("fastmss-config-*/*.yaml"),
                *(job.root / "production").glob("*/fastmss-config-*/*.yaml"),
            ]
            if observed:
                break
            time.sleep(0.01)
        assert observed, "Expected to observe the frozen config during native execution"
        second = pool.submit(generator.process, tasks[1])
        assert generator._identity() == baseline
        assert not list(job.inputs.glob("fastmss-config-*"))
        assert all(path.is_relative_to(job.root / "production") for path in observed)
        rows = [first.result(timeout=40), second.result(timeout=40)]
    assert len({row.data.fastmss_request_id for row in rows}) == 2
    assert generator._identity() == baseline


@pytest.mark.parametrize("save_spk", [False, True])
def test_unique_ids_across_splits_preserve_native_speaker_exports(
    real_native_job: SimpleNamespace, save_spk: bool
) -> None:
    import numpy as np
    import soundfile as sf

    job = real_native_job
    manifest = job.inputs / "source_train.jsonl.gz"
    with gzip.open(manifest, "rt") as stream:
        cuts = [json.loads(line) for line in stream]
    labels = ["é" * 99 + f"-{index}" if save_spk else f"group/speaker-{index}" for index in range(2)]
    for index, cut in enumerate(cuts):
        cut["supervisions"][0]["speaker"] = labels[index]
        with gzip.open(job.inputs / f"source_{['train', 'dev'][index]}.jsonl.gz", "wt") as stream:
            stream.write(json.dumps(cut) + "\n")
    job.config.write_text(
        job.config.read_text().replace("dset_splits: [train]", "dset_splits: [train, dev]")
        + f"\nsave_spk: {str(save_spk).lower()}\n"
    )
    recipe = request_recipe(job)
    rows = build_stages(recipe)[0][1].process_batch(source_rows(recipe)[:1])
    assert len(rows) == 1
    assert {line.split()[7] for line in Path(rows[0].data.rttm_filepath).read_text().splitlines()} == set(labels)
    tracks = list((job.root / "production").glob("*/audio/*-spk-*.wav"))
    assert len(tracks) == (2 if save_spk else 0)
    peaks = {}
    for index, speaker in enumerate(labels):
        if save_spk:
            track = next(path for path in tracks if path.name.endswith(f"-spk-{speaker}.wav"))
            audio, rate = sf.read(track)
            peak = np.fft.rfftfreq(len(audio), 1 / rate)[np.argmax(np.abs(np.fft.rfft(audio)))]
            assert abs(peak - (220 + index * 110)) < 6
            peaks[speaker] = float(peak)
    (job.root / "export-control.json").write_text(
        json.dumps({"save_spk": save_spk, "tracks": len(tracks), "peaks": peaks})
    )
