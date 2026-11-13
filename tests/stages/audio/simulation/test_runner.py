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

"""Verify transcript assembly and unsupported upstream configurations."""

import gzip
import json
import runpy
import sys
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_curator.stages.audio.simulation import _runner
from nemo_curator.stages.audio.simulation._runner import enrich_manifest, validate_config


@pytest.mark.parametrize(
    "config",
    [
        {"min_max_spk": [1, 1]},
        {"min_max_spk": [1, 4]},
        {"min_max_spk": [3, 2]},
        {"save_anechoic": True, "save_spk": True, "reverberate": True},
        {"save_anechoic": True, "save_spk": True, "reverberate": True, "reverb_prob": 0},
        {"save_anechoic": True, "reverberate": False, "reverb_prob": 1},
    ],
)
def test_unsafe_engine_settings_are_rejected(config: dict) -> None:
    with pytest.raises(ValueError, match="FastMSS"):
        validate_config(config)


@pytest.mark.parametrize(
    "config",
    [
        {"min_max_spk": [2, 2]},
        {"save_anechoic": False, "reverberate": False},
        {"save_anechoic": True, "save_spk": True, "reverberate": True, "reverb_prob": 1},
    ],
)
def test_supported_settings_are_accepted(config: dict) -> None:
    validate_config(config)


@pytest.mark.parametrize("missing_text", [False, True])
def test_transcripts_use_recording_identity_and_start_order(tmp_path: Path, missing_text: bool) -> None:
    manifest = tmp_path / "nemo_manifest.json"
    rows = [
        {"audio_filepath": "/audio/renamed.wav", "duration": 4, "custom": "preserved"},
        {"audio_filepath": "/audio/other.wav", "text": "existing transcript"},
    ]
    original = "".join(json.dumps(row) + "\n" for row in rows)
    manifest.write_text(original)
    directory = tmp_path / "manifests"
    directory.mkdir()
    with gzip.open(directory / "synth-source-train-recordings.jsonl.gz", "wt") as output:
        output.write(json.dumps({"id": "meeting", "sources": [{"source": "/audio/renamed.wav"}]}))
    segments = [
        {"recording_id": "meeting", "id": "c", "start": 1, "duration": 2, "text": "later"},
        {"recording_id": "meeting", "id": "b", "start": 0, "duration": 2, "text": "overlap"},
        {"recording_id": "unrelated", "id": "x", "start": 0, "text": "exclude"},
        {"recording_id": "meeting", "id": "a", "start": 0, "duration": 2, "text": " привет "},
    ]
    if missing_text:
        segments[0]["text"] = None
    with gzip.open(directory / "synth-source-train-supervisions.jsonl.gz", "wt", encoding="utf-8") as output:
        output.write("".join(json.dumps(segment) + "\n" for segment in segments))
    if missing_text:
        with pytest.raises(ValueError, match="missing supervision transcripts"):
            enrich_manifest(tmp_path)
        assert manifest.read_text() == original
    else:
        enrich_manifest(tmp_path)
        actual = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
        assert actual == [{**rows[0], "text": "привет overlap later"}, rows[1]]


def test_missing_supervisions_fail_instead_of_emitting_unlabelled_meetings(tmp_path: Path) -> None:
    (tmp_path / "nemo_manifest.json").write_text(json.dumps({"audio_filepath": "/audio/meeting.wav"}))
    with pytest.raises(ValueError, match="missing supervision transcripts"):
        enrich_manifest(tmp_path)


@pytest.fixture
def runner_recipe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Use real Hydra composition and script execution without simulator dependencies."""
    root = tmp_path / "installed toolkit"
    recipes = root / "recipes"
    recipes.mkdir(parents=True)
    (recipes / "default.yaml").write_text("n_meetings: 7\nmin_max_spk: [2, 2]\noutput_dir: null\n")
    script = recipes / "sim.py"
    script.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "from hydra import compose, initialize_config_dir\n"
        "from omegaconf import OmegaConf\n"
        "args = dict(arg.split('=', 1) for arg in sys.argv[1:] if arg.startswith('--'))\n"
        "with initialize_config_dir(config_dir=args['--config-path'], version_base=None):\n"
        "    cfg = compose(config_name=args['--config-name'], overrides=[arg for arg in sys.argv[1:] if not arg.startswith('--')])\n"
        "    observed = OmegaConf.to_container(cfg, resolve=True)\n"
        "output = Path(cfg.output_dir)\n"
        "output.mkdir()\n"
        "observed.update(script=sys.argv[0], import_dir=sys.path[0], root=os.environ['FASTMSS_ROOT'])\n"
        "(output / 'observed.json').write_text(json.dumps(observed))\n"
        "(output / 'nemo_manifest.json').write_text(json.dumps({'text': 'existing reference', 'custom': 'preserved'}) + '\\n')\n"
    )
    configs = tmp_path / "user configs"
    configs.mkdir()
    output = tmp_path / "generated meetings"
    config = configs / "meeting.yaml"
    config.write_text(
        "defaults: [default, _self_]\n"
        "hydra:\n  searchpath: ['file://${oc.env:FASTMSS_ROOT}/recipes']\n"
        f"output_dir: {json.dumps(str(output))}\n"
        f"data_dir: {json.dumps(str(tmp_path / 'protected data'))}\n"
        f"rir_dir: {json.dumps(str(tmp_path / 'protected rirs'))}\n"
    )
    package = SimpleNamespace(files=["recipes/sim.py", "recipes/default.yaml"], locate_file=lambda path: root / path)
    original_distribution = metadata.distribution
    monkeypatch.setattr(
        metadata, "distribution", lambda name: package if name == "fastmss" else original_distribution(name)
    )
    monkeypatch.setenv("FASTMSS_ROOT", str(root))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", ["_runner.py", str(script), f"--config-path={configs}", "--config-name=meeting"])
    return SimpleNamespace(root=root, script=script, config=config, output=output)


@pytest.mark.parametrize(
    ("installed", "isolate", "cap"), [(False, False, -1), (False, True, 2), (True, False, -1), (True, True, 2)]
)
def test_runner_executes_selected_recipe_with_effective_defaults_and_limits(
    runner_recipe: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, installed: bool, isolate: bool, cap: int
) -> None:
    original = runner_recipe.config.read_bytes()
    if installed:
        sys.argv[1] = "recipes/sim.py"
        sys.argv.append("--installed-fastmss")
        monkeypatch.setenv("FASTMSS_ROOT", "/obsolete-installation")
    if isolate:
        sys.argv.append("--isolate-work-dirs")
    sys.argv.append(f"--max-meetings={cap}")
    _runner.main()
    observed = json.loads((runner_recipe.output / "observed.json").read_text())
    assert observed["n_meetings"] == (7 if cap == -1 else cap)
    assert observed["min_max_spk"] == [2, 2]
    assert observed["root"] == str(runner_recipe.root)
    assert observed["script"] == str(runner_recipe.script)
    assert observed["import_dir"] == str(runner_recipe.script.parent)
    assert Path(observed["data_dir"]) == (
        runner_recipe.output / "data" if isolate else runner_recipe.config.parent.parent / "protected data"
    )
    assert Path(observed["rir_dir"]) == (
        runner_recipe.output / "rir" if isolate else runner_recipe.config.parent.parent / "protected rirs"
    )
    assert json.loads((runner_recipe.output / "nemo_manifest.json").read_text()) == {
        "text": "existing reference",
        "custom": "preserved",
    }
    assert runner_recipe.config.read_bytes() == original
    assert not (runner_recipe.config.parent.parent / "protected data").exists()
    assert not (runner_recipe.config.parent.parent / "protected rirs").exists()


def test_runner_script_entrypoint_executes_recipe(runner_recipe: SimpleNamespace) -> None:
    runpy.run_path(_runner.__file__, run_name="__main__")
    assert json.loads((runner_recipe.output / "observed.json").read_text())["n_meetings"] == 7


@pytest.mark.parametrize("cap", [0, -2])
def test_runner_rejects_invalid_cap_before_executing_recipe(runner_recipe: SimpleNamespace, cap: int) -> None:
    sys.argv.append(f"--max-meetings={cap}")
    with pytest.raises(ValueError, match="max_meetings must be positive"):
        _runner.main()
    assert not runner_recipe.output.exists()


@pytest.mark.parametrize("count", ["0", "-1", "true", "2.5"])
def test_runner_rejects_invalid_meeting_count_before_execution(runner_recipe: SimpleNamespace, count: str) -> None:
    sys.argv.extend(["--max-meetings=1", f"n_meetings={count}"])
    with pytest.raises(ValueError, match="n_meetings must be a positive integer"):
        _runner.main()
    assert not runner_recipe.output.exists()


def test_runner_reports_missing_worker_distribution(
    runner_recipe: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(name: str) -> None:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "distribution", missing)
    sys.argv.append("--installed-fastmss")
    with pytest.raises(RuntimeError, match="install nemo-curator") as error:
        _runner.main()
    assert isinstance(error.value.__cause__, metadata.PackageNotFoundError)
    assert not runner_recipe.output.exists()


@pytest.mark.parametrize("files", [None, ["recipes/another.py"]])
def test_runner_rejects_distribution_without_requested_recipe(
    runner_recipe: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, files: list[str] | None
) -> None:
    monkeypatch.setattr(metadata, "distribution", lambda _name: SimpleNamespace(files=files))
    sys.argv[1] = "recipes/sim.py"
    sys.argv.append("--installed-fastmss")
    with pytest.raises(FileNotFoundError, match="does not provide the recipe"):
        _runner.main()
    assert not runner_recipe.output.exists()


@pytest.mark.parametrize("empty", [False, True])
def test_enrichment_leaves_missing_or_empty_manifests_for_parent_diagnostics(tmp_path: Path, empty: bool) -> None:
    manifest = tmp_path / "nemo_manifest.json"
    if empty:
        manifest.touch()
    enrich_manifest(tmp_path)
    assert manifest.exists() == empty
    if empty:
        assert manifest.read_bytes() == b""


@pytest.mark.parametrize("override", ["n_meetings=8", "duration=5"])
def test_planned_native_bounds_are_verified_before_recipe_execution(
    runner_recipe: SimpleNamespace, override: str
) -> None:
    sys.argv.extend(
        [
            "--expected-meetings=7",
            "--expected-duration=2",
            "++duration=2",
            override if override.startswith("n_meetings") else "++duration=5",
        ]
    )
    with pytest.raises(ValueError, match="changed since request planning"):
        _runner.main()
    assert not runner_recipe.output.exists()
