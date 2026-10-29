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

"""Configured persistence determines which outputs can become reuse artifacts."""

import builtins
import json
from pathlib import Path

import pytest

from nemo_curator.stages.audio.audio_agent import _verb_publish, artifacts, reuse
from nemo_curator.stages.audio.audio_agent.recipe import Recipe, StageRef, build_stages


@pytest.fixture
def sortformer_recipe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Recipe:
    monkeypatch.setenv("AUDIO_AGENT_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("AUDIO_AGENT_WORKSPACE", str(tmp_path))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name in ("mel_basis.npy", "learnable_sil_emb.npy", "engine.plan", "runtime.py"):
        (bundle / name).write_bytes(b"original model input")
    config = bundle / "config.json"
    config.write_text(json.dumps({"mel_basis": "mel_basis.npy", "learnable_sil_emb": "learnable_sil_emb.npy"}))
    other = tmp_path / "working-directory"
    other.mkdir()
    monkeypatch.chdir(other)
    return Recipe.from_dict(
        {
            "stages": [
                {"ref": "ManifestReader", "params": {"manifest_path": str(tmp_path / "input.jsonl")}},
                {
                    "ref": "InferenceSortformerStage",
                    "params": {
                        "adapter_target": "nemo_curator.models.audio.speaker_diarization.sortformer_tensorrt.TensorRTSortformerAdapter",
                        "adapter_kwargs": {
                            "engine_path": str(bundle / "engine.plan"),
                            "config_path": str(config),
                            "runtime_module_path": str(bundle / "runtime.py"),
                        },
                    },
                },
                {"ref": "ManifestCheckpointStage", "params": {"output_path": str(tmp_path / "checkpoint.jsonl")}},
            ]
        }
    ).freeze()


@pytest.mark.parametrize("asset", ["mel_basis", "learnable_sil_emb"])
@pytest.mark.parametrize("change", ["edit", "remove", "unreadable"])
@pytest.mark.parametrize("absolute", [False, True])
def test_sortformer_bundle_asset_changes_prevent_checkpoint_reuse(
    monkeypatch: pytest.MonkeyPatch, sortformer_recipe: Recipe, asset: str, change: str, absolute: bool
) -> None:
    recipe = sortformer_recipe
    config_path = Path(recipe.stages[1].params["adapter_kwargs"]["config_path"])
    config = json.loads(config_path.read_text())
    payload = config_path.parent / config[asset]
    if absolute:
        config[asset] = str(payload)
        config_path.write_text(json.dumps(config))
    before = artifacts.plan_steps(recipe, "stat:fixture")
    assert [p.step_key for p in before] == artifacts.step_keys(recipe, "stat:fixture")
    checkpoint = before[-1]
    Path(checkpoint.uri).write_text('{"diar_segments": []}\n')
    artifacts.publish(
        artifacts.Artifact(
            step_key=checkpoint.step_key,
            input_key=checkpoint.input_key,
            stage_ref=checkpoint.stage_ref,
            stage_index=checkpoint.index,
            semantic_params=checkpoint.semantic_params,
            uri=checkpoint.uri,
            kind=checkpoint.kind,
            dataset_key="stat:fixture",
            fingerprint_tier="stat",
            impl_version=checkpoint.impl_version,
            model_version=checkpoint.model_version,
            deterministic=checkpoint.deterministic,
        )
    )
    assert reuse.scan(recipe, dataset_key="stat:fixture")["decision"] == "already_done"

    if change == "edit":
        payload.write_bytes(b"modified model input")
    elif change == "remove":
        payload.unlink()
    else:
        original_open = builtins.open

        def deny_asset_read(file: object, *args: object, **kwargs: object) -> object:
            if file in (payload, str(payload)):
                message = "model asset is unreadable"
                raise PermissionError(message)
            return original_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", deny_asset_read)
    after = artifacts.plan_steps(recipe, "stat:fixture")

    assert before[0].step_key == after[0].step_key
    assert before[1].model_version != after[1].model_version
    assert all(a.step_key != b.step_key for a, b in zip(before[1:], after[1:], strict=True))
    assert reuse.scan(recipe, dataset_key="stat:fixture")["decision"] == "fresh"


@pytest.mark.parametrize("optional_value", [None, "omit"])
def test_sortformer_bundle_without_optional_silence_has_stable_identity(
    tmp_path: Path, sortformer_recipe: Recipe, optional_value: str | None
) -> None:
    config_path = tmp_path / "bundle/config.json"
    config = json.loads(config_path.read_text())
    if optional_value == "omit":
        config.pop("learnable_sil_emb")
    else:
        config["learnable_sil_emb"] = optional_value
    config_path.write_text(json.dumps(config))
    (config_path.parent / "learnable_sil_emb.npy").unlink()

    assert artifacts.step_keys(sortformer_recipe, "stat:fixture") == artifacts.step_keys(
        sortformer_recipe, "stat:fixture"
    )


@pytest.mark.parametrize(
    "content",
    [None, "{", "[]", "{}", '{"mel_basis": 1}', '{"mel_basis": "mel_basis.npy", "learnable_sil_emb": false}'],
)
def test_uninspectable_sortformer_config_cannot_match_an_earlier_identity(
    tmp_path: Path, sortformer_recipe: Recipe, content: str | None
) -> None:
    config_path = tmp_path / "bundle/config.json"
    if content is None:
        config_path.unlink()
    else:
        config_path.write_text(content)

    assert artifacts.step_keys(sortformer_recipe, "stat:fixture") != artifacts.step_keys(
        sortformer_recipe, "stat:fixture"
    )


@pytest.mark.parametrize("write_to_disk", [False, True])
def test_resampler_artifact_requires_enabled_disk_output(tmp_path: Path, write_to_disk: bool) -> None:
    output = tmp_path / "audio"
    output.mkdir()
    (output / "old.wav").write_bytes(b"old content")
    stage = StageRef(
        ref="ResampleAudioStage",
        params={"resampled_audio_dir": str(output), "write_to_disk": write_to_disk, "keep_waveform_in_task": True},
    )
    assert artifacts.output_uri(stage) == ((str(output), "audio_dir") if write_to_disk else ("", "unknown"))


def test_disabled_sink_cannot_publish_unrelated_existing_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AUDIO_AGENT_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("AUDIO_AGENT_WORKSPACE", str(tmp_path))
    output = tmp_path / "disabled"
    output.mkdir()
    (output / "old.wav").write_bytes(b"old content")
    recipe = Recipe.from_dict(
        {
            "stages": [
                {"ref": "ManifestReader", "params": {"manifest_path": str(tmp_path / "input.jsonl")}},
                {
                    "ref": "ResampleAudioStage",
                    "params": {
                        "resampled_audio_dir": str(output),
                        "write_to_disk": False,
                        "keep_waveform_in_task": True,
                    },
                },
            ]
        }
    ).freeze()
    stages, issues = build_stages(recipe)
    assert stages
    assert not issues
    assert not artifacts.plan_steps(recipe, "stat:fixture")[-1].persists()
    published = _verb_publish._publish_artifacts(
        recipe,
        stages,
        dataset_key="stat:fixture",
        fingerprint_tier="stat",
        per_stage={},
        run_id="fixture",
        input_count=1,
        data_profile=None,
        started_at="",
        ended_at="",
    )
    assert published == []
    assert not artifacts.list_artifacts()
    assert reuse.scan(recipe, dataset_key="stat:fixture")["decision"] != "already_done"


def test_dry_run_snippet_archive_is_not_an_artifact(tmp_path: Path) -> None:
    stage = StageRef(
        ref="SnippetExtractionStage",
        params={"output_dir": str(tmp_path), "output_audio_tar_path": str(tmp_path / "old.tar"), "dry_run": True},
    )
    assert artifacts.output_uri(stage) == ("", "unknown")


def test_uninspectable_stage_cannot_claim_an_artifact(tmp_path: Path) -> None:
    assert artifacts.output_uri(StageRef(ref="UnknownStage", params={"output_path": str(tmp_path / "old.jsonl")})) == (
        "",
        "unknown",
    )


@pytest.mark.parametrize("kind", ["file", "directory", "tokenizer"])
def test_local_model_bytes_invalidate_prefix_and_suffix(tmp_path: Path, kind: str) -> None:
    dependency = tmp_path / "model"
    if kind != "file":
        dependency.mkdir()
        payload = dependency / "weights.bin"
    else:
        payload = dependency
    payload.write_bytes(b"old weights")
    params = {"model_name": "remote/model", "tokenizer_path" if kind == "tokenizer" else "model_path": str(dependency)}
    recipe = Recipe.from_dict(
        {
            "stages": [
                {"ref": "ASRStage", "params": params},
                {"ref": "ManifestCheckpointStage", "params": {"output_path": str(tmp_path / "out.jsonl")}},
            ]
        }
    )
    before = artifacts.plan_steps(recipe, "dataset")
    payload.write_bytes(b"new weights")
    after = artifacts.plan_steps(recipe, "dataset")
    assert [p.step_key for p in before] != [p.step_key for p in after]
    assert all(a.step_key != b.step_key for a, b in zip(before, after, strict=True))


@pytest.fixture
def data_filter_recipe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Recipe, Path]:
    from nemo_curator.stages.audio.advanced_pipelines.audio_data_filter import config

    monkeypatch.setenv("AUDIO_AGENT_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("AUDIO_AGENT_WORKSPACE", str(tmp_path))
    defaults = tmp_path / "defaults.yaml"
    defaults.write_bytes(config._DEFAULT_CONFIG_PATH.read_bytes())
    monkeypatch.setattr(config, "_DEFAULT_CONFIG_PATH", defaults)
    recipe = Recipe.from_dict(
        {
            "stages": [
                {"ref": "ManifestReader", "params": {"manifest_path": str(tmp_path / "input.jsonl")}},
                {"ref": "AudioDataFilterStage", "params": {}},
                {"ref": "ManifestCheckpointStage", "params": {"output_path": str(tmp_path / "result.jsonl")}},
            ]
        }
    ).freeze()
    return recipe, defaults


@pytest.mark.parametrize("override", ["none", "config", "config_path"])
def test_changed_data_filter_defaults_invalidate_checkpoint_reuse(
    data_filter_recipe: tuple[Recipe, Path], override: str
) -> None:
    import yaml

    from nemo_curator.stages.audio.advanced_pipelines.audio_data_filter import AudioDataFilterStage

    recipe, defaults = data_filter_recipe
    if override == "config":
        recipe.stages[1].params["config"] = {"sigmos": {"enable": False}}
    elif override == "config_path":
        user_config = defaults.parent / "overrides.yaml"
        user_config.write_text("sigmos:\n  enable: false\n")
        recipe.stages[1].params["config_path"] = str(user_config)
    recipe.freeze()
    before = artifacts.plan_steps(recipe, "stat:fixture")
    assert [p.step_key for p in before] == artifacts.step_keys(recipe, "stat:fixture")
    checkpoint = before[-1]
    Path(checkpoint.uri).write_text('{"audio_filepath": "fixture.wav", "utmos_mos": 3.6}\n')
    artifacts.publish(
        artifacts.Artifact(
            step_key=checkpoint.step_key,
            input_key=checkpoint.input_key,
            stage_ref=checkpoint.stage_ref,
            stage_index=checkpoint.index,
            semantic_params=checkpoint.semantic_params,
            uri=checkpoint.uri,
            kind=checkpoint.kind,
            dataset_key="stat:fixture",
            fingerprint_tier="stat",
            impl_version=checkpoint.impl_version,
            model_version=checkpoint.model_version,
            deterministic=checkpoint.deterministic,
        )
    )
    assert reuse.scan(recipe, dataset_key="stat:fixture")["decision"] == "already_done"

    config = yaml.safe_load(defaults.read_text())
    config["utmos"]["mos_threshold"] = 3.8
    defaults.write_text(yaml.safe_dump(config))
    configured = AudioDataFilterStage(**recipe.stages[1].params)
    assert configured._cfg["utmos"]["mos_threshold"] == 3.8
    after = artifacts.plan_steps(recipe, "stat:fixture")

    assert before[0].step_key == after[0].step_key
    assert before[1].model_version != after[1].model_version
    assert all(a.step_key != b.step_key for a, b in zip(before[1:], after[1:], strict=True))
    assert reuse.scan(recipe, dataset_key="stat:fixture")["decision"] == "fresh"


@pytest.mark.parametrize("unavailable", ["missing", "unreadable"])
def test_unavailable_filter_defaults_cannot_match_reuse_identity(
    data_filter_recipe: tuple[Recipe, Path], monkeypatch: pytest.MonkeyPatch, unavailable: str
) -> None:
    recipe, defaults = data_filter_recipe
    if unavailable == "missing":
        defaults.unlink()
    else:
        original_open = builtins.open

        def deny_defaults_read(file: object, *args: object, **kwargs: object) -> object:
            if file in (defaults, str(defaults)):
                message = "filter defaults are unreadable"
                raise PermissionError(message)
            return original_open(file, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", deny_defaults_read)
    before = artifacts.plan_steps(recipe, "stat:fixture")
    after = artifacts.plan_steps(recipe, "stat:fixture")
    assert before[0].step_key == after[0].step_key
    assert before[1].model_version != after[1].model_version
    assert all(a.step_key != b.step_key for a, b in zip(before[1:], after[1:], strict=True))


def test_overridden_filter_default_does_not_invalidate_unchanged_computation(
    data_filter_recipe: tuple[Recipe, Path],
) -> None:
    import yaml

    recipe, defaults = data_filter_recipe
    recipe.stages[1].params["config"] = {"utmos": {"mos_threshold": 4.0}}
    before = artifacts.step_keys(recipe, "stat:fixture")
    config = yaml.safe_load(defaults.read_text())
    config["utmos"]["mos_threshold"] = 3.8
    defaults.write_text(yaml.safe_dump(config))
    assert artifacts.step_keys(recipe, "stat:fixture") == before


@pytest.mark.parametrize("location", ["defaults", "overrides"])
def test_filter_yaml_adapter_input_changes_invalidate_affected_suffix(
    data_filter_recipe: tuple[Recipe, Path], location: str
) -> None:
    import yaml

    from nemo_curator.stages.audio.advanced_pipelines.audio_data_filter import AudioDataFilterStage

    recipe, defaults = data_filter_recipe
    engine = defaults.parent / "silero.plan"
    engine.write_bytes(b"original engine")
    vad = {
        "adapter_target": "nemo_curator.models.audio.vad.silero_tensorrt.TensorRTSileroVADAdapter",
        "adapter_kwargs": {"engine_path": str(engine)},
    }
    if location == "defaults":
        config = yaml.safe_load(defaults.read_text())
        config["vad"].update(vad)
        defaults.write_text(yaml.safe_dump(config))
    else:
        overrides = defaults.parent / "overrides.yaml"
        overrides.write_text(yaml.safe_dump({"vad": vad}))
        recipe.stages[1].params["config_path"] = str(overrides)
    configured = AudioDataFilterStage(**recipe.stages[1].params)
    delegate = configured._make_vad(configured._cfg, suffix="", nested=False)
    assert delegate.adapter_kwargs["engine_path"] == str(engine)
    before = artifacts.plan_steps(recipe, "stat:fixture")
    assert [p.step_key for p in before] == artifacts.step_keys(recipe, "stat:fixture")
    engine.write_bytes(b"modified engine")
    after = artifacts.plan_steps(recipe, "stat:fixture")
    assert before[0].step_key == after[0].step_key
    assert all(a.step_key != b.step_key for a, b in zip(before[1:], after[1:], strict=True))


@pytest.mark.parametrize(
    ("parameter", "filename"),
    [("model_path", "missing.nemo"), ("engine_path", "missing.plan"), ("runtime_module_path", "missing_runtime.py")],
)
@pytest.mark.parametrize("relative", [False, True])
def test_unreadable_local_dependency_never_reuses_same_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, parameter: str, filename: str, relative: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    params = {"adapter_kwargs": {parameter: filename if relative else str(tmp_path / filename)}}
    assert artifacts.model_version(params) != artifacts.model_version(params)


@pytest.mark.parametrize(
    ("stage_ref", "adapter_target", "parameter"),
    [
        (
            "VADSegmentationStage",
            "nemo_curator.models.audio.vad.silero_tensorrt.TensorRTSileroVADAdapter",
            "engine_path",
        ),
        (
            "InferenceSortformerStage",
            "nemo_curator.models.audio.speaker_diarization.sortformer_tensorrt.TensorRTSortformerAdapter",
            "engine_path",
        ),
        (
            "InferenceSortformerStage",
            "nemo_curator.models.audio.speaker_diarization.sortformer_tensorrt.TensorRTSortformerAdapter",
            "runtime_module_path",
        ),
    ],
)
def test_replaced_adapter_files_invalidate_only_the_affected_suffix(
    tmp_path: Path, sortformer_recipe: Recipe, stage_ref: str, adapter_target: str, parameter: str
) -> None:
    dependency = tmp_path / "adapter-input.bin"
    dependency.write_bytes(b"first version")
    adapter_kwargs = {parameter: str(dependency)}
    if stage_ref == "InferenceSortformerStage":
        adapter_kwargs = {**sortformer_recipe.stages[1].params["adapter_kwargs"], **adapter_kwargs}
    recipe = Recipe.from_dict(
        {
            "stages": [
                {"ref": "ManifestReader", "params": {"manifest_path": str(tmp_path / "input.jsonl")}},
                {
                    "ref": stage_ref,
                    "params": {"adapter_target": adapter_target, "adapter_kwargs": adapter_kwargs},
                },
                {"ref": "ManifestCheckpointStage", "params": {"output_path": str(tmp_path / "out.jsonl")}},
            ]
        }
    )
    before = artifacts.plan_steps(recipe, "dataset")
    assert artifacts.step_keys(recipe, "dataset") == [step.step_key for step in before]

    dependency.write_bytes(b"other version")
    after = artifacts.plan_steps(recipe, "dataset")

    assert before[0].step_key == after[0].step_key
    assert before[1].model_version != after[1].model_version
    assert all(a.step_key != b.step_key for a, b in zip(before[1:], after[1:], strict=True))


def test_replaced_weights_change_identity_in_fresh_process(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    weights = tmp_path / "model.nemo"
    weights.write_bytes(b"first model")
    script = "from nemo_curator.stages.audio.audio_agent.artifacts import model_version; import sys; print(model_version({'model_path':sys.argv[1]}))"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(artifacts.__file__).parents[4]) + os.pathsep + env.get("PYTHONPATH", "")
    before = subprocess.check_output(  # noqa: S603 - fixed script and local fixture
        [sys.executable, "-c", script, str(weights)], env=env, cwd=tmp_path, text=True, timeout=30
    )
    weights.write_bytes(b"other model")
    after = subprocess.check_output(  # noqa: S603 - fixed script and local fixture
        [sys.executable, "-c", script, str(weights)], env=env, cwd=tmp_path, text=True, timeout=30
    )
    assert before != after


@pytest.mark.parametrize("internal_alias", [False, True])
def test_linked_tokenizer_payload_changes_reuse_identity(tmp_path: Path, internal_alias: bool) -> None:
    from transformers import AutoTokenizer

    root = tmp_path / "tokenizer"
    root.mkdir()
    shared = root / "shared-vocab" if internal_alias else tmp_path / "shared-vocab"
    shared.mkdir()
    (root / "vocab").symlink_to(shared, target_is_directory=True)
    payload = shared / "vocab.txt"
    (root / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "BertTokenizer", "vocab_file": str(root / "vocab/vocab.txt")}),
        encoding="utf-8",
    )
    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    payload.write_text("\n".join([*special, "hello"]) + "\n", encoding="utf-8")
    first = AutoTokenizer.from_pretrained(str(root), local_files_only=True)
    params = {"tokenizer_path": str(root)}
    before = artifacts.model_version(params)
    assert before == artifacts.model_version(params)
    payload.write_text("\n".join([*special, "world"]) + "\n", encoding="utf-8")
    second = AutoTokenizer.from_pretrained(str(root), local_files_only=True)
    assert first.is_fast
    assert second.is_fast
    assert first.encode("hello") != second.encode("hello")
    assert before != artifacts.model_version(params)


@pytest.mark.parametrize("indirect", [False, True])
def test_linked_directory_cycles_refuse_reuse(tmp_path: Path, indirect: bool) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"weights")
    if indirect:
        other = model / "nested"
        other.mkdir()
        (other / "cycle").symlink_to(model, target_is_directory=True)
    else:
        (model / "cycle").symlink_to(model, target_is_directory=True)
    assert artifacts.content_digest(str(model)) is None
    params = {"model_path": str(model)}
    assert artifacts.model_version(params) != artifacts.model_version(params)


def test_required_sibling_proof_is_not_optional_for_legacy_records(tmp_path: Path) -> None:
    recipe = Recipe.from_dict(
        {
            "stages": [
                {"ref": "ManifestReader", "params": {"manifest_path": str(tmp_path / "source.jsonl")}},
                {"ref": "ManifestWriterStage", "params": {"output_path": str(tmp_path / "data.jsonl")}},
            ]
        }
    )
    step = artifacts.plan_steps(recipe, "data")[-1]
    audit = str(tmp_path / "audit.jsonl")
    step.required_outputs = (step.uri, audit)
    artifact = artifacts.Artifact(step_key=step.step_key, uri=step.uri)
    assert artifacts.plan_invalid_reasons(step, artifact) == [f"required output has no dependency proof: {audit!r}"]
    artifact.dependencies[audit] = "digest-checked-by-invalid-reasons"
    assert artifacts.plan_invalid_reasons(step, artifact) == []
