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

"""Fail-closed smoke bounds that do not require Ray or audio dependencies."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from nemo_curator.audio_agent import calibration_store, verbs
from nemo_curator.audio_agent.contracts import SmokeReport
from nemo_curator.audio_agent.recipe import Recipe
from nemo_curator.stages.audio.common import ManifestCheckpointStage
from nemo_curator.tasks import AudioTask, DocumentBatch
from nemo_curator.utils.performance_utils import StagePerfStats


def _source_recipe(ref: str, params: dict[str, Any]) -> dict[str, Any]:
    return {"stages": [{"ref": ref, "params": params}]}


def _stub_smoke_runtime(
    monkeypatch: pytest.MonkeyPatch,
    inspect_source,  # noqa: ANN001
    inspect_stages=None,  # noqa: ANN001
) -> None:
    """Keep the test at the verb boundary while replacing only execution."""
    monkeypatch.delenv("AUDIO_AGENT_WORKSPACE", raising=False)
    monkeypatch.setattr(verbs, "_profile_binding", lambda _binding: None)
    monkeypatch.setattr(verbs, "probe_env", lambda: object())
    monkeypatch.setattr(
        verbs,
        "_plan_resources",
        lambda *_args, **_kwargs: SimpleNamespace(
            mode="batch",
            escalations=[],
            machine_fingerprint="smoke-test-machine",
        ),
    )

    def capture(stages, _mode, _executor, **_kwargs):  # noqa: ANN001, ANN202
        inspect_source(stages[0])
        if inspect_stages is not None:
            inspect_stages(stages)
        return [], "batch"

    monkeypatch.setattr(verbs, "_run_pipeline_autofallback", capture)


@pytest.mark.parametrize("sample", [0, -1, True, 1.5, "2"])
def test_smoke_rejects_non_positive_or_non_integer_sample(sample: Any) -> None:  # noqa: ANN401
    result = verbs.smoke(
        _source_recipe("ManifestReader", {"manifest_path": "/does/not/matter.jsonl"}),
        sample=sample,
    )

    assert result["status"] == "refused"
    assert "positive integer" in result["reason"]


def test_manifest_directory_is_concatenated_and_capped_in_execution_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "b.jsonl").write_text(
        '{"audio_filepath":"b1.wav"}\n{"audio_filepath":"b2.wav"}\n',
        encoding="utf-8",
    )
    (manifests / "a.jsonl").write_text(
        '{"audio_filepath":"a1.wav"}\n{"audio_filepath":"a2.wav"}\n',
        encoding="utf-8",
    )
    captured: list[dict[str, Any]] = []

    def inspect_source(source) -> None:  # noqa: ANN001
        assert source.__class__.__name__ == "ManifestReader"
        with open(source.manifest_path, encoding="utf-8") as bounded:
            captured.extend(json.loads(line) for line in bounded)

    _stub_smoke_runtime(monkeypatch, inspect_source)
    result = verbs.smoke(
        _source_recipe("ManifestReader", {"manifest_path": str(manifests)}),
        sample=3,
    )

    assert result["ran"] is True
    assert result["input_count"] == 3
    assert [row["audio_filepath"] for row in captured] == ["a1.wav", "a2.wav", "b1.wav"]
    assert any("resolved local manifest" in note for note in result["notes"])


def test_remote_manifest_selector_refuses_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        verbs,
        "_run_pipeline_autofallback",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("remote selector reached execution")),
    )

    result = verbs.smoke(
        _source_recipe(
            "ManifestReader",
            {"manifest_path": "s3://example-bucket/manifests/*.jsonl"},
        ),
        sample=2,
    )

    assert result["status"] == "refused"
    assert "remote or mixed selectors" in result["reason"]


def test_malformed_multi_manifest_refuses_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bad = tmp_path / "a.jsonl"
    good = tmp_path / "b.jsonl"
    bad.write_text("{bad json}\n", encoding="utf-8")
    good.write_text('{"audio_filepath":"ok.wav"}\n', encoding="utf-8")
    monkeypatch.setattr(verbs, "_profile_binding", lambda _binding: None)
    monkeypatch.setattr(
        verbs,
        "_run_pipeline_autofallback",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("malformed selector reached execution")),
    )

    result = verbs.smoke(
        _source_recipe(
            "ManifestReader",
            {"manifest_path": [str(bad), str(good)]},
        ),
        sample=2,
    )

    assert result["status"] == "refused"
    assert "invalid JSON" in result["reason"]


@pytest.mark.parametrize(
    ("ref", "params"),
    [
        (
            "CreateInitialManifestFleursStage",
            {
                "lang": "en_us",
                "split": "test",
                "raw_data_dir": "{root}",
                "auto_download": True,
            },
        ),
        (
            "CreateInitialManifestReadSpeechStage",
            {
                "raw_data_dir": "{root}",
                "auto_download": True,
            },
        ),
    ],
)
def test_unstaged_download_sources_refuse_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    ref: str,
    params: dict[str, Any],
) -> None:
    root = tmp_path / "unstaged"
    authored = {key: str(root) if value == "{root}" else value for key, value in params.items()}
    monkeypatch.setattr(
        verbs,
        "_run_pipeline_autofallback",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("download-capable source reached execution")),
    )

    result = verbs.smoke(_source_recipe(ref, authored), sample=2)

    assert result["status"] == "refused"
    assert "pre-stage" in result["reason"]


def test_prestaged_fleurs_is_adapted_to_a_bounded_local_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    language_dir = tmp_path / "fleurs" / "en_us"
    audio_root = language_dir / "test"
    audio_root.mkdir(parents=True)
    (language_dir / "test.tsv").write_text(
        "0\ta.wav\tfirst\n1\tb.wav\tsecond\n2\tc.wav\tthird\n",
        encoding="utf-8",
    )
    captured: list[dict[str, Any]] = []

    def inspect_source(source) -> None:  # noqa: ANN001
        assert source.__class__.__name__ == "ManifestReader"
        with open(source.manifest_path, encoding="utf-8") as bounded:
            captured.extend(json.loads(line) for line in bounded)

    _stub_smoke_runtime(monkeypatch, inspect_source)
    result = verbs.smoke(
        _source_recipe(
            "CreateInitialManifestFleursStage",
            {
                "lang": "en_us",
                "split": "test",
                "raw_data_dir": str(tmp_path / "fleurs"),
                "filepath_key": "path",
                "text_key": "transcript",
            },
        ),
        sample=2,
    )

    assert result["ran"] is True
    assert result["input_count"] == 2
    assert captured == [
        {"path": str(audio_root / "a.wav"), "transcript": "first"},
        {"path": str(audio_root / "b.wav"), "transcript": "second"},
    ]


@pytest.mark.parametrize(
    ("ref", "params"),
    [
        (
            "CreateInitialManifestAudioFolderStage",
            {"data_dir": "{root}"},
        ),
        (
            "CreateInitialManifestReadSpeechStage",
            {"raw_data_dir": "{root}", "auto_download": False},
        ),
    ],
)
def test_local_folder_sources_receive_an_ephemeral_sample_cap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    ref: str,
    params: dict[str, Any],
) -> None:
    root = tmp_path / "audio"
    root.mkdir()
    (root / "clip.wav").write_bytes(b"RIFF")
    authored = {key: str(root) if value == "{root}" else value for key, value in params.items()}
    seen: list[int] = []

    def inspect_source(source) -> None:  # noqa: ANN001
        seen.append(source.max_samples)

    _stub_smoke_runtime(monkeypatch, inspect_source)
    result = verbs.smoke(_source_recipe(ref, authored), sample=1)

    assert result["ran"] is True
    assert seen == [1]
    assert params.get("max_samples") is None


@pytest.mark.parametrize("raise_after_setup", [False, True])
def test_smoke_never_truncates_the_production_manifest_and_cleans_its_sandbox(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raise_after_setup: bool,
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text('{"audio_filepath":"clip.wav"}\n', encoding="utf-8")
    production = tmp_path / "production.jsonl"
    production.write_text("KEEP-ME\n", encoding="utf-8")
    recipe = {
        "stages": [
            {"ref": "ManifestReader", "params": {"manifest_path": str(source)}},
            {
                "ref": "ManifestWriterStage",
                "params": {"output_path": str(production)},
            },
        ]
    }
    authored = json.loads(json.dumps(recipe))
    sandbox_outputs: list[str] = []

    monkeypatch.setattr(verbs, "_profile_binding", lambda _binding: None)
    monkeypatch.setattr(verbs, "probe_env", lambda: object())
    monkeypatch.setattr(
        verbs,
        "_plan_resources",
        lambda *_args, **_kwargs: SimpleNamespace(
            mode="batch",
            escalations=[],
            machine_fingerprint="smoke-test-machine",
        ),
    )

    def exercise_lifecycle(stages, _mode, _executor, **_kwargs):  # noqa: ANN001, ANN202
        writer = stages[-1]
        sandbox_outputs.append(writer.output_path)
        assert writer.output_path != str(production)
        writer.setup_on_node()
        writer.setup()
        if raise_after_setup:
            raise RuntimeError("injected failure after writer setup")  # noqa: EM101
        return [], "batch"

    monkeypatch.setattr(verbs, "_run_pipeline_autofallback", exercise_lifecycle)
    result = verbs.smoke(recipe, sample=1)

    assert production.read_text(encoding="utf-8") == "KEEP-ME\n"
    assert recipe == authored
    assert sandbox_outputs
    assert all(not os.path.exists(path) for path in sandbox_outputs)
    assert result["ran"] is (not raise_after_setup)
    assert "smoke_token" not in result


def test_remote_writer_destination_is_not_touched_by_smoke(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import fsspec

    source = tmp_path / "source.jsonl"
    source.write_text('{"audio_filepath":"clip.wav"}\n', encoding="utf-8")
    production = "memory://audio-agent-smoke/production.jsonl"
    fs, remote_path = fsspec.core.url_to_fs(production)
    with fs.open(remote_path, "w", encoding="utf-8") as output:
        output.write("KEEP-REMOTE\n")

    _stub_smoke_runtime(
        monkeypatch,
        lambda _source: None,
    )
    result = verbs.smoke(
        {
            "stages": [
                {
                    "ref": "ManifestReader",
                    "params": {"manifest_path": str(source)},
                },
                {
                    "ref": "ManifestWriterStage",
                    "params": {"output_path": production},
                },
            ]
        },
        sample=1,
    )

    with fs.open(remote_path, encoding="utf-8") as output:
        assert output.read() == b"KEEP-REMOTE\n"
    assert result["ran"] is True


def test_output_isolation_covers_aliases_implicit_paths_and_hidden_writes(
    tmp_path: Path,
) -> None:
    shared = str(tmp_path / "shared.jsonl")
    authored = {
        "stages": [
            {"ref": "ManifestWriterStage", "params": {"output_path": shared}},
            {"ref": "SnippetManifestWriterStage", "params": {"output_path": shared}},
            {
                "ref": "SnippetExtractionStage",
                "params": {
                    "output_dir": str(tmp_path / "snippets"),
                    "output_audio_tar_path": str(tmp_path / "audio.tar"),
                },
            },
            {
                "ref": "MonoConversionStage",
                "params": {"write_to_disk": True},
            },
            {
                "ref": "PyAnnoteDiarizationStage",
                "params": {"hf_token": "secret"},
            },
            {
                "ref": "CreateInitialManifestReadSpeechStage",
                "params": {
                    "raw_data_dir": str(tmp_path / "readspeech"),
                    "auto_download": True,
                },
            },
            {"ref": "SplitASRAlignJoinStage", "params": {}},
        ]
    }
    recipe = Recipe.from_dict(authored)
    report = SmokeReport(sample=1)

    isolated = verbs._isolate_smoke_outputs(
        verbs._SmokeBound(recipe),
        report,
    )

    assert isolated.recipe is not None
    root = str(isolated.output_root)
    stages = isolated.recipe.stages
    assert stages[0].params["output_path"] == stages[1].params["output_path"]
    assert stages[2].params["output_audio_tar_path"].endswith(".tar")
    assert verbs._inside_smoke_root(stages[2].params["output_dir"], root)
    assert verbs._inside_smoke_root(stages[3].params["output_dir"], root)
    assert stages[4].params["write_rttm"] is False
    assert stages[5].params["raw_data_dir"] == str(tmp_path / "readspeech")
    assert stages[5].params["auto_download"] is False
    assert verbs._inside_smoke_root(stages[6].params["output_dir"], root)
    verbs._cleanup(list(isolated.tmp_paths))
    assert not os.path.exists(root)


@pytest.mark.parametrize("split_ref", ["SplitLongAudioStage", "SplitASRAlignJoinStage"])
def test_direct_and_composite_split_writers_are_isolated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    split_ref: str,
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        '{"audio_filepath":"clip.wav","duration":7200,"segments":[]}\n',
        encoding="utf-8",
    )
    captured: list[str] = []

    def inspect_stages(stages) -> None:  # noqa: ANN001
        captured.append(stages[1].output_dir)

    _stub_smoke_runtime(
        monkeypatch,
        lambda _source: None,
        inspect_stages,
    )
    result = verbs.smoke(
        {
            "stages": [
                {
                    "ref": "ManifestReader",
                    "params": {"manifest_path": str(source)},
                },
                {"ref": split_ref, "params": {}},
            ]
        },
        sample=1,
    )

    assert result["ran"] is True
    assert captured and all(not os.path.exists(path) for path in captured)  # noqa: PT018


def test_unknown_disk_writer_fails_the_future_stage_guard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from nemo_curator.stages.audio import agent as foundation
    from nemo_curator.stages.audio._agent._agent_ready import Gates, StageContract

    # A stage that claims writes_to_disk without naming WHERE cannot be sandboxed. Guessing
    # which of its params look path-like would risk a smoke writing into the caller's real
    # output tree, so an undeclared writer must refuse -- this is the property that had to
    # survive moving the declaration from a central table onto the stage itself.
    monkeypatch.setattr(
        foundation,
        "build_contract",
        lambda _stage: StageContract(gates=Gates(writes_to_disk=True)),
    )

    issues = verbs._smoke_write_issues([object()], str(tmp_path))

    assert len(issues) == 1
    assert "does not declare output_path_params" in issues[0]


def test_an_empty_declaration_is_accepted_from_the_stage_it_is_true_of(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``[]`` is a positive claim and must stay distinguishable from "never declared".

    ``CreateInitialManifestReadSpeechStage`` writes to disk through no redirectable path
    parameter, so an empty declaration is the honest answer -- and collapsing it into the
    undeclared case would refuse a stage that is in fact fine.
    """
    from nemo_curator.stages.audio import agent as foundation
    from nemo_curator.stages.audio._agent._agent_ready import Gates, StageContract

    class CreateInitialManifestReadSpeechStage:
        auto_download = False

    monkeypatch.setattr(
        foundation,
        "build_contract",
        lambda _stage: StageContract(gates=Gates(writes_to_disk=True, output_path_params=[])),
    )

    assert verbs._smoke_write_issues([CreateInitialManifestReadSpeechStage()], str(tmp_path)) == []


def test_an_empty_declaration_from_any_other_writer_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Accepted from anyone, ``[]`` is the easiest way past this check entirely.

    A writer with a hardcoded or derived destination declares an empty list, the redirect
    loop has nothing to iterate, and the smoke is pronounced isolated while the stage writes
    into the caller's real output tree. Isolation that cannot be proven fails closed, exactly
    as it does for a writer that never declared at all -- so the exemption is a name, not a
    shape anyone can adopt.
    """
    from nemo_curator.stages.audio import agent as foundation
    from nemo_curator.stages.audio._agent._agent_ready import Gates, StageContract

    monkeypatch.setattr(
        foundation,
        "build_contract",
        lambda _stage: StageContract(gates=Gates(writes_to_disk=True, output_path_params=[])),
    )

    issues = verbs._smoke_write_issues([object()], str(tmp_path))

    assert len(issues) == 1
    assert "empty output_path_params" in issues[0]


def test_smoke_token_is_issued_only_after_sampled_goals_are_met(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text('{"audio_filepath":"clip.wav"}\n', encoding="utf-8")

    monkeypatch.setattr(verbs, "_profile_binding", lambda _binding: None)
    monkeypatch.setattr(verbs, "probe_env", lambda: object())
    monkeypatch.setattr(
        verbs,
        "_plan_resources",
        lambda *_args, **_kwargs: SimpleNamespace(
            mode="batch",
            escalations=[],
            machine_fingerprint="smoke-test-machine",
        ),
    )
    monkeypatch.setattr(
        verbs,
        "_run_pipeline_autofallback",
        lambda *_args, **_kwargs: (
            [
                SimpleNamespace(
                    data={"audio_filepath": "clip.wav"},
                    num_items=1,
                    _stage_perf=[],
                )
            ],
            "batch",
        ),
    )

    result = verbs.smoke(
        _source_recipe("ManifestReader", {"manifest_path": str(source)}),
        sample=1,
    )

    assert result["goals_met"] is True
    assert result["smoke_token"]
    assert "smoke_token_status" not in result


def test_a_smokes_measurements_are_waiting_for_the_next_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The measurements have to reach the planner without the caller passing them back.

    The smoke token cannot carry them -- it is an HMAC over the config hash, not an address --
    so smoke stores them under that same hash and run picks them up.
    """
    source = tmp_path / "source.jsonl"
    source.write_text('{"audio_filepath":"clip.wav"}\n', encoding="utf-8")
    monkeypatch.setenv("AUDIO_AGENT_RUNS_DIR", str(tmp_path / "runs"))

    monkeypatch.setattr(verbs, "_profile_binding", lambda _binding: None)
    monkeypatch.setattr(verbs, "probe_env", lambda: object())
    monkeypatch.setattr(
        verbs,
        "_plan_resources",
        lambda *_args, **_kwargs: SimpleNamespace(
            mode="batch",
            escalations=[],
            machine_fingerprint="smoke-test-machine",
        ),
    )
    monkeypatch.setattr(
        verbs,
        "_run_pipeline_autofallback",
        lambda *_args, **_kwargs: (
            [
                SimpleNamespace(
                    data={"audio_filepath": "clip.wav"},
                    num_items=1,
                    _stage_perf=[
                        StagePerfStats(
                            stage_name="ManifestReader",
                            custom_metrics={"peak_host_mem_gb": 9.0},
                        )
                    ],
                )
            ],
            "batch",
        ),
    )

    result = verbs.smoke(
        _source_recipe("ManifestReader", {"manifest_path": str(source)}),
        sample=1,
    )

    assert result["calibration"]["ManifestReader"]["host_mem_gb"] == 9.0
    assert result["calibration_stored"] is True

    resolved, note = verbs._calibration_for_run(None, result["config_hash"])

    assert resolved["calibration"] == result["calibration"]
    assert resolved["machine_fingerprint"] == "smoke-test-machine"
    assert "none passed" in note
    assert calibration_store.load("a-different-recipe") is None


def test_document_batch_required_output_is_checked_beyond_preview_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(json.dumps({"audio_filepath": f"clip-{i}.wav"}) + "\n" for i in range(4)),
        encoding="utf-8",
    )
    batch = DocumentBatch(
        dataset_name="smoke",
        data=pd.DataFrame(
            [
                {"audio_filepath": "clip-0.wav", "text": "one"},
                {"audio_filepath": "clip-1.wav", "text": "two"},
                {"audio_filepath": "clip-2.wav", "text": "three"},
                # DataFrame represents the absent field as NaN. This fourth
                # row is intentionally outside the three-row report preview.
                {"audio_filepath": "clip-3.wav"},
            ]
        ),
    )

    monkeypatch.setattr(verbs, "_profile_binding", lambda _binding: None)
    monkeypatch.setattr(verbs, "probe_env", lambda: object())
    monkeypatch.setattr(
        verbs,
        "_plan_resources",
        lambda *_args, **_kwargs: SimpleNamespace(
            mode="batch",
            feasible=True,
            escalations=[],
            machine_fingerprint="smoke-test-machine",
        ),
    )
    monkeypatch.setattr(
        verbs,
        "_run_pipeline_autofallback",
        lambda *_args, **_kwargs: ([batch], "batch"),
    )

    result = verbs.smoke(
        {
            "stages": [
                {
                    "ref": "ManifestReader",
                    "params": {"manifest_path": str(source)},
                },
                {"ref": "AudioToDocumentStage", "params": {}},
            ],
            "acceptance_criteria": [
                {
                    "id": "transcript",
                    "type": "output_completeness",
                    "check": {"field": "text"},
                    "severity": "must",
                }
            ],
        },
        sample=4,
    )

    assert result["ran"] is True
    assert result["retained"] == 4
    assert len(result["examples"]) == 3
    assert all("text" in row for row in result["examples"])
    assert result["goals_met"] is False
    assert any("MISSING or EMPTY" in note and "text" in note for note in result["notes"])
    assert "smoke_token" not in result
    assert result["smoke_token_status"].startswith("not_issued")


def test_smoke_runs_pretrain_driver_lifecycle_inside_the_sandbox(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text('{"audio_filepath":"clip.wav"}\n', encoding="utf-8")
    calls: list[str] = []

    class _Finalizer:
        @staticmethod
        def prepare() -> None:
            calls.append("prepare")

        @staticmethod
        def finalize() -> None:
            calls.append("finalize")

    _stub_smoke_runtime(monkeypatch, lambda _source: calls.append("execute"))
    monkeypatch.setattr(
        verbs,
        "_pretrain_finalizer",
        lambda _stages: (_Finalizer(), ""),
    )

    result = verbs.smoke(
        _source_recipe("ManifestReader", {"manifest_path": str(source)}),
        sample=1,
    )

    assert result["ran"] is True
    assert calls == ["prepare", "execute", "finalize"]
    assert "alm_pretrain_prepare=completed" in result["notes"]
    assert "alm_pretrain_finalize=completed" in result["notes"]


def test_smoke_counts_serialized_pretrain_rows_not_origin_stubs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text('{"audio_filepath":"clip.wav"}\n', encoding="utf-8")

    class _Finalizer:
        @staticmethod
        def prepare() -> None:
            return None

        @staticmethod
        def finalize(*, successful: bool = True) -> int:
            assert successful is True
            return 0

    _stub_smoke_runtime(monkeypatch, lambda _source: None)
    monkeypatch.setattr(
        verbs,
        "_pretrain_finalizer",
        lambda _stages: (_Finalizer(), ""),
    )
    monkeypatch.setattr(
        verbs,
        "_run_pipeline_autofallback",
        lambda *_args, **_kwargs: (
            [SimpleNamespace(num_items=1, data={"is_stub": True})],
            "batch",
        ),
    )

    result = verbs.smoke(
        _source_recipe("ManifestReader", {"manifest_path": str(source)}),
        sample=1,
    )

    assert result["ran"] is True
    assert result["retained"] == 0
    assert result["goals_met"] is False
    assert "smoke_token" not in result


def test_streaming_fallback_resets_attempt_outputs_before_batch_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def execute(_stages, executor, *, checkpoint_path=None):  # noqa: ANN001, ANN202, ARG001
        calls.append(str(executor))
        if len(calls) == 1:
            raise RuntimeError("streaming mode requires batch mode: not enough GPU capacity")  # noqa: EM101
        return []

    monkeypatch.setattr(verbs, "_make_executor", lambda mode: mode)
    monkeypatch.setattr(verbs, "_run_pipeline", execute)

    results, used_mode = verbs._run_pipeline_autofallback(
        [],
        "streaming",
        None,
        before_retry=lambda: calls.append("reset"),
    )

    assert results == []
    assert used_mode == "batch"
    assert calls == ["streaming", "reset", "batch"]


def test_streaming_fallback_resets_owned_partial_checkpoint_before_batch_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "checkpoint.jsonl"
    checkpoint = ManifestCheckpointStage(output_path=str(output))
    attempts = 0

    def execute(_stages, _executor, *, checkpoint_path=None):  # noqa: ANN001, ANN202, ARG001
        nonlocal attempts
        attempts += 1
        # Xenna executes a serialized worker copy, so retry ownership must be
        # durable evidence the original driver-side stage can verify.
        worker_checkpoint = copy.deepcopy(checkpoint)
        worker_checkpoint.setup()
        worker_checkpoint.process(AudioTask(data={"attempt": attempts}))
        if attempts == 1:
            raise RuntimeError("streaming mode requires batch mode: not enough GPU capacity")  # noqa: EM101
        return []

    monkeypatch.setattr(verbs, "_make_executor", lambda mode: mode)
    monkeypatch.setattr(verbs, "_run_pipeline", execute)

    results, used_mode = verbs._run_pipeline_autofallback(
        [checkpoint],
        "streaming",
        None,
        before_retry=verbs._automatic_retry_reset_hook([checkpoint]),
    )

    assert results == []
    assert used_mode == "batch"
    assert output.read_text(encoding="utf-8") == '{"attempt": 2}\n'
    assert not Path(f"{output}._RETRY_OWNER").exists()


def test_resource_planning_failure_is_structured_and_cleans_output_sandbox(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text('{"audio_filepath":"clip.wav"}\n', encoding="utf-8")
    roots: list[str] = []
    isolate = verbs._isolate_smoke_outputs

    def capture_root(bound, report):  # noqa: ANN001, ANN202
        isolated = isolate(bound, report)
        roots.append(str(isolated.output_root))
        return isolated

    monkeypatch.setattr(verbs, "_profile_binding", lambda _binding: None)
    monkeypatch.setattr(verbs, "_isolate_smoke_outputs", capture_root)
    monkeypatch.setattr(
        verbs,
        "_plan_resources",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("planner unavailable")),
    )

    result = verbs.smoke(
        _source_recipe("ManifestReader", {"manifest_path": str(source)}),
        sample=1,
    )

    assert result["status"] == "error"
    assert "resource planning failed" in result["reason"]
    assert roots and all(not os.path.exists(root) for root in roots)  # noqa: PT018
    assert "smoke_token" not in result
