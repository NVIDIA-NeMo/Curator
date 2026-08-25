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

"""Unit tests for verb-level guardrails: the confirm gate, workspace lock, require-smoke,
resolve, and row-accurate evidence counting (no GPU / Ray execution needed)."""

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from nemo_curator import audio_agent as aa
from nemo_curator.audio_agent import cli, run_store, verbs
from nemo_curator.audio_agent.recipe import Recipe
from nemo_curator.audio_agent.report import _row_count

_READER = {"ref": "ManifestReader", "params": {"manifest_path": "/tmp/m.jsonl"}}  # noqa: S108
_WRITER = {"ref": "ManifestWriterStage", "params": {"output_path": "/tmp/out.jsonl"}}  # noqa: S108
_RECIPE = {"stages": [_READER, {"ref": "GetAudioDurationStage", "params": {}}, _WRITER]}


class TestRunConfirmGate:
    def test_refuses_without_confirmation(self) -> None:
        r = aa.run(_RECIPE, confirm=False)
        assert r["status"] == "refused"
        assert "confirmation" in r["reason"].lower()
        assert "config_hash" in r

    def test_refuses_on_hash_mismatch(self) -> None:
        r = aa.run(_RECIPE, confirm="not-the-real-hash")
        assert r["status"] == "refused"
        assert "integrity" in r["reason"].lower()

    def test_falsy_non_true_confirm_does_not_bypass_the_gate(self) -> None:
        # A JSON-RPC ``confirm: null`` (-> None) or any other falsy-but-not-True value must
        # NOT slip past into a silent full run: None/0/[]/{} fail the confirmation gate, and
        # "" is a non-matching hash caught by the integrity gate. All must be refused.
        for bad in (None, 0, "", [], {}):
            r = aa.run(_RECIPE, confirm=bad)
            assert r["status"] == "refused", f"confirm={bad!r} should be refused"


class TestRunWorkspaceLock:
    def test_refuses_path_outside_workspace(self, monkeypatch, tmp_path) -> None:  # noqa: ANN001
        monkeypatch.setenv("AUDIO_AGENT_WORKSPACE", str(tmp_path))
        r = aa.run(_RECIPE, confirm=True, data="/etc/passwd")
        assert r["status"] == "refused"
        assert "workspace" in r["reason"].lower()

    def test_unconfirmed_run_does_not_profile_an_outside_source(self, monkeypatch, tmp_path) -> None:  # noqa: ANN001
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        source = tmp_path / "outside.jsonl"
        source.write_text("{}\n", encoding="utf-8")
        recipe = {
            "stages": [
                {"ref": "ManifestReader", "params": {"manifest_path": str(source)}},
            ]
        }
        monkeypatch.setenv("AUDIO_AGENT_WORKSPACE", str(workspace))
        monkeypatch.setattr(
            verbs,
            "_dataset_binding",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("outside source was inspected")),
        )

        result = aa.run(recipe, confirm=False)

        assert result["status"] == "refused"
        assert "workspace" in result["reason"].lower()

    def test_validate_does_not_profile_an_outside_source(self, monkeypatch, tmp_path) -> None:  # noqa: ANN001
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        source = tmp_path / "outside.jsonl"
        source.write_text("{}\n", encoding="utf-8")
        recipe = {
            "stages": [
                {"ref": "ManifestReader", "params": {"manifest_path": str(source)}},
            ]
        }
        monkeypatch.setenv("AUDIO_AGENT_WORKSPACE", str(workspace))
        monkeypatch.setattr(
            verbs,
            "_dataset_binding",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("outside source was inspected")),
        )

        result = aa.validate(recipe)

        assert result["runnable"] is False
        assert result["issues"][0]["code"] == "path_outside_workspace"

    def test_semantic_path_fields_do_not_trip_the_workspace_lock(self, monkeypatch, tmp_path) -> None:  # noqa: ANN001
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        manifest = tmp_path / "long.jsonl"
        manifest.write_text(
            '{"id":"row-1","custom_audio":"clip.wav"}\n',
            encoding="utf-8",
        )
        recipe = {
            "stages": [
                {
                    "ref": "ReadLongFormManifestStage",
                    "params": {
                        "input_manifest": str(manifest),
                        "audio_dir": str(audio_dir),
                        "audio_filepath_key": "custom_audio",
                        "audio_path_resolution": "relative",
                    },
                }
            ]
        }
        monkeypatch.setenv("AUDIO_AGENT_WORKSPACE", str(tmp_path))

        result = aa.validate(recipe)

        assert not any(issue["code"] == "path_outside_workspace" for issue in result["issues"])


class TestVerbInputBinding:
    @staticmethod
    def _recipe(source: str) -> dict:
        return {
            "stages": [
                {"ref": "ManifestReader", "params": {"manifest_path": source}},
                {"ref": "GetAudioDurationStage", "params": {}},
                {"ref": "ManifestWriterStage", "params": {"output_path": "/tmp/out.jsonl"}},  # noqa: S108
            ]
        }

    @staticmethod
    def _manifest(path) -> str:  # noqa: ANN001
        path.write_text('{"audio_filepath": "/tmp/clip.wav"}\n', encoding="utf-8")
        return str(path)

    def test_validate_rejects_same_content_at_a_different_source(self, tmp_path) -> None:  # noqa: ANN001
        configured = self._manifest(tmp_path / "configured.jsonl")
        asserted = self._manifest(tmp_path / "asserted.jsonl")

        verdict = aa.validate(self._recipe(configured), data=asserted)

        assert verdict["runnable"] is False
        assert any(issue["code"] == "data_source_mismatch" for issue in verdict["issues"])
        assert verdict["data_binding"]["primary_path"] == configured

    def test_omitted_data_is_derived_from_the_recipe(self, tmp_path) -> None:  # noqa: ANN001
        configured = self._manifest(tmp_path / "configured.jsonl")

        verdict = aa.validate(self._recipe(configured))

        assert verdict["data_binding"]["status"] == "resolved"
        assert verdict["data_binding"]["profile_source"] == configured
        assert not any(issue["code"] == "data_source_missing" for issue in verdict["issues"])

    def test_mismatched_smoke_refuses_before_bounding_or_execution(self, monkeypatch, tmp_path) -> None:  # noqa: ANN001
        configured = self._manifest(tmp_path / "configured.jsonl")
        asserted = self._manifest(tmp_path / "asserted.jsonl")

        def must_not_bound(*_args, **_kwargs):  # noqa: ANN202
            raise AssertionError("smoke attempted to bound a mismatched source")  # noqa: EM101

        monkeypatch.setattr(verbs, "_bound_recipe", must_not_bound)
        result = verbs.smoke(self._recipe(configured), data=asserted)

        assert result["status"] == "refused"
        assert result["data_binding"]["status"] == "mismatch"

    def test_confirmed_mismatched_run_never_calls_executor(self, monkeypatch, tmp_path) -> None:  # noqa: ANN001
        configured = self._manifest(tmp_path / "configured.jsonl")
        asserted = self._manifest(tmp_path / "asserted.jsonl")

        def must_not_run(*_args, **_kwargs):  # noqa: ANN202
            raise AssertionError("executor was called for a mismatched source")  # noqa: EM101

        monkeypatch.delenv("AUDIO_AGENT_REQUIRE_SMOKE", raising=False)
        monkeypatch.setattr(verbs, "_run_pipeline_autofallback", must_not_run)
        result = verbs.run(self._recipe(configured), confirm=True, data=asserted)

        assert result["status"] == "refused"
        assert result["data_binding"]["status"] == "mismatch"

    def test_reuse_scan_derives_a_nonempty_key_without_data(self, monkeypatch, tmp_path) -> None:  # noqa: ANN001
        configured = self._manifest(tmp_path / "configured.jsonl")
        monkeypatch.setenv("AUDIO_AGENT_RUNS_DIR", str(tmp_path / "runs"))

        result = verbs.reuse_scan(self._recipe(configured))

        assert result["dataset_key"].startswith(("stat:", "shape:"))
        assert result["data_binding"]["primary_path"] == configured

    def test_report_rejects_a_denominator_from_another_source(self, tmp_path) -> None:  # noqa: ANN001
        configured = self._manifest(tmp_path / "configured.jsonl")
        asserted = self._manifest(tmp_path / "asserted.jsonl")
        output = tmp_path / "output.jsonl"
        output.write_text('{"audio_filepath": "/tmp/clip.wav"}\n', encoding="utf-8")

        result = verbs.report(str(output), recipe=self._recipe(configured), data=asserted)

        assert result["status"] == "refused"
        assert result["data_binding"]["status"] == "mismatch"


class TestMalformedSourceBinding:
    @staticmethod
    def _recipe(source: str, output: str) -> dict:
        return {
            "stages": [
                {"ref": "ManifestReader", "params": {"manifest_path": source}},
                {"ref": "ManifestWriterStage", "params": {"output_path": output}},
            ]
        }

    def test_all_recipe_verbs_fail_structured_before_execution(self, tmp_path) -> None:  # noqa: ANN001
        source = tmp_path / "bad.jsonl"
        source.write_bytes(b"\xff\xfe")
        output = tmp_path / "out.jsonl"
        output.write_text("{}\n", encoding="utf-8")
        recipe = self._recipe(str(source), str(output))

        def unexpected_executor(*_args, **_kwargs):  # noqa: ANN202
            raise AssertionError("malformed source reached the executor")  # noqa: EM101

        verdict = aa.validate(recipe)
        smoke = aa.smoke(recipe, executor=unexpected_executor)
        run = aa.run(recipe, confirm=True, executor=unexpected_executor)
        scan = aa.reuse_scan(recipe)
        continuation = aa.plan_continuation(
            recipe,
            execute=True,
            choice="fresh",
            confirm=True,
        )
        report = aa.report(str(output), recipe=recipe)

        assert verdict["runnable"] is False
        assert any(issue["code"] == "data_source_unreadable" for issue in verdict["issues"])
        assert smoke["status"] == "refused"
        assert run["status"] == "refused"
        assert scan["decision"] == "fresh"
        assert scan["dataset_key"] == ""
        assert "prior work was not considered" in scan["rationale"]
        assert continuation["status"] == "refused"
        assert report["status"] == "refused"


class TestPostHocReportIntegrity:
    @staticmethod
    def _recipe(source: Path, output: Path, *, criteria: list[dict] | None = None) -> dict:
        recipe = {
            "stages": [
                {
                    "ref": "ManifestReader",
                    "params": {"manifest_path": str(source)},
                },
                {
                    "ref": "ManifestWriterStage",
                    "params": {"output_path": str(output)},
                },
            ]
        }
        if criteria is not None:
            recipe["acceptance_criteria"] = criteria
        return recipe

    def test_missing_output_is_an_explicit_error(self, tmp_path) -> None:  # noqa: ANN001
        result = verbs.report(str(tmp_path / "missing.jsonl"))

        assert result["status"] == "error"
        assert result["output_scan"]["status"] == "missing"
        assert "could not be read" in result["reason"]

    def test_malformed_output_is_not_reported_clean(self, monkeypatch, tmp_path) -> None:  # noqa: ANN001
        output = tmp_path / "bad.jsonl"
        output.write_text('{"ok": 1}\n{bad json}\n', encoding="utf-8")
        monkeypatch.setattr(
            verbs,
            "probe_env",
            lambda: SimpleNamespace(to_dict=dict),
        )

        result = verbs.report(str(output))

        assert result["status"] == "error"
        assert result["accepted"] == 1
        assert result["output_scan"]["malformed_rows"] == 1
        assert result["failure_reasons"][0]["code"] == "terminal_output_incomplete"

    def test_recipe_refuses_an_unrelated_output(self, tmp_path) -> None:  # noqa: ANN001
        source = tmp_path / "source.jsonl"
        source.write_text('{"audio_filepath":"clip.wav"}\n', encoding="utf-8")
        declared = tmp_path / "declared.jsonl"
        declared.write_text('{"audio_filepath":"clip.wav"}\n', encoding="utf-8")
        unrelated = tmp_path / "unrelated.jsonl"
        unrelated.write_text('{"audio_filepath":"other.wav"}\n', encoding="utf-8")

        result = verbs.report(
            str(unrelated),
            recipe=self._recipe(source, declared),
        )

        assert result["status"] == "refused"
        assert result["declared_terminal_outputs"] == [str(declared)]
        assert result["config_hash"]

    def test_recipe_identity_and_acceptance_are_bound_to_terminal_rows(
        self,
        monkeypatch,  # noqa: ANN001
        tmp_path,  # noqa: ANN001
    ) -> None:
        source = tmp_path / "source.jsonl"
        source.write_text('{"audio_filepath":"clip.wav"}\n', encoding="utf-8")
        output = tmp_path / "out.jsonl"
        output.write_text('{"audio_filepath":"clip.wav","text":""}\n', encoding="utf-8")
        monkeypatch.setattr(
            verbs,
            "probe_env",
            lambda: SimpleNamespace(to_dict=dict),
        )
        criteria = [
            {
                "id": "text",
                "type": "output_completeness",
                "check": {"field": "text"},
                "severity": "must",
            }
        ]

        result = verbs.report(
            str(output),
            recipe=self._recipe(source, output, criteria=criteria),
        )

        assert result["status"] == "ok"
        assert result["recipe_id"]
        assert result["config_hash"]
        assert result["accepted"] == 1
        assert result["acceptance"]["overall"] == "not_met"
        assert result["acceptance"]["criteria"][0]["status"] == "not_met"

    def test_aggregate_acceptance_uses_complete_terminal_row_mean(
        self,
        monkeypatch,  # noqa: ANN001
        tmp_path,  # noqa: ANN001
    ) -> None:
        source = tmp_path / "source.jsonl"
        source.write_text(
            '{"audio_filepath":"a.wav"}\n{"audio_filepath":"b.wav"}\n',
            encoding="utf-8",
        )
        output = tmp_path / "out.jsonl"
        output.write_text(
            '{"audio_filepath":"a.wav","utmos_mos":3.0}\n{"audio_filepath":"b.wav","utmos_mos":5.0}\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(
            verbs,
            "probe_env",
            lambda: SimpleNamespace(to_dict=dict),
        )
        criteria = [
            {
                "id": "quality",
                "type": "quality_standard",
                "check": {
                    "scope": "aggregate",
                    "field": "utmos_mos",
                    "op": ">=",
                    "value": 4.0,
                },
                "severity": "must",
            }
        ]

        result = verbs.report(
            str(output),
            recipe=self._recipe(source, output, criteria=criteria),
        )

        assert result["output_scan"]["fields"]["utmos_mos"]["mean"] == 4.0
        assert result["acceptance"]["overall"] == "met"
        assert "utmos_mos=4.0" in result["acceptance"]["criteria"][0]["evidence"]

    def test_report_does_not_manufacture_unknown_source_counts(
        self,
        monkeypatch,  # noqa: ANN001
        tmp_path,  # noqa: ANN001
    ) -> None:
        output = tmp_path / "out.jsonl"
        output.write_text('{"audio_filepath":"a.wav"}\n', encoding="utf-8")
        monkeypatch.setattr(
            verbs,
            "probe_env",
            lambda: SimpleNamespace(to_dict=dict),
        )

        result = verbs.report(str(output))

        assert result["accepted"] == 1
        assert result["output_rows"] == 1
        assert result["input_count"] is None
        assert result["source_items"] is None
        assert result["rejected"] is None

    def test_non_manifest_directory_returns_inventory_not_a_false_error(
        self,
        monkeypatch,  # noqa: ANN001
        tmp_path,  # noqa: ANN001
    ) -> None:
        output = tmp_path / "clips"
        output.mkdir()
        (output / "a.wav").write_bytes(b"RIFF-a")
        (output / "b.flac").write_bytes(b"fLaC-b")
        monkeypatch.setattr(
            verbs,
            "probe_env",
            lambda: SimpleNamespace(to_dict=dict),
        )

        result = verbs.report(str(output))

        assert result["status"] == "ok"
        assert result["accepted"] is None
        assert result["output_rows"] is None
        assert result["output_files"] == 2
        assert result["output_inventory"]["status"] == "complete"
        assert result["output_inventory"]["suffixes"] == {".flac": 1, ".wav": 1}


class TestPretrainFinalizerContract:
    @staticmethod
    def _stage(name: str, **attrs: object) -> object:
        cls = type(name, (), {})
        stage = cls()
        for key, value in attrs.items():
            setattr(stage, key, value)
        return stage

    def test_partial_shard_pipeline_is_refused(self) -> None:
        finalizer, error = verbs._pretrain_finalizer(
            [
                self._stage(
                    "SnippetManifestWriterStage",
                    output_path="/tmp/snippets.jsonl",  # noqa: S108
                )
            ]
        )

        assert finalizer is None
        assert "missing" in error
        assert "SnippetExtractionStage" in error

    def test_complete_shard_pipeline_resolves_driver_outputs(self) -> None:
        finalizer, error = verbs._pretrain_finalizer(
            [
                self._stage(
                    "SnippetExtractionStage",
                    output_audio_tar_path="/tmp/snippets.tar",  # noqa: S108
                    audio_filepath_key="clip_path",
                ),
                self._stage(
                    "SnippetManifestWriterStage",
                    output_path="/tmp/snippets.jsonl",  # noqa: S108
                ),
                self._stage(
                    "PretrainMetricsAggregatorStage",
                    output_path="/tmp/metrics.json",  # noqa: S108
                ),
            ]
        )

        assert error == ""
        assert finalizer == verbs._PretrainFinalizer(
            manifest_path="/tmp/snippets.jsonl",  # noqa: S108
            metrics_path="/tmp/metrics.json",  # noqa: S108
            audio_tar_path="/tmp/snippets.tar",  # noqa: S108
            audio_filepath_key="clip_path",
        )


class TestCliExitSemantics:
    def test_structured_smoke_refusal_returns_nonzero(
        self,
        tmp_path,  # noqa: ANN001
        capsys,  # noqa: ANN001
    ) -> None:
        recipe = tmp_path / "recipe.yaml"
        recipe.write_text(yaml.safe_dump(_RECIPE), encoding="utf-8")

        rc = cli.main(
            [
                "smoke",
                "--recipe",
                str(recipe),
                "--sample",
                "0",
            ]
        )
        result = json.loads(capsys.readouterr().out)

        assert rc == 1
        assert result["status"] == "refused"

    def test_structured_lookup_error_returns_nonzero(self, capsys) -> None:  # noqa: ANN001
        rc = cli.main(["describe", "DefinitelyNotAnAudioStage"])
        result = json.loads(capsys.readouterr().out)

        assert rc == 1
        assert "error" in result

    def test_missing_report_output_returns_nonzero(
        self,
        tmp_path,  # noqa: ANN001
        capsys,  # noqa: ANN001
    ) -> None:
        rc = cli.main(["report", "--output", str(tmp_path / "missing.jsonl")])
        result = json.loads(capsys.readouterr().out)

        assert rc == 1
        assert result["status"] == "error"


class TestContinuationExecutionEvidence:
    def test_fresh_branch_forwards_smoke_checkpoint_and_calibration(
        self,
        monkeypatch,  # noqa: ANN001
    ) -> None:
        captured: dict[str, object] = {}

        def fake_run(_recipe: object, **kwargs: object) -> dict[str, str]:
            captured.update(kwargs)
            return {"status": "completed"}

        monkeypatch.setattr(verbs, "run", fake_run)
        rec = Recipe.from_dict(_RECIPE).freeze()

        result = verbs._execute_plan(
            rec,
            {"mode": "full_rerun", "source": "none"},
            choice="fresh",
            data="/tmp/m.jsonl",  # noqa: S108
            confirm=rec.config_hash or True,
            output_dir=None,
            bootstrap_ray=False,
            goal={"task": "continue"},
            parent=None,
            continuation_mod=object(),
            checkpoint_path="/tmp/checkpoint",  # noqa: S108
            smoke_token="proof",  # noqa: S106
            calibration={"Stage": {"source": "measured"}},
        )

        assert result["status"] == "completed"
        assert captured["checkpoint_path"] == "/tmp/checkpoint"  # noqa: S108
        assert captured["smoke_token"] == "proof"  # noqa: S105
        assert captured["calibration"] == {"Stage": {"source": "measured"}}


class TestRunRequireSmoke:
    def test_refuses_without_smoke_token(self, monkeypatch) -> None:  # noqa: ANN001
        monkeypatch.setenv("AUDIO_AGENT_REQUIRE_SMOKE", "1")
        monkeypatch.delenv("AUDIO_AGENT_WORKSPACE", raising=False)
        r = aa.run(_RECIPE, confirm=True)
        assert r["status"] == "refused"
        assert "smoke" in r["reason"].lower()


class TestRunAcceptanceResult:
    def test_run_returns_the_same_acceptance_result_it_records(
        self,
        monkeypatch,  # noqa: ANN001
        tmp_path,  # noqa: ANN001
    ) -> None:
        env = SimpleNamespace(
            has_gpu=False,
            gpu_count=0,
            to_dict=dict,
        )
        resource_plan = SimpleNamespace(
            feasible=True,
            mode="batch",
            machine_fingerprint="machine",
            escalations=[],
            to_dict=lambda: {"mode": "batch"},
        )
        report = SimpleNamespace(
            accepted=1,
            input_count=1,
            output_paths=[],
            per_stage_metrics={},
            to_dict=lambda: {"accepted": 1, "input_count": 1},
        )
        acceptance = {
            "overall": "met",
            "criteria": [{"id": "kept", "status": "met"}],
        }
        acceptance_calls = 0
        recorded: dict = {}

        def fake_acceptance(*_args, **_kwargs):  # noqa: ANN202
            nonlocal acceptance_calls
            acceptance_calls += 1
            return acceptance

        def fake_record(*_args, **kwargs):  # noqa: ANN202
            recorded.update(kwargs)
            return "run-test"

        monkeypatch.delenv("AUDIO_AGENT_REQUIRE_SMOKE", raising=False)
        monkeypatch.setattr(verbs, "probe_env", lambda: env)
        monkeypatch.setattr(verbs, "build_stages", lambda _rec: ([object()], []))
        monkeypatch.setattr(
            verbs,
            "_plan_resources",
            lambda *_args, **_kwargs: resource_plan,
        )
        monkeypatch.setattr(
            verbs,
            "_run_pipeline_autofallback",
            lambda *_args, **_kwargs: ([object()], "batch"),
        )
        monkeypatch.setattr(verbs, "build_run_report", lambda **_kwargs: report)
        monkeypatch.setattr(verbs, "_produced_roles_keys", lambda *_args, **_kw: ([], []))
        monkeypatch.setattr(verbs, "_acceptance_result", fake_acceptance)
        monkeypatch.setattr(verbs, "_publish_artifacts", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(verbs, "_record_run", fake_record)
        monkeypatch.setattr(run_store, "new_run_id", lambda _config_hash: "run-test")
        manifest = tmp_path / "source.jsonl"
        manifest.write_text('{"audio_filepath": "/tmp/a.wav"}\n', encoding="utf-8")

        result = verbs.run(
            {
                **_RECIPE,
                "stages": [
                    {"ref": "ManifestReader", "params": {"manifest_path": str(manifest)}},
                    *_RECIPE["stages"][1:],
                ],
                "acceptance_criteria": [
                    {
                        "id": "kept",
                        "type": "yield",
                        "check": {"op": ">=", "value": 1},
                    }
                ],
            },
            confirm=True,
        )

        assert acceptance_calls == 1
        assert result["acceptance"] == acceptance
        assert recorded["acceptance_result"] == acceptance


class TestResolve:
    def test_resolves_label_to_concrete_params(self) -> None:
        r = aa.resolve("UTMOSFilterStage", label="studio")
        assert isinstance(r, dict)
        assert "mos_threshold" in str(r)  # a concrete threshold was resolved


class TestRowCount:
    def test_counts_rows_not_tasks(self) -> None:
        class _T:
            def __init__(self, n: int) -> None:
                self.num_items = n

        assert _row_count([_T(1), _T(1)]) == 2  # AudioTask-like: 1 row each
        assert _row_count([_T(500)]) == 500  # DocumentBatch-like: one task, 500 rows
        assert _row_count(None) == 0


class TestCatalogDisclosure:
    """A shorter catalog must never be mistaken for a smaller library."""

    def test_a_healthy_environment_reports_no_unavailable_key(self) -> None:
        """The key is absent when everything imported, so healthy output is unchanged."""
        assert "unavailable" not in aa.discover()

    def test_a_module_that_failed_to_import_is_disclosed(self, monkeypatch) -> None:  # noqa: ANN001
        """On a supported CPU-only install the ASR stages are simply gone; the host must be
        able to say 'unavailable here' rather than 'this cannot be done'."""
        from nemo_curator.stages.audio._agent import _catalog

        monkeypatch.setattr(
            _catalog,
            "_SKIPPED",
            [
                {
                    "module": "nemo_curator.stages.audio.inference.asr.stage",
                    "error": "ModuleNotFoundError: No module named 'nemo'",
                }
            ],
        )
        result = aa.discover()
        assert result["unavailable"][0]["module"].endswith("asr.stage")
        assert "ModuleNotFoundError" in result["unavailable"][0]["error"]
        assert result["count"] == len(result["stages"])  # the present stages still list normally


class TestDataInformedConfig:
    """Path B: values the DATASET fixes, bound onto the stage -- never onto a stage default.

    Stage defaults are what tutorials and hand-written pipelines rely on, so the agent
    configures the recipe it builds instead of changing what the stage does by default.
    """

    def _manifest(self, tmp_path, name, row, rate=16000):  # noqa: ANN001, ANN202
        import numpy as np
        import soundfile as sf

        wav = tmp_path / "a.wav"
        sf.write(str(wav), np.zeros(rate, dtype="float32"), rate)
        row = {k: (str(wav) if v == "@wav" else v) for k, v in row.items()}
        path = tmp_path / name
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return str(path)

    def test_the_observed_rate_is_bound_so_the_default_does_not_discard_the_corpus(self, tmp_path) -> None:  # noqa: ANN001
        """MonoConversion VERIFIES the rate and drops non-matching rows; its 48 kHz default
        would silently discard a 16 kHz corpus."""
        manifest = self._manifest(tmp_path, "nemo.jsonl", {"audio_filepath": "@wav", "text": "hi"})
        result = aa.resolve("MonoConversionStage", data=manifest)
        assert result["params"]["output_sample_rate"] == 16000
        assert result["asks"] == []

    def test_a_folder_source_needs_no_column_question(self, tmp_path) -> None:  # noqa: ANN001
        """The agent creates the manifest for a folder, so the schema is known by construction."""
        import numpy as np
        import soundfile as sf

        sf.write(str(tmp_path / "a.wav"), np.zeros(16000, dtype="float32"), 16000)
        result = aa.resolve("MonoConversionStage", data=str(tmp_path))
        assert result["params"]["output_sample_rate"] == 16000
        assert result["asks"] == []

    def test_an_explicit_value_outranks_the_inferred_one(self, tmp_path) -> None:  # noqa: ANN001
        manifest = self._manifest(tmp_path, "nemo.jsonl", {"audio_filepath": "@wav", "text": "hi"})
        result = aa.resolve("MonoConversionStage", explicit={"output_sample_rate": 48000}, data=manifest)
        assert result["params"]["output_sample_rate"] == 48000

    def test_an_overridden_inference_leaves_no_trace_claiming_it_applied(self, tmp_path) -> None:  # noqa: ANN001
        """The value was always the user's, but the strategy trail also gained a data_informed
        entry stating 16000 and why it was chosen -- an audit record of a binding that never
        happened. On the one param a user is most likely to have pinned deliberately, as a
        strict gate, that reads as the agent having quietly widened it to fit the data.
        """
        manifest = self._manifest(tmp_path, "nemo.jsonl", {"audio_filepath": "@wav", "text": "hi"})
        result = aa.resolve("MonoConversionStage", explicit={"output_sample_rate": 48000}, data=manifest)
        rate_entries = [e for e in result["strategy"] if e["param"] == "output_sample_rate"]
        assert [e["value"] for e in rate_entries] == [48000]

    def test_the_derivation_is_recorded_as_recomputable(self, tmp_path) -> None:  # noqa: ANN001
        """A data-derived value must be stamped so a different dataset recomputes it."""
        manifest = self._manifest(tmp_path, "nemo.jsonl", {"audio_filepath": "@wav", "text": "hi"})
        entry = next(
            e
            for e in aa.resolve("MonoConversionStage", data=manifest)["strategy"]
            if e["param"] == "output_sample_rate"
        )
        assert entry["mode"] == "data_informed"
        assert entry["recompute_on"] == "data_change"

    def test_path_a_alone_is_unchanged(self) -> None:
        assert aa.resolve("UTMOSFilterStage", label="studio")["params"] == {"mos_threshold": 4.0}

    def test_a_dataset_outside_the_workspace_is_refused_like_every_other_verb(self, monkeypatch, tmp_path) -> None:  # noqa: ANN001
        """``resolve`` profiles ``data`` off the filesystem, so it owes the same lock the
        other data-taking verbs enforce. It was the one verb without the check -- harmless
        only while no adapter could pass ``data``, and a hole the moment one could."""
        outside = self._manifest(tmp_path, "nemo.jsonl", {"audio_filepath": "@wav", "text": "hi"})
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        monkeypatch.setenv("AUDIO_AGENT_WORKSPACE", str(workspace))
        result = aa.resolve("MonoConversionStage", data=outside)
        assert result["status"] == "refused"
        assert "outside the allowed workspace" in result["reason"]
        assert "params" not in result

    def test_a_profile_that_read_no_audio_says_so(self, tmp_path) -> None:  # noqa: ANN001
        """An empty audio profile must not be mistaken for a healthy one."""
        from nemo_curator.audio_agent.profiler import profile_data

        manifest = self._manifest(tmp_path, "cv.jsonl", {"path": "@wav", "sentence": "hi"})
        notes = profile_data(manifest).to_dict()["notes"]
        assert any("no rows carried an audio path" in n for n in notes)
        assert profile_data(manifest, audio_filepath_key="path").to_dict()["notes"] == []
