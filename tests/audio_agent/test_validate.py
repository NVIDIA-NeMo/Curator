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

"""Unit tests for the deterministic grounding layer (validate / checks)."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

from nemo_curator import audio_agent as aa
from nemo_curator.audio_agent import cli, verbs
from nemo_curator.audio_agent.verbs import validate

_READER_MANIFEST = Path(__file__).resolve().parents[1] / "fixtures/audio/alm/sample_input.jsonl"
_READER = {"ref": "ManifestReader", "params": {"manifest_path": str(_READER_MANIFEST)}}
_WRITER = {"ref": "ManifestWriterStage", "params": {"output_path": "/tmp/out.jsonl"}}  # noqa: S108
_DOCUMENT_WRITER = {
    "ref": "DocumentBatchJsonlWriterStage",
    "params": {"output_path": "/tmp/document-batch-out.jsonl"},  # noqa: S108
}
_DURATION = {"ref": "GetAudioDurationStage", "params": {}}


def _validate(stages: list[dict], **kw: object) -> dict:
    return aa.validate({"stages": stages}, **kw)


def _codes(verdict: dict) -> set[str]:
    return {
        issue["code"] for pool in ("issues", "card_violations", "gate_flags") for issue in (verdict.get(pool) or [])
    }


class TestValidateContract:
    def test_returns_json_dict(self) -> None:
        r = _validate([_READER, _DURATION, _WRITER])
        assert isinstance(r, dict)
        assert "status" in r

    def test_valid_recipe_is_runnable(self) -> None:
        r = _validate([_READER, _DURATION, _WRITER])
        assert r.get("runnable") is not False
        assert r.get("status") != "fail"

    def test_clean_validation_requires_separate_semantic_review(self) -> None:
        r = _validate(
            [
                _READER,
                _DURATION,
                {
                    "ref": "PreserveByValueStage",
                    "params": {
                        "input_value_key": "duration",
                        "target_value": 1,
                        "operator": "ge",
                    },
                },
                _WRITER,
            ]
        )

        assert r["validation_scope"] == "mechanical_runnability_not_intent_approval"
        assert r["semantic_review"]["review_required"] is True
        assert r["semantic_review"]["intent_interpretation_performed"] is False
        duration_edge = next(
            edge
            for edge in r["semantic_review"]["lineage"]
            if edge["consumer"]["stage"] == "PreserveByValueStage" and edge["read"]["key"] == "duration"
        )
        assert duration_edge["latest_upstream_producer"]["stage"] == "GetAudioDurationStage"
        expected_hash = aa.Recipe.from_dict(
            {
                "stages": [
                    _READER,
                    _DURATION,
                    {
                        "ref": "PreserveByValueStage",
                        "params": {
                            "input_value_key": "duration",
                            "target_value": 1,
                            "operator": "ge",
                        },
                    },
                    _WRITER,
                ]
            }
        ).compute_hash()
        assert r["semantic_review"]["recipe"]["config_hash"] == expected_hash
        assert (
            r["semantic_review"]["required_response"]["recipe_config_hash"]["source"]
            == "semantic_review.recipe.config_hash"
        )
        assert "mechanically" in r["summary"]


class TestDataFlowChecks:
    def test_tensor_into_sink_flagged(self) -> None:
        # A resident-waveform producer feeding a JSON sink without AudioToDocument.
        r = _validate([_READER, {"ref": "SpeakerSeparationStage", "params": {}}, _WRITER])
        assert "tensor_into_sink" in _codes(r)
        assert r.get("status") == "fail"

    def test_sanitized_flow_clears_tensor_into_sink(self) -> None:
        r = _validate(
            [
                _READER,
                {"ref": "SpeakerSeparationStage", "params": {}},
                {"ref": "AudioToDocumentStage", "params": {}},
                _DOCUMENT_WRITER,
            ]
        )
        assert "tensor_into_sink" not in _codes(r)
        assert "task_type_mismatch" not in _codes(r)

    def test_audio_to_document_requires_a_document_batch_sink(self) -> None:
        r = _validate([_READER, {"ref": "AudioToDocumentStage", "params": {}}, _WRITER])
        assert "task_type_mismatch" in _codes(r)
        assert r.get("status") == "fail"
        issue = next(issue for issue in r["issues"] if issue["code"] == "task_type_mismatch")
        assert "DocumentBatchJsonlWriterStage" in issue["fix"]
        assert "insert a converter (e.g. AudioToDocumentStage)" not in issue["fix"]

    def test_audio_task_cannot_feed_document_batch_sink_directly(self) -> None:
        r = _validate([_READER, _DURATION, _DOCUMENT_WRITER])
        assert "task_type_mismatch" in _codes(r)
        assert r.get("status") == "fail"

    def test_document_batch_writer_is_compatible_with_converter(self) -> None:
        r = _validate([_READER, _DURATION, {"ref": "AudioToDocumentStage", "params": {}}, _DOCUMENT_WRITER])
        assert "task_type_mismatch" not in _codes(r)
        assert r.get("status") != "fail"

    def test_snippet_writer_rejects_a_resident_tensor(self) -> None:
        snippet_writer = {
            "ref": "SnippetManifestWriterStage",
            "params": {"output_path": "/tmp/snippets.jsonl"},  # noqa: S108
        }
        r = _validate([_READER, {"ref": "SpeakerSeparationStage", "params": {}}, snippet_writer])
        assert "tensor_into_sink" in _codes(r)
        assert r.get("status") == "fail"

    def test_audio_data_filter_declares_its_audio_task_carrier(self) -> None:
        contract = aa.describe("AudioDataFilterStage")["contract"]
        assert contract["accepts_task_type"] == "AudioTask"
        assert contract["produces_task_type"] == "AudioTask"


class TestCheckIsolation:
    def test_malformed_num_speakers_does_not_crash(self) -> None:
        # H3: a non-numeric card-constrained param must yield a JSON verdict, not a traceback.
        r = _validate([_READER, {"ref": "InferenceSortformerStage", "params": {"num_speakers": "two"}}, _WRITER])
        assert isinstance(r, dict)
        assert "status" in r


class TestEnvironmentGates:
    def test_gpu_unavailable_is_reported_once(self, monkeypatch) -> None:  # noqa: ANN001
        monkeypatch.setattr(
            verbs,
            "probe_env",
            lambda: SimpleNamespace(
                has_gpu=False,
                gpu_count=0,
                has_ffmpeg=True,
                available_secrets=[],
            ),
        )
        result = _validate(
            [
                _READER,
                {
                    "ref": "ASRStage",
                    "params": {
                        "adapter_target": "nemo_curator.models.asr.nemo_asr.NeMoASRAdapter",
                        "model_id": "nvidia/test-model",
                        "audio_filepath_key": "audio_filepath",
                    },
                },
                _WRITER,
            ]
        )
        matches = [
            (pool, issue)
            for pool in ("issues", "card_violations", "gate_flags")
            for issue in result.get(pool, [])
            if issue["code"] == "gpu_unavailable"
        ]

        assert len(matches) == 1
        assert matches[0][0] == "gate_flags"


class TestOutputCompleteness:
    def test_missing_producer_flagged(self) -> None:
        r = _validate([_READER, _DURATION, _WRITER], expected_outputs=["nonexistent_metric_xyz"])
        assert "missing_output_producer" in _codes(r)

    def test_satisfied_by_produced_key(self) -> None:
        # H5: 'duration' is produced by GetAudioDurationStage -> not flagged.
        r = _validate([_READER, _DURATION, _WRITER], expected_outputs=["duration"])
        assert "missing_output_producer" not in _codes(r)


class TestAcceptanceContractBinding:
    _CRITERIA: ClassVar[list[dict]] = [
        {
            "id": "duration",
            "type": "output_completeness",
            "check": {"field": "duration"},
            "severity": "must",
        }
    ]

    @staticmethod
    def _recipe(criteria: list[dict] | None = None) -> dict:
        recipe = {"stages": [_READER, _DURATION, _WRITER]}
        if criteria is not None:
            recipe["acceptance_criteria"] = criteria
        return recipe

    def test_separate_contract_cannot_validate_a_contractless_recipe(self) -> None:
        verdict = aa.validate(
            self._recipe(),
            acceptance_criteria=self._CRITERIA,
        )

        assert "acceptance_contract_not_embedded" in _codes(verdict)
        assert verdict["runnable"] is False

    def test_matching_embedded_contract_passes_the_binding_check(self) -> None:
        verdict = aa.validate(
            self._recipe(self._CRITERIA),
            acceptance_criteria=self._CRITERIA,
        )

        assert "acceptance_contract_not_embedded" not in _codes(verdict)

    def test_different_explicit_contract_is_rejected(self) -> None:
        different = [
            {
                "id": "transcript",
                "type": "output_completeness",
                "check": {"field": "pred_text"},
                "severity": "must",
            }
        ]
        verdict = aa.validate(
            self._recipe(self._CRITERIA),
            acceptance_criteria=different,
        )

        assert "acceptance_contract_not_embedded" in _codes(verdict)
        assert verdict["runnable"] is False

    def test_cli_separate_contract_is_also_fail_closed(
        self,
        tmp_path: Path,
        capsys,  # noqa: ANN001
    ) -> None:
        import json

        import yaml

        recipe_path = tmp_path / "recipe.yaml"
        criteria_path = tmp_path / "criteria.yaml"
        recipe_path.write_text(
            yaml.safe_dump(self._recipe()),
            encoding="utf-8",
        )
        criteria_path.write_text(
            yaml.safe_dump({"acceptance_criteria": self._CRITERIA}),
            encoding="utf-8",
        )

        rc = cli.main(
            [
                "validate",
                "--recipe",
                str(recipe_path),
                "--acceptance-criteria",
                str(criteria_path),
            ]
        )
        verdict = json.loads(capsys.readouterr().out)

        assert rc == 0
        assert "acceptance_contract_not_embedded" in _codes(verdict)
        assert verdict["runnable"] is False


class TestDiarizationContinuity:
    def test_vad_before_diarizer_without_rejoin_flagged(self) -> None:
        r = _validate(
            [
                _READER,
                {"ref": "VADSegmentationStage", "params": {}},
                {"ref": "InferenceSortformerStage", "params": {}},
                _WRITER,
            ]
        )
        assert "diarization_needs_continuous_audio" in _codes(r)

    def test_diarizer_on_continuous_audio_ok(self) -> None:
        # Diarizing before any VAD is fine (continuous waveform).
        r = _validate([_READER, {"ref": "InferenceSortformerStage", "params": {}}, _WRITER])
        assert "diarization_needs_continuous_audio" not in _codes(r)


class TestSampleRateFollowsThePipeline:
    """A model's supported rates must be judged against what reaches IT, not the source files.

    Comparing every stage's card against the SOURCE profile warned that 48 kHz input was
    unsupported by a 16 kHz model even when a resample sat immediately upstream -- telling the
    user a correct pipeline was broken, which is how a validator loses its authority.
    """

    @staticmethod
    def _corpus(tmp_path: Path, rate: int) -> str:
        import json

        import numpy as np
        import soundfile as sf

        wav = tmp_path / f"clip{rate}.wav"
        sf.write(str(wav), np.zeros(rate, dtype="float32"), rate)
        manifest = tmp_path / "in.jsonl"
        manifest.write_text(json.dumps({"audio_filepath": str(wav), "duration": 1.0}) + "\n", encoding="utf-8")
        return str(manifest)

    @staticmethod
    def _resample(tmp_path: Path, target: int | None) -> dict:
        params: dict = {"resampled_audio_dir": str(tmp_path / f"rs{target}")}
        if target is not None:
            params["target_sample_rate"] = target
        return {"ref": "ResampleAudioStage", "params": params}

    # InferenceSortformerStage's card declares supported_sample_rates: [16000].
    _SIXTEEN_K_ONLY: ClassVar[dict] = {"ref": "InferenceSortformerStage", "params": {}}

    def _rate_warnings(self, stages: list[dict], manifest: str) -> list[dict]:
        verdict = aa.validate({"stages": stages}, data=manifest)
        return [
            i
            for pool in ("issues", "card_violations", "gate_flags")
            for i in (verdict.get(pool) or [])
            if i["code"] == "card_sample_rate"
        ]

    def test_an_upstream_resample_silences_the_warning(self, tmp_path: Path) -> None:
        manifest = self._corpus(tmp_path, 48000)
        reader = {"ref": "ManifestReader", "params": {"manifest_path": manifest}}
        stages = [reader, self._resample(tmp_path, 16000), self._SIXTEEN_K_ONLY]
        assert self._rate_warnings(stages, manifest) == []

    def test_without_a_resample_it_still_warns(self, tmp_path: Path) -> None:
        manifest = self._corpus(tmp_path, 48000)
        reader = {"ref": "ManifestReader", "params": {"manifest_path": manifest}}
        warnings = self._rate_warnings([reader, self._SIXTEEN_K_ONLY], manifest)
        assert len(warnings) == 1
        assert "48000" in warnings[0]["message"]

    def test_an_omitted_target_resolves_to_the_stage_default(self, tmp_path: Path) -> None:
        # ResampleAudioStage converts to 16 kHz by default, whether or not the recipe says so.
        manifest = self._corpus(tmp_path, 48000)
        reader = {"ref": "ManifestReader", "params": {"manifest_path": manifest}}
        stages = [reader, self._resample(tmp_path, None), self._SIXTEEN_K_ONLY]
        assert self._rate_warnings(stages, manifest) == []

    def test_resampling_to_an_unsupported_rate_warns_with_the_effective_rate(self, tmp_path: Path) -> None:
        manifest = self._corpus(tmp_path, 48000)
        reader = {"ref": "ManifestReader", "params": {"manifest_path": manifest}}
        stages = [reader, self._resample(tmp_path, 8000), self._SIXTEEN_K_ONLY]
        warnings = self._rate_warnings(stages, manifest)
        assert len(warnings) == 1
        assert "8000" in warnings[0]["message"]
        assert "source was [48000]" in warnings[0]["message"]

    def test_mono_conversion_is_not_mistaken_for_a_resample(self, tmp_path: Path) -> None:
        # MonoConversionStage takes an output_sample_rate, but its card is explicit that this is
        # a rate to VERIFY against and that it never resamples. Treating it as a conversion would
        # tell the planner the audio had been converted when it had only been checked.
        manifest = self._corpus(tmp_path, 48000)
        reader = {"ref": "ManifestReader", "params": {"manifest_path": manifest}}
        mono = {"ref": "MonoConversionStage", "params": {"output_sample_rate": 16000, "strict_sample_rate": False}}
        warnings = self._rate_warnings([reader, mono, self._SIXTEEN_K_ONLY], manifest)
        assert len(warnings) == 1
        assert "48000" in warnings[0]["message"]


class TestSourceSchemaHonesty:
    """A manifest whose columns cannot satisfy the stages must not validate as sound."""

    def _recipe(self, manifest: str, params: dict | None = None) -> dict:
        return {
            "stages": [
                {"ref": "ManifestReader", "params": {"manifest_path": manifest}},
                {"ref": "GetAudioDurationStage", "params": params or {}},
            ]
        }

    def _write(self, tmp_path, name: str, row: dict) -> str:  # noqa: ANN001
        path = tmp_path / name
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return str(path)

    def test_a_manifest_without_an_audio_path_column_is_refused(self, tmp_path) -> None:  # noqa: ANN001
        """The 'validates green, yields zero rows' class: the reader declares
        audio_filepath but is schema-agnostic, so the declaration is not evidence.

        Refused rather than inferred around. ``audio_filepath`` is the NeMo manifest key
        and meeting it is the caller's contract; guessing which other column holds the
        audio means guessing wrong on some dataset and curating the wrong field silently.
        """
        manifest = self._write(tmp_path, "cv.jsonl", {"path": "/a.wav", "sentence": "hi"})
        verdict = validate(self._recipe(manifest), data=manifest)
        issue = next(i for i in verdict["issues"] if i["code"] == "source_schema_mismatch")
        assert issue["severity"] == "error"
        assert verdict["runnable"] is False
        # The message has to be actionable: name the columns actually present, and the fix.
        assert "'path'" in issue["message"]
        assert "'sentence'" in issue["message"]
        assert "audio_filepath" in issue["fix"]

    def test_a_stage_pointed_at_the_real_column_is_not_flagged(self, tmp_path) -> None:  # noqa: ANN001
        manifest = self._write(tmp_path, "cv.jsonl", {"path": "/a.wav", "sentence": "hi"})
        verdict = validate(self._recipe(manifest, {"audio_filepath_key": "path"}), data=manifest)
        assert "source_schema_mismatch" not in [i["code"] for i in verdict["issues"]]

    def test_a_conventional_manifest_is_not_flagged(self, tmp_path) -> None:  # noqa: ANN001
        manifest = self._write(tmp_path, "nemo.jsonl", {"audio_filepath": "/a.wav", "text": "hi"})
        verdict = validate(self._recipe(manifest), data=manifest)
        assert "source_schema_mismatch" not in [i["code"] for i in verdict["issues"]]

    def test_the_recipes_own_manifest_is_evidence_even_without_an_explicit_data_argument(self, tmp_path) -> None:  # noqa: ANN001
        """validate binds and profiles the source named in the recipe, so the mismatch is
        caught whether or not the caller passes ``data``."""
        manifest = self._write(tmp_path, "cv.jsonl", {"path": "/a.wav"})
        verdict = validate(self._recipe(manifest))  # no data= argument
        assert "source_schema_mismatch" in [i["code"] for i in verdict["issues"]]

    def test_no_observed_columns_means_no_claim(self, tmp_path) -> None:  # noqa: ANN001
        """An empty manifest yields no columns; with no evidence the check stays silent
        rather than inventing a mismatch."""
        manifest = self._write_raw(tmp_path, "empty.jsonl", "")
        verdict = validate(self._recipe(manifest), data=manifest)
        assert "source_schema_mismatch" not in [i["code"] for i in verdict["issues"]]

    def _write_raw(self, tmp_path, name: str, text: str) -> str:  # noqa: ANN001
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return str(path)
