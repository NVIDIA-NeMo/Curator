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

"""Dataset identity is derived from source-stage execution truth, never guessed."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from nemo_curator.audio_agent.input_identity import (
    SUPPORTED_SOURCE_REFS,
    canonical_source,
    resolve_dataset_binding,
)
from nemo_curator.audio_agent.recipe import Recipe

if TYPE_CHECKING:
    from pathlib import Path


def _recipe(
    ref: str,
    params: dict[str, object],
    *,
    inputs: dict[str, object] | None = None,
    tail: list[dict[str, object]] | None = None,
) -> Recipe:
    stages: list[dict[str, object]] = [{"ref": ref, "params": params}]
    stages.extend(tail or [])
    return Recipe.from_dict({"stages": stages, "inputs": inputs or {}})


class TestCanonicalSource:
    def test_local_and_file_uri_aliases_have_one_identity(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        manifest = tmp_path / "data" / "manifest.jsonl"
        manifest.parent.mkdir()
        manifest.write_text("", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        assert canonical_source("data/../data/manifest.jsonl") == str(manifest)
        assert canonical_source(manifest.as_uri()) == str(manifest)

    def test_remote_uri_is_lexical_and_never_normalized(self) -> None:
        uri = "s3://bucket/a/../manifest.jsonl?versionId=abc"
        assert canonical_source(uri) == uri

    @pytest.mark.parametrize(
        "value",
        ["", "REQUIRED_manifest_path", "REQUIRED: folder", "{{ data_dir }}", "${DATA}", "/data/{split}/audio"],
    )
    def test_unresolved_values_are_rejected(self, value: str) -> None:
        with pytest.raises(ValueError, match=r"empty|unresolved"):
            canonical_source(value)


class TestSourceBoundary:
    def test_supported_source_table_is_closed(self) -> None:
        assert SUPPORTED_SOURCE_REFS == (
            "ManifestReader",
            "CreateInitialManifestAudioFolderStage",
            "CreateInitialManifestFleursStage",
            "CreateInitialManifestReadSpeechStage",
            "ReadLongFormManifestStage",
        )

    def test_empty_and_unknown_first_stage_are_not_guessed(self) -> None:
        assert resolve_dataset_binding(Recipe()).status == "missing"
        unsupported = resolve_dataset_binding(_recipe("ManifestReaderStage", {}))
        assert unsupported.status == "unsupported"
        assert unsupported.source_ref == "ManifestReaderStage"

    def test_source_after_index_zero_is_ambiguous(self, tmp_path: Path) -> None:
        manifest = tmp_path / "m.jsonl"
        manifest.write_text("", encoding="utf-8")
        recipe = Recipe.from_dict(
            {
                "stages": [
                    {"ref": "GetAudioDurationStage", "params": {}},
                    {"ref": "ManifestReader", "params": {"manifest_path": str(manifest)}},
                ]
            }
        )
        binding = resolve_dataset_binding(recipe)
        assert binding.status == "ambiguous"
        assert binding.source_index == 1

    def test_multiple_sources_are_ambiguous(self, tmp_path: Path) -> None:
        manifest = tmp_path / "m.jsonl"
        manifest.write_text("", encoding="utf-8")
        recipe = _recipe(
            "ManifestReader",
            {"manifest_path": str(manifest)},
            tail=[{"ref": "CreateInitialManifestAudioFolderStage", "params": {"data_dir": str(tmp_path)}}],
        )
        binding = resolve_dataset_binding(recipe)
        assert binding.status == "ambiguous"
        assert "multiple" in binding.reason


class TestManifestReader:
    def test_single_manifest_matches_canonical_input_assertions(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text("{}\n", encoding="utf-8")
        recipe = _recipe(
            "ManifestReader",
            {"manifest_path": manifest.as_uri()},
            inputs={"manifest_path": str(manifest), "output_manifest": "REQUIRED: ignored output"},
        )

        binding = resolve_dataset_binding(recipe, data=str(manifest))

        assert binding.status == "resolved"
        assert binding.configured_paths == (str(manifest),)
        assert binding.primary_path == str(manifest)
        assert binding.profile_source == str(manifest)
        assert binding.configured_path == str(manifest)
        assert binding.to_dict()["configured_paths"] == [str(manifest)]

    def test_inputs_and_cli_are_assertions_not_substitutions(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text("", encoding="utf-8")
        unresolved = _recipe(
            "ManifestReader",
            {"manifest_path": "REQUIRED_manifest_path"},
            inputs={"manifest_path": str(manifest)},
        )
        assert resolve_dataset_binding(unresolved, data=str(manifest)).status == "missing"

        other = tmp_path / "other.jsonl"
        other.write_text("", encoding="utf-8")
        configured = _recipe("ManifestReader", {"manifest_path": str(manifest)})
        assert resolve_dataset_binding(configured, data=str(other)).status == "mismatch"

    def test_conflicting_recipe_input_is_a_mismatch(self, tmp_path: Path) -> None:
        configured = tmp_path / "configured.jsonl"
        asserted = tmp_path / "asserted.jsonl"
        configured.write_text("", encoding="utf-8")
        asserted.write_text("", encoding="utf-8")
        recipe = _recipe(
            "ManifestReader",
            {"manifest_path": str(configured)},
            inputs={"manifest_path": str(asserted)},
        )
        binding = resolve_dataset_binding(recipe)
        assert binding.status == "mismatch"
        assert "inputs.manifest_path" in binding.reason

    def test_multi_manifest_has_no_singular_profile_target(self, tmp_path: Path) -> None:
        manifests = [tmp_path / "a.jsonl", tmp_path / "b.jsonl"]
        for manifest in manifests:
            manifest.write_text("", encoding="utf-8")
        recipe = _recipe("ManifestReader", {"manifest_path": [str(path) for path in manifests]})

        binding = resolve_dataset_binding(recipe)
        asserted_member = resolve_dataset_binding(recipe, data=str(manifests[0]))

        assert binding.status == "ambiguous"
        assert binding.configured_paths == tuple(map(str, manifests))
        assert binding.primary_path is None
        assert binding.profile_source is None
        assert binding.selected_manifest_files == tuple(map(str, manifests))
        assert asserted_member.status == "ambiguous"

    def test_multi_manifest_requires_every_local_member(self, tmp_path: Path) -> None:
        present = tmp_path / "present.jsonl"
        present.write_text("", encoding="utf-8")
        missing = tmp_path / "missing.jsonl"

        binding = resolve_dataset_binding(_recipe("ManifestReader", {"manifest_path": [str(present), str(missing)]}))

        assert binding.status == "missing"
        assert binding.selected_manifest_files == ()
        assert "manifest_path[1]" in binding.reason
        assert "selects no local files" in binding.reason

    def test_multi_manifest_cli_outside_list_is_a_mismatch(self, tmp_path: Path) -> None:
        manifests = [tmp_path / "a.jsonl", tmp_path / "b.jsonl"]
        for manifest in manifests:
            manifest.write_text("", encoding="utf-8")
        other = tmp_path / "other.jsonl"
        other.write_text("", encoding="utf-8")
        recipe = _recipe("ManifestReader", {"manifest_path": [str(path) for path in manifests]})
        assert resolve_dataset_binding(recipe, data=str(other)).status == "mismatch"

    def test_missing_local_and_remote_lexical_sources(self, tmp_path: Path) -> None:
        local = resolve_dataset_binding(_recipe("ManifestReader", {"manifest_path": str(tmp_path / "missing.jsonl")}))
        remote = resolve_dataset_binding(
            _recipe("ManifestReader", {"manifest_path": "s3://bucket/path/../manifest.jsonl"})
        )
        assert local.status == "missing"
        assert remote.status == "resolved"
        assert remote.primary_path == "s3://bucket/path/../manifest.jsonl"
        assert remote.profile_source is None
        assert remote.selected_manifest_files == ()

    def test_directory_and_glob_selectors_are_not_profiled_as_audio(self, tmp_path: Path) -> None:
        manifest = tmp_path / "a.jsonl"
        manifest.write_text("", encoding="utf-8")

        directory = resolve_dataset_binding(_recipe("ManifestReader", {"manifest_path": str(tmp_path)}))
        selector = resolve_dataset_binding(_recipe("ManifestReader", {"manifest_path": str(tmp_path / "*.jsonl")}))

        assert directory.status == selector.status == "resolved"
        assert directory.profile_source is selector.profile_source is None
        assert directory.selected_manifest_files == selector.selected_manifest_files == (str(manifest),)

    def test_scalar_directory_recurses_but_explicit_list_member_does_not(self, tmp_path: Path) -> None:
        nested = tmp_path / "nested"
        nested.mkdir()
        manifest = nested / "manifest.jsonl"
        manifest.write_text("", encoding="utf-8")

        scalar = resolve_dataset_binding(_recipe("ManifestReader", {"manifest_path": str(tmp_path)}))
        listed = resolve_dataset_binding(_recipe("ManifestReader", {"manifest_path": [str(tmp_path)]}))

        assert scalar.status == "resolved"
        assert scalar.selected_manifest_files == (str(manifest),)
        assert listed.status == "missing"
        assert "manifest_path[0]" in listed.reason

    def test_selectors_apply_configured_extensions_and_preserve_execution_order(self, tmp_path: Path) -> None:
        json_manifest = tmp_path / "b.jsonl"
        json_manifest.write_text("", encoding="utf-8")
        upper_json_manifest = tmp_path / "a.JSON"
        upper_json_manifest.write_text("", encoding="utf-8")
        text_manifest = tmp_path / "c.txt"
        text_manifest.write_text("", encoding="utf-8")

        default = resolve_dataset_binding(_recipe("ManifestReader", {"manifest_path": str(tmp_path / "*")}))
        custom = resolve_dataset_binding(
            _recipe(
                "ManifestReader",
                {
                    "manifest_path": str(tmp_path / "*"),
                    "file_extensions": [".txt"],
                },
            )
        )

        assert default.status == custom.status == "resolved"
        assert default.selected_manifest_files == (str(upper_json_manifest), str(json_manifest))
        assert custom.selected_manifest_files == (str(text_manifest),)

    def test_existing_wrong_extension_and_no_match_glob_are_missing(self, tmp_path: Path) -> None:
        wrong_extension = tmp_path / "manifest.txt"
        wrong_extension.write_text("", encoding="utf-8")

        wrong = resolve_dataset_binding(_recipe("ManifestReader", {"manifest_path": str(wrong_extension)}))
        no_match = resolve_dataset_binding(_recipe("ManifestReader", {"manifest_path": str(tmp_path / "*.jsonl")}))

        assert wrong.status == no_match.status == "missing"
        assert "file_extensions" in wrong.reason
        assert "selects no local files" in no_match.reason

    def test_explicit_none_extensions_preserves_partitioner_default(self, tmp_path: Path) -> None:
        parquet = tmp_path / "manifest.parquet"
        parquet.write_bytes(b"parquet")

        omitted = resolve_dataset_binding(_recipe("ManifestReader", {"manifest_path": str(parquet)}))
        explicit_none = resolve_dataset_binding(
            _recipe(
                "ManifestReader",
                {
                    "manifest_path": str(parquet),
                    "file_extensions": None,
                },
            )
        )

        assert omitted.status == "missing"
        assert explicit_none.status == "resolved"
        assert explicit_none.selected_manifest_files == (str(parquet),)

    def test_mixed_remote_list_remains_lexical_and_unkeyed(self, tmp_path: Path) -> None:
        local = tmp_path / "local.jsonl"
        local.write_text("", encoding="utf-8")

        binding = resolve_dataset_binding(
            _recipe(
                "ManifestReader",
                {"manifest_path": ["s3://bucket/remote.jsonl", str(local)]},
            )
        )

        assert binding.status == "ambiguous"
        assert binding.selected_manifest_files == ()

    def test_missing_selector_refuses_smoke_before_pipeline_bounding(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from nemo_curator.audio_agent import verbs

        def must_not_bound(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("smoke attempted to bound a missing manifest selector")  # noqa: EM101

        monkeypatch.setattr(verbs, "_bound_recipe", must_not_bound)
        result = verbs.smoke(_recipe("ManifestReader", {"manifest_path": str(tmp_path / "*.jsonl")}))

        assert result["status"] == "refused"
        assert result["data_binding"]["status"] == "missing"


class TestFolderAndGeneratedSources:
    def test_audio_folder_is_local_and_exact(self, tmp_path: Path) -> None:
        binding = resolve_dataset_binding(
            _recipe(
                "CreateInitialManifestAudioFolderStage",
                {"data_dir": str(tmp_path)},
                inputs={"data_dir": tmp_path.as_uri()},
            ),
            data=tmp_path.as_uri(),
        )
        assert binding.status == "resolved"
        assert binding.profile_source == str(tmp_path)
        assert binding.profile_kwargs == {
            "folder_extensions": [".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"],
            "recursive": True,
            "max_files": -1,
            "exclude_stage_intermediates": False,
        }

        remote = resolve_dataset_binding(
            _recipe("CreateInitialManifestAudioFolderStage", {"data_dir": "s3://bucket/audio"})
        )
        assert remote.status == "unsupported"

    @pytest.mark.parametrize(
        ("ref", "params"),
        [
            ("CreateInitialManifestAudioFolderStage", {"data_dir": "file:///tmp/audio"}),
            (
                "CreateInitialManifestFleursStage",
                {
                    "raw_data_dir": "file:///tmp/fleurs",
                    "lang": "en_us",
                    "split": "dev",
                },
            ),
            (
                "CreateInitialManifestReadSpeechStage",
                {"raw_data_dir": "file:///tmp/readspeech"},
            ),
            (
                "ReadLongFormManifestStage",
                {
                    "input_manifest": "file:///tmp/long.jsonl",
                    "audio_dir": "/tmp/audio",  # noqa: S108
                },
            ),
        ],
    )
    def test_os_path_sources_do_not_claim_uri_aliases_execute(
        self,
        ref: str,
        params: dict[str, object],
    ) -> None:
        binding = resolve_dataset_binding(_recipe(ref, params))
        assert binding.status == "unsupported"
        assert "URI syntax" in binding.reason

    def test_os_path_source_does_not_expand_tilde(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        literal = tmp_path / "~" / "audio"
        literal.mkdir(parents=True)
        monkeypatch.chdir(tmp_path)

        binding = resolve_dataset_binding(_recipe("CreateInitialManifestAudioFolderStage", {"data_dir": "~/audio"}))

        assert binding.status == "resolved"
        assert binding.primary_path == str(literal)

    @pytest.mark.parametrize(("auto_download", "expected"), [(True, "resolved"), (False, "missing")])
    def test_unstaged_fleurs_distinguishes_generatable_source(
        self,
        tmp_path: Path,
        auto_download: bool,
        expected: str,
    ) -> None:
        recipe = _recipe(
            "CreateInitialManifestFleursStage",
            {
                "raw_data_dir": str(tmp_path),
                "lang": "en_us",
                "split": "dev",
                "auto_download": auto_download,
            },
        )
        binding = resolve_dataset_binding(recipe)
        assert binding.status == expected
        assert binding.generated is auto_download
        assert binding.primary_path == str(tmp_path / "en_us")
        assert binding.profile_source is None

    def test_fleurs_profiles_effective_language_root_when_staged(self, tmp_path: Path) -> None:
        effective = tmp_path / "hy_am"
        (effective / "test").mkdir(parents=True)
        (effective / "test.tsv").write_text("", encoding="utf-8")
        recipe = _recipe(
            "CreateInitialManifestFleursStage",
            {"raw_data_dir": str(tmp_path), "lang": "hy_am", "split": "test"},
            inputs={"raw_data_dir": tmp_path.as_uri()},
        )
        binding = resolve_dataset_binding(recipe, data=str(effective))
        assert binding.profile_source == str(effective / "test")
        assert binding.profile_kwargs == {
            "identity_files": [str(effective / "test.tsv")],
            "exclude_stage_intermediates": False,
        }

    def test_readspeech_discovers_the_same_nested_wav_root_as_the_stage(self, tmp_path: Path) -> None:
        effective = tmp_path / "data" / "mnt" / "dnsv5" / "clean" / "read_speech"
        effective.mkdir(parents=True)
        (effective / "sample.wav").write_bytes(b"RIFF")
        binding = resolve_dataset_binding(
            _recipe(
                "CreateInitialManifestReadSpeechStage",
                {"raw_data_dir": str(tmp_path), "auto_download": False},
            )
        )
        assert binding.status == "resolved"
        assert binding.primary_path == str(effective)
        assert binding.profile_source == str(effective)
        assert binding.profile_kwargs == {
            "folder_extensions": [".wav"],
            "recursive": True,
            "max_files": 5000,
            "case_sensitive_extensions": True,
            "exclude_stage_intermediates": False,
        }

    def test_unstaged_readspeech_can_be_generated_without_downloading(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "not-created"
        binding = resolve_dataset_binding(
            _recipe(
                "CreateInitialManifestReadSpeechStage",
                {"raw_data_dir": str(raw_dir), "auto_download": True},
            )
        )
        assert binding.status == "resolved"
        assert binding.generated
        assert binding.profile_source is None
        assert not raw_dir.exists()


class TestLongFormSource:
    def test_manifest_is_primary_and_audio_resolution_is_profile_context(self, tmp_path: Path) -> None:
        manifest = tmp_path / "long.jsonl"
        manifest.write_text("{}\n", encoding="utf-8")
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        recipe = _recipe(
            "ReadLongFormManifestStage",
            {
                "input_manifest": str(manifest),
                "audio_dir": str(audio_dir),
                "audio_path_resolution": "relative",
                "audio_filepath_key": "path",
            },
            inputs={"manifest_path": manifest.as_uri(), "audio_dir": str(audio_dir)},
        )

        binding = resolve_dataset_binding(recipe, data=manifest.as_uri())

        assert binding.status == "resolved"
        assert binding.configured_paths == (str(manifest), str(audio_dir))
        assert binding.primary_path == str(manifest)
        assert binding.profile_source == str(manifest)
        assert binding.profile_kwargs == {
            "audio_dir": str(audio_dir),
            "audio_path_resolution": "relative",
            "audio_filepath_key": "path",
        }

    def test_as_is_resolution_does_not_require_audio_dir_to_exist(self, tmp_path: Path) -> None:
        manifest = tmp_path / "long.jsonl"
        manifest.write_text("", encoding="utf-8")
        binding = resolve_dataset_binding(
            _recipe(
                "ReadLongFormManifestStage",
                {
                    "input_manifest": str(manifest),
                    "audio_dir": str(tmp_path / "unused"),
                    "audio_path_resolution": "as_is",
                },
            )
        )
        assert binding.status == "resolved"

    def test_invalid_resolution_and_missing_manifest_are_not_resolved(self, tmp_path: Path) -> None:
        invalid = resolve_dataset_binding(
            _recipe(
                "ReadLongFormManifestStage",
                {
                    "input_manifest": str(tmp_path / "missing.jsonl"),
                    "audio_dir": str(tmp_path),
                    "audio_path_resolution": "invented",
                },
            )
        )
        missing = resolve_dataset_binding(
            _recipe(
                "ReadLongFormManifestStage",
                {"input_manifest": str(tmp_path / "missing.jsonl"), "audio_dir": str(tmp_path)},
            )
        )
        assert invalid.status == "unsupported"
        assert missing.status == "missing"

    def test_resolution_does_not_mutate_recipe_or_stage_defaults(self, tmp_path: Path) -> None:
        manifest = tmp_path / "long.jsonl"
        manifest.write_text("", encoding="utf-8")
        recipe = _recipe(
            "ReadLongFormManifestStage",
            {"input_manifest": str(manifest), "audio_dir": str(tmp_path)},
        )
        before = recipe.to_dict()
        resolve_dataset_binding(recipe)
        assert recipe.to_dict() == before
