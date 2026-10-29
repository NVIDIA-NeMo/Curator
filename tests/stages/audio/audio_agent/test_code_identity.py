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

"""Code identity is per stage, so a commit elsewhere cannot empty the artifact store.

The stamp used to be ``nemo_curator.__version__``, which ends in the repository's git SHA.
It was wrong in both directions: a README commit invalidated every artifact, and editing a
stage without committing invalidated none. These pin down the replacement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from nemo_curator.stages.audio.audio_agent import artifacts, code_identity
from nemo_curator.stages.audio.audio_agent.recipe import Recipe

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_STAGE = "nemo_curator.stages.audio.fake_stage"
_TREE = {
    "stages/audio/fake_stage.py": (
        "from nemo_curator.stages.audio import fake_helper\n"
        "from nemo_curator.tasks.audio_task import AudioTask\n"
        "import third_party_thing\n"
    ),
    "stages/audio/fake_helper.py": "VALUE = 1\n",
    "stages/audio/__init__.py": "",
    "tasks/audio_task.py": "class AudioTask:\n    pass\n",
    # Never imported by the stage: the whole point is that editing this changes nothing.
    "stages/video/unrelated.py": "SOMETHING = 1\n",
    "stages/audio/audio_agent/__init__.py": "",
    "stages/audio/audio_agent/_verb_pretrain.py": "DRIVER_VERSION = 1\n",
    "stages/audio/alm/pretrain/finalize.py": "MERGER_VERSION = 1\n",
    "models/asr/dynamic_adapter.py": (
        "from nemo_curator.models.asr import adapter_helper\nclass DynamicAdapter:\n    VALUE = 1\n"
    ),
    "models/asr/adapter_helper.py": "VALUE = 1\n",
}


@pytest.fixture
def package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[str, str], None]:
    """A fake ``nemo_curator`` source tree the closure walk reads for real."""
    for rel, src in _TREE.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(src, encoding="utf-8")

    monkeypatch.setattr(
        "nemo_curator.stages.audio.audio_agent._resolve.resolve_stage_class",
        lambda _ref: type("FakeStage", (), {"__module__": _STAGE}),
    )

    def edit(rel: str, src: str) -> None:
        (tmp_path / rel).write_text(src, encoding="utf-8")
        code_identity._reset_caches()
        code_identity._root_cache.append(str(tmp_path))

    code_identity._reset_caches()
    code_identity._root_cache.append(str(tmp_path))
    yield edit
    code_identity._reset_caches()


class TestWhatTheStampCovers:
    @pytest.mark.parametrize(
        "stage", ["SnippetExtractionStage", "SnippetManifestWriterStage", "PretrainMetricsAggregatorStage"]
    )
    def test_shared_merger_changes_invalidate_alm_artifacts(
        self, package: Callable[[str, str], None], stage: str
    ) -> None:
        before = code_identity.impl_version(stage)
        assert not code_identity.is_fallback(before)
        package("stages/audio/alm/pretrain/finalize.py", "MERGER_VERSION = 2\n")
        assert code_identity.impl_version(stage) != before

    @pytest.mark.parametrize(
        "stage", ["SnippetExtractionStage", "SnippetManifestWriterStage", "PretrainMetricsAggregatorStage"]
    )
    def test_driver_finalization_changes_invalidate_alm_artifacts(
        self, package: Callable[[str, str], None], stage: str
    ) -> None:
        before = code_identity.impl_version(stage)
        assert not code_identity.is_fallback(before)
        package("stages/audio/audio_agent/_verb_pretrain.py", "DRIVER_VERSION = 2\n")
        assert code_identity.impl_version(stage) != before

    def test_a_stages_own_source_decides_its_version(self, package: Callable[[str, str], None]) -> None:
        before = code_identity.impl_version("FakeStage")
        package("stages/audio/fake_stage.py", _TREE["stages/audio/fake_stage.py"] + "CHANGED = 1\n")

        assert code_identity.impl_version("FakeStage") != before

    def test_a_module_the_stage_imports_decides_it_too(self, package: Callable[[str, str], None]) -> None:
        # A helper is where behaviour hides: reading only the stage's own file would serve
        # results produced by arithmetic that has since changed.
        before = code_identity.impl_version("FakeStage")
        package("stages/audio/fake_helper.py", "VALUE = 2\n")

        assert code_identity.impl_version("FakeStage") != before

    def test_code_the_stage_never_imports_does_not(self, package: Callable[[str, str], None]) -> None:
        # The whole point of the change: a commit to another modality must leave audio alone.
        before = code_identity.impl_version("FakeStage")
        package("stages/video/unrelated.py", "SOMETHING = 2\n")

        assert code_identity.impl_version("FakeStage") == before

    def test_an_import_inside_a_function_body_still_counts(
        self,
        package: Callable[[str, str], None],
    ) -> None:
        # Stages import their heavy dependencies inside setup(), so a walk over the module
        # namespace would miss exactly the code that does the work.
        package(
            "stages/audio/fake_stage.py",
            "def setup():\n    from nemo_curator.stages.audio import fake_helper\n    return fake_helper\n",
        )
        before = code_identity.impl_version("FakeStage")
        package("stages/audio/fake_helper.py", "VALUE = 3\n")

        assert code_identity.impl_version("FakeStage") != before

    def test_the_same_sources_always_give_the_same_answer(
        self,
        package: Callable[[str, str], None],
    ) -> None:
        first = code_identity.impl_version("FakeStage")
        package("stages/audio/fake_helper.py", _TREE["stages/audio/fake_helper.py"])  # rewrite, same bytes

        assert code_identity.impl_version("FakeStage") == first


class TestWhatHappensWhenItCannotTell:
    def test_an_unresolvable_stage_falls_back_to_the_package_build(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Unknown code must over-invalidate, never over-reuse: the package version moves with
        # every commit, which is the safe direction to be wrong in.
        code_identity._reset_caches()
        monkeypatch.setattr(
            "nemo_curator.stages.audio.audio_agent._resolve.resolve_stage_class",
            lambda _ref: (_ for _ in ()).throw(KeyError("no such stage")),
        )

        assert code_identity.impl_version("NoSuchStage") == f"pkg:{artifacts.code_version()}"
        code_identity._reset_caches()

    def test_an_unbounded_closure_falls_back_rather_than_hashing_part_of_it(
        self,
        package: Callable[[str, str], None],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Hashing a truncated closure would silently drop the file that changed.
        monkeypatch.setattr(code_identity, "_CLOSURE_LIMIT", 1)
        code_identity._version_cache.clear()

        assert code_identity.impl_version("FakeStage").startswith("pkg:")


class TestWhatItMeansForReuse:
    def test_editing_one_stage_invalidates_it_and_everything_below_it(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rec = Recipe.from_dict(
            {
                "stages": [
                    {"ref": "ManifestReader", "params": {"manifest_path": "/data/m.jsonl"}},
                    {"ref": "GetAudioDurationStage", "params": {}},
                    {"ref": "ManifestWriterStage", "params": {"output_path": "/data/out.jsonl"}},
                ]
            }
        ).freeze()
        before = artifacts.step_keys(rec, "stat:abc")
        real = artifacts.impl_version
        monkeypatch.setattr(
            artifacts,
            "impl_version",
            lambda ref: "impl:edited" if ref == "GetAudioDurationStage" else real(ref),
        )

        after = artifacts.step_keys(rec, "stat:abc")

        assert after[0] == before[0]  # the reader is untouched code, so its work survives
        assert after[1] != before[1]
        assert after[2] != before[2]  # the chain carries the change down

    def test_two_stages_from_one_module_still_key_apart(self) -> None:
        # Sharing a source file means sharing an impl_version; the stage ref is what keeps
        # their step keys distinct.
        assert code_identity.impl_version("ManifestWriterStage") == code_identity.impl_version("ManifestReader")
        rec = Recipe.from_dict(
            {
                "stages": [
                    {"ref": "ManifestReader", "params": {"manifest_path": "/data/m.jsonl"}},
                    {"ref": "ManifestWriterStage", "params": {"output_path": "/data/out.jsonl"}},
                ]
            }
        ).freeze()
        keys = artifacts.step_keys(rec, "stat:abc")

        assert keys[0] != keys[1]


@pytest.mark.parametrize("edited_module", ["dynamic_adapter", "adapter_helper"])
def test_default_adapter_edit_invalidates_only_its_step_and_suffix(
    package: Callable[[str, str], None], monkeypatch: pytest.MonkeyPatch, edited_module: str
) -> None:
    target = "nemo_curator.models.asr.dynamic_adapter.DynamicAdapter"
    monkeypatch.setattr(
        "nemo_curator.stages.audio.audio_agent._resolve.resolve_stage_class",
        lambda ref: type(
            ref,
            (),
            {"__module__": _STAGE, **({"adapter_target": target} if ref == "DefaultAdapterStage" else {})},
        ),
    )
    recipe = Recipe.from_dict(
        {"stages": [{"ref": "FakeStage"}, {"ref": "DefaultAdapterStage"}, {"ref": "FakeStage"}]}
    ).freeze()
    before = artifacts.step_keys(recipe, "stat:fixture")
    package(f"models/asr/{edited_module}.py", _TREE[f"models/asr/{edited_module}.py"] + "CHANGED = 2\n")
    after = artifacts.step_keys(recipe, "stat:fixture")
    assert before[0] == after[0]
    assert before[1] != after[1]
    assert before[2] != after[2]


def test_default_vad_binds_the_same_adapter_implementation_as_an_explicit_selection() -> None:
    from nemo_curator.stages.audio.segmentation.vad_segmentation import VADSegmentationStage

    implicit = code_identity.impl_version("VADSegmentationStage")
    explicit = code_identity.impl_version("VADSegmentationStage", adapter_target=VADSegmentationStage.adapter_target)
    assert not code_identity.is_fallback(implicit)
    assert implicit == explicit


@pytest.mark.parametrize("edited_module", ["dynamic_adapter", "adapter_helper"])
def test_filter_configured_adapter_code_is_part_of_its_effective_identity(
    package: Callable[[str, str], None], edited_module: str
) -> None:
    params = {"config": {"vad": {"adapter_target": "nemo_curator.models.asr.dynamic_adapter.DynamicAdapter"}}}
    before = artifacts.model_version(params, stage_ref="AudioDataFilterStage")
    assert before == artifacts.model_version(params, stage_ref="AudioDataFilterStage")
    package(f"models/asr/{edited_module}.py", _TREE[f"models/asr/{edited_module}.py"] + "CHANGED = 2\n")
    assert artifacts.model_version(params, stage_ref="AudioDataFilterStage") != before


@pytest.mark.usefixtures("package")
def test_filter_with_unavailable_adapter_code_cannot_prove_reusable_identity() -> None:
    params = {"config": {"vad": {"adapter_target": "nemo_curator.models.asr.missing.MissingAdapter"}}}
    assert artifacts.model_version(params, stage_ref="AudioDataFilterStage") != artifacts.model_version(
        params, stage_ref="AudioDataFilterStage"
    )


@pytest.mark.parametrize("edited_module", ["dynamic_adapter", "adapter_helper"])
def test_configured_adapter_edit_invalidates_only_its_step_and_suffix(
    package: Callable[[str, str], None], edited_module: str
) -> None:
    target = "nemo_curator.models.asr.dynamic_adapter.DynamicAdapter"
    recipe = Recipe.from_dict(
        {
            "stages": [
                {"ref": "FakeStage"},
                {"ref": "FakeStage", "params": {"adapter_target": target}},
                {"ref": "FakeStage"},
            ]
        }
    ).freeze()
    before = artifacts.step_keys(recipe, "stat:fixture")
    stamp = code_identity.impl_version("FakeStage", adapter_target=target)
    artifact = artifacts.Artifact(
        stage_ref="FakeStage", step_key=before[1], impl_version=stamp, semantic_params={"adapter_target": target}
    )
    assert not artifacts._impl_reasons(artifact)

    package(f"models/asr/{edited_module}.py", "VALUE = 2\n")

    after = artifacts.step_keys(recipe, "stat:fixture")
    assert before[0] == after[0]
    assert before[1] != after[1]
    assert before[2] != after[2]
    assert artifacts._impl_reasons(artifact) == ["FakeStage's implementation changed since this artifact was produced"]


@pytest.mark.parametrize("target", ["third_party.Adapter", "nemo_curator.missing.Adapter", "Adapter"])
@pytest.mark.usefixtures("package")
def test_unidentifiable_adapter_refuses_reuse_even_with_matching_fallback(
    target: str,
) -> None:
    stamp = code_identity.impl_version("FakeStage", adapter_target=target)
    assert code_identity.is_fallback(stamp)
    artifact = artifacts.Artifact(
        stage_ref="FakeStage", step_key="fixture", impl_version=stamp, semantic_params={"adapter_target": target}
    )

    assert "implementation is unproven" in artifacts._impl_reasons(artifact)[0]
