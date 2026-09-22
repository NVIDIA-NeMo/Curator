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

"""Validation can see inside a composite, and stops asserting things it cannot see.

The failure these guard against: a pipeline diarizing into ``diar_segments`` fed
``SplitASRAlignJoinStage``, whose inner ``SplitLongAudioStage`` requires ``segments``. Validation
passed clean because the composite declared an empty contract, so two models downloaded, a GPU
diarization pass ran, and only then did the splitter refuse to start.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nemo_curator.stages.audio._agent._composite import expand_composites
from nemo_curator.stages.audio._agent._planning import _forwarding_param, validate_pipeline
from nemo_curator.stages.audio.alm.alm_data_builder import ALMDataBuilderStage
from nemo_curator.stages.audio.common import ManifestReader, ManifestWriterStage
from nemo_curator.stages.audio.inference.speaker_diarization.sortformer import InferenceSortformerStage
from nemo_curator.stages.audio.tagging.resample_audio import ResampleAudioStage
from nemo_curator.stages.audio.tagging.split import SplitASRAlignJoinStage

if TYPE_CHECKING:
    import pytest

_SEEDED = {"audio_filepath", "duration", "resampled_audio_filepath"}


def _codes(report) -> set[str]:  # noqa: ANN001
    return {i.code for i in report.issues}


class TestExpansion:
    def test_a_composite_resolves_to_the_stages_that_will_run(self) -> None:
        expansion = expand_composites([SplitASRAlignJoinStage()])
        assert expansion.fully_resolved
        assert [type(i.stage).__name__ for i in expansion.stages] == [
            "SplitLongAudioStage",
            "NeMoASRAlignerStage",
            "JoinSplitAudioMetadataStage",
        ]
        # Every leaf traces back to the one stage the caller actually wrote.
        assert {i.recipe_index for i in expansion.stages} == {0}

    def test_expansion_carries_the_configured_value_not_the_default(self) -> None:
        # The point of expanding the CONFIGURED composite: a check on the defaults would
        # describe a pipeline nobody asked to run.
        expansion = expand_composites([SplitASRAlignJoinStage(segments_key="diar_segments")])
        assert all(getattr(i.stage, "segments_key", "diar_segments") == "diar_segments" for i in expansion.stages)

    def test_a_leaf_is_reported_as_itself(self) -> None:
        expansion = expand_composites([InferenceSortformerStage()])
        assert expansion.fully_resolved
        assert len(expansion.stages) == 1
        assert expansion.stages[0].composite_ref is None
        assert expansion.stages[0].label == "InferenceSortformerStage"

    def test_a_composite_that_cannot_decompose_yields_no_invented_leaf(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Patched on the instance rather than via a subclass: defining a stage subclass registers
        # it in the global agent-ready registry, which then leaks into every later test.
        stage = SplitASRAlignJoinStage()

        def boom() -> list:
            msg = "cannot plan"
            raise RuntimeError(msg)

        monkeypatch.setattr(stage, "decompose_and_apply_with", boom)
        expansion = expand_composites([stage])
        assert expansion.stages == []
        assert not expansion.fully_resolved
        assert "RuntimeError" in expansion.opaque[0]

    def test_a_single_child_composite_is_an_error_not_a_substitution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``Pipeline._decompose_stages`` only substitutes children when there is more than
        one, so a single-child decomposition leaves the COMPOSITE in the execution list and
        ``CompositeStage.process`` raises on the first task. Substituting the child here would
        model a pipeline the backend never runs and hand back a clean verdict for a recipe that
        dies on contact -- after the user confirmed a full-scale run.
        """
        stage = SplitASRAlignJoinStage()
        only_child = ResampleAudioStage(resampled_audio_dir="/tmp/x")  # noqa: S108
        monkeypatch.setattr(stage, "decompose_and_apply_with", lambda: [only_child])

        expansion = expand_composites([stage])

        assert expansion.stages == [], "no leaf is invented for a stage that cannot run"
        assert not expansion.fully_resolved
        assert 0 not in expansion.opaque, "this is knowable, not merely unresolved"
        assert "single stage" in expansion.unrunnable[0]

        report = validate_pipeline([stage], initial_roles=set(_SEEDED))
        assert not report.ok, "a recipe the executor will refuse must not validate clean"
        assert "composite_unrunnable" in _codes(report)

    def test_a_single_child_that_is_itself_a_composite_is_still_unrunnable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The executor's nested-composite rejection lives inside its ``len(sub_stages) > 1``
        branch, so it is never reached for a single child.

        Asking the nested question first inverted the verdict here: reported as opaque, "we
        cannot tell", when the executor can tell perfectly well -- it leaves the outer
        composite in the list and raises on the first task.
        """
        stage = SplitASRAlignJoinStage()
        monkeypatch.setattr(stage, "decompose_and_apply_with", lambda: [SplitASRAlignJoinStage()])

        expansion = expand_composites([stage])

        assert 0 not in expansion.opaque, "this is knowable, not merely unresolved"
        assert "single stage" in expansion.unrunnable[0]


class TestInnerRequirementsAreVisible:
    def test_the_key_mismatch_that_survived_validation_is_now_reported(self) -> None:
        report = validate_pipeline(
            [InferenceSortformerStage(), SplitASRAlignJoinStage()],
            initial_keys=_SEEDED,
        )
        assert "unsatisfied_reads_in_composite" in _codes(report)

    def test_the_report_names_the_inner_stage_and_the_parameter_that_fixes_it(self) -> None:
        report = validate_pipeline(
            [InferenceSortformerStage(), SplitASRAlignJoinStage()],
            initial_keys=_SEEDED,
        )
        issue = next(i for i in report.issues if i.code == "unsatisfied_reads_in_composite")
        # Naming only the inner stage leaves the caller stuck: they configured the composite and
        # cannot reach SplitLongAudioStage directly.
        assert "SplitASRAlignJoinStage" in issue.stage_name
        assert "SplitLongAudioStage" in issue.stage_name
        assert "segments_key" in issue.message

    def test_configuring_the_composite_correctly_silences_it(self) -> None:
        report = validate_pipeline(
            [InferenceSortformerStage(), SplitASRAlignJoinStage(segments_key="diar_segments")],
            initial_keys=_SEEDED,
        )
        assert "unsatisfied_reads_in_composite" not in _codes(report)

    def test_a_list_valued_parameter_does_not_crash_the_search_for_a_remedy(self) -> None:
        """Hunting for the composite parameter to blame tests each shared field against the set
        of missing keys, and testing a value against a set hashes it. Real stages share list and
        dict parameters (``ManifestReader`` and its ``FilePartitioningStage`` child already share
        ``file_extensions`` and ``storage_options``), so that raises TypeError -- which escapes
        ``run_checks`` and kills the verb, handing the caller a traceback instead of a verdict
        over a remedy hint that was never going to apply to a list anyway.
        """
        composite = ManifestReader(manifest_path="/tmp/m.jsonl")  # noqa: S108
        inner = next(child for child in composite.decompose() if hasattr(child, "file_extensions"))
        assert isinstance(inner.file_extensions, list), "the shared field really is unhashable"

        assert _forwarding_param(inner, composite, {"segments"}) is None

    def test_a_shared_string_parameter_is_still_offered_as_the_remedy(self) -> None:
        composite = SplitASRAlignJoinStage()
        inner = composite.decompose()[0]

        assert _forwarding_param(inner, composite, {inner.segments_key}) == "segments_key"

    def test_an_inner_requirement_is_a_warning_not_a_block(self) -> None:
        # Expansion is new; it earns the right to hard-fail a pipeline only after it has been
        # shown not to false-positive. Until then it informs and does not stop anyone.
        report = validate_pipeline(
            [InferenceSortformerStage(), SplitASRAlignJoinStage()],
            initial_keys=_SEEDED,
        )
        assert report.ok, report.summary()


class TestUnreadableChildrenStayOpaque:
    def test_a_composite_containing_unannotated_plumbing_does_not_blame_the_caller(self) -> None:
        # ManifestReader expands through FilePartitioningStage, which has no describe() at all.
        # Reporting that as an error would fail the simplest working pipeline there is, naming a
        # stage the caller never wrote.
        from nemo_curator.stages.audio.common import ManifestReader

        report = validate_pipeline([ManifestReader(manifest_path="x.jsonl")], initial_keys=_SEEDED)
        assert "contract_error" not in _codes(report)
        assert "composite" in _codes(report)
        assert report.ok, report.summary()

    def test_the_siblings_that_do_describe_themselves_are_still_used(self) -> None:
        """One illegible child made the whole composite invisible.

        ``ManifestReader`` expands through ``FilePartitioningStage``, which has no describe(),
        so the reader that starts nearly every recipe contributed NOTHING -- not the keys its
        legible siblings write, not their gate checks. Unknown is the right verdict for the
        unknown part only; the rest is fact and is worth keeping.
        """
        from nemo_curator.stages.audio.common import ManifestReader

        report = validate_pipeline([ManifestReader(manifest_path="x.jsonl")], initial_keys=set())

        assert report.produced_keys, "the legible siblings' writes were thrown away with the rest"
        assert report.ok, report.summary()


class TestDroppedTensorDoesNotReachTheSink:
    """A hard block that fires on data the pipeline has already discarded is worse than none:
    the only way past it is to fake a value, which is what nearly shipped."""

    def _pipeline(self, drop: str) -> list:
        return [
            ResampleAudioStage(
                resampled_audio_dir="rs",
                target_sample_rate=16000,
                write_to_disk=True,
                update_audio_filepath=True,
                keep_waveform_in_task=True,
            ),
            ALMDataBuilderStage(
                min_speakers=1,
                min_bandwidth=0,
                audio_sample_rate_key="sample_rate",
                drop_fields_top_level=drop,
            ),
            ManifestWriterStage(output_path="out.jsonl"),
        ]

    def test_a_resident_waveform_reaching_a_json_sink_is_still_an_error(self) -> None:
        report = validate_pipeline(self._pipeline("words,segments"), initial_keys={*_SEEDED, "segments"})
        assert "tensor_into_sink" in _codes(report)

    def test_dropping_the_waveform_clears_the_sink_gate(self) -> None:
        report = validate_pipeline(
            self._pipeline("words,segments,waveform"),
            initial_keys={*_SEEDED, "segments"},
        )
        assert "tensor_into_sink" not in _codes(report)


class TestKeyIdentitySatisfiesAReadRoleNamingDoesNot:
    def test_reading_a_key_that_exists_is_not_a_break(self) -> None:
        # A diarizer writes diar_segments under the role `diar_segments`; a consumer configured to
        # read diar_segments calls that slot `segments`. Runtime reads by key, so this runs.
        report = validate_pipeline(
            [InferenceSortformerStage(), SplitASRAlignJoinStage(segments_key="diar_segments")],
            initial_keys=_SEEDED,
        )
        assert not report.errors, report.summary()

    def test_an_empty_role_seed_also_means_an_empty_key_seed(self) -> None:
        # Both seeds describe one input task, so a caller who says "no roles" is not also saying
        # "but the default columns are present".
        from nemo_curator.stages.audio.common import GetAudioDurationStage

        report = validate_pipeline([GetAudioDurationStage()], initial_roles=set())
        assert not report.ok

    def test_a_role_with_no_column_of_that_name_is_not_seeded_as_one(self) -> None:
        """Roles and key values are different vocabularies. They coincide for the conventional
        ones -- ``audio_filepath``, ``pred_text`` -- and part company for the rest.

        A caller describing a task that carries speaker information says the role
        ``speaker``; the column is called something else entirely (``speaker_id``, ``spk``).
        Seeding the role name straight into the key set invented a column the task does not
        carry, and the literal-key route then declared satisfied a read of it.
        """
        from nemo_curator.stages.audio._agent._roles import role_for_value

        assert role_for_value("speaker") == "unknown", "no column is conventionally named this"

        report = validate_pipeline(
            [ManifestWriterStage(output_path="out.jsonl")],
            initial_roles={"audio_filepath", "speaker"},
        )

        assert "speaker" not in report.produced_keys
        assert "audio_filepath" in report.produced_keys, "a role that IS its own column stays"
