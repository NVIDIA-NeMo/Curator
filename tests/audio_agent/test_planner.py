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

"""Unit tests for the resource planner: mode selection + composite expansion."""

import pytest

from nemo_curator.audio_agent import planner
from nemo_curator.audio_agent.context import assemble
from nemo_curator.audio_agent.profiler import probe_env
from nemo_curator.audio_agent.recipe import Recipe, build_stages
from nemo_curator.stages.audio import agent as foundation


def _plan(stages: list[dict]):  # noqa: ANN202
    built, _issues = build_stages(Recipe.from_dict({"stages": stages}))
    assert built is not None, f"recipe failed to build: {stages}"
    contracts = [foundation.build_contract(s) for s in built]
    return planner.plan(built, contracts, probe_env())


class TestPlanner:
    def test_cpu_only_recipe_streams(self) -> None:
        p = _plan(
            [
                {"ref": "ManifestReader", "params": {"manifest_path": "/tmp/m.jsonl"}},  # noqa: S108
                {"ref": "GetAudioDurationStage", "params": {}},
            ]
        )
        assert p.mode == "streaming"
        assert p.feasible is True

    def test_composite_is_expanded_for_resource_planning(self) -> None:
        # SplitASRAlignJoinStage is a composite hiding an inner GPU ASR aligner; the
        # planner must flatten it (via decompose()) so the concurrent GPU reservation is
        # not under-counted. The expansion note proves the composite was accounted for.
        p = _plan(
            [
                {"ref": "ManifestReader", "params": {"manifest_path": "/tmp/m.jsonl"}},  # noqa: S108
                {"ref": "ResampleAudioStage", "params": {"resampled_audio_dir": "/tmp/r"}},  # noqa: S108
                {"ref": "VADSegmentationStage", "params": {}},
                {"ref": "SplitASRAlignJoinStage", "params": {}},
                {"ref": "ManifestWriterStage", "params": {"output_path": "/tmp/o.jsonl"}},  # noqa: S108
            ]
        )
        assert any("expanded" in n for n in p.notes)

    def test_estimate_reports_reservation_sum(self) -> None:
        p = _plan(
            [
                {"ref": "ManifestReader", "params": {"manifest_path": "/tmp/m.jsonl"}},  # noqa: S108
                {"ref": "GetAudioDurationStage", "params": {}},
            ]
        )
        assert "sum_gpu_reservation" in p.estimate
        assert p.estimate["sum_gpu_reservation"] >= 0


def test_planning_context_carries_the_workflow_preference() -> None:
    preference = {
        "schema_version": 1,
        "curation_mode": "fast_first",
        "source": "inferred_from_request",
    }

    packet = assemble(
        {"task": "quality_filter"},
        include_env=False,
        planning_preference=preference,
    ).to_dict()

    assert packet["planning_preference"] == preference


@pytest.mark.parametrize(
    "preference",
    [
        ["refine_later"],
        {
            "schema_version": 1,
            "curation_mode": ["refine_later"],
            "source": "explicit_user_choice",
        },
        {
            "schema_version": 1,
            "curation_mode": "fast_first",
            "source": {"kind": "inferred_from_request"},
        },
    ],
)
def test_planning_context_rejects_malformed_preference_with_value_error(
    preference: object,
) -> None:
    with pytest.raises(ValueError, match="planning_preference"):
        assemble(
            {"task": "quality_filter"},
            include_env=False,
            planning_preference=preference,  # type: ignore[arg-type]
        )
