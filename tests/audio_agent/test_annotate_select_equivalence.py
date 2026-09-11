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

"""Annotating and then selecting must keep the rows a native filter would have kept.

The agent plans quality gates two ways. A stage can filter natively, dropping rows as it
scores them, or it can annotate -- writing the score and keeping every row -- and leave the
dropping to a later selector. Those two shapes are what make the action-keyed design usable:
the agent picks annotate when a downstream stage still needs the rows, and it may only do so
because the outcome is the same either way.

These live here rather than beside the filters because each one drives TWO stages and asserts
a property of the pipeline, not of a module: the module tests own what SIGMOS and UTMOS score,
this owns whether the two routes agree. Relocated from tests/stages/audio/filtering/ for that
reason -- they were the bulk of those files' growth and belong to the agent's concern.
"""

from unittest.mock import MagicMock, patch

import torch

from nemo_curator.stages.audio.common import PreserveByValueConditionsStage, PreserveByValueStage
from nemo_curator.stages.audio.filtering.sigmos import SIGMOSFilterStage
from nemo_curator.stages.audio.filtering.utmos import UTMOSFilterStage
from nemo_curator.tasks import AudioTask

_GOOD_SCORES = {
    "MOS_NOISE": 4.5,
    "MOS_OVRL": 4.0,
    "MOS_SIG": 4.2,
    "MOS_COL": 4.1,
    "MOS_DISC": 4.3,
    "MOS_LOUD": 3.8,
    "MOS_REVERB": 4.0,
}

_BAD_SCORES = {
    "MOS_NOISE": 2.0,
    "MOS_OVRL": 2.0,
    "MOS_SIG": 2.0,
    "MOS_COL": 2.0,
    "MOS_DISC": 2.0,
    "MOS_LOUD": 2.0,
    "MOS_REVERB": 2.0,
}


def _make_task(duration_s: float = 1.0, sample_rate: int = 48000) -> AudioTask:
    num_samples = int(duration_s * sample_rate)
    return AudioTask(
        data={"waveform": torch.randn(1, num_samples), "sample_rate": sample_rate},
        dataset_name="test",
    )


def _make_mock_model(scores: dict) -> MagicMock:
    model = MagicMock()
    model.run.return_value = scores
    return model


class TestSIGMOSAnnotateThenSelect:
    """Annotating with SIGMOS then selecting matches what native filtering keeps."""

    @patch.object(SIGMOSFilterStage, "_initialize_model")
    def test_task_annotate_then_compound_selector_matches_all_native_thresholds(
        self,
        mock_init: MagicMock,
    ) -> None:
        keys = {
            "noise_key": "quality_noise",
            "ovrl_key": "quality_ovrl",
            "reverb_key": "quality_reverb",
        }
        thresholds = {
            "noise_threshold": 4.0,
            "ovrl_threshold": 3.5,
            "reverb_threshold": 3.8,
            "sig_threshold": None,
            "col_threshold": None,
            "disc_threshold": None,
            "loud_threshold": None,
        }
        cases = [
            ("passing", _make_task(), _GOOD_SCORES),
            (
                "noise_fails",
                _make_task(),
                {**_GOOD_SCORES, "MOS_NOISE": 3.9},
            ),
            (
                "ovrl_fails",
                _make_task(),
                {**_GOOD_SCORES, "MOS_OVRL": 3.4},
            ),
            (
                "reverb_fails",
                _make_task(),
                {**_GOOD_SCORES, "MOS_REVERB": 3.7},
            ),
            (
                "nan_dimension",
                _make_task(),
                {**_GOOD_SCORES, "MOS_SIG": float("nan")},
            ),
            (
                "positive_inf_dimension",
                _make_task(),
                {**_GOOD_SCORES, "MOS_COL": float("inf")},
            ),
            (
                "negative_inf_dimension",
                _make_task(),
                {**_GOOD_SCORES, "MOS_DISC": float("-inf")},
            ),
            ("unscorable", AudioTask(data={"id": "missing"}), None),
        ]

        for case_id, source, scores in cases:
            source.data["id"] = case_id
            if "dimension" in case_id:
                source.data.update(
                    {
                        "quality_noise": 99.0,
                        "quality_ovrl": 99.0,
                        "quality_reverb": 99.0,
                        "sigmos_sig": 99.0,
                        "sigmos_col": 99.0,
                        "sigmos_disc": 99.0,
                        "sigmos_loud": 99.0,
                    }
                )
            native = SIGMOSFilterStage(
                action="filter",
                mode="task",
                **thresholds,
                **keys,
            )
            annotate = SIGMOSFilterStage(
                action="annotate",
                mode="task",
                **thresholds,
                **keys,
            )
            if scores is not None:
                native._model = _make_mock_model(scores)
                annotate._model = _make_mock_model(scores)

            native_result = native.process(AudioTask(data=dict(source.data), dataset_name=source.dataset_name))
            annotated = annotate.process(AudioTask(data=dict(source.data), dataset_name=source.dataset_name))
            assert isinstance(annotated, AudioTask)
            if "dimension" in case_id:
                assert not any(key.startswith(("quality_", "sigmos_")) for key in annotated.data)
            selected = PreserveByValueConditionsStage(
                conditions=[
                    {
                        "input_value_key": "quality_noise",
                        "target_value": 4.0,
                        "operator": "ge",
                    },
                    {
                        "input_value_key": "quality_ovrl",
                        "target_value": 3.5,
                        "operator": "ge",
                    },
                    {
                        "input_value_key": "quality_reverb",
                        "target_value": 3.8,
                        "operator": "ge",
                    },
                ],
                missing_value_policy="drop",
            ).process_batch([annotated])

            assert bool(selected) is isinstance(native_result, AudioTask)

    @patch.object(SIGMOSFilterStage, "_initialize_model")
    def test_segment_annotate_then_compound_selector_matches_native_filter(
        self,
        mock_init: MagicMock,
    ) -> None:
        sample_rate = 48000
        thresholds = {
            "noise_threshold": 4.0,
            "ovrl_threshold": 3.5,
            "sig_threshold": None,
            "col_threshold": None,
            "disc_threshold": None,
            "loud_threshold": None,
            "reverb_threshold": 3.8,
        }
        keys = {
            "noise_key": "quality_noise",
            "ovrl_key": "quality_ovrl",
            "reverb_key": "quality_reverb",
        }

        def make_parent() -> AudioTask:
            return AudioTask(
                data={
                    "recording": "r1",
                    "clips": [
                        {
                            "id": "pass",
                            "waveform": torch.randn(1, sample_rate),
                            "sample_rate": sample_rate,
                        },
                        {
                            "id": "fail",
                            "waveform": torch.randn(1, sample_rate),
                            "sample_rate": sample_rate,
                        },
                        {
                            "id": "nan",
                            "waveform": torch.randn(1, sample_rate),
                            "sample_rate": sample_rate,
                            "quality_noise": 99.0,
                            "sigmos_sig": 99.0,
                        },
                        {
                            "id": "inf",
                            "waveform": torch.randn(1, sample_rate),
                            "sample_rate": sample_rate,
                            "quality_noise": 99.0,
                            "sigmos_sig": 99.0,
                        },
                        {"id": "unscorable"},
                    ],
                },
                dataset_name="test",
            )

        def sequence_model() -> MagicMock:
            model = MagicMock()
            model.run.side_effect = [
                _GOOD_SCORES,
                {**_GOOD_SCORES, "MOS_REVERB": 3.7},
                {**_GOOD_SCORES, "MOS_SIG": float("nan")},
                {**_GOOD_SCORES, "MOS_COL": float("inf")},
            ]
            return model

        native = SIGMOSFilterStage(
            action="filter",
            mode="segments",
            segments_key="clips",
            **thresholds,
            **keys,
        )
        annotate = SIGMOSFilterStage(
            action="annotate",
            mode="segments",
            segments_key="clips",
            **thresholds,
            **keys,
        )
        native._model = sequence_model()
        annotate._model = sequence_model()

        native_result = native.process(make_parent())
        annotated = annotate.process(make_parent())
        assert isinstance(native_result, AudioTask)
        assert isinstance(annotated, AudioTask)
        invalid = {item["id"]: item for item in annotated.data["clips"] if item["id"] in {"nan", "inf"}}
        assert all(not any(key.startswith(("quality_", "sigmos_")) for key in item) for item in invalid.values())
        selected = PreserveByValueConditionsStage(
            conditions=[
                {"input_value_key": "quality_noise", "target_value": 4.0, "operator": "ge"},
                {"input_value_key": "quality_ovrl", "target_value": 3.5, "operator": "ge"},
                {"input_value_key": "quality_reverb", "target_value": 3.8, "operator": "ge"},
            ],
            missing_value_policy="drop",
            items_key="clips",
            drop_parent_if_empty=True,
        ).process_batch([annotated])

        assert [item["id"] for item in native_result.data["clips"]] == ["pass"]
        assert [item["id"] for item in selected[0].data["clips"]] == ["pass"]


def _utmos_task(duration_s: float = 1.0, sample_rate: int = 16000) -> AudioTask:
    num_samples = int(duration_s * sample_rate)
    return AudioTask(
        data={"waveform": torch.randn(1, num_samples), "sample_rate": sample_rate},
        dataset_name="test",
    )


def _utmos_mock_model(score: float) -> MagicMock:
    model = MagicMock()
    model.return_value = torch.tensor([score])
    model.parameters = lambda: iter([torch.tensor([0.0])])
    return model


class TestUTMOSAnnotateThenSelect:
    """Annotating with UTMOS then selecting matches what native filtering keeps."""

    @patch("nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage._ensure_model")
    def test_task_annotate_then_drop_selector_matches_native_filter(
        self,
        mock_ensure: MagicMock,
    ) -> None:
        threshold = 3.5
        cases = [
            ("passing", _utmos_task(), 4.2),
            ("failing", _utmos_task(), 2.8),
            ("nan", _utmos_task(), float("nan")),
            ("positive_inf", _utmos_task(), float("inf")),
            ("negative_inf", _utmos_task(), float("-inf")),
            ("unscorable", AudioTask(data={"id": "missing"}), None),
        ]

        for case_id, source, score in cases:
            source.data["id"] = case_id
            if case_id in {"nan", "positive_inf", "negative_inf"}:
                source.data["custom_utmos"] = 99.0
            native = UTMOSFilterStage(
                action="filter",
                mode="task",
                mos_threshold=threshold,
                score_key="custom_utmos",
            )
            annotate = UTMOSFilterStage(
                action="annotate",
                mode="task",
                mos_threshold=threshold,
                score_key="custom_utmos",
            )
            if score is not None:
                native._model = _utmos_mock_model(score)
                annotate._model = _utmos_mock_model(score)

            native_result = native.process(AudioTask(data=dict(source.data), dataset_name=source.dataset_name))
            annotated = annotate.process(AudioTask(data=dict(source.data), dataset_name=source.dataset_name))
            assert isinstance(annotated, AudioTask)
            if case_id in {"nan", "positive_inf", "negative_inf"}:
                assert "custom_utmos" not in annotated.data
            selected = PreserveByValueStage(
                input_value_key="custom_utmos",
                target_value=threshold,
                operator="ge",
                missing_value_policy="drop",
            ).process_batch([annotated])

            assert bool(selected) is isinstance(native_result, AudioTask)

    @patch("nemo_curator.stages.audio.filtering.utmos.UTMOSFilterStage._ensure_model")
    def test_segment_annotate_then_generic_selector_matches_native_filter(
        self,
        mock_ensure: MagicMock,
    ) -> None:
        threshold = 3.5
        sample_rate = 16000

        def make_parent() -> AudioTask:
            return AudioTask(
                data={
                    "recording": "r1",
                    "clips": [
                        {
                            "id": "pass",
                            "waveform": torch.randn(1, sample_rate),
                            "sample_rate": sample_rate,
                        },
                        {
                            "id": "fail",
                            "waveform": torch.randn(1, sample_rate),
                            "sample_rate": sample_rate,
                        },
                        {
                            "id": "nan",
                            "waveform": torch.randn(1, sample_rate),
                            "sample_rate": sample_rate,
                            "quality": 99.0,
                        },
                        {
                            "id": "inf",
                            "waveform": torch.randn(1, sample_rate),
                            "sample_rate": sample_rate,
                            "quality": 99.0,
                        },
                        {"id": "unscorable"},
                    ],
                },
                dataset_name="test",
            )

        def sequence_model() -> MagicMock:
            scores = iter([4.2, 2.8, float("nan"), float("inf")])
            model = MagicMock(side_effect=lambda *_args, **_kwargs: torch.tensor([next(scores)]))
            model.parameters = lambda: iter([torch.tensor([0.0])])
            return model

        native = UTMOSFilterStage(
            action="filter",
            mode="segments",
            segments_key="clips",
            score_key="quality",
            mos_threshold=threshold,
        )
        annotate = UTMOSFilterStage(
            action="annotate",
            mode="segments",
            segments_key="clips",
            score_key="quality",
            mos_threshold=threshold,
        )
        native._model = sequence_model()
        annotate._model = sequence_model()

        native_result = native.process(make_parent())
        annotated = annotate.process(make_parent())
        assert isinstance(native_result, AudioTask)
        assert isinstance(annotated, AudioTask)
        invalid = {item["id"]: item for item in annotated.data["clips"] if item["id"] in {"nan", "inf"}}
        assert all("quality" not in item for item in invalid.values())
        selected = PreserveByValueConditionsStage(
            [{"input_value_key": "quality", "target_value": threshold, "operator": "ge"}],
            missing_value_policy="drop",
            items_key="clips",
            drop_parent_if_empty=True,
        ).process_batch([annotated])

        assert [item["id"] for item in native_result.data["clips"]] == ["pass"]
        assert [item["id"] for item in selected[0].data["clips"]] == ["pass"]

        native_all_fail = UTMOSFilterStage(
            action="filter",
            mode="segments",
            segments_key="clips",
            mos_threshold=threshold,
        )
        annotate_all_fail = UTMOSFilterStage(
            action="annotate",
            mode="segments",
            segments_key="clips",
            mos_threshold=threshold,
        )
        native_all_fail._model = _utmos_mock_model(2.0)
        annotate_all_fail._model = _utmos_mock_model(2.0)
        native_empty = native_all_fail.process(
            AudioTask(data={"clips": [{"waveform": torch.randn(1, sample_rate), "sample_rate": sample_rate}]})
        )
        annotated_empty = annotate_all_fail.process(
            AudioTask(data={"clips": [{"waveform": torch.randn(1, sample_rate), "sample_rate": sample_rate}]})
        )
        selected_empty = PreserveByValueConditionsStage(
            [{"input_value_key": "utmos_mos", "target_value": threshold, "operator": "ge"}],
            missing_value_policy="drop",
            items_key="clips",
            drop_parent_if_empty=True,
        ).process_batch([annotated_empty])
        assert native_empty == []
        assert selected_empty == []
