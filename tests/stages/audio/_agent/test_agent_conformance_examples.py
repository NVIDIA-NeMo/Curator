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

"""EXEMPLAR per-stage conformance tests — copy these as templates.

Each stage owner adds one ``assert_agent_ready(...)`` test for their stage (see
``nemo_curator/stages/audio/AGENT_READY.md``). These three cover the common
patterns on CPU with no models:

  * 1:1 transform reading a file              -> MonoConversionStage
  * 1:1 annotate writing one metric            -> GetAudioDurationStage
  * filter (batch-only, may drop items)        -> PreserveByValueStage

``assert_agent_ready`` runs the stage on the fixture and verifies the declared
contract matches runtime: declared writes appear, no undeclared top-level keys
leak, cardinality matches, and reads are satisfiable by role. Model/GPU stages
follow the same shape but build the fixture + fake model via the stub harness in
``test_agent_simulation_pipelines.py``.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003

import numpy as np
import soundfile as sf
import torch

from nemo_curator.stages.audio._agent._conformance import assert_agent_ready, assert_residency_consumption
from nemo_curator.stages.audio.common import GetAudioDurationStage, PreserveByValueStage
from nemo_curator.stages.audio.preprocessing.mono_conversion import MonoConversionStage
from nemo_curator.tasks import AudioTask


def _write_wav(path: Path, *, duration_sec: float = 1.0, sample_rate: int = 48000) -> str:
    samples = np.zeros(int(duration_sec * sample_rate), dtype="float32")
    sf.write(str(path), samples, sample_rate)
    return str(path)


# --- TEMPLATE 1: a 1:1 transform that reads a file -------------------------- #
def test_example_transform_stage_conformance(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "a.wav", sample_rate=48000)  # MonoConversion default sr

    def fixture() -> AudioTask:
        return AudioTask(dataset_name="t", data={"audio_filepath": wav})

    # ``strict_sample_rate`` (the default) drops rows whose rate differs, which is a
    # filter; relaxing it gives the plain 1:1 transform this template is meant to show.
    assert_agent_ready(
        MonoConversionStage(strict_sample_rate=False),
        fixture,
        expected_cardinality="1:1",
        available_keys={"audio_filepath"},
    )


# --- TEMPLATE 2: a 1:1 stage that annotates one metric ---------------------- #
def test_example_metric_stage_conformance(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "b.wav")

    def fixture() -> AudioTask:
        return AudioTask(dataset_name="t", data={"audio_filepath": wav})

    assert_agent_ready(
        GetAudioDurationStage(),
        fixture,
        expected_cardinality="1:1",
        available_keys={"audio_filepath"},
    )


# --- TEMPLATE 4: per-residency consumption (advertised residency == code) --- #
def test_example_residency_consumption(tmp_path: Path) -> None:
    """A residency-configurable stage must actually consume each residency it advertises."""
    wav = _write_wav(tmp_path / "r.wav", sample_rate=48000)

    def file_fixture() -> AudioTask:
        return AudioTask(dataset_name="t", data={"audio_filepath": wav})

    def waveform_fixture() -> AudioTask:
        return AudioTask(dataset_name="t", data={"waveform": torch.zeros(2, 48000), "sample_rate": 48000})

    assert_residency_consumption(
        lambda r: MonoConversionStage(input_residency=r),
        file_fixture=file_fixture,
        waveform_fixture=waveform_fixture,
    )


# --- TEMPLATE 3: a filter (batch-only; may drop items) ---------------------- #
def test_example_filter_stage_conformance() -> None:
    def fixture() -> AudioTask:
        # value passes the filter (keep == True), so the task survives
        return AudioTask(dataset_name="t", data={"keep": True})

    assert_agent_ready(
        PreserveByValueStage("keep", target_value=True, operator="eq"),
        fixture,
        expected_cardinality="filter",
        available_keys={"keep"},
    )
