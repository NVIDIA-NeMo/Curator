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

"""Canonical agent-ready vertical slice: ``MonoConversionStage``.

Shows that the two foundation modules work together end to end:

* ``_agent_ready.py`` — the stage inherits ``AgentReady`` and declares a
  ``StageContract`` via ``describe()`` (reads/writes, accepted audio forms,
  gates, configurable ``*_key`` fields).
* ``_residency.py`` — ``process()`` resolves audio from either an in-memory
  waveform or a file, and materialized temp files are cleaned up.

``choose_input_form`` below is a *minimal consumer* that reads only the
declared contract to decide file- vs waveform-based execution. It stands in
for the registry/orchestrator (which lands in a later PR); the equivalent
production path is ``_agent_registry.build_contract(stage)`` +
``_planning.validate_pipeline`` reading the same ``StageContract``.

Pseudocode for the real boundary::

    contract = build_contract(stage)          # describe() + auto params/roles/dispatch
    # build_contract also fills registry-derived fields not set by describe():
    #   contract.params      (from the constructor signature)
    #   contract.key_roles   (from the *_key field names)
    #   contract.dispatch    (per-item vs batched, from process/process_batch)
    if "waveform" in contract.reads.accepts and item.has(stage.waveform_key):
        run_on_waveform(...)
    elif "file" in contract.reads.accepts and item.has(stage.audio_filepath_key):
        run_on_file(...)
"""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from nemo_curator.stages.audio._residency import cleanup_temp_files, resolve_audio_path
from nemo_curator.stages.audio.preprocessing.mono_conversion import MonoConversionStage
from nemo_curator.tasks import AudioTask

SR = 48000


def _write_stereo_wav(path: Path, seconds: float = 0.1) -> None:
    """Write a small real stereo WAV so the file path actually resolves."""
    data = np.random.randn(int(SR * seconds), 2).astype("float32")
    sf.write(path, data, SR)


def choose_input_form(stage: MonoConversionStage, item: dict) -> str:
    """Minimal contract consumer: pick execution form from the declared contract.

    Uses ONLY ``stage.describe()`` (the agent-facing contract) plus the item's
    keys — no knowledge of the stage internals.
    """
    contract = stage.describe()
    accepts = set(contract.reads.accepts or [])
    if "waveform" in accepts and item.get(stage.waveform_key) is not None:
        return "waveform"
    if "file" in accepts and item.get(stage.audio_filepath_key):
        return "file"
    return "none"


class TestMonoConversionAgentSlice:
    def test_contract_declares_both_forms(self) -> None:
        contract = MonoConversionStage().describe()
        # Accepted audio residencies are declared, so a consumer can branch.
        assert set(contract.reads.accepts) == {"file", "waveform"}
        # Configurable output keys are surfaced on the contract.
        assert "is_mono" in contract.writes.data_keys

    def test_dispatch_is_per_item(self) -> None:
        # Dispatch: MonoConversion runs per item via process() (it is not
        # batch-only). The concrete ``dispatch`` field on StageContract is
        # filled by the registry (build_contract / static_contract) in the
        # orchestrator PR; here we assert the property it is derived from.
        assert getattr(MonoConversionStage, "BATCH_ONLY", False) is False

    def test_consumer_picks_file_vs_waveform(self, tmp_path: Path) -> None:
        stage = MonoConversionStage(output_sample_rate=SR)
        wav = tmp_path / "a.wav"
        _write_stereo_wav(wav)
        assert choose_input_form(stage, {"audio_filepath": wav.as_posix()}) == "file"
        assert choose_input_form(stage, {"waveform": torch.randn(2, SR), "sample_rate": SR}) == "waveform"

    def test_runs_on_file_input(self, tmp_path: Path) -> None:
        wav = tmp_path / "stereo.wav"
        _write_stereo_wav(wav)
        stage = MonoConversionStage(output_sample_rate=SR)
        out = stage.process(AudioTask(data={"audio_filepath": wav.as_posix()}))
        assert isinstance(out, AudioTask)
        assert out.data["is_mono"] is True
        assert out.data["waveform"].shape[0] == 1
        assert out.data["sample_rate"] == SR

    def test_runs_on_waveform_input(self) -> None:
        stage = MonoConversionStage(output_sample_rate=SR)
        out = stage.process(AudioTask(data={"waveform": torch.randn(2, SR), "sample_rate": SR}))
        assert isinstance(out, AudioTask)
        assert out.data["is_mono"] is True
        assert out.data["waveform"].shape[0] == 1

    def test_both_forms_give_equivalent_shape(self, tmp_path: Path) -> None:
        stage = MonoConversionStage(output_sample_rate=SR)
        wav = tmp_path / "s.wav"
        _write_stereo_wav(wav, seconds=0.1)
        from_file = stage.process(AudioTask(data={"audio_filepath": wav.as_posix()}))
        from_wave = stage.process(AudioTask(data={"waveform": torch.randn(2, int(SR * 0.1)), "sample_rate": SR}))
        assert from_file.data["waveform"].shape[0] == from_wave.data["waveform"].shape[0] == 1
        assert from_file.data["is_mono"] == from_wave.data["is_mono"] is True

    def test_waveform_temp_file_is_created_then_cleaned_up(self) -> None:
        # resolve_audio_path materializes a temp WAV from an in-memory waveform;
        # cleanup_temp_files removes it (the temp-file lifecycle used by stages
        # that need a real path, e.g. resample/ASR/diarization).
        temp_paths: list[str] = []
        item = {"waveform": np.random.randn(SR).astype("float32"), "sample_rate": SR}
        path = resolve_audio_path(item, residency="waveform", register_temp=temp_paths)
        assert path is not None
        assert path.endswith(".wav")
        assert Path(path).exists()
        assert temp_paths == [path]

        cleanup_temp_files(temp_paths)
        assert not Path(path).exists()
