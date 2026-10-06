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

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path  # noqa: TC003
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nemo_curator.stages.audio.inference.speaker_diarization.sortformer import (
    InferenceSortformerStage,
    _parse_sortformer_segments,
    _write_rttm,
)
from nemo_curator.tasks import AudioTask
from tests.stages.audio.inference import review_helpers as rh

if TYPE_CHECKING:
    from collections.abc import Iterator


class TestParseSortformerSegments:
    def test_parses_string_segments(self) -> None:
        raw = ["0.00 2.70 speaker_0", "0.80 13.60 speaker_1"]
        out = _parse_sortformer_segments(raw)
        assert len(out) == 2
        assert out[0] == {"start": 0.0, "end": 2.7, "speaker": "speaker_0"}
        assert out[1] == {"start": 0.8, "end": 13.6, "speaker": "speaker_1"}

    def test_parses_object_segments(self) -> None:
        seg1 = SimpleNamespace(start=1.0, end=3.5, speaker="speaker_0")
        seg2 = SimpleNamespace(start=4.0, end=7.2, speaker="speaker_1")
        out = _parse_sortformer_segments([seg1, seg2])
        assert out[0] == {"start": 1.0, "end": 3.5, "speaker": "speaker_0"}
        assert out[1] == {"start": 4.0, "end": 7.2, "speaker": "speaker_1"}

    def test_parses_object_with_label_attr(self) -> None:
        seg = SimpleNamespace(start=0.5, end=1.5, label="spk_2")
        out = _parse_sortformer_segments([seg])
        assert out[0]["speaker"] == "spk_2"

    def test_parses_tuple_segments(self) -> None:
        raw = [(0.0, 2.0, "speaker_0"), (3.0, 5.0, "speaker_1")]
        out = _parse_sortformer_segments(raw)
        assert len(out) == 2
        assert out[0] == {"start": 0.0, "end": 2.0, "speaker": "speaker_0"}

    def test_empty_list_returns_empty(self) -> None:
        assert _parse_sortformer_segments([]) == []

    def test_unrecognised_format_warns(self) -> None:
        out = _parse_sortformer_segments([42])
        assert out == []


class TestWriteRttm:
    def test_writes_rttm_file(self, tmp_path: Path) -> None:
        segments = [
            {"start": 0.0, "end": 2.5, "speaker": "speaker_0"},
            {"start": 3.0, "end": 5.0, "speaker": "speaker_1"},
        ]
        _write_rttm(segments, "test_session", str(tmp_path))
        rttm_path = tmp_path / "test_session.rttm"
        assert rttm_path.exists()
        lines = rttm_path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert lines[0].startswith("SPEAKER test_session 1 0.000 2.500")
        assert "speaker_0" in lines[0]
        assert lines[1].startswith("SPEAKER test_session 1 3.000 2.000")
        assert "speaker_1" in lines[1]

    @pytest.mark.parametrize("session", ["shard/clip", r"shard\clip", "../escaped", "/absolute", "日本" * 100])
    def test_unsafe_names_stay_inside_export_directory(self, session: str, tmp_path: Path) -> None:
        output = tmp_path / "rttm"
        _write_rttm([{"start": 0.0, "end": 1.0, "speaker": "speaker_0"}], session, str(output))
        files = list(tmp_path.rglob("*.rttm"))
        assert len(files) == 1
        assert files[0].parent == output
        assert len(files[0].name.encode("utf-8")) <= 255
        assert files[0].read_text().startswith(f"SPEAKER {session} 1 ")

    def test_sanitized_labels_cannot_alias_each_other_or_literal_names(self, tmp_path: Path) -> None:
        segments = [{"start": 0.0, "end": 1.0, "speaker": "speaker_0"}]
        _write_rttm(segments, "shard/clip", str(tmp_path))
        encoded_name = next(tmp_path.glob("*.rttm")).stem
        for name in ("shard_clip", r"shard\clip", encoded_name):
            _write_rttm(segments, name, str(tmp_path))
        assert len(list(tmp_path.glob("*.rttm"))) == 4

    def test_ordinary_names_remain_unchanged(self, tmp_path: Path) -> None:
        _write_rttm([], "_clip-01.v2", str(tmp_path))
        assert (tmp_path / "_clip-01.v2.rttm").is_file()


class TestInferenceSortformerStage:
    def test_preserves_explicit_non_resumable_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(InferenceSortformerStage, "is_resumable", False)
        assert InferenceSortformerStage().is_resumable is False

    def test_rejects_unknown_rttm_naming_policy(self) -> None:
        with pytest.raises(ValueError, match="rttm_naming"):
            InferenceSortformerStage(rttm_naming="shard")

    def test_setup_on_node_pre_caches_model(self) -> None:
        stage = InferenceSortformerStage(model_name="nvidia/diar_streaming_sortformer_4spk-v2")
        with patch("nemo_curator.stages.audio.inference.speaker_diarization.sortformer.snapshot_download") as mock_dl:
            stage.setup_on_node()
            mock_dl.assert_called_once_with(repo_id="nvidia/diar_streaming_sortformer_4spk-v2", cache_dir=None)

    def test_setup_on_node_skips_for_local_path(self) -> None:
        stage = InferenceSortformerStage(model_path="/local/model.nemo")
        with patch("nemo_curator.stages.audio.inference.speaker_diarization.sortformer.snapshot_download") as mock_dl:
            stage.setup_on_node()
            mock_dl.assert_not_called()

    def test_setup_skips_when_model_provided(self) -> None:
        mock_model = MagicMock()
        mock_model.sortformer_modules = MagicMock()
        stage = InferenceSortformerStage(diar_model=mock_model)
        stage.setup()
        assert mock_model.sortformer_modules.chunk_len == 340

    def test_streaming_config_applied(self) -> None:
        mock_model = MagicMock()
        mock_model.sortformer_modules = MagicMock()
        stage = InferenceSortformerStage(
            diar_model=mock_model,
            chunk_len=124,
            chunk_right_context=1,
            fifo_len=124,
            spkcache_update_period=124,
            spkcache_len=200,
        )
        stage.setup()
        sm = mock_model.sortformer_modules
        assert sm.chunk_len == 124
        assert sm.chunk_right_context == 1
        assert sm.fifo_len == 124
        assert sm.spkcache_update_period == 124
        assert sm.spkcache_len == 200

    def _make_mock_model(self, fake_segments_per_file: list[list[str]]) -> MagicMock:
        mock_model = MagicMock()
        mock_model.sortformer_modules = MagicMock()
        mock_model.diarize.return_value = fake_segments_per_file
        return mock_model

    def test_process_audio_task(self) -> None:
        fake_output = [
            ["0.00 2.70 speaker_0", "0.80 13.60 speaker_1"],
        ]
        mock_model = self._make_mock_model(fake_output)
        stage = InferenceSortformerStage(diar_model=mock_model)

        task = AudioTask(
            data={"audio_filepath": "/test/audio1.wav"},
        )
        result = stage.process(task)

        assert isinstance(result, AudioTask)
        assert result.data["audio_filepath"] == "/test/audio1.wav"
        assert result.data["diar_segments"] == [
            {"start": 0.0, "end": 2.7, "speaker": "speaker_0"},
            {"start": 0.8, "end": 13.6, "speaker": "speaker_1"},
        ]
        mock_model.diarize.assert_called_once_with(
            audio=["/test/audio1.wav"],
            batch_size=1,
        )

    def test_process_writes_rttm(self, tmp_path: Path) -> None:
        fake_output = [["0.00 2.50 speaker_0"]]
        mock_model = self._make_mock_model(fake_output)
        stage = InferenceSortformerStage(
            diar_model=mock_model,
            rttm_out_dir=str(tmp_path),
        )

        task = AudioTask(data={"audio_filepath": "/test/my_audio.wav"})
        stage.process(task)

        rttm_file = tmp_path / "my_audio.rttm"
        assert rttm_file.exists()
        content = rttm_file.read_text()
        assert "SPEAKER my_audio" in content

    def test_process_preserves_existing_data(self) -> None:
        fake_output = [["0.00 1.00 speaker_0"]]
        mock_model = self._make_mock_model(fake_output)
        stage = InferenceSortformerStage(diar_model=mock_model)

        task = AudioTask(
            data={"audio_filepath": "/test/audio1.wav", "extra_key": "extra_value"},
        )
        result = stage.process(task)
        assert result.data["extra_key"] == "extra_value"
        assert "diar_segments" in result.data

    def test_process_uses_session_name_from_data(self, tmp_path: Path) -> None:
        fake_output = [["0.00 1.00 speaker_0"]]
        mock_model = self._make_mock_model(fake_output)
        stage = InferenceSortformerStage(
            diar_model=mock_model,
            rttm_out_dir=str(tmp_path),
        )

        task = AudioTask(
            data={"audio_filepath": "/test/audio1.wav", "session_name": "sess_42"},
        )
        stage.process(task)
        assert (tmp_path / "sess_42.rttm").exists()

    def test_resident_waveform_ignores_stale_file_path_for_identity(self, tmp_path: Path) -> None:
        mock_model = self._make_mock_model([["0.00 0.50 speaker_0"]])
        stage = InferenceSortformerStage(
            diar_model=mock_model,
            input_residency="waveform",
            fanout=True,
            rttm_out_dir=str(tmp_path),
        )
        stale_path = "/stale/unrelated.wav"
        task = AudioTask(
            data={
                "audio_filepath": stale_path,
                "waveform": np.zeros(16000, dtype=np.float32),
                "sample_rate": 16000,
            },
        )

        with patch(
            "nemo_curator.stages.audio.inference.speaker_diarization.sortformer.resolve_audio_path",
            return_value="/tmp/materialized-resident.wav",  # noqa: S108
        ):
            result = stage.process(task)

        assert isinstance(result, list)
        assert len(result) == 1
        identity = result[0].data["original_file"]
        assert identity != stale_path
        assert identity.startswith("audio_")
        assert [path.stem for path in tmp_path.glob("*.rttm")] == [identity]


def test_cpu_injected_sortformer_does_not_claim_gpu_requirement() -> None:
    cpu_injected = rh.build_contract(
        rh.InferenceSortformerStage(diar_model=rh.MagicMock(), resources=rh.Resources(gpus=0))
    )
    restored = rh.build_contract(
        rh.InferenceSortformerStage(model_path="/models/local.nemo", resources=rh.Resources(gpus=0))
    )
    assert cpu_injected.gates.requires_gpu is False
    assert restored.gates.requires_gpu is True


@pytest.mark.parametrize("session_name", [None, "shared-session"])
def test_source_hash_exports_duplicate_basenames_without_overwriting_and_retries_subset(
    session_name: str | None, tmp_path: Path
) -> None:
    model = MagicMock()

    def diarize(*, audio: list[str], batch_size: int) -> list[list[str]]:
        assert batch_size == 1
        return [["0.00 1.00 speaker_0"]] if audio[0] == "/corpus/a/utt1.wav" else [["0.00 1.00 speaker_1"]]

    model.diarize.side_effect = diarize
    params = {"diar_model": model, "rttm_out_dir": str(tmp_path), "rttm_naming": "source_hash"}
    stage = InferenceSortformerStage(**params)
    tasks = [AudioTask(data={"audio_filepath": f"/corpus/{shard}/utt1.wav"}) for shard in ("a", "b")]
    if session_name is not None:
        for task in tasks:
            task.data["session_name"] = session_name
    for task in tasks:
        result = stage.process(task)
        assert result.data["audio_filepath"] == task.data["audio_filepath"]
        assert result.data["diar_segments"]
        assert "num_speakers" not in result.data
    before = {path.name: path.read_text() for path in tmp_path.glob("*.rttm")}
    assert len(before) == 2
    assert any("speaker_0" in text for text in before.values())
    assert any("speaker_1" in text for text in before.values())
    # New workers, new task IDs, and reversed order must preserve artifact identities.
    for task in reversed(tasks):
        InferenceSortformerStage(**params).process(AudioTask(data=dict(task.data)))
    assert {path.name: path.read_text() for path in tmp_path.glob("*.rttm")} == before
    model.diarize.side_effect = None
    model.diarize.return_value = [["0.00 1.00 speaker_2"]]
    InferenceSortformerStage(**params).process(AudioTask(data=dict(tasks[0].data)))
    after = {path.name: path.read_text() for path in tmp_path.glob("*.rttm")}
    assert set(after) == set(before)
    for name, content in before.items():
        assert ("speaker_2" in after[name]) if "speaker_0" in content else after[name] == content


@pytest.mark.parametrize("residency", ["waveform", "auto"])
@pytest.mark.parametrize("identity_key", ["session_name", "audio_item_id"])
def test_source_hash_resident_inputs_use_content_not_stale_or_temporary_paths(
    residency: str, identity_key: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage, seen = rh._make_stage("sortformer", monkeypatch, input_residency=residency, rttm_out_dir=str(tmp_path))
    stage.rttm_naming = "source_hash"
    data = {
        "audio_filepath": "/stale/utt1.wav",
        "original_file": "/stale/utt1.wav",
        identity_key: "shared/clip",
        "sample_rate": rh._SAMPLE_RATE,
    }
    for value in (0.0, 1.0):
        stage.process(AudioTask(data={**data, "waveform": np.full((1, 12), value, dtype=np.float32)}))
    before = {path.name: path.read_text() for path in tmp_path.glob("*.rttm")}
    assert len(before) == 2
    retry, _ = rh._make_stage("sortformer", monkeypatch, input_residency=residency, rttm_out_dir=str(tmp_path))
    retry.rttm_naming = "source_hash"
    retry.process(AudioTask(data={**data, "waveform": np.zeros((1, 12), dtype=np.float32)}))
    assert {path.name: path.read_text() for path in tmp_path.glob("*.rttm")} == before
    assert seen == [12, 12]


def test_unsafe_audio_item_id_is_sanitized_through_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stage, _ = rh._make_stage(
        "sortformer", monkeypatch, input_residency="waveform", rttm_out_dir=str(tmp_path / "rttm")
    )
    result = stage.process(
        AudioTask(
            data={
                "waveform": np.zeros((1, 12), dtype=np.float32),
                "sample_rate": rh._SAMPLE_RATE,
                "audio_item_id": "../clip",
            }
        )
    )
    assert result.data["diar_segments"]
    assert len(list((tmp_path / "rttm").glob("*.rttm"))) == 1
    assert not (tmp_path / "clip.rttm").exists()


@pytest.mark.parametrize(("naming", "independent"), [("legacy", False), ("source_hash", True)])
def test_rttm_reuse_gate_matches_naming_policy(naming: str, independent: bool, tmp_path: Path) -> None:
    stage = InferenceSortformerStage(rttm_out_dir=str(tmp_path), rttm_naming=naming)
    contract = rh.build_contract(stage)
    assert contract.gates.per_row_independent is independent
    assert contract.gates.writes_to_disk is True
    assert contract.gates.output_path_params == ["rttm_out_dir"]
    policy = next(param for param in contract.params if param.name == "rttm_naming")
    assert policy.default == "legacy"
    assert policy.choices == ["legacy", "source_hash"]


@pytest.mark.parametrize("residency", ["file", "waveform"])
@pytest.mark.parametrize("fanout", [False, True])
def test_source_hash_export_preserves_agent_contract_at_runtime(
    residency: str, fanout: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_path = tmp_path / "input.wav"
    waveform = np.zeros((1, 12), dtype=np.float32)
    rh._write_audio(audio_path, waveform)
    stage, _ = rh._make_stage(
        "sortformer", monkeypatch, input_residency=residency, fanout=fanout, rttm_out_dir=str(tmp_path / "rttm")
    )
    stage.rttm_naming = "source_hash"
    data = (
        {"audio_filepath": str(audio_path)}
        if residency == "file"
        else {"waveform": waveform, "sample_rate": rh._SAMPLE_RATE}
    )
    contract = rh.assert_agent_ready(stage, fixture_factory=lambda: AudioTask(data=dict(data)))
    assert contract.cardinality == ("1:N fan-out" if fanout else "1:1")
    assert len(list((tmp_path / "rttm").glob("*.rttm"))) == 1


@pytest.fixture(scope="module")
def gpu_sortformer_stage() -> Iterator[InferenceSortformerStage]:
    if not rh.torch.cuda.is_available():
        pytest.skip("CUDA is required for real Sortformer inference")
    stage = InferenceSortformerStage()
    stage.setup_on_node()
    stage.setup()
    assert next(stage.diar_model.parameters()).is_cuda
    yield stage
    stage.diar_model = None
    rh.torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.parametrize("session", [None, "../shard/clip"])
def test_gpu_sortformer_legacy_exports(
    session: str | None, gpu_sortformer_stage: InferenceSortformerStage, wav_filepath: Path, tmp_path: Path
) -> None:
    output = tmp_path / "rttm"
    stage = replace(gpu_sortformer_stage, rttm_out_dir=str(output))
    data = {"audio_filepath": str(wav_filepath)}
    if session is not None:
        data["session_name"] = session
    result = stage.process(AudioTask(data=data))
    segments = result.data[stage.diar_segments_key]
    assert segments
    duration = rh.sf.info(wav_filepath).duration
    assert all(0 <= segment["start"] < segment["end"] <= duration + 0.1 for segment in segments)
    files = list(tmp_path.rglob("*.rttm"))
    assert len(files) == 1
    assert files[0].parent == output
    if session is None:
        assert files[0].name == f"{wav_filepath.stem}.rttm"
    lines = files[0].read_text().splitlines()
    assert len(lines) == len(segments)
    assert all(line.split()[1] == (session or wav_filepath.stem) for line in lines)
    assert "num_speakers" not in result.data


@pytest.mark.gpu
def test_gpu_sortformer_source_hash_exports_and_subset_retry(
    gpu_sortformer_stage: InferenceSortformerStage, wav_filepath: Path, tmp_path: Path
) -> None:
    inputs = []
    for shard in ("a", "b"):
        path = tmp_path / shard / "utt1.wav"
        path.parent.mkdir()
        shutil.copyfile(wav_filepath, path)
        inputs.append(path)
    baseline = gpu_sortformer_stage.process(AudioTask(data={"audio_filepath": str(inputs[0])}))
    output = tmp_path / "rttm"
    stage = replace(gpu_sortformer_stage, rttm_out_dir=str(output), rttm_naming="source_hash")
    for path in inputs:
        result = stage.process(AudioTask(data={"audio_filepath": str(path)}))
        assert result.data[stage.diar_segments_key] == baseline.data[stage.diar_segments_key]
    before = {path.name: path.read_bytes() for path in output.glob("*.rttm")}
    assert len(before) == 2
    assert all(name.startswith("utt1~") for name in before)
    result = replace(stage).process(AudioTask(data={"audio_filepath": str(inputs[0])}))
    assert result.data[stage.diar_segments_key] == baseline.data[stage.diar_segments_key]
    assert {path.name: path.read_bytes() for path in output.glob("*.rttm")} == before


@pytest.mark.gpu
def test_gpu_sortformer_resident_fanout_with_source_hash(
    gpu_sortformer_stage: InferenceSortformerStage, wav_filepath: Path, tmp_path: Path
) -> None:
    decoded, sample_rate = rh.sf.read(wav_filepath, dtype="float32")
    waveform = decoded[np.newaxis, :]
    data = {
        "waveform": waveform,
        "sample_rate": sample_rate,
        "audio_item_id": "shard/clip",
        "audio_filepath": "/stale/ignored.wav",
    }
    baseline = replace(gpu_sortformer_stage, input_residency="waveform").process(AudioTask(data=dict(data)))
    output = tmp_path / "rttm"
    stage = replace(
        gpu_sortformer_stage,
        input_residency="waveform",
        fanout=True,
        rttm_out_dir=str(output),
        rttm_naming="source_hash",
    )
    children = stage.process(AudioTask(data=dict(data)))
    assert isinstance(children, list)
    assert len(children) == len(baseline.data[stage.diar_segments_key]) > 0
    assert stage.is_resumable is False
    for child, segment in zip(children, baseline.data[stage.diar_segments_key], strict=True):
        assert "audio_filepath" not in child.data
        assert child.data[stage.speaker_key] == segment["speaker"]
        assert child.data[stage.sample_rate_key] == sample_rate
        assert child.data[stage.original_file_key] == "shard/clip"
        assert child.data[stage.waveform_key].shape[1] > 0
        assert not np.shares_memory(child.data[stage.waveform_key], waveform)
    assert len(list(output.glob("*.rttm"))) == 1
