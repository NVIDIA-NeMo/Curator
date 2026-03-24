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


from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from nemo_curator.stages.audio.inference.sortformer import (
    InferenceSortformerStage,
    _parse_sortformer_segments,
    _write_rttm,
)
from nemo_curator.tasks import AudioBatch


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


class TestInferenceSortformerStage:
    def test_stage_properties(self) -> None:
        stage = InferenceSortformerStage()
        assert stage.name == "Sortformer_inference"
        assert stage.inputs() == (["data"], [])
        assert stage.outputs() == (["data"], ["audio_filepath", "diar_segments"])

    def test_custom_keys(self) -> None:
        stage = InferenceSortformerStage(
            filepath_key="audio_path",
            diar_segments_key="segments",
        )
        assert stage.outputs() == (["data"], ["audio_path", "segments"])

    def test_setup_on_node_pre_caches_model(self) -> None:
        stage = InferenceSortformerStage(model_name="nvidia/diar_streaming_sortformer_4spk-v2")
        with patch("nemo_curator.stages.audio.inference.sortformer.snapshot_download") as mock_dl:
            stage.setup_on_node()
            mock_dl.assert_called_once_with(repo_id="nvidia/diar_streaming_sortformer_4spk-v2", cache_dir=None)

    def test_setup_on_node_skips_for_local_path(self) -> None:
        stage = InferenceSortformerStage(model_path="/local/model.nemo")
        with patch("nemo_curator.stages.audio.inference.sortformer.snapshot_download") as mock_dl:
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

    def test_process_audio_batch(self) -> None:
        fake_output = [
            ["0.00 2.70 speaker_0", "0.80 13.60 speaker_1"],
            ["1.00 4.00 speaker_0"],
        ]
        mock_model = self._make_mock_model(fake_output)
        stage = InferenceSortformerStage(diar_model=mock_model)

        task = AudioBatch(
            data=[
                {"audio_filepath": "/test/audio1.wav"},
                {"audio_filepath": "/test/audio2.wav"},
            ],
        )
        result = stage.process(task)

        assert isinstance(result, AudioBatch)
        assert len(result.data) == 2
        assert result.data[0]["audio_filepath"] == "/test/audio1.wav"
        assert result.data[0]["diar_segments"] == [
            {"start": 0.0, "end": 2.7, "speaker": "speaker_0"},
            {"start": 0.8, "end": 13.6, "speaker": "speaker_1"},
        ]
        assert result.data[1]["diar_segments"] == [
            {"start": 1.0, "end": 4.0, "speaker": "speaker_0"},
        ]
        assert result.task_id.endswith("_sortformer")
        mock_model.diarize.assert_called_once_with(
            audio=["/test/audio1.wav", "/test/audio2.wav"],
            batch_size=1,
        )

    def test_process_writes_rttm(self, tmp_path: Path) -> None:
        fake_output = [["0.00 2.50 speaker_0"]]
        mock_model = self._make_mock_model(fake_output)
        stage = InferenceSortformerStage(
            diar_model=mock_model,
            rttm_out_dir=str(tmp_path),
        )

        task = AudioBatch(data=[{"audio_filepath": "/test/my_audio.wav"}])
        stage.process(task)

        rttm_file = tmp_path / "my_audio.rttm"
        assert rttm_file.exists()
        content = rttm_file.read_text()
        assert "SPEAKER my_audio" in content

    def test_process_preserves_existing_data(self) -> None:
        fake_output = [["0.00 1.00 speaker_0"]]
        mock_model = self._make_mock_model(fake_output)
        stage = InferenceSortformerStage(diar_model=mock_model)

        task = AudioBatch(
            data=[{"audio_filepath": "/test/audio1.wav", "extra_key": "extra_value"}],
        )
        result = stage.process(task)
        assert result.data[0]["extra_key"] == "extra_value"
        assert "diar_segments" in result.data[0]

    def test_process_uses_session_name_from_data(self, tmp_path: Path) -> None:
        fake_output = [["0.00 1.00 speaker_0"]]
        mock_model = self._make_mock_model(fake_output)
        stage = InferenceSortformerStage(
            diar_model=mock_model,
            rttm_out_dir=str(tmp_path),
        )

        task = AudioBatch(
            data=[{"audio_filepath": "/test/audio1.wav", "session_name": "sess_42"}],
        )
        stage.process(task)
        assert (tmp_path / "sess_42.rttm").exists()
