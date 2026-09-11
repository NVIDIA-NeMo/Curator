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
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_curator.stages.audio._agent._agent_registry import build_contract
from nemo_curator.stages.audio._agent._conformance import assert_agent_ready
from nemo_curator.stages.audio._agent._planning import validate_pipeline
from nemo_curator.stages.audio.tagging.inference.nemo_asr_align import NeMoASRAlignerStage
from nemo_curator.stages.audio.tagging.merge_alignment_diarization import (
    MergeAlignmentDiarizationStage,
)
from nemo_curator.stages.audio.tagging.split import JoinSplitAudioMetadataStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


def _stub_asr(stage: NeMoASRAlignerStage) -> MagicMock:
    model = MagicMock()
    model.transcribe.return_value = [MagicMock()]
    stage._asr_model = model
    stage._override_cfg = MagicMock()
    stage.get_alignments_text = MagicMock(
        return_value=([{"word": "hello", "start": 0.0, "end": 0.5, "confidence": 0.9}], "hello")
    )
    return model


class TestNeMoASRAlignerStage:
    def test_io_contracts(self) -> None:
        full_audio_stage = NeMoASRAlignerStage(infer_segment_only=False)
        assert full_audio_stage.inputs() == (
            ["data"],
            ["duration", "segments", "split_filepaths", "split_metadata"],
        )
        assert full_audio_stage.outputs() == (
            ["data"],
            ["duration", "segments", "split_filepaths", "split_metadata"],
        )

        segment_stage = NeMoASRAlignerStage(infer_segment_only=True)
        assert segment_stage.inputs() == (["data"], ["resampled_audio_filepath", "segments"])
        assert segment_stage.outputs() == (["data"], ["resampled_audio_filepath", "segments"])

    def test_describe_scopes_configured_full_and_segment_writes(self) -> None:
        full = build_contract(
            NeMoASRAlignerStage(
                infer_segment_only=False,
                split_metadata_key="chunks",
                split_filepaths_key="chunk_paths",
                text_key="transcript",
                alignment_key="token_alignment",
                words_key="must_not_be_claimed",
            )
        )
        assert full.cardinality == "1:1 nested-list"
        assert full.iteration_key == "chunks"
        assert full.writes.data_keys == []
        assert full.writes.segment_data_keys == ["transcript", "token_alignment"]
        assert "must_not_be_claimed" not in full.writes.segment_data_keys
        assert len(full.conditional_writes) == 1
        assert full.conditional_writes[0].writes.data_keys == ["transcript", "token_alignment"]
        assert full.conditional_writes[0].writes.segment_data_keys == []

        segment_only = build_contract(
            NeMoASRAlignerStage(
                infer_segment_only=True,
                segments_key="utterances",
                text_key="transcript",
                words_key="tokens",
            )
        )
        assert segment_only.cardinality == "1:1 nested-list"
        assert segment_only.iteration_key == "utterances"
        assert segment_only.writes.data_keys == []
        assert segment_only.writes.segment_data_keys == ["transcript", "tokens"]
        assert segment_only.conditional_writes == []

    def test_agent_ready_full_normal_split_writes_nested_metadata(self) -> None:
        stage = NeMoASRAlignerStage(
            resources=Resources(cpus=1.0),
            split_filepaths_key="chunk_paths",
            split_metadata_key="chunks",
            text_key="transcript",
            alignment_key="token_alignment",
        )
        _stub_asr(stage)
        task = AudioTask(
            dataset_name="test",
            data={
                "duration": 2.0,
                "segments": [],
                "chunk_paths": ["chunk.wav"],
                "chunks": [{"start": 0.0, "end": 2.0}],
            },
        )

        assert_agent_ready(
            stage,
            lambda: task,
            expected_cardinality="1:1 nested-list",
            available_keys={"duration", "segments", "chunk_paths", "chunks"},
        )

        assert task.data["chunks"][0]["transcript"] == "hello"
        assert task.data["chunks"][0]["token_alignment"][0]["word"] == "hello"
        assert "transcript" not in task.data
        assert "token_alignment" not in task.data

    def test_agent_ready_full_no_split_and_missing_metadata_write_top_level(self) -> None:
        no_split_stage = NeMoASRAlignerStage(resources=Resources(cpus=1.0))
        no_split_model = _stub_asr(no_split_stage)
        no_split_task = AudioTask(
            dataset_name="test",
            data={"duration": 0.0, "segments": [], "split_filepaths": [], "split_metadata": []},
        )
        assert_agent_ready(
            no_split_stage,
            lambda: no_split_task,
            expected_cardinality="1:1 nested-list",
            available_keys={"duration", "segments", "split_filepaths", "split_metadata"},
        )
        assert no_split_task.data["text"] == ""
        assert no_split_task.data["alignment"] == []
        no_split_model.transcribe.assert_not_called()

        fallback_stage = NeMoASRAlignerStage(resources=Resources(cpus=1.0))
        _stub_asr(fallback_stage)
        fallback_task = AudioTask(
            dataset_name="test",
            data={
                "duration": 2.0,
                "segments": [],
                "split_filepaths": ["chunk.wav"],
                "split_metadata": [],
            },
        )
        assert_agent_ready(
            fallback_stage,
            lambda: fallback_task,
            expected_cardinality="1:1 nested-list",
            available_keys={"duration", "segments", "split_filepaths", "split_metadata"},
        )
        assert fallback_task.data["text"] == "hello"
        assert fallback_task.data["alignment"][0]["word"] == "hello"

    def test_agent_ready_segment_only_writes_configured_segments(self) -> None:
        stage = NeMoASRAlignerStage(
            infer_segment_only=True,
            resources=Resources(cpus=1.0),
            segments_key="utterances",
            text_key="transcript",
            words_key="tokens",
        )
        _stub_asr(stage)
        task = AudioTask(
            dataset_name="test",
            data={
                "resampled_audio_filepath": "audio.wav",
                "utterances": [{"start": 1.0, "end": 3.0}],
            },
        )

        with patch(
            "nemo_curator.stages.audio.tagging.inference.nemo_asr_align.torchaudio.load",
            return_value=(torch.zeros(1, 32000), 16000),
        ):
            assert_agent_ready(
                stage,
                lambda: task,
                expected_cardinality="1:1 nested-list",
                available_keys={"resampled_audio_filepath", "utterances"},
            )

        assert task.data["utterances"][0]["transcript"] == "hello"
        assert task.data["utterances"][0]["tokens"][0]["start"] == 1.0

    def test_nested_full_alignment_does_not_plan_as_top_level_merge_input(self) -> None:
        aligner = NeMoASRAlignerStage(resources=Resources(cpus=1.0))
        merger = MergeAlignmentDiarizationStage()
        report = validate_pipeline(
            [aligner, merger],
            initial_keys={"duration", "segments", "split_filepaths", "split_metadata"},
            initial_task_type="AudioTask",
        )
        assert not report.ok
        assert any(
            issue.stage_index == 1 and issue.code == "unsatisfied_reads" for issue in report.issues
        )

        _stub_asr(aligner)
        task = AudioTask(
            dataset_name="test",
            data={
                "duration": 2.0,
                "segments": [{"speaker": "s1", "start": 0.0, "end": 2.0}],
                "split_filepaths": ["chunk.wav"],
                "split_metadata": [{"start": 0.0, "end": 2.0}],
            },
        )
        aligner.process(task)
        assert task.data["split_metadata"][0]["alignment"][0]["word"] == "hello"
        assert "alignment" not in task.data

        with pytest.raises(ValueError, match="failed validation"):
            merger.process_batch([task])

    def test_tutorial_full_asr_join_merge_chain(self) -> None:
        aligner = NeMoASRAlignerStage(resources=Resources(cpus=1.0))
        joiner = JoinSplitAudioMetadataStage()
        merger = MergeAlignmentDiarizationStage()
        stages = [aligner, joiner, merger]
        initial_keys = {
            "duration",
            "segments",
            "split_filepaths",
            "split_metadata",
            "split_offsets",
            "split_timestamps",
        }

        report = validate_pipeline(
            stages,
            initial_roles={"duration", "segments"},
            initial_keys=initial_keys,
            initial_task_type="AudioTask",
        )

        assert report.ok
        assert report.keys_ok
        assert "text" in report.produced_keys
        assert "alignment" in report.produced_keys
        assert "split_filepaths" not in report.produced_keys
        assert "split_metadata" not in report.produced_keys

        _stub_asr(aligner)
        task = AudioTask(
            dataset_name="test",
            data={
                "duration": 2.0,
                "segments": [{"speaker": "s1", "start": 0.0, "end": 2.0}],
                "split_filepaths": ["chunk.wav"],
                "split_metadata": [{"start": 0.0, "end": 2.0}],
                "split_offsets": [0.0],
                "split_timestamps": [],
            },
        )

        for stage in stages:
            stage.process(task)

        assert task.data["text"] == "hello"
        assert task.data["alignment"][0]["word"] == "hello"
        assert task.data["segments"][0]["text"] == "hello"
        assert task.data["segments"][0]["words"] == task.data["alignment"]
        assert "split_filepaths" not in task.data
        assert "split_metadata" not in task.data

    def test_setup_configures_rnnt_cuda_graphs(self) -> None:
        model = MagicMock()
        stage = NeMoASRAlignerStage(
            decoder_type="rnnt",
            is_fastconformer=False,
            timestamp_type="char",
            use_cuda_graphs=False,
            resources=Resources(cpus=1.0),
            _asr_model=model,
        )

        stage.setup()

        decoding_cfg = model.change_decoding_strategy.call_args.kwargs["decoding_cfg"]
        assert decoding_cfg.rnnt_timestamp_type == "char"
        assert decoding_cfg.greedy.use_cuda_graph_decoder is False

    def test_process_full_audio(self, tmpdir: Any, wav_filepath: Path) -> None:  # noqa: ANN401
        stage = NeMoASRAlignerStage(
            model_name="nvidia/stt_en_fastconformer_ctc_large",
            is_fastconformer=True,
            decoder_type="ctc",
            resources=Resources(cpus=1.0),
        )
        stage.setup()

        tasks = [
            AudioTask(
                data={
                    "audio_filepath": str(wav_filepath),
                    "split_filepaths": [str(wav_filepath)],
                    "split_metadata": [
                        {
                            "start": 0,
                            "end": 10,
                            "resampled_audio_filepath": str(wav_filepath),
                        }
                    ],
                }
            )
        ]
        results = stage.process_batch(tasks)

        assert len(results) == 1
        entry = results[0].data
        split = entry["split_metadata"][0]
        assert "text" in split
        assert "alignment" in split
        assert isinstance(split["text"], str)
        assert isinstance(split["alignment"], list)
        assert split["text"] != ""
        assert len(split["alignment"]) > 10
