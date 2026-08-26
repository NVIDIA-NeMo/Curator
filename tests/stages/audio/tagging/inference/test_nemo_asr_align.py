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
from unittest.mock import MagicMock

from nemo_curator.stages.audio.tagging.inference.nemo_asr_align import NeMoASRAlignerStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask


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
