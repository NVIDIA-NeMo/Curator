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

from nemo_curator.stages.audio.tagging.inference.nemo_asr_align import NeMoASRAlignerStage
from nemo_curator.tasks import AudioTask


class TestNeMoASRAlignerStage:
    def test_process_full_audio(self, tmpdir: Any, wav_filepath: Path) -> None:  # noqa: ANN401
        stage = NeMoASRAlignerStage(
            model_name="nvidia/stt_en_fastconformer_ctc_large",
            is_fastconformer=True,
            decoder_type="ctc",
            device="cpu",
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
