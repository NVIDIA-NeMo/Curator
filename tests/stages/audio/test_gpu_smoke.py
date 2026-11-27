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

"""Real-model GPU smoke tests for audio stages whose unit tests only use mocks.

Each test loads the default model on one GPU, runs one clip through the stage
lifecycle, and checks that the stage wrote its output. They download weights on
first run.
"""

import os
from pathlib import Path

import pytest
import soundfile as sf

from nemo_curator.stages.audio.filtering import BandFilterStage, SIGMOSFilterStage, UTMOSFilterStage
from nemo_curator.stages.audio.inference.indic_conformer_hybrid import InferenceIndicConformerHybridStage
from nemo_curator.stages.audio.inference.indic_transcribe import InferenceIndicTranscribeStage
from nemo_curator.stages.audio.inference.lid.stage import AudioLIDInferenceStage
from nemo_curator.stages.audio.inference.sed.stage import SEDInferenceStage
from nemo_curator.stages.audio.inference.speaker_diarization.stage import InferenceSortformerStage
from nemo_curator.stages.audio.preprocessing.indic_asr import PrepareIndicASRInputStage
from nemo_curator.stages.audio.segmentation import SpeakerSeparationStage
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask

pytestmark = pytest.mark.gpu


def _run(stage: ProcessingStage, task: AudioTask) -> list[AudioTask]:
    stage.setup_on_node()
    stage.setup()
    out = stage.process_batch([task])
    return [t for item in out for t in (item if isinstance(item, list) else [item])]


@pytest.mark.parametrize(
    ("stage_cls", "key"),
    [(BandFilterStage, "band_prediction"), (SIGMOSFilterStage, "sigmos_ovrl"), (UTMOSFilterStage, "utmos_mos")],
    ids=["band", "sigmos", "utmos"],
)
def test_quality_filters_annotate(stage_cls: type[ProcessingStage], key: str, wav_filepath: Path) -> None:
    (out,) = _run(stage_cls(action="annotate"), AudioTask(data={"audio_filepath": str(wav_filepath)}))
    assert key in out.data


def test_sortformer_diarizes(wav_filepath: Path) -> None:
    (out,) = _run(InferenceSortformerStage(), AudioTask(data={"audio_filepath": str(wav_filepath)}))
    assert out.data["diar_segments"]
    assert out.data["num_speakers"] >= 1


def test_speechbrain_lid(wav_filepath: Path) -> None:
    stage = AudioLIDInferenceStage(
        adapter_target="nemo_curator.models.audio.lid.speechbrain.SpeechBrainLIDAdapter",
        model_id="SpeechBrainLangID",
        tag="primary",
    )
    (out,) = _run(stage, AudioTask(data={"audio_filepath": str(wav_filepath)}))
    assert out.data["lid"]["SpeechBrainLangID"]["language"]


def test_panns_sed(wav_filepath: Path, tmp_path: Path) -> None:
    stage = SEDInferenceStage(
        adapter_target="nemo_curator.models.audio.sed.panns.PANNsSEDAdapter", output_dir=str(tmp_path)
    )
    (out,) = _run(stage, AudioTask(data={"audio_filepath": str(wav_filepath)}))
    assert out.data["sed_valid_frames"] > 0


def test_speaker_separation(wav_filepath: Path) -> None:
    out = _run(SpeakerSeparationStage(), AudioTask(data={"audio_filepath": str(wav_filepath)}))
    assert out
    assert all("speaker_id" in t.data for t in out)


def _hindi_task(wav_filepath: Path) -> AudioTask:
    info = sf.info(str(wav_filepath))
    task = AudioTask(
        data={
            "audio_filepath": str(wav_filepath),
            "audio_item_id": "clip",
            "duration": info.frames / info.samplerate,
            "source_lang": "hi",
            "text": "reference",
        }
    )
    return PrepareIndicASRInputStage().process(task)


def test_indic_conformer_hybrid(wav_filepath: Path) -> None:
    (out,) = _run(InferenceIndicConformerHybridStage(), _hindi_task(wav_filepath))
    assert isinstance(out.data["asr_prediction"], str)


@pytest.mark.skipif(
    not os.environ.get("INDIC_TRANSCRIBE_ENGINE_DIR"), reason="needs an Indic Transcribe TensorRT-LLM engine"
)
def test_indic_transcribe(wav_filepath: Path) -> None:
    stage = InferenceIndicTranscribeStage(engine_dir=os.environ["INDIC_TRANSCRIBE_ENGINE_DIR"])
    (out,) = _run(stage, _hindi_task(wav_filepath))
    assert isinstance(out.data["asr_prediction"], str)
