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

from nemo_curator.stages.audio.tts.early_cutoff.endpoint_gate import EarlyCutOffEndpointGateStage
from nemo_curator.stages.audio.tts.early_cutoff.phonemization import PhonemizedReference
from nemo_curator.tasks import AudioTask


def _ready_stage(**kwargs: object) -> EarlyCutOffEndpointGateStage:
    stage = EarlyCutOffEndpointGateStage(**kwargs)
    stage._setup_error = None
    stage._phonemizer = object()
    stage._detector = object()
    stage._heads = {"en": object(), "multi": object()}
    return stage


def test_setup_error_annotates_without_dropping() -> None:
    stage = EarlyCutOffEndpointGateStage()
    stage._setup_error = "RuntimeError: missing checkpoint"
    task = AudioTask(data={"tn_raw": "hello", "source_lang": "en"})
    result = stage.process(task)
    assert result is task
    annotation = result.data["early_cut_off_detection"]
    assert annotation["decision"] == "REVIEW"
    assert annotation["error"]["code"] == "model_unavailable"
    assert annotation["cascade_step"] == "gpu_endpoint_gate"
    assert annotation["annotation_only"] is True


def test_missing_text() -> None:
    stage = _ready_stage()
    task = AudioTask(data={"source_lang": "en"})
    stage.process(task)
    assert task.data["early_cut_off_detection"]["error"]["code"] == "missing_text"


def test_unsupported_language() -> None:
    stage = _ready_stage()
    task = AudioTask(data={"tn_raw": "hello", "source_lang": "mt"})
    stage.process(task)
    assert task.data["early_cut_off_detection"]["error"]["code"] == "unsupported_language"


def test_ineligible_reference_is_indeterminate() -> None:
    class _Phonemizer:
        def phonemize(self, text: str, language: str, *, precomputed_ipa: str | None = None) -> PhonemizedReference:
            del text, language, precomputed_ipa
            return PhonemizedReference(
                normalized_text="",
                phones=(),
                token_ids=(),
                phone_to_word=(),
                words=(),
                phonemizer_language="en-us",
                source="empty",
            )

    class _Detector:
        def _empty_result(self, reference: PhonemizedReference, *, status: str, error: dict) -> dict:
            return {"analysis_status": status, "error": error, "decision": "REVIEW", "phones": len(reference.phones)}

    stage = _ready_stage()
    stage._phonemizer = _Phonemizer()
    stage._detector = _Detector()
    task = AudioTask(data={"tn_raw": "hello", "source_lang": "en"})
    stage.process(task)
    annotation = task.data["early_cut_off_detection"]
    assert annotation["analysis_status"] == "INDETERMINATE"
    assert annotation["error"]["code"] == "unsupported_reference"
    assert annotation["reference_eligible_for_drop"] is False


def test_ipa_dict_is_unwrapped() -> None:
    stage = EarlyCutOffEndpointGateStage(ipa_key="ipa")
    assert stage._resolve_ipa({"ipa": {"ipa": "h ə l oʊ", "error": None}}) == "h ə l oʊ"
    assert stage._resolve_ipa({"ipa": {"ipa": None, "error": "espeak failed"}}) is None


def test_existing_final_annotation_is_kept() -> None:
    stage = _ready_stage()
    existing = {
        "schema_version": 13,
        "cascade_version": "endpoint_mfa_v13",
        "cascade_step": "gpu_endpoint_gate",
        "analysis_status": "OK",
        "decision": "OK",
        "annotation_only": True,
    }
    task = AudioTask(data={"tn_raw": "hello", "source_lang": "en", "early_cut_off_detection": existing})
    stage.process(task)
    assert task.data["early_cut_off_detection"] == existing
