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

from nemo_curator.stages.audio.tts.fields import get_dotted
from nemo_curator.stages.audio.tts.ipa import ManifestIpaStage
from nemo_curator.tasks import AudioTask


class FakeEspeak:
    def __init__(self, voice: str = "en", *, fail: bool = False) -> None:
        self.voice = voice
        self.exe = "espeak-ng"
        self.fallback_exe = None
        self.fail = fail
        self.calls: list[str] = []

    def text_to_ipa(self, text: str) -> str:
        self.calls.append(text)
        if self.fail:
            msg = "conversion failed"
            raise RuntimeError(msg)
        return f"ipa:{text}"


def _stage(runner: FakeEspeak | None = None, **kwargs: object) -> ManifestIpaStage:
    stage = ManifestIpaStage(**kwargs)
    stage._runner = runner or FakeEspeak()
    return stage


def test_converts_flat_tn_raw_and_caches_repeated_values() -> None:
    runner = FakeEspeak()
    stage = _stage(runner)
    t1 = AudioTask(data={"tn_raw": " repeated ", "source_lang": "en"})
    t2 = AudioTask(data={"tn_raw": "repeated", "source_lang": "en"})

    r1 = stage.process(t1)
    r2 = stage.process(t2)

    assert runner.calls == ["repeated"]
    assert r1.data["ipa"] == {"ipa": "ipa:repeated", "error": None}
    assert r2.data["ipa"] == {"ipa": "ipa:repeated", "error": None}


def test_falls_back_to_itn_text() -> None:
    runner = FakeEspeak()
    stage = _stage(runner)
    task = AudioTask(data={"itn_text": "hello there", "source_lang": "en"})
    result = stage.process(task)
    assert runner.calls == ["hello there"]
    assert result.data["ipa"]["ipa"] == "ipa:hello there"


def test_nested_granary_v2_tn_raw_on_segments() -> None:
    runner = FakeEspeak()
    stage = _stage(runner, output_key="GranaryHifi.OrigAudioPipeline.IPA")
    sample = {
        "segments": [
            {"GranaryV2": {"tn_raw": "hello"}},
            {"GranaryV2": {"tn_raw": "hello"}},
        ]
    }
    result = stage.process(AudioTask(data=sample))
    assert runner.calls == ["hello"]
    for item in result.data["segments"]:
        assert get_dotted(item, "GranaryHifi.OrigAudioPipeline.IPA") == {"ipa": "ipa:hello", "error": None}


def test_preserves_existing_nonempty_ipa() -> None:
    runner = FakeEspeak()
    stage = _stage(runner)
    task = AudioTask(data={"tn_raw": "hello", "ipa": {"ipa": "existing", "error": None}})
    stage.process(task)
    assert runner.calls == []
    assert task.data["ipa"] == {"ipa": "existing", "error": None}


def test_overwrite_recomputes_existing_ipa() -> None:
    runner = FakeEspeak()
    stage = _stage(runner, overwrite=True)
    task = AudioTask(data={"tn_raw": "hello", "ipa": {"ipa": "existing", "error": None}})
    stage.process(task)
    assert runner.calls == ["hello"]
    assert task.data["ipa"] == {"ipa": "ipa:hello", "error": None}


def test_text_validation_errors() -> None:
    stage = _stage()
    missing = stage.process(AudioTask(data={}))
    empty = stage.process(AudioTask(data={"tn_raw": "  "}))
    invalid = stage.process(AudioTask(data={"tn_raw": 123}))
    assert missing.data["ipa"] == {"ipa": None, "error": "missing_text"}
    assert empty.data["ipa"] == {"ipa": None, "error": "empty_text"}
    assert invalid.data["ipa"] == {"ipa": None, "error": "invalid_text"}


def test_records_ipa_failure() -> None:
    stage = _stage(FakeEspeak(fail=True))
    result = stage.process(AudioTask(data={"tn_raw": "hello"}))
    assert result.data["ipa"] == {"ipa": None, "error": "ipa_failed"}
