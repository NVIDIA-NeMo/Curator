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

from nemo_curator.stages.audio.io.nemo_speech_reader import (
    NeMoSpeechDiscoveryStage,
    _dedup_entries_by_stem,
)
from nemo_curator.tasks import EmptyTask


def test_dedup_keeps_preferred_container_and_distinct_directories() -> None:
    entries = [
        {"audio_filepath": "s3://bucket/a/clip.wav"},
        {"audio_filepath": "s3://bucket/a/clip.opus"},
        {"audio_filepath": "s3://bucket/b/clip.wav"},
    ]

    result = _dedup_entries_by_stem(entries, "shard")

    assert [entry["audio_filepath"] for entry in result] == [
        "s3://bucket/a/clip.opus",
        "s3://bucket/b/clip.wav",
    ]


def test_discovery_skips_done_shards(tmp_path: Path) -> None:
    manifest = tmp_path / "corpus" / "manifest.jsonl"
    manifest.parent.mkdir()
    manifest.write_text('{"audio_filepath": "a.wav"}\n', encoding="utf-8")
    config = tmp_path / "input.yaml"
    config.write_text(
        f"- input_cfg:\n  - corpus: corpus\n    manifest_filepath: {manifest}\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"
    output.mkdir()
    done = output / "corpus" / "manifest.jsonl.done"
    done.parent.mkdir()
    done.write_text("1\n", encoding="utf-8")
    stage = NeMoSpeechDiscoveryStage(yaml_path=str(config), output_dir=str(output))

    assert stage.process(EmptyTask()) == []


def test_discovery_uses_current_framework_task_contract(tmp_path: Path) -> None:
    manifest = tmp_path / "corpus" / "manifest.jsonl"
    manifest.parent.mkdir()
    manifest.write_text('{"audio_filepath": "a.wav"}\n', encoding="utf-8")
    config = tmp_path / "input.yaml"
    config.write_text(
        f"- input_cfg:\n  - corpus: corpus\n    manifest_filepath: {manifest}\n",
        encoding="utf-8",
    )
    stage = NeMoSpeechDiscoveryStage(yaml_path=str(config))

    tasks = stage.process(EmptyTask())

    assert len(tasks) == 1
    assert tasks[0].data == ["a.wav"]
    assert tasks[0].reader_config["shard_key"] == "corpus/manifest"
    assert tasks[0].task_id == ""
