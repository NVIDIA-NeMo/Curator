# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Exercise the tutorial's actual CTM-to-Lhotse-to-FastMSS handoff."""

import argparse
import json
import wave
from pathlib import Path

import pytest

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask
from tutorials.audio.fastmss.mfa_to_fastmss import prepare, read_rows, read_word_alignment


@pytest.fixture
def aligned_manifest(tmp_path: Path) -> Path:
    rows = []
    for i in range(2):
        audio = tmp_path / f"speaker-{i}.wav"
        with wave.open(str(audio), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(b"\0\0" * 16000)
        ctm = tmp_path / f"speaker-{i}.ctm"
        ctm.write_text(f"u{i} 1 0.1 0.3 HELLO\nu{i} 1 0.5 0.4 WORLD\n")
        rows.append(
            {
                "audio_filepath": str(audio),
                "text": "Hello world.",
                "speaker": f"s{i}",
                "mfa_skipped": False,
                "ctm_filepath": str(ctm),
            }
        )
    manifest = tmp_path / "aligned.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return manifest


def test_preparation_outputs_real_embedded_word_aligned_cutset(aligned_manifest: Path, tmp_path: Path) -> None:
    from lhotse import CutSet

    before = aligned_manifest.read_bytes()
    config = prepare(
        argparse.Namespace(
            aligned_manifest=aligned_manifest, output_dir=tmp_path / "prepared", meetings=4, duration=10
        )
    )
    cuts = list(CutSet.from_file(config.parent / "speech_train.jsonl.gz"))
    assert len(cuts) == 2
    assert {cut.supervisions[0].speaker for cut in cuts} == {"s0", "s1"}
    assert all(cut.supervisions[0].text == "Hello world." for cut in cuts)
    for cut in cuts:
        assert [item.symbol for item in cut.supervisions[0].alignment["word"]] == ["HELLO", "WORLD"]
        assert cut.supervisions[0].alignment["word"][0].start == 0.1
        assert Path(cut.recording.sources[0].source).is_relative_to(config.parent)
        assert not Path(cut.recording.sources[0].source).is_symlink()
    assert aligned_manifest.read_bytes() == before


def test_prepared_cutset_is_accepted_by_native_request_planner(aligned_manifest: Path, tmp_path: Path) -> None:
    from importlib.metadata import distribution

    from nemo_curator.stages.audio.simulation import write_fastmss_requests

    config = prepare(
        argparse.Namespace(
            aligned_manifest=aligned_manifest, output_dir=tmp_path / "prepared", meetings=4, duration=10
        )
    )
    package = distribution("fastmss")
    requests = tmp_path / "plan.jsonl"
    count = write_fastmss_requests(
        [str(config)],
        str(requests),
        native_inputs_path=str(config.parent),
        engine_path=str(package.locate_file("fastmss")),
        recipe_path=str(package.locate_file("recipes")),
        worker_lock_path=str(tmp_path / "worker.json"),
    )
    assert count == len(requests.read_text().splitlines()) == 4
    assert not (tmp_path / "generated").exists()


@pytest.mark.parametrize("bad_row", [{"mfa_skipped": True}, {"ctm_filepath": ""}])
def test_mfa_fallbacks_cannot_be_presented_as_word_alignment(
    aligned_manifest: Path, tmp_path: Path, bad_row: dict
) -> None:
    rows = [json.loads(line) for line in aligned_manifest.read_text().splitlines()]
    rows[0].update(bad_row)
    aligned_manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="successful MFA"):
        prepare(argparse.Namespace(aligned_manifest=aligned_manifest, output_dir=output, meetings=4, duration=10))
    assert not output.exists()


@pytest.mark.parametrize(
    "ctm",
    [
        "u 1 0.9 0.2 hello\n",
        "u 1 0 0 hello\n",
        "u 1 nan 0.1 hello\n",
        "u 1 0.2 0.4 hello\nu 1 0.1 0.3 world\n",
        "u 1 0 0.1 hello\nv 1 0.2 0.1 world\n",
        "",
    ],
)
def test_invalid_ctm_is_rejected_before_generation(tmp_path: Path, ctm: str) -> None:
    path = tmp_path / "words.ctm"
    path.write_text(ctm)
    with pytest.raises(ValueError, match=r"alignment|Word|clip"):
        read_word_alignment(path, 1)


def test_speaker_identity_is_required_before_mfa(aligned_manifest: Path) -> None:
    rows = [json.loads(line) for line in aligned_manifest.read_text().splitlines()]
    rows[1]["speaker"] = rows[0]["speaker"]
    aligned_manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="two distinct speakers"):
        read_rows(aligned_manifest)


@pytest.mark.parametrize("speaker", ["Speaker 0", "Speaker\t0", "Speaker\u00a00"])
def test_speaker_ids_incompatible_with_rttm_are_rejected_before_mfa(aligned_manifest: Path, speaker: str) -> None:
    rows = [json.loads(line) for line in aligned_manifest.read_text().splitlines()]
    rows[0]["speaker"] = speaker
    aligned_manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="without whitespace"):
        read_rows(aligned_manifest)


@pytest.fixture(scope="module")
def generated_meeting(tmp_path_factory: pytest.TempPathFactory) -> AudioTask:
    import math
    from array import array
    from importlib.metadata import distribution

    from nemo_curator.stages.audio.simulation import FastMSSRequestStage, write_fastmss_requests
    from nemo_curator.tasks import AudioTask

    root = tmp_path_factory.mktemp("native-component-compatibility")
    manifest = aligned_manifest.__wrapped__(root)
    for i, row in enumerate(json.loads(line) for line in manifest.read_text().splitlines()):
        with wave.open(row["audio_filepath"], "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(
                array(
                    "h", (int(1500 * math.sin(2 * math.pi * (220 + i * 110) * n / 16000)) for n in range(16000))
                ).tobytes()
            )
    config = prepare(
        argparse.Namespace(aligned_manifest=manifest, output_dir=root / "prepared", meetings=2, duration=2)
    )
    package = distribution("fastmss")
    roots = {
        "native_inputs_path": str(config.parent),
        "engine_path": str(package.locate_file("fastmss")),
        "recipe_path": str(package.locate_file("recipes")),
        "worker_lock_path": str(root / "worker-lock.json"),
    }
    requests = root / "requests.jsonl"
    write_fastmss_requests([str(config)], str(requests), **roots)
    tasks = [AudioTask(data=json.loads(line)) for line in requests.read_text().splitlines()]
    return FastMSSRequestStage(output_dir=str(root / "generated"), **roots).process_batch(tasks)[0]


def _consumer_stage(consumer: str, output: Path, audio_key: str, duration_key: str) -> ProcessingStage:
    from nemo_curator.stages.audio import agent

    if consumer == "mono":
        return agent.get_agent_ready_stage_class("MonoConversionStage")(
            output_sample_rate=16000, audio_filepath_key=audio_key, duration_key=duration_key
        )
    if consumer == "resample":
        return agent.get_agent_ready_stage_class("ResampleAudioStage")(
            resampled_audio_dir=str(output / "resampled"),
            target_sample_rate=8000,
            audio_filepath_key=audio_key,
            resampled_audio_filepath_key="converted_audio",
            duration_key=duration_key,
        )
    if consumer in {"checkpoint", "manifest"}:
        name = "ManifestCheckpointStage" if consumer == "checkpoint" else "ManifestWriterStage"
        return agent.get_agent_ready_stage_class(name)(output_path=str(output / "rows.jsonl"))
    if consumer == "documents":
        return agent.get_agent_ready_stage_class("AudioToDocumentStage")(strict_json=True)
    return agent.get_agent_ready_stage_class("SampleRateFilterStage")(
        min_sample_rate=16000, audio_filepath_key=audio_key
    )


@pytest.mark.parametrize("renamed", [False, True])
@pytest.mark.parametrize("consumer", ["mono", "resample", "checkpoint", "manifest", "documents", "sample_rate"])
def test_real_native_meetings_compose_with_existing_components(
    generated_meeting: AudioTask, tmp_path: Path, consumer: str, renamed: bool
) -> None:
    import copy

    from nemo_curator.stages.audio import agent
    from nemo_curator.tasks import FileGroupTask

    keys = (
        {"audio_filepath": "mixture", "duration": "seconds", "text": "reference", "rttm_filepath": "speaker_reference"}
        if renamed
        else {}
    )
    data = {keys.get(key, key): value for key, value in copy.deepcopy(generated_meeting.data).items()}
    task = AudioTask(dataset_name="native_meeting", data=data)
    audio_key, duration_key, text_key = (keys.get(key, key) for key in ("audio_filepath", "duration", "text"))
    reference = data[text_key]
    stage = _consumer_stage(consumer, tmp_path, audio_key, duration_key)
    stage.setup()
    try:
        rows = stage.process_batch([task])
        assert len(rows) == 1
        assert rows[0] is not None
        if hasattr(stage, "finalize"):
            stage.finalize()
    finally:
        stage.teardown()
    if consumer == "documents":
        writer = agent.get_agent_ready_stage_class("DocumentBatchJsonlWriterStage")(
            output_path=str(tmp_path / "documents.jsonl")
        )
        writer.setup()
        try:
            writer.process_batch(rows)
        finally:
            writer.teardown()
        saved = json.loads((tmp_path / "documents.jsonl").read_text().splitlines()[0])
        assert saved[text_key] == reference
    else:
        assert rows[0].data[text_key] == reference
    if consumer == "mono":
        assert rows[0].data.sample_rate == 16000
        assert "waveform" in rows[0].data
    elif consumer == "resample":
        from lhotse import Recording

        assert Recording.from_file(rows[0].data.converted_audio).sampling_rate == 8000
    elif consumer in {"checkpoint", "manifest"}:
        reader = agent.get_agent_ready_stage_class("ManifestReaderStage")()
        restored = reader.process(FileGroupTask(dataset_name="read_back", data=[str(tmp_path / "rows.jsonl")]))[0]
        assert getattr(restored.data, audio_key) == rows[0].data[audio_key]
        assert restored.data[text_key] == reference
        assert restored.data.fastmss_request_id == rows[0].data.fastmss_request_id
