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

import hashlib
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile as sf

import nemo_curator.stages.audio.io.nemo_speech_reader as reader_module
from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio.io._nemo_speech_state import shard_registration_path
from nemo_curator.stages.audio.io.nemo_speech_reader import (
    NeMoSpeechAudioReader,
    NeMoSpeechDiscoveryStage,
    NeMoSpeechReaderStage,
    _descriptors_from_entry,
    _load_input_cfg,
    _parse_input_cfg,
)
from nemo_curator.stages.audio.io.nemo_speech_writer import (
    NeMoSpeechWriterStage,
    finalize_nemo_speech_output,
)
from nemo_curator.tasks import EmptyTask, FileGroupTask


def _identity_expand(value: str | list[str]) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else [str(value)]


def _input_cfg(manifest_path: str, *, corpus: str = "dataset", language: str = "en") -> list[dict[str, Any]]:
    return [
        {
            "input_cfg": [
                {
                    "type": "nemo",
                    "corpus": corpus,
                    "language": language,
                    "manifest_filepath": manifest_path,
                }
            ]
        }
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8")


class _FakeCut:
    def __init__(
        self,
        audio_path: str,
        *,
        waveform: np.ndarray | None = None,
        row_index: int | None = None,
    ) -> None:
        self.id = Path(audio_path).stem
        self.start = 0.0
        self.duration = 1.0
        self.recording = SimpleNamespace(
            id=audio_path,
            sampling_rate=16_000,
            sources=[SimpleNamespace(source=audio_path)],
        )
        self.custom = {"audio_filepath": audio_path}
        if row_index is not None:
            self.custom["_nemo_curator_row_index"] = row_index
        self._waveform = waveform if waveform is not None else np.zeros((1, 16_000), dtype=np.float32)

    def load_audio(self) -> np.ndarray:
        return self._waveform


def test_loads_yaml_wrapper_and_inline_config_takes_precedence(tmp_path: Path) -> None:
    config = _input_cfg("/data/dataset/manifest.jsonl")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "input_cfg:\n"
        "  - type: nemo\n"
        "    corpus: dataset\n"
        "    language: en\n"
        "    manifest_filepath: /data/dataset/manifest.jsonl\n",
        encoding="utf-8",
    )

    assert _load_input_cfg(str(config_path)) == config
    assert _load_input_cfg("/does/not/exist.yaml", config) is config


@pytest.mark.parametrize("config", [None, [], {}, ["not-a-mapping"]])
def test_rejects_missing_or_malformed_top_level_config(config: Any) -> None:  # noqa: ANN401
    if config is None:
        with pytest.raises(ValueError, match="input_cfg or yaml_path"):
            _load_input_cfg(None)
    else:
        with pytest.raises(ValueError, match=r"non-empty NeMo input_cfg list|must be a mapping"):
            _load_input_cfg(None, config)


def test_parse_filters_corpus_and_language_before_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reader_module, "_expand_sharded_path", _identity_expand)
    config = [
        {
            "input_cfg": [
                {
                    "type": "nemo",
                    "corpus": "keep",
                    "language": "en",
                    "manifest_filepath": "/data/keep/manifest.jsonl",
                },
                {
                    "type": "unsupported-but-filtered",
                    "corpus": "drop",
                    "language": "de",
                    "manifest_filepath": "/data/drop/manifest.jsonl",
                },
            ]
        }
    ]

    assert _parse_input_cfg(config, corpus_filter=["keep"], language_filter=["en"]) == [
        {
            "corpus": "keep",
            "language": "en",
            "manifest_path": "/data/keep/manifest.jsonl",
            "tar_path": None,
            "shard_key_prefix": None,
        }
    ]


def test_pairs_expanded_tarred_shards(monkeypatch: pytest.MonkeyPatch) -> None:
    expansions = {
        "manifests": ["manifest_0.json", "manifest_1.json"],
        "tars": ["audio_0.tar", "audio_1.tar"],
    }
    monkeypatch.setattr(reader_module, "_expand_sharded_path", lambda value: expansions[str(value)])

    descriptors = _descriptors_from_entry(
        {
            "type": "nemo_tarred",
            "corpus": "dataset",
            "language": "en",
            "manifest_filepath": "manifests",
            "tarred_audio_filepaths": "tars",
            "shard_key_prefix": "catalog/en/dataset",
        }
    )

    assert [(item["manifest_path"], item["tar_path"]) for item in descriptors] == [
        ("manifest_0.json", "audio_0.tar"),
        ("manifest_1.json", "audio_1.tar"),
    ]
    assert all(item["shard_key_prefix"] == "catalog/en/dataset" for item in descriptors)


def test_rejects_mismatched_manifest_and_tar_expansions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        reader_module,
        "_expand_sharded_path",
        lambda value: ["manifest_0.json", "manifest_1.json"] if value == "manifests" else ["audio_0.tar"],
    )

    with pytest.raises(ValueError, match="shard count mismatch"):
        _descriptors_from_entry(
            {
                "type": "nemo_tarred",
                "manifest_filepath": "manifests",
                "tarred_audio_filepaths": "tars",
            }
        )


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"type": "nemo"}, "missing manifest_filepath"),
        ({"type": "other", "manifest_filepath": "manifest.jsonl"}, "Unsupported"),
        ({"type": "nemo_tarred", "manifest_filepath": "manifest.jsonl"}, "requires tarred_audio_filepaths"),
        (
            {
                "type": "nemo",
                "manifest_filepath": "manifest.jsonl",
                "tarred_audio_filepaths": "audio.tar",
            },
            "requires type: nemo_tarred",
        ),
    ],
)
def test_rejects_invalid_entry_contract(
    monkeypatch: pytest.MonkeyPatch,
    entry: dict[str, Any],
    message: str,
) -> None:
    monkeypatch.setattr(reader_module, "_expand_sharded_path", _identity_expand)
    with pytest.raises(ValueError, match=message):
        _descriptors_from_entry(entry)


def test_valid_completion_marker_skips_shard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(reader_module, "_expand_sharded_path", _identity_expand)
    output_dir = tmp_path / "output"
    shard_key = "dataset/manifest"
    output_manifest = output_dir / f"{shard_key}.jsonl"
    _write_jsonl(output_manifest, [{"row": 0}, {"row": 1}])
    marker = output_dir / f"{shard_key}.jsonl.done"
    marker.write_text(
        json.dumps(
            {
                "version": 1,
                "shard_key": shard_key,
                "expected_inputs": 2,
                "completed_inputs": 2,
                "manifest_rows": 2,
                "manifest_sha256": hashlib.sha256(output_manifest.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    stage = NeMoSpeechDiscoveryStage(
        input_cfg=_input_cfg("/inputs/dataset/manifest.jsonl"),
        output_dir=str(output_dir),
    )

    assert stage.process(EmptyTask()) == []


def test_invalid_marker_replays_and_cleans_partial_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(reader_module, "_expand_sharded_path", _identity_expand)
    output_dir = tmp_path / "output"
    shard_key = "dataset/manifest"
    output_manifest = output_dir / f"{shard_key}.jsonl"
    _write_jsonl(output_manifest, [{"partial": True}])
    marker = output_dir / f"{shard_key}.jsonl.done"
    marker.write_text(
        json.dumps(
            {
                "version": 1,
                "shard_key": shard_key,
                "expected_inputs": 2,
                "completed_inputs": 2,
                "manifest_rows": 2,
            }
        ),
        encoding="utf-8",
    )
    legacy_progress = output_dir / ".dataset_manifest.shard_progress.json"
    legacy_progress.write_text('{"seen": ["0"]}', encoding="utf-8")
    receipts = output_dir / ".nemo_curator" / "nemo_speech_rows" / shard_key
    receipts.mkdir(parents=True)
    (receipts / "0.json").write_text("{}", encoding="utf-8")
    opus = output_dir / shard_key / "audio.opus"
    opus.parent.mkdir(parents=True, exist_ok=True)
    opus.write_bytes(b"complete-opus")

    stage = NeMoSpeechDiscoveryStage(
        input_cfg=_input_cfg("/inputs/dataset/manifest.jsonl"),
        output_dir=str(output_dir),
    )
    tasks = stage.process(EmptyTask())

    assert len(tasks) == 1
    assert tasks[0].task_id == ""
    assert tasks[0].data == ["/inputs/dataset/manifest.jsonl"]
    assert tasks[0].reader_config == {
        "corpus": "dataset",
        "language": "en",
        "manifest_path": "/inputs/dataset/manifest.jsonl",
        "tar_path": None,
        "shard_key": shard_key,
    }
    assert tasks[0].get_deterministic_id()
    assert not output_manifest.exists()
    assert not legacy_progress.exists()
    assert not receipts.exists()
    assert opus.read_bytes() == b"complete-opus"


def test_restart_cleanup_preserves_completed_nested_shard_receipts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(reader_module, "_expand_sharded_path", _identity_expand)
    output_dir = tmp_path / "output"
    parent_key = "dataset/foo"
    child_key = "dataset/foo/bar"
    receipt_root = output_dir / ".nemo_curator" / "nemo_speech_rows"
    parent_receipt = receipt_root / parent_key / "parent.success.json"
    child_receipt = receipt_root / child_key / "child.success.json"
    parent_receipt.parent.mkdir(parents=True)
    child_receipt.parent.mkdir(parents=True)
    parent_receipt.write_text("{}", encoding="utf-8")
    child_receipt.write_text("{}", encoding="utf-8")
    child_manifest = output_dir / f"{child_key}.jsonl"
    _write_jsonl(child_manifest, [{"row": 0}])
    child_marker = output_dir / f"{child_key}.jsonl.done"
    child_marker.write_text(
        json.dumps(
            {
                "version": 1,
                "shard_key": child_key,
                "expected_inputs": 1,
                "completed_inputs": 1,
                "manifest_rows": 1,
                "manifest_sha256": hashlib.sha256(child_manifest.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    config = [
        {
            "input_cfg": [
                {
                    "type": "nemo",
                    "corpus": "dataset",
                    "manifest_filepath": "/inputs/dataset/foo.jsonl",
                },
                {
                    "type": "nemo",
                    "corpus": "dataset",
                    "manifest_filepath": "/inputs/dataset/foo/bar.jsonl",
                },
            ]
        }
    ]

    tasks = NeMoSpeechDiscoveryStage(input_cfg=config, output_dir=str(output_dir)).process(EmptyTask())

    assert [task.reader_config["shard_key"] for task in tasks] == [parent_key]
    assert not parent_receipt.exists()
    assert child_receipt.is_file()


@pytest.mark.parametrize("marker_payload", ["", " \n", "[]"])
def test_invalid_completion_markers_are_ignored(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    marker_payload: str,
) -> None:
    monkeypatch.setattr(reader_module, "_expand_sharded_path", _identity_expand)
    output_dir = tmp_path / "output"
    shard_key = "dataset/manifest"
    _write_jsonl(output_dir / f"{shard_key}.jsonl", [{"row": 0}])
    marker = output_dir / f"{shard_key}.jsonl.done"
    marker.write_text(marker_payload, encoding="utf-8")
    stage = NeMoSpeechDiscoveryStage(
        input_cfg=_input_cfg("/inputs/dataset/manifest.jsonl"),
        output_dir=str(output_dir),
        cleanup_partial=False,
    )

    assert len(stage.process(EmptyTask())) == 1


def test_completion_marker_manifest_digest_mismatch_replays_shard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(reader_module, "_expand_sharded_path", _identity_expand)
    output_dir = tmp_path / "output"
    shard_key = "dataset/manifest"
    output_manifest = output_dir / f"{shard_key}.jsonl"
    _write_jsonl(output_manifest, [{"row": "current"}])
    marker = output_dir / f"{shard_key}.jsonl.done"
    marker.write_text(
        json.dumps(
            {
                "version": 1,
                "shard_key": shard_key,
                "expected_inputs": 1,
                "completed_inputs": 1,
                "manifest_rows": 1,
                "manifest_sha256": hashlib.sha256(b'{"row":"different"}\n').hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    stage = NeMoSpeechDiscoveryStage(
        input_cfg=_input_cfg("/inputs/dataset/manifest.jsonl"),
        output_dir=str(output_dir),
        cleanup_partial=False,
    )

    assert len(stage.process(EmptyTask())) == 1
    assert output_manifest.is_file()


def test_composite_preserves_discovery_and_reader_controls() -> None:
    reader = NeMoSpeechAudioReader(
        input_cfg=_input_cfg("/inputs/dataset/manifest.jsonl"),
        corpus_filter=["dataset"],
        language_filter=["en"],
        output_dir="/output",
        cleanup_partial=False,
        resume_mode="checkpoint",
        max_audio_duration_sec=123.0,
        keep_waveform=False,
        reader_workers=3,
    )
    discovery, decode = reader.decompose()

    assert isinstance(discovery, NeMoSpeechDiscoveryStage)
    assert discovery.corpus_filter == ["dataset"]
    assert discovery.language_filter == ["en"]
    assert discovery.output_dir == "/output"
    assert discovery.cleanup_partial is False
    assert discovery.resume_mode == "checkpoint"
    assert isinstance(decode, NeMoSpeechReaderStage)
    assert decode.output_dir == "/output"
    assert decode.max_audio_duration_sec == 123.0
    assert decode.keep_waveform is False
    assert decode.num_workers() == 3


def test_reader_preserves_manifest_identity_when_cuts_are_out_of_order_or_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = [
        {"audio_filepath": "a.wav", "duration": 1.0, "label": "row-a"},
        {"audio_filepath": "b.wav", "duration": 1.0, "label": "row-b"},
        {"audio_filepath": "c.wav", "duration": 1.0, "label": "row-c"},
    ]
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest, rows)
    cuts = [
        _FakeCut("c.wav", waveform=np.full((2, 8), 3.0, dtype=np.float32), row_index=2),
        _FakeCut("a.wav", waveform=np.full((1, 8), 1.0, dtype=np.float32), row_index=0),
    ]
    stage = NeMoSpeechReaderStage()
    monkeypatch.setattr(stage, "_make_tar_cutset", lambda _manifest, _tar: cuts)
    source = FileGroupTask(
        dataset_name="dataset",
        data=[str(manifest), "audio.tar"],
        reader_config={
            "corpus": "dataset",
            "language": "en",
            "manifest_path": str(manifest),
            "tar_path": "audio.tar",
            "shard_key": "dataset/manifest",
        },
        _metadata={"source": "test"},
    )

    results = stage.process(source)

    assert len(results) == len(rows)
    assert [task.task_id for task in results] == ["", "", ""]
    assert all(task._metadata["_shard_key"] == "dataset/manifest" for task in results)
    assert all(task._metadata["_shard_total"] == 3 for task in results)
    assert all(task._metadata["source"] == "test" for task in results)
    by_input = {task._metadata["_shard_input_id"]: task for task in results}
    assert set(by_input) == {"0", "1", "2"}
    assert by_input["0"].data["audio_filepath"] == "a.wav"
    assert by_input["0"].data["label"] == "row-a"
    assert np.all(by_input["0"].data["waveform"] == 1.0)
    assert by_input["1"].data["audio_filepath"] == "b.wav"
    assert by_input["1"].data["label"] == "row-b"
    assert by_input["1"].data["read_error"] is True
    assert by_input["2"].data["audio_filepath"] == "c.wav"
    assert by_input["2"].data["label"] == "row-c"
    assert np.all(by_input["2"].data["waveform"] == 3.0)


def test_reader_decodes_lightweight_local_nemo_manifest(tmp_path: Path) -> None:
    sample_rate = 8_000
    waveform = np.linspace(-0.25, 0.25, 80, dtype=np.float32)
    audio_path = tmp_path / "tone.wav"
    sf.write(audio_path, waveform, sample_rate, subtype="FLOAT")
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "audio_filepath": str(audio_path),
                "duration": len(waveform) / sample_rate,
                "sampling_rate": sample_rate,
                "text": "",
                "tag": "kept",
            }
        ],
    )
    source = FileGroupTask(
        dataset_name="dataset",
        data=[str(manifest)],
        reader_config={
            "corpus": "dataset",
            "language": "en",
            "manifest_path": str(manifest),
            "tar_path": None,
            "shard_key": "dataset/manifest",
        },
    )

    output_dir = tmp_path / "output"
    results = NeMoSpeechReaderStage(output_dir=str(output_dir)).process(source)

    assert len(results) == 1
    result = results[0]
    assert result.task_id == ""
    assert result.data["audio_filepath"] == str(audio_path)
    assert result.data["original_file"] == str(audio_path)
    assert result.data["sampling_rate"] == sample_rate
    assert result.data["sample_rate"] == sample_rate
    assert result.data["num_channels"] == 1
    assert result.data["corpus"] == "dataset"
    assert result.data["source_lang"] == "en"
    assert result.data["tag"] == "kept"
    assert result._metadata == {
        "_shard_key": "dataset/manifest",
        "_shard_total": 1,
        "_shard_input_id": "0",
    }
    np.testing.assert_allclose(result.data["waveform"], waveform)
    registration = json.loads(shard_registration_path(output_dir, "dataset/manifest").read_text(encoding="utf-8"))
    assert registration == {
        "expected_inputs": 1,
        "shard_key": "dataset/manifest",
        "version": 1,
    }


def test_relative_non_tar_source_survives_manifest_only_round_trip(tmp_path: Path) -> None:
    sample_rate = 16_000
    waveform = np.linspace(-0.25, 0.25, 800, dtype=np.float32)
    source_dir = tmp_path / "source"
    audio_path = source_dir / "audio" / "tone.wav"
    audio_path.parent.mkdir(parents=True)
    sf.write(audio_path, waveform, sample_rate, subtype="FLOAT")
    manifest = source_dir / "manifest.jsonl"
    relative_audio_path = "audio/tone.wav"
    _write_jsonl(
        manifest,
        [
            {
                "audio_filepath": relative_audio_path,
                "duration": len(waveform) / sample_rate,
                "sampling_rate": sample_rate,
                "text": "relative",
            }
        ],
    )
    source = FileGroupTask(
        dataset_name="dataset",
        data=[str(manifest)],
        reader_config={
            "corpus": "dataset",
            "language": "en",
            "manifest_path": str(manifest),
            "tar_path": None,
            "shard_key": "dataset/manifest",
        },
    )

    first_read = NeMoSpeechReaderStage().process(source)

    assert len(first_read) == 1
    assert first_read[0].data["audio_filepath"] == relative_audio_path
    assert first_read[0].data["original_file"] == str(audio_path)
    first_read[0]._metadata["_shard_output_id"] = "row:0"

    output_dir = tmp_path / "output"
    writer = NeMoSpeechWriterStage(output_dir=str(output_dir), save_audio=False)
    writer.setup()
    writer.process(first_read[0])
    finalized = finalize_nemo_speech_output(str(output_dir))

    assert not list(output_dir.rglob("*.opus"))
    output_manifest = Path(finalized[0])
    output_rows = [json.loads(line) for line in output_manifest.read_text(encoding="utf-8").splitlines()]
    assert output_rows[0]["audio_filepath"] == str(audio_path)
    assert output_rows[0]["original_audio_filepath"] == str(audio_path)

    reread_source = FileGroupTask(
        dataset_name="dataset",
        data=[str(output_manifest)],
        reader_config={
            "corpus": "dataset",
            "language": "en",
            "manifest_path": str(output_manifest),
            "tar_path": None,
            "shard_key": "roundtrip/manifest",
        },
    )
    reread = NeMoSpeechReaderStage().process(reread_source)
    assert len(reread) == 1
    assert not reread[0].data.get("read_error", False)
    assert reread[0].data["audio_filepath"] == str(audio_path)
    assert reread[0].data["original_file"] == str(audio_path)
    np.testing.assert_allclose(reread[0].data["waveform"], waveform)


def test_non_tar_reader_decodes_sampling_rate_row_with_positive_offset(tmp_path: Path) -> None:
    sample_rate = 8_000
    waveform = np.linspace(-0.5, 0.5, sample_rate, dtype=np.float32)
    audio_path = tmp_path / "tone.wav"
    sf.write(audio_path, waveform, sample_rate, subtype="FLOAT")
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "audio_filepath": str(audio_path),
                "duration": 0.25,
                "offset": 0.5,
                "sampling_rate": sample_rate,
                "text": "segment",
            }
        ],
    )
    source = FileGroupTask(
        dataset_name="dataset",
        data=[str(manifest)],
        reader_config={
            "corpus": "dataset",
            "language": "en",
            "manifest_path": str(manifest),
            "tar_path": None,
            "shard_key": "dataset/manifest",
        },
    )

    results = NeMoSpeechReaderStage().process(source)

    assert len(results) == 1
    result = results[0]
    assert "read_error" not in result.data
    assert result.data["offset"] == 0.5
    assert result.data["duration"] == 0.25
    assert result.data["sampling_rate"] == sample_rate
    np.testing.assert_allclose(
        result.data["waveform"],
        waveform[sample_rate // 2 : 3 * sample_rate // 4],
        atol=4e-5,
    )


def test_done_marker_mode_is_not_pipeline_checkpoint_resumable(tmp_path: Path) -> None:
    stage = NeMoSpeechDiscoveryStage(
        input_cfg=_input_cfg("/inputs/dataset/manifest.jsonl"),
        output_dir=str(tmp_path / "output"),
        resume_mode="done_markers",
    )
    checkpoint_stage = NeMoSpeechDiscoveryStage(
        input_cfg=_input_cfg("/inputs/dataset/manifest.jsonl"),
        output_dir=str(tmp_path / "output"),
        resume_mode="checkpoint",
    )

    assert stage.is_resumable is False
    assert checkpoint_stage.is_resumable is True


def test_done_marker_mode_rejects_checkpoint_before_partial_cleanup(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    partial = output_dir / "dataset/manifest.jsonl"
    _write_jsonl(partial, [{"partial": True}])
    pipeline = Pipeline(
        name="invalid_dual_resume",
        stages=[
            NeMoSpeechAudioReader(
                input_cfg=_input_cfg("/inputs/dataset/manifest.jsonl"),
                output_dir=str(output_dir),
                resume_mode="done_markers",
            )
        ],
    )

    with pytest.raises(ValueError, match="not marked resumable"):
        pipeline.run(checkpoint_path=tmp_path / "checkpoint")

    assert partial.is_file()


def test_duplicate_derived_shard_keys_are_rejected_before_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(reader_module, "_expand_sharded_path", _identity_expand)
    config = [
        {
            "input_cfg": [
                {
                    "type": "nemo",
                    "corpus": "dataset",
                    "manifest_filepath": "/one/dataset/manifest.jsonl",
                },
                {
                    "type": "nemo",
                    "corpus": "dataset",
                    "manifest_filepath": "/two/dataset/manifest.jsonl",
                },
            ]
        }
    ]
    output_dir = tmp_path / "output"
    partial = output_dir / "dataset/manifest.jsonl"
    _write_jsonl(partial, [{"partial": True}])

    with pytest.raises(ValueError, match="duplicate shard key"):
        NeMoSpeechDiscoveryStage(input_cfg=config, output_dir=str(output_dir)).process(EmptyTask())

    assert partial.is_file()


def test_logical_shard_key_participates_in_source_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(reader_module, "_expand_sharded_path", _identity_expand)
    manifest = "/inputs/shared/manifest.jsonl"
    config = [
        {
            "input_cfg": [
                {
                    "type": "nemo",
                    "corpus": "dataset-a",
                    "manifest_filepath": manifest,
                    "shard_key_prefix": "dataset-a/train",
                },
                {
                    "type": "nemo",
                    "corpus": "dataset-b",
                    "manifest_filepath": manifest,
                    "shard_key_prefix": "dataset-b/train",
                },
            ]
        }
    ]
    tasks = NeMoSpeechDiscoveryStage(
        input_cfg=config,
        output_dir=str(tmp_path / "output"),
        resume_mode="checkpoint",
    ).process(EmptyTask())

    assert len(tasks) == 2
    assert tasks[0].data == tasks[1].data
    assert tasks[0].get_deterministic_id() != tasks[1].get_deterministic_id()


def test_non_tar_reader_isolates_iterator_construction_failure(tmp_path: Path) -> None:
    sample_rate = 8_000
    good_waveform = np.linspace(-0.1, 0.1, 80, dtype=np.float32)
    good_audio = tmp_path / "good.wav"
    sf.write(good_audio, good_waveform, sample_rate, subtype="FLOAT")
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"audio_filepath": str(tmp_path / "missing.wav"), "duration": 0.01, "text": "bad"},
        {
            "audio_filepath": str(good_audio),
            "duration": 0.01,
            "sampling_rate": sample_rate,
            "text": "good",
        },
    ]
    _write_jsonl(manifest, rows)
    source = FileGroupTask(
        dataset_name="dataset",
        data=[str(manifest)],
        reader_config={
            "corpus": "dataset",
            "language": "en",
            "manifest_path": str(manifest),
            "tar_path": None,
            "shard_key": "dataset/manifest",
        },
    )

    results = NeMoSpeechReaderStage().process(source)

    assert results[0].data["read_error"] is True
    assert results[0]._metadata["_shard_input_id"] == "0"
    assert results[1].data["text"] == "good"
    np.testing.assert_allclose(results[1].data["waveform"], good_waveform)


def test_real_tarred_reader_preserves_exact_row_identity_and_missing_placeholder(tmp_path: Path) -> None:
    sample_rate = 8_000
    waveform = np.linspace(-0.5, 0.5, sample_rate, dtype=np.float32)
    wave_bytes = io.BytesIO()
    sf.write(wave_bytes, waveform, sample_rate, format="WAV", subtype="FLOAT")
    tar_path = tmp_path / "audio_0.tar"
    with tarfile.open(tar_path, "w") as archive:
        info = tarfile.TarInfo("clip.wav")
        payload = wave_bytes.getvalue()
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    manifest = tmp_path / "manifest_0.json"
    rows = [
        {
            "audio_filepath": "clip-sub1.wav",
            "duration": 0.25,
            "offset": 0.0,
            "text": "first",
            "shard_id": 0,
        },
        {
            "audio_filepath": "clip-sub2.wav",
            "duration": 0.25,
            "offset": 0.5,
            "text": "second",
            "shard_id": 0,
        },
        {
            "audio_filepath": "missing.wav",
            "duration": 0.1,
            "text": "missing",
            "shard_id": 0,
        },
    ]
    _write_jsonl(manifest, rows)
    source = FileGroupTask(
        dataset_name="dataset",
        data=[str(manifest), str(tar_path)],
        reader_config={
            "corpus": "dataset",
            "language": "en",
            "manifest_path": str(manifest),
            "tar_path": str(tar_path),
            "shard_key": "dataset/manifest_0",
        },
    )

    results = NeMoSpeechReaderStage().process(source)

    assert [result._metadata["_shard_input_id"] for result in results] == ["0", "1", "2"]
    assert [result.data["text"] for result in results] == ["first", "second", "missing"]
    assert results[2].data["read_error"] is True
    assert "_nemo_curator_row_index" not in results[0].data
    assert results[0].data["manifest_origin"] == str(manifest)
    assert results[1].data["manifest_origin"] == str(manifest)
    np.testing.assert_allclose(results[0].data["waveform"], waveform[: sample_rate // 4], atol=4e-5)
    np.testing.assert_allclose(
        results[1].data["waveform"],
        waveform[sample_rate // 2 : 3 * sample_rate // 4],
        atol=4e-5,
    )


def test_empty_manifest_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("", encoding="utf-8")
    source = FileGroupTask(
        dataset_name="dataset",
        data=[str(manifest)],
        reader_config={
            "corpus": "dataset",
            "language": "en",
            "manifest_path": str(manifest),
            "tar_path": None,
            "shard_key": "dataset/manifest",
        },
    )

    with pytest.raises(ValueError, match="empty shards are unsupported"):
        NeMoSpeechReaderStage().process(source)


def test_public_lazy_exports_are_available() -> None:
    from nemo_curator.stages.audio import NeMoSpeechAudioReader as AudioReaderExport
    from nemo_curator.stages.audio.io import NeMoSpeechWriterStage as WriterExport

    assert AudioReaderExport is NeMoSpeechAudioReader
    assert WriterExport.__name__ == "NeMoSpeechWriterStage"


def test_pipeline_reader_writer_terminal_sink_end_to_end(tmp_path: Path) -> None:
    sample_rate = 16_000
    waveform = np.linspace(-0.2, 0.2, 800, dtype=np.float32)
    audio_path = tmp_path / "tone.wav"
    sf.write(audio_path, waveform, sample_rate, subtype="FLOAT")
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "audio_filepath": str(audio_path),
                "duration": len(waveform) / sample_rate,
                "sampling_rate": sample_rate,
                "text": "pipeline",
            }
        ],
    )
    output_dir = tmp_path / "output"
    pipeline = Pipeline(name="nemo_speech_io_e2e")
    pipeline.add_stage(
        NeMoSpeechAudioReader(
            input_cfg=[
                {
                    "type": "nemo",
                    "corpus": "dataset",
                    "manifest_filepath": str(manifest),
                    "shard_key_prefix": "dataset/test",
                }
            ],
            output_dir=str(output_dir),
        )
    )
    pipeline.add_stage(NeMoSpeechWriterStage(output_dir=str(output_dir)))

    results = pipeline.run(XennaExecutor())

    assert results == []
    finalized = finalize_nemo_speech_output(str(output_dir))
    assert len(finalized) == 1
    manifest_path = Path(finalized[0])
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["text"] == "pipeline"
    assert rows[0]["audio_filepath"].startswith(f"{manifest_path.stem}/audio/")
    assert (manifest_path.parent / rows[0]["audio_filepath"]).is_file()

    reread_source = FileGroupTask(
        dataset_name="dataset",
        data=[str(manifest_path)],
        reader_config={
            "corpus": "dataset",
            "language": "",
            "manifest_path": str(manifest_path),
            "tar_path": None,
            "shard_key": "roundtrip/output",
        },
    )
    reread = NeMoSpeechReaderStage().process(reread_source)
    assert len(reread) == 1
    assert not reread[0].data.get("read_error", False)
    assert reread[0].data["text"] == "pipeline"
    assert reread[0].data["sample_rate"] == sample_rate
    assert len(reread[0].data["waveform"]) == len(waveform)
