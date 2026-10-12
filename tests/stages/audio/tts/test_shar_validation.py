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

import gzip
import io
import json
import tarfile
from pathlib import Path

import pytest
import yaml

from nemo_curator.stages.audio.tts.codec import CodecEncoder, extract_codes_for_registry
from nemo_curator.stages.audio.tts.shar import build_shars_for_manifest, load_registry
from nemo_curator.stages.audio.tts.shar_validation import (
    VALIDATION_REPORT,
    PublishLayout,
    SharValidationError,
    SharValidationStage,
    publish_shars,
    validate_registry,
)
from nemo_curator.tasks import FileGroupTask
from tests.stages.audio.tts.test_codec import FakeCodec
from tests.stages.audio.tts.test_shar import make_manifest


def coded_registry(tmp_path: Path) -> Path:
    manifest = make_manifest(tmp_path, n_videos=2, n_segments=3)
    registry = build_shars_for_manifest(manifest, tmp_path / "shars", dataset="D", max_per_shard=4)
    extract_codes_for_registry(registry, CodecEncoder(FakeCodec(), "cpu"))
    return registry


def _rewrite_tar_without_last(path: Path) -> None:
    with tarfile.open(path) as src:
        members = [(m, src.extractfile(m).read()) for m in src.getmembers()]
    with tarfile.open(path, "w") as dst:
        for member, payload in members[:-2]:
            dst.addfile(member, io.BytesIO(payload))


def test_valid_registry_and_report(tmp_path: Path) -> None:
    registry = coded_registry(tmp_path)
    report = validate_registry(registry)
    assert report["valid"], report["problems"]
    assert report["num_cuts"] == 6
    assert json.loads((registry.parent / VALIDATION_REPORT).read_text())["valid"] is True


def test_detects_misalignment_missing_codes_and_missing_ipa(tmp_path: Path) -> None:
    registry = coded_registry(tmp_path)
    shard = load_registry(registry)["shards"][0]
    _rewrite_tar_without_last(Path(shard["context_audio"]))
    report = validate_registry(registry)
    assert not report["valid"]
    assert any("context_audio payload: count mismatch" in p for p in report["problems"])

    registry2 = coded_registry(tmp_path / "b")
    shard2 = load_registry(registry2)["shards"][1]
    cuts_path = Path(shard2["cuts"])
    with gzip.open(cuts_path, "rt") as f:
        rows = [json.loads(line) for line in f]
    rows[0]["supervisions"][0]["custom"]["ipa"] = ""
    with gzip.open(cuts_path, "wt") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    assert any("no IPA" in p for p in validate_registry(registry2)["problems"])
    rows[0]["supervisions"][0]["custom"]["valid_ipa"] = False  # flagged invalid -> acceptable
    with gzip.open(cuts_path, "wt") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    assert validate_registry(registry2)["valid"]

    codes_dir = Path(shard2["dir"]) / "25fpsSpectralCodecBWE" / "target_codes"
    next(codes_dir.iterdir()).unlink()
    assert any("missing streams" in p for p in validate_registry(registry2)["problems"])
    assert validate_registry(registry2, require_codes=False)["valid"]


def test_validation_stage(tmp_path: Path) -> None:
    registry = coded_registry(tmp_path)
    task = FileGroupTask(dataset_name="d", data=[str(registry)])
    assert SharValidationStage().process(task).data == [str(registry)]
    _rewrite_tar_without_last(Path(load_registry(registry)["shards"][0]["target_audio"]))
    with pytest.raises(SharValidationError):
        SharValidationStage().process(task)
    assert SharValidationStage(fail_on_error=False).process(task).data == []


def test_publish_hardlinks_layout_and_is_idempotent(tmp_path: Path) -> None:
    from lhotse import CutSet

    registry_a = coded_registry(tmp_path / "a")
    registry_b = coded_registry(tmp_path / "b")
    root = tmp_path / "published"
    layout = PublishLayout.for_dataset("ytc")
    assert layout == PublishLayout(dataset="YTC", speech_domain="YouTube speech")
    result = publish_shars([registry_a, registry_b], root, layout)
    assert result == {"output_root": str(root), "shards": 4, "status": "published"}

    pair = root / "it" / "GranaryHiFi" / "YTC" / "16000"
    cuts = sorted((pair / "cuts_with_ipaNG_early_cut_off_filter").glob("cuts.*.jsonl.gz"))
    assert [p.name for p in cuts] == [f"cuts.{i:06d}.jsonl.gz" for i in range(4)]
    for sub in ("target_audio", "context_audio"):
        assert len(list((pair / sub).glob("recording.*.tar"))) == 4
    first_source = load_registry(registry_a)["shards"][0]
    assert (pair / "target_audio" / "recording.000000.tar").stat().st_ino == Path(
        first_source["target_audio"]
    ).stat().st_ino

    inventory = yaml.safe_load((root / "cfgs" / "it" / "GranaryHiFi_YTC_16000_train.yaml").read_text())
    assert inventory["shar_base"] == "${oc.env:AUDIO_ROOT}/it/GranaryHiFi/YTC/16000"
    assert inventory["num_shards"] == 4
    assert inventory["codecs"] == {"spectralCodecBWE25fps": "25fpsSpectralCodecBWE"}
    assert inventory["cut_variants"] == {
        "audioContext_with_ipa_wo_early_cut_off": {
            "cuts_dir": "cuts_with_ipaNG_early_cut_off_filter",
            "has_audio_context": True,
        }
    }
    assert len((root / "index.jsonl").read_text().splitlines()) == 4

    # training-style read of the published layout (cuts + audio + codes, 6-digit shard ranges)
    fields = {
        "cuts": [str(pair / "cuts_with_ipaNG_early_cut_off_filter" / f"cuts.{i:06d}.jsonl.gz") for i in range(4)],
        "target_audio": [str(pair / "target_audio" / f"recording.{i:06d}.tar") for i in range(4)],
        "context_audio": [str(pair / "context_audio" / f"recording.{i:06d}.tar") for i in range(4)],
        "target_codes": [
            str(pair / "25fpsSpectralCodecBWE" / "target_codes" / f"codes.{i:06d}.tar") for i in range(4)
        ],
        "context_codes": [
            str(pair / "25fpsSpectralCodecBWE" / "context_codes" / f"codes.{i:06d}.tar") for i in range(4)
        ],
    }
    loaded = list(CutSet.from_shar(fields=fields))
    assert len(loaded) == 12
    assert loaded[0].load_target_codes().shape[0] == 3
    assert loaded[0].load_target_audio().shape[0] == 1

    assert publish_shars([registry_a, registry_b], root, layout)["status"] == "already_published"
    with pytest.raises(FileExistsError):
        publish_shars([registry_a], root, layout)
    assert not list(tmp_path.glob(".published.tmp.*"))


def test_publish_refuses_invalid(tmp_path: Path) -> None:
    registry = coded_registry(tmp_path)
    _rewrite_tar_without_last(Path(load_registry(registry)["shards"][0]["target_audio"]))
    with pytest.raises(SharValidationError):
        publish_shars([registry], tmp_path / "pub")
    assert not (tmp_path / "pub").exists()


def test_publish_layout_for_dataset() -> None:
    assert PublishLayout.for_dataset("VoxPopuli").speech_domain == "European Parliament speech"
    assert PublishLayout.for_dataset("librilight").dataset == "LibriLight"
    assert PublishLayout.for_dataset("YODAS", speech_domain="web").speech_domain == "web"
    with pytest.raises(ValueError, match="Unknown GranaryHiFi dataset"):
        PublishLayout.for_dataset("mls")


def test_publish_without_codes(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path, n_videos=1, n_segments=3)
    registry = build_shars_for_manifest(manifest, tmp_path / "shars", dataset="D")
    task = FileGroupTask(dataset_name="d", data=[str(registry)])
    assert SharValidationStage(require_codes=False).process(task).data == [str(registry)]
    root = tmp_path / "pub"
    result = publish_shars([registry], root, PublishLayout.for_dataset("YODAS"), require_codes=False)
    assert result["status"] == "published"
    report = json.loads((registry.parent / VALIDATION_REPORT).read_text())
    assert report["valid"] is True
    assert report["require_codes"] is False
    pair = root / "it" / "GranaryHiFi" / "YODAS" / "16000"
    assert not (pair / "25fpsSpectralCodecBWE").exists()
    inventory = yaml.safe_load((root / "cfgs" / "it" / "GranaryHiFi_YODAS_16000_train.yaml").read_text())
    assert inventory["codecs"] == {}
    with pytest.raises(SharValidationError):
        publish_shars([registry], tmp_path / "pub2")  # codes required by default


def test_speaker_grouping_validation(tmp_path: Path) -> None:
    from nemo_curator.stages.audio.tts.shar_validation import validate_speaker_grouping

    registry_path = coded_registry(tmp_path)
    registry = load_registry(registry_path)
    assert validate_speaker_grouping(registry) == []

    # point a vid0 cut's context at a vid1 recording (other speaker, other shard)
    shard0 = registry["shards"][0]
    with gzip.open(registry["shards"][1]["cuts"], "rt") as f:
        other_rec = json.loads(f.readline())["recording"]["id"]
    with gzip.open(shard0["cuts"], "rt") as f:
        rows = [json.loads(line) for line in f]
    rows[0]["supervisions"][0]["custom"]["context_recording_id"] = other_rec
    with gzip.open(shard0["cuts"], "wt") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    problems = validate_registry(registry_path)["problems"]
    assert any("belongs to speaker" in p for p in problems)


def test_speaker_spanning_shards_is_reported(tmp_path: Path) -> None:
    from nemo_curator.stages.audio.tts.shar_validation import validate_speaker_grouping

    manifest = make_manifest(tmp_path, n_videos=1, n_segments=4)
    registry = load_registry(
        build_shars_for_manifest(
            manifest, tmp_path / "shars", dataset="D", max_per_shard=2, keep_speaker_in_one_shard=False
        )
    )
    assert len(registry["shards"]) == 2
    assert validate_speaker_grouping(registry, require_one_shard=False) == []
    problems = validate_speaker_grouping(registry, require_one_shard=True)
    assert any("spans shards" in p for p in problems)
    assert any("is in shard" in p for p in problems)
