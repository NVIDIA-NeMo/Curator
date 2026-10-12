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
import json
from pathlib import Path

import pytest

from nemo_curator.stages.audio.tts.shar import (
    REGISTRY_NAME,
    LhotseSharBuildStage,
    SharEntry,
    SharFieldMap,
    apply_training_filter,
    build_shar_entries,
    build_shars_for_manifest,
    deduplicate_entries,
    load_registry,
    manifest_output_dir,
    pack_best_fit_decreasing,
    plan_shards,
    read_cut_ids,
    read_tar_ids,
    row_language,
    shard_is_aligned,
    split_balanced,
)
from nemo_curator.stages.audio.tts.shar_materialize import marker_path
from nemo_curator.tasks import FileGroupTask
from tests.stages.audio.tts.test_audio_io import make_tar_segments

# Key order of supervision.custom in the published v2610.granary cuts (cuts_with_ipaNG_early_cut_off_filter).
PUBLISHED_CUSTOM_KEYS = [
    "sampling_rate",
    "context_audio_duration",
    "utt_id",
    "sed_weighted_speech_rate",
    "sed_weighted_model_confidence",
    "sed_weighted_classifier_confidence",
    "diarized_speech_rate",
    "MOS",
    "bandwidth",
    "tar_member_offset",
    "tar_member_bytes",
    "cluster_similarity",
    "cluster_margin",
    "context_recording_id",
    "context_audio_speaker_similarity",
    "context_audio_text_overlap",
    "context_tar_member_offset",
    "context_tar_member_bytes",
    "context_audio_text",
    "ipa",
    "valid_tn_raw_transcript",
    "valid_ipa",
    "early_cut_off",
    "valid_context_audio_text",
]


def annotate(segments: list[dict], speaker: str, *, mos: float = 3.0) -> None:
    """Attach the annotation-pipeline outputs the SHAR builder consumes; context = next segment (cyclic)."""
    for i, seg in enumerate(segments):
        ctx = segments[(i + 1) % len(segments)]
        meta = ctx["GranaryHifi"]["AudioMetadata"]
        seg["sed"] = {
            "weighted_speech_rate": 0.9,
            "weighted_model_confidence": 0.4,
            "weighted_classifier_confidence": 0.3,
            "error": None,
        }
        seg["diarization"] = {"diarized_speech_rate": 0.8, "num_speakers": 1, "error": None}
        seg["utmosv2"] = {"score": mos, "error": None}
        seg["bandwidth"] = {"bandwidth": 8000, "bandwidth_sample_rate": 16000, "error": None}
        seg["speaker_cluster"] = {
            "speaker_id": speaker,
            "cluster_similarity": 0.9,
            "cluster_margin": None,
            "error": None,
        }
        seg["context"] = {
            "context_recording_id": meta["tar_member"],
            "context_audio_duration": ctx["duration"],
            "context_audio_text": ctx["GranaryV2"]["pnc_text"],
            "context_audio_speaker_similarity": 0.8,
            "context_audio_text_overlap": 0.1,
            "context_audio_offset": meta["tar_member_offset"],
            "context_audio_bytes": meta["tar_member_bytes"],
            "error": None,
        }
        seg["ipa"] = {"ipa": f"ipa {i}", "error": None}
        seg["tts_flags"] = {
            "valid_tn_raw_transcript": True,
            "valid_ipa": True,
            "valid_context_audio_text": True,
            "early_cut_off": False,
            "context_audio_text_overlap": 0.12,
        }


def make_manifest(
    tmp_path: Path,
    n_videos: int = 2,
    n_segments: int = 3,
    sample_rate: int = 16000,
    languages: tuple[str, ...] = ("it",),
) -> Path:
    """Write an annotated manifest ``<tmp>/it000/manifest.jsonl``; video ``v`` gets ``source_lang=languages[v % n]``."""
    rows = []
    for v in range(n_videos):
        vdir = tmp_path / f"v{v}"
        vdir.mkdir(parents=True)
        specs = [(f"vid{v}_{s:03d}.wav", 0.4 + 0.1 * s, sample_rate) for s in range(n_segments)]
        segments = make_tar_segments(vdir, specs)
        annotate(segments, f"vid{v}_speaker_0")
        rows.append(
            {
                "orig_sample_id": f"vid{v}",
                "orig_sample_rate": sample_rate,
                "source_lang": languages[v % len(languages)],
                "segments": segments,
            }
        )
    manifest = tmp_path / "it000" / "manifest.jsonl"
    manifest.parent.mkdir()
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return manifest


def _entry(cut_id: str, video: str, speaker: str, member: str, mos: float = 1.0) -> SharEntry:
    return SharEntry(
        language="it",
        sampling_rate=16000,
        video_id=video,
        speaker_id=speaker,
        audio_filepath=member,
        target_cut={"id": cut_id, "supervisions": [{"custom": {"MOS": mos}}]},
        context_cut={},
        tar_path="/x.tar",
        target_offset=0,
        target_size=1,
        context_offset=0,
        context_size=1,
    )


def test_split_balanced_and_best_fit() -> None:
    assert [len(c) for c in split_balanced(list(range(10)), 4)] == [4, 3, 3]
    assert split_balanced([], 4) == []
    with pytest.raises(ValueError, match="positive"):
        split_balanced([1], 0)
    shards = pack_best_fit_decreasing([[1, 2, 3], [4], [5, 6], [7, 8, 9, 10]], 5)
    assert sorted(len(s) for s in shards) == [5, 5]  # 4+1 and 3+2
    assert all(len(s) <= 5 for s in shards)


def test_plan_shards_keeps_speakers_together_and_is_deterministic() -> None:
    entries = [_entry(f"c{v}{i}", f"vid{v}", f"vid{v}_spk", f"vid{v}_{i:02d}.wav") for v in range(5) for i in range(3)]
    plan = plan_shards(entries, max_per_shard=6, seed=17)
    assert list(plan) == [("it", 16000)]
    shards = plan[("it", 16000)]
    assert sum(len(s) for s in shards) == 15
    for shard in shards:
        assert len(shard) <= 6
        for speaker in {e.speaker_id for e in shard}:
            members = [e.audio_filepath for e in shard if e.speaker_id == speaker]
            assert members == sorted(members)
            assert len(members) == 3  # a 3-segment speaker is never split across shards
    again = plan_shards(list(reversed(entries)), max_per_shard=6, seed=17)
    assert [[e.cut_id for e in s] for s in again[("it", 16000)]] == [[e.cut_id for e in s] for s in shards]


def test_deduplicate_keeps_higher_mos_and_rejects_conflicts() -> None:
    kept, dups = deduplicate_entries([_entry("a", "v", "s", "m.wav", 1.0), _entry("a", "v", "s", "m.wav", 2.0)])
    assert dups == 1
    assert kept[0].mos == 2.0
    with pytest.raises(ValueError, match="Conflicting"):
        deduplicate_entries([_entry("a", "v", "s", "m.wav"), _entry("a", "v", "s", "other.wav")])


def test_build_entries_final_cut_format(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path, n_videos=1, n_segments=2)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    entries, counts = build_shar_entries(rows, dataset="GranaryHiFi_YTC (orig_audio)", language="it")
    assert counts == {"segments": 2, "without_context": 0, "duplicates": 0}
    cut = entries[0].target_cut
    assert cut["id"] == "cut-rec-vid0_000-0.00-0.40"
    sup = cut["supervisions"][0]
    assert sup["id"] == "sup-rec-vid0_000"
    assert sup["text"] == "text of vid0_000.wav"  # spoken-form tn_raw, not pnc_text
    assert sup["language"] == "it"
    assert sup["speaker"] == "| Language:it Dataset:GranaryHiFi_YTC (orig_audio) Speaker:vid0_speaker_0 |"
    custom = sup["custom"]
    assert list(custom) == PUBLISHED_CUSTOM_KEYS
    assert custom["context_recording_id"] == "rec-vid0_001"
    assert custom["context_audio_text"] == "text of vid0_001.wav"  # context tn_raw looked up in the same row
    assert custom["context_audio_text_overlap"] == 0.12  # from tts_flags
    assert custom["utt_id"] == "vid0_000.wav"
    assert custom["sampling_rate"] == 16000
    assert cut["recording"]["sources"][0]["source"].endswith("shard_0.tar:vid0_000.wav")
    assert entries[0].context_cut["id"] == "context_cut-rec-vid0_001-0.00-0.50"


def test_missing_fields_policy_and_no_context(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path, n_videos=1, n_segments=2)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    del rows[0]["segments"][0]["sed"]
    with pytest.raises(KeyError, match="sed_weighted_speech_rate"):
        build_shar_entries(rows, dataset="D", language="it")
    entries, _ = build_shar_entries(rows, dataset="D", language="it", missing_field_policy="null")
    assert entries[0].target_cut["supervisions"][0]["custom"]["sed_weighted_speech_rate"] is None
    rows[0]["segments"][1]["context"]["error"] = "no_context_audio_below_min_ssim"
    entries, counts = build_shar_entries(rows, dataset="D", language="it", missing_field_policy="null")
    assert counts["without_context"] == 1
    assert len(entries) == 1


def _rows(manifest: Path) -> list[dict]:
    return [json.loads(line) for line in manifest.read_text().splitlines()]


def _write_rows(manifest: Path, rows: list[dict]) -> None:
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_annotation_error_counts_as_missing(tmp_path: Path) -> None:
    rows = _rows(make_manifest(tmp_path, n_videos=1, n_segments=2))
    rows[0]["segments"][0]["bandwidth"] = {
        "bandwidth": None,
        "bandwidth_sample_rate": None,
        "error": "bandwidth_failed",
    }
    with pytest.raises(KeyError, match="bandwidth"):
        build_shar_entries(rows, dataset="D", language="it")
    entries, _ = build_shar_entries(rows, dataset="D", language="it", missing_field_policy="null")
    assert entries[0].target_cut["supervisions"][0]["custom"]["bandwidth"] is None
    # a failed IPA annotation is not "missing": ipa=None and the flags carry valid_ipa=False
    rows[0]["segments"][1]["ipa"] = {"ipa": None, "error": "espeak_failed"}
    entries, _ = build_shar_entries(rows, dataset="D", language="it", missing_field_policy="null")
    assert entries[1].target_cut["supervisions"][0]["custom"]["ipa"] is None


def test_language_precedence() -> None:
    fields = SharFieldMap()
    assert row_language({"source_lang": "fr"}, fields, None) == "fr"
    assert row_language({"source_lang": "fr"}, fields, "de") == "de"  # explicit argument wins
    with pytest.raises(ValueError, match="No language"):
        row_language({}, fields, None)


def test_librilight_preset(tmp_path: Path) -> None:
    rows = _rows(make_manifest(tmp_path, n_videos=1, n_segments=2))
    rows[0].update(speaker_id="spk42", chapter_id="ch7")
    for seg in rows[0]["segments"]:
        del seg["speaker_cluster"]
    with pytest.raises(ValueError, match="Unknown SHAR field preset"):
        SharFieldMap.for_preset("nope")
    entries, _ = build_shar_entries(rows, dataset="L", fields=SharFieldMap.for_preset("librilight"))
    custom = entries[0].target_cut["supervisions"][0]["custom"]
    assert "cluster_similarity" not in custom
    assert "cluster_margin" not in custom
    assert custom["sampling_rate"] == 16000
    assert entries[0].speaker_id == "spk42"
    assert entries[0].video_id == "ch7"


def test_build_shars_roundtrip(tmp_path: Path) -> None:
    from lhotse import CutSet

    manifest = make_manifest(tmp_path, n_videos=2, n_segments=3)
    out = tmp_path / "shars"
    registry = load_registry(build_shars_for_manifest(manifest, out, dataset="D", max_per_shard=4))
    assert registry["counts"]["written_cuts"] == 6
    assert [s["num_items"] for s in registry["shards"]] == [3, 3]  # each 3-segment video stays in one shard
    rate_dir = out / "it" / "16000"
    for shard in registry["shards"]:
        assert shard["dir"] == str(rate_dir)
        assert shard_is_aligned(rate_dir, shard["shard_id"])
        target_ids, target_meta = read_tar_ids(Path(shard["target_audio"]))
        assert target_ids == target_meta == read_cut_ids(Path(shard["cuts"]))
    assert not (rate_dir / ".tmp_materialize").exists()

    cuts = CutSet.from_shar(
        fields={
            "cuts": [registry["shards"][0]["cuts"]],
            "recording": [registry["shards"][0]["target_audio"]],
            "context_recording": [registry["shards"][0]["context_audio"]],
        }
    )
    for cut in cuts:
        audio = cut.load_audio()
        assert audio.shape[0] == 1
        assert abs(audio.shape[1] / 16000 - cut.duration) < 1e-3
        assert cut.context_recording.load_audio().shape[1] > 0
        assert apply_training_filter(cut)


def test_multi_language_manifest_does_not_overwrite(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path, n_videos=2, n_segments=2, languages=("it", "fr"))
    out = tmp_path / "shars"
    registry = load_registry(build_shars_for_manifest(manifest, out, dataset="D"))
    by_lang = {s["language"]: s for s in registry["shards"]}
    assert set(by_lang) == {"it", "fr"}
    assert by_lang["it"]["dir"] != by_lang["fr"]["dir"]
    for lang, video in (("it", "vid0"), ("fr", "vid1")):
        ids = read_cut_ids(Path(by_lang[lang]["cuts"]))
        assert len(ids) == 2
        assert all(f"rec-{video}_" in cut_id for cut_id in ids)
        assert shard_is_aligned(Path(by_lang[lang]["dir"]), 0, ids)
    # explicit language overrides the row language
    forced = load_registry(build_shars_for_manifest(manifest, tmp_path / "forced", dataset="D", language="de"))
    assert {s["language"] for s in forced["shards"]} == {"de"}


def test_resume_rebuilds_on_changed_annotations(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path, n_videos=2, n_segments=3)
    out = tmp_path / "shars"
    registry_path = build_shars_for_manifest(manifest, out, dataset="D", max_per_shard=4)
    rate_dir = out / "it" / "16000"
    mtimes = {p: p.stat().st_mtime_ns for p in rate_dir.rglob("*.tar")}

    build_shars_for_manifest(manifest, out, dataset="D", max_per_shard=4)
    counts = load_registry(registry_path)["counts"]
    assert (counts["reused_shards"], counts["built_shards"]) == (2, 0)
    assert {p: p.stat().st_mtime_ns for p in rate_dir.rglob("*.tar")} == mtimes

    # re-annotation (same cut ids): flag + MOS change on vid0 -> only vid0's shard is rebuilt with the new values
    rows = _rows(manifest)
    rows[0]["segments"][0]["tts_flags"]["early_cut_off"] = True
    rows[0]["segments"][1]["utmosv2"]["score"] = 4.5
    _write_rows(manifest, rows)
    registry = load_registry(build_shars_for_manifest(manifest, out, dataset="D", max_per_shard=4))
    assert (registry["counts"]["reused_shards"], registry["counts"]["built_shards"]) == (1, 1)
    customs = {}
    for shard in registry["shards"]:
        with gzip.open(shard["cuts"], "rt") as f:
            for line in f:
                cut = json.loads(line)
                customs[cut["id"]] = cut["supervisions"][0]["custom"]
    assert customs["cut-rec-vid0_000-0.00-0.40"]["early_cut_off"] is True
    assert customs["cut-rec-vid0_001-0.00-0.50"]["MOS"] == 4.5

    # a shard without its completion marker (e.g. crash between os.replace calls) is rebuilt
    marker_path(rate_dir, 0).unlink()
    assert not shard_is_aligned(rate_dir, 0)
    registry = load_registry(build_shars_for_manifest(manifest, out, dataset="D", max_per_shard=4))
    assert registry["counts"]["built_shards"] == 1
    assert shard_is_aligned(rate_dir, 0)


def test_bad_audio_entries_are_skipped_and_recorded(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path, n_videos=1, n_segments=4)
    rows = _rows(manifest)
    rows[0]["segments"][0]["duration"] = 2.0  # decoded 0.4 s vs manifest 2.0 s
    meta = rows[0]["segments"][1]["GranaryHifi"]["AudioMetadata"]
    meta["tar_member_bytes"] += 10**6  # read past EOF -> short read
    rows[0]["segments"][0]["context"]["context_audio_bytes"] = meta["tar_member_bytes"]
    _write_rows(manifest, rows)
    registry = load_registry(build_shars_for_manifest(manifest, tmp_path / "out", dataset="D"))
    shard = registry["shards"][0]
    reasons = {s["cut_id"]: s["reason"] for s in shard["skipped"]}
    assert reasons["cut-rec-vid0_000-0.00-2.00"] == "duration_mismatch"
    assert reasons["cut-rec-vid0_001-0.00-0.50"] == "short_read"
    assert shard["num_items"] == 2
    assert registry["counts"]["skipped_cuts"] == 2
    assert len(read_cut_ids(Path(shard["cuts"]))) == 2


def test_training_filter() -> None:
    def cut(**custom: object) -> dict:
        base = {
            "valid_tn_raw_transcript": True,
            "valid_ipa": True,
            "valid_context_audio_text": True,
            "early_cut_off": False,
            "context_audio_text_overlap": 0.2,
        }
        return {"supervisions": [{"custom": {**base, **custom}}]}

    assert apply_training_filter(cut())
    assert not apply_training_filter(cut(early_cut_off=True))
    assert not apply_training_filter(cut(valid_ipa=False))
    assert not apply_training_filter(cut(context_audio_text_overlap=0.5))
    assert not apply_training_filter(cut(context_audio_text_overlap=None))
    assert not apply_training_filter({"supervisions": []})


def test_manifest_output_dir_is_unique_and_absolute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    a = manifest_output_dir("out", "/data/run1/it/manifest.jsonl")
    b = manifest_output_dir("out", "/data/run2/it/manifest.jsonl")
    assert a != b
    assert a.is_absolute()
    assert a.parent == tmp_path / "out"
    assert a.name.startswith("manifest-")


def test_stage_processes_file_group(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path)
    stage = LhotseSharBuildStage(output_dir=str(tmp_path / "out"), dataset="D")
    result = stage.process(FileGroupTask(dataset_name="d", data=[str(manifest)]))
    assert result.data == [str(manifest_output_dir(tmp_path / "out", manifest) / REGISTRY_NAME)]
    registry = load_registry(result.data[0])
    with gzip.open(registry["shards"][0]["cuts"], "rt") as f:
        assert json.loads(f.readline())["supervisions"][0]["speaker"].startswith("| Language:it Dataset:D ")
    with pytest.raises(ValueError, match="output_dir"):
        LhotseSharBuildStage()
    with pytest.raises(ValueError, match="missing_field_policy"):
        LhotseSharBuildStage(output_dir="x", missing_field_policy="drop")
    libri = LhotseSharBuildStage(output_dir="rel", preset="librilight")
    assert libri.field_map == SharFieldMap.for_preset("librilight")
    assert Path(libri.output_dir).is_absolute()


def _add_enhanced_copies(tmp_path: Path, rows: list[dict], rate: int = 24000, skip: tuple[str, ...] = ()) -> None:
    import numpy as np
    import soundfile as sf

    enh_dir = tmp_path / "enhanced"
    enh_dir.mkdir(exist_ok=True)
    for row in rows:
        for seg in row["segments"]:
            member = seg["GranaryHifi"]["AudioMetadata"]["tar_member"]
            if member in skip:
                continue
            path = enh_dir / member
            t = np.arange(int(seg["duration"] * rate)) / rate
            sf.write(path, (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32), rate)
            seg["enhanced_audio"] = {
                "audio_filepath": str(path),
                "sampling_rate": rate,
                "duration": seg["duration"],
                "method": "resemble_enhance",
                "error": None,
            }


def test_enhanced_audio_source(tmp_path: Path) -> None:
    from lhotse import CutSet

    from nemo_curator.stages.audio.tts.audio_io import TarAudioConfig

    manifest = make_manifest(tmp_path, n_videos=2, n_segments=3)
    rows = _rows(manifest)
    _add_enhanced_copies(tmp_path, rows)
    _write_rows(manifest, rows)
    enhanced = TarAudioConfig(audio_source="enhanced")
    registry = load_registry(build_shars_for_manifest(manifest, tmp_path / "out", dataset="D", audio_config=enhanced))
    assert registry["audio_source"] == "enhanced"
    assert {s["sampling_rate"] for s in registry["shards"]} == {"24000"}
    shard = registry["shards"][0]
    cuts = list(
        CutSet.from_shar(
            fields={
                "cuts": [shard["cuts"]],
                "recording": [shard["target_audio"]],
                "context_recording": [shard["context_audio"]],
            }
        )
    )
    with gzip.open(shard["cuts"], "rt") as f:
        raw = [json.loads(line) for line in f]
    custom = raw[0]["supervisions"][0]["custom"]
    assert list(custom)[: len(PUBLISHED_CUSTOM_KEYS)] == PUBLISHED_CUSTOM_KEYS
    assert list(custom)[len(PUBLISHED_CUSTOM_KEYS) :] == ["audio_source", "enhancement_method"]
    assert custom["audio_source"] == "enhanced"
    assert custom["enhancement_method"] == "resemble_enhance"
    assert custom["sampling_rate"] == 24000
    assert raw[0]["recording"]["sources"][0]["source"].startswith(str(tmp_path / "enhanced"))
    for cut in cuts:
        assert cut.sampling_rate == 24000
        assert cut.context_recording.sampling_rate == 24000
        assert abs(cut.load_audio().shape[1] / 24000 - cut.duration) < 1e-3

    # original mode stays byte-identical in custom keys (no audio_source key)
    original = load_registry(build_shars_for_manifest(manifest, tmp_path / "orig", dataset="D"))
    with gzip.open(original["shards"][0]["cuts"], "rt") as f:
        assert list(json.loads(f.readline())["supervisions"][0]["custom"]) == PUBLISHED_CUSTOM_KEYS

    # regenerating an enhanced file changes the shard digest -> the shard is rebuilt
    out = tmp_path / "out"
    before = load_registry(build_shars_for_manifest(manifest, out, dataset="D", audio_config=enhanced))
    assert before["counts"]["built_shards"] == 0
    Path(rows[0]["segments"][0]["enhanced_audio"]["audio_filepath"]).write_bytes(
        Path(rows[0]["segments"][1]["enhanced_audio"]["audio_filepath"]).read_bytes()
    )
    after = load_registry(build_shars_for_manifest(manifest, out, dataset="D", audio_config=enhanced))
    assert after["counts"]["built_shards"] >= 1


def test_enhanced_missing_copy_skip_or_fallback(tmp_path: Path) -> None:
    from nemo_curator.stages.audio.tts.audio_io import TarAudioConfig

    manifest = make_manifest(tmp_path, n_videos=1, n_segments=3)
    rows = _rows(manifest)
    _add_enhanced_copies(tmp_path, rows, skip=("vid0_002.wav",))  # target of cut 2, context of cut 1
    _write_rows(manifest, rows)
    registry = load_registry(
        build_shars_for_manifest(
            manifest, tmp_path / "skip", dataset="D", audio_config=TarAudioConfig(audio_source="enhanced")
        )
    )
    assert registry["counts"]["skipped_missing_enhanced_audio"] == 2
    assert registry["counts"]["planned_cuts"] == 1

    registry = load_registry(
        build_shars_for_manifest(
            manifest,
            tmp_path / "fallback",
            dataset="D",
            audio_config=TarAudioConfig(audio_source="enhanced", enhanced_fallback=True),
        )
    )
    by_rate = {s["sampling_rate"]: s for s in registry["shards"]}
    assert set(by_rate) == {"16000", "24000"}
    with gzip.open(by_rate["16000"]["cuts"], "rt") as f:
        customs = [json.loads(line)["supervisions"][0]["custom"] for line in f]
    assert {c["audio_source"] for c in customs} == {"original"}
    assert all("enhancement_method" not in c for c in customs)


def test_keep_speaker_in_one_shard() -> None:
    entries = [_entry(f"a{i}", "vidA", "spkA", f"a_{i:02d}.wav") for i in range(10)]
    entries += [_entry(f"b{i}", "vidB", "spkB", f"b_{i:02d}.wav") for i in range(3)]
    kept = plan_shards(entries, max_per_shard=4)[("it", 16000)]
    assert sorted(len(s) for s in kept) == [3, 10]  # oversized shard for spkA, never split
    assert all(len({e.speaker_id for e in s}) == 1 for s in kept)
    split = plan_shards(entries, max_per_shard=4, keep_speaker_in_one_shard=False)[("it", 16000)]
    assert max(len(s) for s in split) <= 4
    # one speaker across two videos (e.g. LibriLight chapters) still lands in one shard
    multi = [_entry(f"c{v}{i}", f"ch{v}", "spkC", f"c{v}_{i}.wav") for v in range(3) for i in range(2)]
    assert [len(s) for s in plan_shards(multi, max_per_shard=4)[("it", 16000)]] == [6]


def test_context_speaker_mismatch_is_skipped(tmp_path: Path) -> None:
    rows = _rows(make_manifest(tmp_path, n_videos=1, n_segments=3))
    rows[0]["segments"][1]["speaker_cluster"]["speaker_id"] = "vid0_speaker_1"  # context of cut 0
    entries, counts = build_shar_entries(rows, dataset="D", language="it")
    assert counts["skipped_context_speaker_mismatch"] == 2  # cut 0 -> ctx speaker_1; cut 1 -> ctx speaker_0
    assert len(entries) == 1
