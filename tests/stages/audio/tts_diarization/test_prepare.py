# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check source selection and the aligned-manifest/simulator boundary."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from nemo_curator.config.run import create_pipeline_from_yaml
from nemo_curator.stages.audio.alignment import MFAAlignmentStage
from nemo_curator.stages.audio.simulation import FastMSSManifestReaderStage, FastMSSSimulationStage
from nemo_curator.tasks import AudioTask, FileGroupTask
from tutorials.audio.tts_diarization.prepare import export_cutset, prepare_jobs, prepare_recipes, read_rows, write_rows


@pytest.fixture
def catalogs(tmp_path: Path) -> tuple[Path, Path]:
    reference = tmp_path / "reference.wav"
    sf.write(reference, np.zeros(16000 * 5), 16000)
    texts, references = tmp_path / "texts.jsonl", tmp_path / "references.jsonl"
    write_rows(
        texts,
        [{"utterance_id": str(i), "speaker": "a", "text": f"Example {i}", "chapter_id": "chapter"} for i in range(6)],
    )
    write_rows(
        references,
        [
            {
                "speaker": speaker,
                "reference_audio": reference.name,
                "split": split,
                "language": language,
                "channel": "ihm",
            }
            for speaker, split, language in (("a", "train", "fr"), ("b", "test", "de"))
        ],
    )
    return texts, references


def test_job_pairing_is_deterministic_and_retains_source_identity(catalogs: tuple[Path, Path], tmp_path: Path):
    texts, references = catalogs
    outputs = [tmp_path / f"jobs{i}.jsonl" for i in range(2)]
    for output in outputs:
        assert prepare_jobs(texts, references, output, "direct") == 6
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    jobs = read_rows(outputs[0])
    assert len({row["utterance_id"] for row in jobs}) == len(jobs)
    assert {row["source_utterance_id"] for row in jobs} == {str(i) for i in range(6)}
    assert all(row["speaker"] == row["source_speaker"] == "a" for row in jobs)
    assert all(row["chapter_id"] == "chapter" and row["reference_channel"] == "ihm" for row in jobs)
    assert all(Path(row["reference_audio"]).is_absolute() for row in jobs)


def test_target_selection_excludes_test_references_unless_requested(catalogs: tuple[Path, Path], tmp_path: Path):
    texts, references = catalogs
    output = tmp_path / "train.jsonl"
    assert prepare_jobs(texts, references, output, "target-domain", utterances_per_speaker=3) == 3
    assert {row["reference_split"] for row in read_rows(output)} == {"train"}
    output = tmp_path / "test.jsonl"
    assert (
        prepare_jobs(
            texts,
            references,
            output,
            "target-domain",
            utterances_per_speaker=3,
            languages=["de"],
            include_test_references=True,
        )
        == 3
    )
    assert {row["speaker"] for row in read_rows(output)} == {"b"}
    assert {row["reference_split"] for row in read_rows(output)} == {"test"}


@pytest.mark.parametrize("strategy", ["cross-lingual", "target-domain"])
def test_sampling_requires_enough_distinct_texts(catalogs: tuple[Path, Path], tmp_path: Path, strategy: str):
    with pytest.raises(ValueError, match="text pool"):
        prepare_jobs(*catalogs, tmp_path / "jobs.jsonl", strategy)


def test_jobs_reject_short_references_and_duplicate_text_ids(catalogs: tuple[Path, Path], tmp_path: Path):
    texts, references = catalogs
    sf.write(tmp_path / "reference.wav", np.zeros(16000 * 4), 16000)
    with pytest.raises(ValueError, match="shorter than 5"):
        prepare_jobs(texts, references, tmp_path / "jobs.jsonl", "direct")
    texts.write_text(
        json.dumps({"utterance_id": "same", "text": "hello"})
        + "\n"
        + json.dumps({"utterance_id": "same", "text": "again"})
        + "\n"
    )
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_jobs(texts, references, tmp_path / "jobs.jsonl", "direct")


@pytest.fixture
def aligned_manifest(tmp_path: Path) -> Path:
    pytest.importorskip("lhotse")
    from praatio import textgrid

    alignment = MFAAlignmentStage(output_dir=str(tmp_path / "mfa"), copy_models_to_local=False)
    alignment.setup()
    rows = []
    for i in range(4):
        waveform = tmp_path / f"speaker{i}.wav"
        sf.write(waveform, 0.05 * np.sin(2 * np.pi * (220 + 100 * i) * np.arange(24000) / 24000), 24000)
        tg_path = tmp_path / f"speaker{i}.TextGrid"
        tg = textgrid.Textgrid()
        tg.addTier(textgrid.IntervalTier("words", [(0.1, 0.4, "hello"), (0.5, 0.9, "world")], 0, 1))
        tg.save(str(tg_path), format="long_textgrid", includeBlankSpaces=True)
        task = AudioTask(
            data={
                "utterance_id": f"utterance{i}",
                "audio_filepath": str(waveform),
                "text": "hello world",
                "speaker": f"speaker{i}",
                "duration": 1,
            }
        )
        alignment._handle_successful_textgrid(f"speaker{i}", task, tg_path, "batch")
        rows.append(dict(task.data))
    manifest = tmp_path / "aligned.jsonl"
    write_rows(manifest, rows)
    return manifest


def test_export_preserves_words_and_real_tts_speaker_identity(aligned_manifest: Path, tmp_path: Path):
    from lhotse import CutSet

    summary = export_cutset(aligned_manifest, tmp_path / "corpus", real=aligned_manifest)
    cuts = list(CutSet.from_file(summary["cutset"]))
    assert summary["utterances"] == {"tts": 4, "real": 4}
    assert summary["speakers"] == 4
    assert len({cut.id for cut in cuts}) == 8
    assert {cut.custom["source_kind"] for cut in cuts} == {"tts", "real"}
    assert all([word.symbol for word in cut.supervisions[0].alignment["word"]] == ["hello", "world"] for cut in cuts)
    with pytest.raises(FileExistsError):
        export_cutset(aligned_manifest, tmp_path / "corpus")


@pytest.mark.parametrize("invalid", ["skipped", "outside", "unordered", "stereo"])
def test_export_rejects_unaligned_or_invalid_sources(aligned_manifest: Path, tmp_path: Path, invalid: str):
    rows = read_rows(aligned_manifest)[:1]
    if invalid == "skipped":
        rows[0]["mfa_skipped"] = True
    elif invalid == "stereo":
        sf.write(rows[0]["audio_filepath"], np.zeros((24000, 2)), 24000)
    else:
        timing = "1.0 1.0" if invalid == "outside" else "0.5 0.4"
        Path(rows[0]["ctm_filepath"]).write_text(f"id 1 {timing} hello\nid 1 0.1 0.1 world\n")
    manifest = tmp_path / "bad.jsonl"
    write_rows(manifest, rows)
    with pytest.raises(ValueError, match=r"successful MFA|outside|ordered|mono"):
        export_cutset(manifest, tmp_path / "corpus")


def test_generated_recipes_execute_with_exported_word_alignments(aligned_manifest: Path, tmp_path: Path):
    pytest.importorskip("fastmss")
    pytest.importorskip("pyroomacoustics")
    export_cutset(aligned_manifest, tmp_path / "corpus")
    recipes = prepare_recipes(
        tmp_path / "corpus",
        tmp_path / "recipes",
        None,
        count_unit=1,
        duration=2,
        augment=False,
        min_utterances=1,
    )
    stage = FastMSSSimulationStage(output_dir=str(tmp_path / "mixtures"), isolate_work_dirs=True, timeout_seconds=90)
    for count, ratio, recipe in zip(range(1, 5), (1, 3, 6, 10), recipes, strict=True):
        outputs = stage.process(FileGroupTask(data=[str(recipe)], dataset_name="paper-smoke"))
        rows = FastMSSManifestReaderStage().process(outputs[0])
        assert len(rows) == ratio
        effective = json.loads((Path(outputs[0].data[0]).parent / "effective_config.json").read_text())
        assert effective["min_max_spk"] == [count, count]
        assert effective["boost_overlap_factor"] is None
        for row in rows:
            assert sf.info(row.data["audio_filepath"]).samplerate == 16000
            assert row.data["text"].strip()
            assert Path(row.data["rttm_filepath"]).stat().st_size > 0


@pytest.mark.parametrize("name", ["tts", "align", "simulate"])
def test_stage_yamls_instantiate_with_shared_runner(tmp_path: Path, name: str):
    config_dir = Path(__file__).resolve().parents[4] / "tutorials/audio/tts_diarization/configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=name)
        OmegaConf.set_struct(cfg, False)
        for key in ("input_manifest", "output_dir", "mfa_root_dir", "recipes_dir"):
            cfg[key] = str(tmp_path / key)
        pipeline = create_pipeline_from_yaml(cfg, log_config=False)
    assert pipeline is not None
