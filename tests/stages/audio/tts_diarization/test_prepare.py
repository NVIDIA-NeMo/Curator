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


@pytest.mark.parametrize("speakers", [1, 4])
def test_default_augmentation_generates_noisy_reverberant_meetings(  # noqa: PLR0915
    aligned_manifest: Path, tmp_path: Path, speakers: int
):
    pytest.importorskip("fastmss")
    pytest.importorskip("pyroomacoustics")
    sample_rate = 16000
    musan = tmp_path / "musan"
    rng = np.random.default_rng(42)
    noise_files = set()
    for subset in ("noise", "music"):
        directory = musan / subset
        directory.mkdir(parents=True)
        path = directory / "sample.wav"
        sf.write(path, rng.normal(0, 0.05, sample_rate * 4), sample_rate)
        noise_files.add(str(path))
    export_cutset(aligned_manifest, tmp_path / "corpus")
    recipes = prepare_recipes(
        tmp_path / "corpus",
        tmp_path / "recipes",
        musan,
        count_unit=1,
        duration=2,
        min_utterances=1,
    )
    recipe = recipes[speakers - 1]
    config = OmegaConf.load(recipe)
    assert config.noise_probability_global == config.reverb_prob == 0.5
    stage = FastMSSSimulationStage(
        output_dir=str(tmp_path / "mixtures"),
        isolate_work_dirs=True,
        max_meetings=1,
        timeout_seconds=90,
        overrides=[
            "noise_probability_global=1",
            "reverb_prob=1",
            "n_rirs=1",
            "n_pos_rirs=4",
            "save_spk=true",
            "save_anechoic=true",
        ],
    )
    outputs = stage.process(FileGroupTask(data=[str(recipe)], dataset_name="augmented-paper-smoke"))
    rows = FastMSSManifestReaderStage().process(outputs[0])
    assert len(rows) == 1
    run_dir = Path(outputs[0].data[0]).parent
    effective = json.loads((run_dir / "effective_config.json").read_text())
    assert effective["add_noise"]
    assert effective["reverberate"]
    assert effective["noise_probability_global"] == effective["reverb_prob"] == 1
    assert set((run_dir / "manifests/noise_files.txt").read_text().splitlines()) == noise_files
    rooms = json.loads((run_dir / "rir/all_rooms.json").read_text())
    assert len(rooms) == 1
    assert len(rooms[0]) == 4
    assert all(Path(path).is_file() for path in rooms[0])

    row = rows[0].data
    audio_path = Path(row["audio_filepath"])
    waveform, rate = sf.read(audio_path)
    assert rate == sample_rate
    assert waveform.ndim == 1
    assert np.isfinite(waveform).all()
    assert np.max(np.abs(waveform)) > 0
    assert row["duration"] == pytest.approx(len(waveform) / rate, abs=1e-5)
    assert row["text"].strip()
    assert row["num_speakers"] == speakers
    labels = {line.split()[7] for line in Path(row["rttm_filepath"]).read_text().splitlines()}
    assert len(labels) == speakers
    assert labels <= {f"speaker{i}" for i in range(4)}
    speech = np.zeros_like(waveform)
    for speaker in labels:
        reverberant, reverb_rate = sf.read(audio_path.with_stem(f"{audio_path.stem}-spk-{speaker}"))
        dry, dry_rate = sf.read(audio_path.with_stem(f"{audio_path.stem}-spk-{speaker}-anechoic"))
        assert reverb_rate == dry_rate == rate
        assert reverberant.shape == dry.shape == waveform.shape
        assert np.isfinite(reverberant).all()
        assert np.isfinite(dry).all()
        assert not np.allclose(reverberant, dry, atol=2 / 32768)
        speech += reverberant
    # PCM16 quantization alone cannot account for the residual mixture energy.
    assert np.sqrt(np.mean((waveform - speech) ** 2)) > (speakers + 1) / 32768


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
