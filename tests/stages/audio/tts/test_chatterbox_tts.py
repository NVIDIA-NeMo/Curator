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

"""Tests for ChatterboxTTSStage."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
import torch

from nemo_curator.stages.audio.tts.chatterbox_tts import (
    _ENGLISH_MODEL_FILES,
    ChatterboxTTSStage,
)
from nemo_curator.tasks import AudioTask
from nemo_curator.utils.performance_utils import StagePerfStats

if TYPE_CHECKING:
    from pathlib import Path

MODULE = "nemo_curator.stages.audio.tts.chatterbox_tts"


@pytest.fixture
def ref_dataset(tmp_path: Path) -> str:
    """Reference voices in wavs/ layout with optional rttms/."""
    wavs_dir = tmp_path / "wavs" / "dialog001"
    wavs_dir.mkdir(parents=True)
    rttms_dir = tmp_path / "rttms" / "dialog001"
    rttms_dir.mkdir(parents=True)

    sr = 16000
    rng = np.random.default_rng()
    for spk in ("spk_A", "spk_B", "spk_C"):
        audio = rng.standard_normal(sr * 5).astype(np.float32)
        sf.write(str(wavs_dir / f"{spk}.wav"), audio, sr)
        rttm_path = rttms_dir / f"{spk}.rttm"
        rttm_path.write_text(
            f"SPEAKER dialog001 1 0.0 2.0 <NA> <NA> {spk} <NA> <NA>\n"
            f"SPEAKER dialog001 1 3.0 1.5 <NA> <NA> {spk} <NA> <NA>\n"
        )
    return str(tmp_path)


@pytest.fixture
def ref_dataset_mls(tmp_path: Path) -> str:
    """Reference voices in MLS layout."""
    sr = 16000
    rng = np.random.default_rng()
    for spk_id in ("1234", "5678", "9012"):
        book_dir = tmp_path / spk_id / "book01"
        book_dir.mkdir(parents=True)
        for seg in range(3):
            audio = rng.standard_normal(sr * 2).astype(np.float32)
            sf.write(str(book_dir / f"{spk_id}_book01_{seg:04d}.flac"), audio, sr)
    return str(tmp_path)


@pytest.fixture
def output_dir(tmp_path: Path) -> str:
    return str(tmp_path / "tts_output")


def _fake_model(sample_rate: int = 24000) -> MagicMock:
    model = MagicMock()
    model.sr = sample_rate  # native output rate, as real Chatterbox models expose it

    def _generate(text: str, **_kwargs: object) -> torch.Tensor:
        duration_sec = max(0.5, len(text) * 0.02)
        n_samples = int(sample_rate * duration_sec)
        t = torch.linspace(0, duration_sec, n_samples)
        return 0.3 * torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)

    model.generate.side_effect = _generate
    return model


def _make_task(
    text: str = "Hello world",
    speaker: str = "Alice",
    conversation_id: str = "conv001",
    task_id: str = "t1",
    **extra_fields: object,
) -> AudioTask:
    data = {
        "utterance": text,
        "speaker": speaker,
        "conversation_id": conversation_id,
        **extra_fields,
    }
    return AudioTask(data=data, task_id=task_id, dataset_name="test")


def _build_stage(
    output_dir: str,
    ref_dataset: str,
    language: str | None = None,
    **overrides: object,
) -> ChatterboxTTSStage:
    kwargs = {
        "output_audio_dir": output_dir,
        "reference_voices_dataset": ref_dataset,
        "language": language,
        "device": "cpu",
    }
    kwargs.update(overrides)
    return ChatterboxTTSStage(**kwargs)


def _inject_model(stage: ChatterboxTTSStage) -> None:
    stage.model = _fake_model()


class TestChatterboxTTSStage:
    """Test suite for ChatterboxTTSStage."""

    @patch("chatterbox.tts.ChatterboxTTS")
    def test_setup_loads_english_model(
        self, mock_cls: MagicMock, output_dir: str, ref_dataset: str
    ) -> None:
        mock_cls.from_pretrained.return_value = _fake_model()
        stage = _build_stage(output_dir, ref_dataset)
        stage.setup()
        mock_cls.from_pretrained.assert_called_once_with(device="cpu")
        assert stage.model is not None
        assert stage.reference_wavs_list is not None
        stage.teardown()
        assert stage.model is None

    @patch("chatterbox.mtl_tts.ChatterboxMultilingualTTS")
    def test_setup_loads_multilingual_model(
        self, mock_cls: MagicMock, output_dir: str, ref_dataset: str
    ) -> None:
        mock_cls.from_pretrained.return_value = _fake_model()
        stage = _build_stage(output_dir, ref_dataset, language="fr")
        stage.setup()
        mock_cls.from_pretrained.assert_called_once_with(device="cpu")
        assert stage.language == "fr"

    def test_multilingual_load_restores_global_state(
        self, output_dir: str, ref_dataset: str
    ) -> None:
        import chatterbox.models.t3.llama_configs as llama_cfgs

        env_before = os.environ.get("TRANSFORMERS_ATTN_IMPLEMENTATION")
        cfgs_before = {
            name: cfg.get("attn_implementation")
            for name, cfg in llama_cfgs.LLAMA_CONFIGS.items()
        }

        stage = _build_stage(output_dir, ref_dataset, language="fr")
        with patch("chatterbox.mtl_tts.ChatterboxMultilingualTTS") as mock_cls:
            mock_cls.from_pretrained.return_value = _fake_model()
            stage.setup()

        assert os.environ["TRANSFORMERS_ATTN_IMPLEMENTATION"] == "eager"
        assert all(
            cfg.get("attn_implementation") == "eager"
            for cfg in llama_cfgs.LLAMA_CONFIGS.values()
        )

        stage.teardown()

        assert os.environ.get("TRANSFORMERS_ATTN_IMPLEMENTATION") == env_before
        cfgs_after = {
            name: cfg.get("attn_implementation")
            for name, cfg in llama_cfgs.LLAMA_CONFIGS.items()
        }
        assert cfgs_after == cfgs_before

    def test_setup_on_node_pre_downloads_english_model(
        self, output_dir: str, ref_dataset: str, tmp_path: Path
    ) -> None:
        cache_dir = str(tmp_path / "hf-cache")
        stage = _build_stage(output_dir, ref_dataset, cache_dir=cache_dir)
        with patch(f"{MODULE}.hf_hub_download") as mock_download:
            stage.setup_on_node()
        assert mock_download.call_count == len(_ENGLISH_MODEL_FILES)
        mock_download.assert_any_call(
            repo_id="ResembleAI/chatterbox",
            filename="ve.safetensors",
            cache_dir=cache_dir,
        )

    def test_setup_on_node_pre_downloads_multilingual_model(
        self, output_dir: str, ref_dataset: str, tmp_path: Path
    ) -> None:
        cache_dir = str(tmp_path / "hf-cache")
        stage = _build_stage(output_dir, ref_dataset, language="fr", cache_dir=cache_dir)
        with patch(f"{MODULE}.snapshot_download") as mock_download:
            stage.setup_on_node()
        mock_download.assert_called_once_with(
            repo_id="ResembleAI/chatterbox",
            repo_type="model",
            revision="main",
            allow_patterns=[
                "ve.pt",
                "t3_23lang.safetensors",
                "s3gen.pt",
                "mtl_tokenizer.json",
                "conds.pt",
                "Cangjie5_TC.json",
            ],
            cache_dir=cache_dir,
            token=os.getenv("HF_TOKEN"),
        )

    def test_stage_contract(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset)
        assert stage.inputs() == ([], ["conversation_id", "speaker", "utterance"])
        assert stage.outputs() == ([], ["audio_filepath", "duration", "reference_voice"])
        with pytest.raises(NotImplementedError):
            stage.process(_make_task())
        with pytest.raises(ValueError, match="Unsupported language"):
            _build_stage(output_dir, ref_dataset, language="xx")

    def test_init_repetition_penalty_defaults(self, output_dir: str, ref_dataset: str) -> None:
        english = _build_stage(output_dir, ref_dataset)
        multilingual = _build_stage(output_dir, ref_dataset, language="fr")
        custom = _build_stage(output_dir, ref_dataset, language="fr", repetition_penalty=1.5)
        assert english.repetition_penalty == 1.2
        assert multilingual.repetition_penalty == 2.0
        assert custom.repetition_penalty == 1.5

    def test_init_fixed_exaggeration_has_no_range(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset, exaggeration=0.7)
        assert stage.exaggeration_range is None
        assert stage._get_exaggeration("any-conv") == 0.7

    def test_process_batch_empty(self, output_dir: str, ref_dataset: str) -> None:
        stage = _build_stage(output_dir, ref_dataset)
        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            stage.setup()
        assert stage.process_batch([]) == []

    def test_process_batch_single_entry(self, output_dir: str, ref_dataset: str) -> None:
        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            stage = _build_stage(output_dir, ref_dataset)
            stage.setup()

            task = _make_task("Hello world", "Alice", "conv001")
            results = stage.process_batch([task])

        assert len(results) == 1
        out = results[0].data
        assert out["speaker"] == "Alice"
        assert out["conversation_id"] == "conv001"
        assert "audio_filepath" in out
        assert "duration" in out
        assert "reference_voice" in out
        assert out["duration"] > 0
        assert os.path.exists(out["audio_filepath"])

    def test_process_batch_multi_turn(self, output_dir: str, ref_dataset: str) -> None:
        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            stage = _build_stage(output_dir, ref_dataset)
            stage.setup()

            tasks = [
                _make_task("Hello Bob", "Alice", "conv001", task_id="t1"),
                _make_task("Hi Alice", "Bob", "conv001", task_id="t2"),
                _make_task("How are you?", "Alice", "conv001", task_id="t3"),
            ]
            results = stage.process_batch(tasks)

        assert len(results) == 3
        assert results[0].data["reference_voice"] == results[2].data["reference_voice"]
        assert results[1].data["reference_voice"] != results[0].data["reference_voice"]
        for r in results:
            assert os.path.exists(r.data["audio_filepath"])

    def test_process_batch_skips_empty_text(self, output_dir: str, ref_dataset: str) -> None:
        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            stage = _build_stage(output_dir, ref_dataset)
            stage.setup()

            tasks = [
                _make_task("Valid", "Alice", "conv001", task_id="t1"),
                _make_task("", "Bob", "conv001", task_id="t2"),
                _make_task("   ", "Charlie", "conv001", task_id="t3"),
            ]
            results = stage.process_batch(tasks)

        assert "audio_filepath" in results[0].data
        assert "audio_filepath" not in results[1].data
        assert "audio_filepath" not in results[2].data

    def test_process_batch_text_field_fallback(self, output_dir: str, ref_dataset: str) -> None:
        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            stage = _build_stage(output_dir, ref_dataset)
            stage.setup()

            task = AudioTask(
                data={"text": "Fallback text", "speaker": "Bob", "conversation_id": "c1"},
                task_id="t1",
                dataset_name="test",
            )
            results = stage.process_batch([task])

        assert "audio_filepath" in results[0].data

    def test_process_batch_preserves_task_metadata_and_fields(
        self, output_dir: str, ref_dataset: str
    ) -> None:
        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            stage = _build_stage(output_dir, ref_dataset)
            stage.setup()

            task = _make_task(
                "Test",
                task_id="my_id",
                overlap=0.3,
                topic="weather",
            )
            task.dataset_name = "my_dataset"
            task._metadata["source"] = "manifest_v1"
            task.add_stage_perf(StagePerfStats(stage_name="upstream_stage", process_time=1.5))
            results = stage.process_batch([task])

        result = results[0]
        # The framework mutates and returns the same input task rather than
        # constructing a new one, so provenance fields set by earlier stages
        # (perf history, metadata) survive rather than resetting to empty.
        assert result is task
        assert result.task_id == "my_id"
        assert result.dataset_name == "my_dataset"
        assert result.data["overlap"] == 0.3
        assert result.data["topic"] == "weather"
        assert result._metadata == {"source": "manifest_v1"}
        assert len(result._stage_perf) == 1
        assert result._stage_perf[0].stage_name == "upstream_stage"

    def test_process_batch_idempotent(self, output_dir: str, ref_dataset: str) -> None:
        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            stage = _build_stage(output_dir, ref_dataset)
            stage.setup()

            task = _make_task("Idempotent test", "Alice", "conv001")
            path1 = stage.process_batch([task])[0].data["audio_filepath"]
            calls_before = stage.model.generate.call_count
            path2 = stage.process_batch([task])[0].data["audio_filepath"]

        assert path1 == path2
        assert stage.model.generate.call_count == calls_before

    def test_process_batch_multilingual(
        self, output_dir: str, ref_dataset_mls: str
    ) -> None:
        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            stage = _build_stage(output_dir, ref_dataset_mls, language="es")
            stage.setup()

            task = _make_task("Hola mundo")
            stage.process_batch([task])

        assert stage.model.generate.call_args.kwargs["language_id"] == "es"

    def test_process_audio_with_rttm_returns_original_when_unusable(
        self, output_dir: str, ref_dataset: str, tmp_path: Path
    ) -> None:
        stage = _build_stage(output_dir, ref_dataset)
        stage._init_temp_dir()
        wav_path = os.path.join(ref_dataset, "wavs", "dialog001", "spk_A.wav")

        # No RTTM file at all (e.g. dialog has no diarization).
        missing_rttm = os.path.join(ref_dataset, "rttms", "dialog001", "does_not_exist.rttm")
        assert stage._process_audio_with_rttm(wav_path, missing_rttm) == wav_path

        # RTTM exists but has no parseable SPEAKER lines.
        malformed_rttm = tmp_path / "malformed.rttm"
        malformed_rttm.write_text("NOT_SPEAKER dialog001 1 0.0 2.0\ntoo short\n")
        assert stage._process_audio_with_rttm(wav_path, str(malformed_rttm)) == wav_path

        # A valid RTTM but audio loading itself fails.
        rttm_path = os.path.join(ref_dataset, "rttms", "dialog001", "spk_A.rttm")
        with patch(f"{MODULE}.ta.load", side_effect=RuntimeError("corrupt audio")):
            assert stage._process_audio_with_rttm(wav_path, rttm_path) == wav_path

    def test_process_audio_with_rttm_truncates_to_max_duration(
        self, output_dir: str, ref_dataset: str
    ) -> None:
        stage = _build_stage(output_dir, ref_dataset, max_reference_duration=1.0)
        stage._init_temp_dir()
        wav_path = os.path.join(ref_dataset, "wavs", "dialog001", "spk_A.wav")
        rttm_path = os.path.join(ref_dataset, "rttms", "dialog001", "spk_A.rttm")

        out_path = stage._process_audio_with_rttm(wav_path, rttm_path)

        assert out_path != wav_path
        audio, sr = sf.read(out_path)
        assert len(audio) / sr == pytest.approx(1.0, abs=0.05)

    def test_get_reference_audio_mls_falls_back_on_errors(
        self, output_dir: str, ref_dataset_mls: str
    ) -> None:
        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            stage = _build_stage(output_dir, ref_dataset_mls, language="es")
            stage.setup()

        # Every segment fails to load.
        with patch(f"{MODULE}.ta.load", side_effect=RuntimeError("bad file")):
            out_path, chosen = stage._get_reference_audio_mls("some_key")
        assert out_path in stage._speaker_audio_map[chosen]

        # Segments load fine but writing the concatenated reference fails.
        with patch(f"{MODULE}.ta.save", side_effect=OSError("disk full")):
            out_path, chosen = stage._get_reference_audio_mls("some_key")
        assert out_path in stage._speaker_audio_map[chosen]

    def test_get_reference_audio_mls_truncates_to_max_duration(
        self, output_dir: str, ref_dataset_mls: str
    ) -> None:
        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            stage = _build_stage(output_dir, ref_dataset_mls, language="es", max_reference_duration=3.0)
            stage.setup()
            out_path, _chosen = stage._get_reference_audio_mls("some_key")

        audio, sr = sf.read(out_path)
        assert len(audio) / sr == pytest.approx(3.0, abs=0.05)

    def test_get_reference_audio_mls_resamples_mixed_rate_segments(
        self, output_dir: str, tmp_path: Path
    ) -> None:
        mls_root = tmp_path / "mls_mixed_rate"
        book_dir = mls_root / "1234" / "book01"
        book_dir.mkdir(parents=True)
        rng = np.random.default_rng()
        sf.write(str(book_dir / "1234_book01_0000.flac"), rng.standard_normal(8000 * 2).astype(np.float32), 8000)
        sf.write(str(book_dir / "1234_book01_0001.flac"), rng.standard_normal(16000 * 2).astype(np.float32), 16000)

        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            stage = _build_stage(output_dir, str(mls_root), language="es")
            stage.setup()
            out_path, chosen = stage._get_reference_audio_mls("some_key")

        assert chosen == "1234"
        audio, sr = sf.read(out_path)
        # Both segments are 2s; if concatenated without resampling to a common
        # rate, one segment's raw sample count is played back under the
        # other's rate, corrupting the total duration.
        assert len(audio) / sr == pytest.approx(4.0, rel=0.05)

    def test_process_batch_mls_reference_layout(
        self, output_dir: str, ref_dataset_mls: str
    ) -> None:
        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            stage = _build_stage(output_dir, ref_dataset_mls, language="ru")
            stage.setup()

            tasks = [
                _make_task("Привет", "Alice", "conv001", task_id="t1"),
                _make_task("Пока", "Bob", "conv001", task_id="t2"),
            ]
            results = stage.process_batch(tasks)

        assert len(results) == 2
        assert results[0].data["reference_voice"] != results[1].data["reference_voice"]

    def test_process_batch_generation_failure_produces_silence(
        self, output_dir: str, ref_dataset: str
    ) -> None:
        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            stage = _build_stage(output_dir, ref_dataset)
            stage.setup()
            stage.model.generate.side_effect = RuntimeError("GPU OOM")

            result = stage.process_batch([_make_task("This will fail")])[0]

        audio, _sr = sf.read(result.data["audio_filepath"])
        assert np.allclose(audio, 0.0)

    def test_normalize_audio_returns_silence_unchanged(
        self, output_dir: str, ref_dataset: str
    ) -> None:
        stage = _build_stage(output_dir, ref_dataset)
        silence = torch.zeros(1, 1000)
        assert torch.equal(stage._normalize_audio(silence), silence)

    def test_normalize_audio_scales_rms_toward_target_level(
        self, output_dir: str, ref_dataset: str
    ) -> None:
        stage = _build_stage(output_dir, ref_dataset, normalize_level=-20.0)
        wav = 0.01 * torch.sin(torch.linspace(0, 100, 16000)).unsqueeze(0)

        normalized = stage._normalize_audio(wav)

        rms_db = 20 * torch.log10(torch.sqrt(torch.mean(normalized**2)) + 1e-8)
        assert rms_db.item() == pytest.approx(-20.0, abs=0.5)

    def test_normalize_audio_clips_peaks_below_one(
        self, output_dir: str, ref_dataset: str
    ) -> None:
        stage = _build_stage(output_dir, ref_dataset, normalize_level=0.0)
        wav = 0.001 * torch.sin(torch.linspace(0, 100, 16000)).unsqueeze(0)

        normalized = stage._normalize_audio(wav)

        assert torch.max(torch.abs(normalized)).item() <= 0.99 + 1e-6

    def test_generate_turn_audio_respects_normalize_audio_flag(
        self, output_dir: str, ref_dataset: str
    ) -> None:
        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            enabled = _build_stage(output_dir, ref_dataset, normalize_audio=True)
            enabled.setup()
            with patch.object(enabled, "_normalize_audio", wraps=enabled._normalize_audio) as mock_norm:
                enabled._generate_turn_audio("Hello", "ref.wav", "conv001")
            mock_norm.assert_called_once()

            disabled = _build_stage(output_dir, ref_dataset, normalize_audio=False)
            disabled.setup()
            with patch.object(disabled, "_normalize_audio") as mock_norm:
                disabled._generate_turn_audio("Hello", "ref.wav", "conv001")
            mock_norm.assert_not_called()

    def test_process_batch_resamples_to_configured_sample_rate(
        self, output_dir: str, ref_dataset: str
    ) -> None:
        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            stage = _build_stage(output_dir, ref_dataset, sample_rate=16000)
            stage.setup()
            assert stage.model.sr == 24000  # model's native rate, unaffected by config

            result = stage.process_batch([_make_task("Hello world", "Alice", "conv001")])[0]
            audio_path = result.data["audio_filepath"]
            audio, file_sr = sf.read(audio_path)

            assert file_sr == 16000
            assert result.data["duration"] == pytest.approx(len(audio) / 16000)

            duration_before = result.data["duration"]
            cache_hit = stage.process_batch([_make_task("Hello world", "Alice", "conv001")])[0]

        assert cache_hit.data["duration"] == pytest.approx(duration_before)

    def test_output_filename_differs_for_each_generation_input(
        self, output_dir: str, ref_dataset: str
    ) -> None:
        stage = _build_stage(output_dir, ref_dataset)
        base = {
            "conversation_id": "conv001",
            "speaker": "Alice",
            "text": "Hello",
            "ref_id": "dialog001/spk_A",
            "ref_content_hash": "hash1",
            "exaggeration": 0.5,
        }
        baseline = ChatterboxTTSStage._output_filename(stage._cache_manifest(**base))

        for override in (
            {"ref_id": "dialog001/spk_B"},
            {"ref_content_hash": "hash2"},
            {"exaggeration": 0.9},
        ):
            filename = ChatterboxTTSStage._output_filename(stage._cache_manifest(**{**base, **override}))
            assert filename != baseline

        # This is the reported repro: language lives on the stage (not the
        # call args) and picks a different model, but it must still change
        # the filename so an English and a French run of the same turn
        # don't collide on the same cached audio.
        stage_fr = _build_stage(output_dir, ref_dataset, language="fr")
        filename_fr = ChatterboxTTSStage._output_filename(stage_fr._cache_manifest(**base))
        assert filename_fr != baseline

    def test_process_batch_regenerates_on_invalid_sidecar(
        self, output_dir: str, ref_dataset: str
    ) -> None:
        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            stage = _build_stage(output_dir, ref_dataset)
            stage.setup()

            task = _make_task("Hello world", "Alice", "conv001")
            audio_path = stage.process_batch([task])[0].data["audio_filepath"]
            sidecar_path = ChatterboxTTSStage._sidecar_path(audio_path)

            # Missing sidecar (e.g. a legacy cache entry) is not trusted.
            os.remove(sidecar_path)
            calls_before = stage.model.generate.call_count
            stage.process_batch([task])
            assert stage.model.generate.call_count == calls_before + 1

            # Sidecar that no longer matches the current config is not trusted either.
            with open(sidecar_path, "w", encoding="utf-8") as f:
                json.dump({"stale": True}, f)
            calls_before = stage.model.generate.call_count
            stage.process_batch([task])
            assert stage.model.generate.call_count == calls_before + 1

            # Sidecar with invalid JSON syntax (not just wrong content).
            with open(sidecar_path, "w", encoding="utf-8") as f:
                f.write("{not valid json::")
            calls_before = stage.model.generate.call_count
            stage.process_batch([task])
            assert stage.model.generate.call_count == calls_before + 1

            # Cached WAV bytes are corrupt even though its sidecar matches.
            audio_path = stage.process_batch([task])[0].data["audio_filepath"]
            with open(audio_path, "wb") as f:
                f.write(b"not a real wav file")
            calls_before = stage.model.generate.call_count
            result = stage.process_batch([task])[0]
            assert stage.model.generate.call_count == calls_before + 1
            assert os.path.exists(result.data["audio_filepath"])

    def test_process_batch_io_error_preserves_original_task(
        self, output_dir: str, ref_dataset: str
    ) -> None:
        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            stage = _build_stage(output_dir, ref_dataset)
            stage.setup()
            task = _make_task("Hello world", "Alice", "conv001")

            with patch.object(stage, "_publish_cache_entry", side_effect=OSError("disk full")):
                results = stage.process_batch([task])

        assert results == [task]
        assert "audio_filepath" not in results[0].data

    def test_hash_file_content_reflects_bytes(self, tmp_path: Path) -> None:
        path_a = tmp_path / "a.bin"
        path_b = tmp_path / "b.bin"
        path_a.write_bytes(b"hello")
        path_b.write_bytes(b"world")

        hash_a = ChatterboxTTSStage._hash_file_content(str(path_a))
        hash_b = ChatterboxTTSStage._hash_file_content(str(path_b))
        assert hash_a != hash_b

        path_b.write_bytes(b"hello")
        assert ChatterboxTTSStage._hash_file_content(str(path_b)) == hash_a

    def test_reference_content_hash_defaults_to_empty_string(
        self, output_dir: str, ref_dataset: str
    ) -> None:
        stage = _build_stage(output_dir, ref_dataset)
        assert stage._reference_content_hash("Alice", "conv001") == ""

    def test_process_batch_regenerates_when_reference_audio_content_changes(
        self, output_dir: str, ref_dataset: str
    ) -> None:
        # Same speaker/conversation deterministically picks the same reference
        # file on both stages; only the file's *content* hash differs (as if
        # the underlying reference audio were edited between runs), which
        # must still invalidate the cache even though the path-derived
        # reference_id is unchanged.
        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            stage1 = _build_stage(output_dir, ref_dataset)
            stage1.setup()
            with patch.object(stage1, "_hash_file_content", return_value="hash_v1"):
                path1 = stage1.process_batch([_make_task("Provenance test", "Alice", "conv001")])[0].data[
                    "audio_filepath"
                ]
            stage1.teardown()

            stage2 = _build_stage(output_dir, ref_dataset)
            stage2.setup()
            with patch.object(stage2, "_hash_file_content", return_value="hash_v2"):
                path2 = stage2.process_batch([_make_task("Provenance test", "Alice", "conv001")])[0].data[
                    "audio_filepath"
                ]

        assert path1 != path2
        assert stage2.model.generate.call_count == 1

    def test_process_batch_writes_matching_sidecar_and_no_leftover_temp_files(
        self, output_dir: str, ref_dataset: str
    ) -> None:
        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            stage = _build_stage(output_dir, ref_dataset)
            stage.setup()
            result = stage.process_batch([_make_task("Hello world", "Alice", "conv001")])[0]

        audio_path = result.data["audio_filepath"]
        sidecar_path = ChatterboxTTSStage._sidecar_path(audio_path)
        with open(sidecar_path, encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["reference_id"] == result.data["reference_voice"]
        assert manifest["language"] is None

        for name in os.listdir(os.path.dirname(audio_path)):
            assert name.endswith((".wav", ".json"))

    def test_process_batch_different_reference_voice_uses_separate_cache(
        self, output_dir: str, ref_dataset: str
    ) -> None:
        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            stage1 = _build_stage(output_dir, ref_dataset)
            stage1.setup()
            with patch.object(
                stage1, "_assign_reference", return_value=("/a.wav", "dialog001/spk_A")
            ):
                result1 = stage1.process_batch([_make_task("Same text", "Alice", "conv001")])[0]
            stage1.teardown()

            stage2 = _build_stage(output_dir, ref_dataset)
            stage2.setup()
            with patch.object(
                stage2, "_assign_reference", return_value=("/b.wav", "dialog001/spk_B")
            ):
                result2 = stage2.process_batch([_make_task("Same text", "Alice", "conv001")])[0]

        assert result1.data["reference_voice"] == "dialog001/spk_A"
        assert result2.data["reference_voice"] == "dialog001/spk_B"
        assert result1.data["audio_filepath"] != result2.data["audio_filepath"]
        assert stage2.model.generate.call_count == 1

    def test_stable_index_is_deterministic(self) -> None:
        idx_a = ChatterboxTTSStage._stable_index("conv001::Alice", 7)
        idx_b = ChatterboxTTSStage._stable_index("conv001::Alice", 7)
        assert idx_a == idx_b
        assert 0 <= idx_a < 7

        seen = {ChatterboxTTSStage._stable_index(f"conv001::speaker{i}", 100) for i in range(50)}
        assert len(seen) > 25

    def test_stable_unit_interval_is_deterministic(self) -> None:
        val_a = ChatterboxTTSStage._stable_unit_interval("exaggeration::conv001")
        val_b = ChatterboxTTSStage._stable_unit_interval("exaggeration::conv001")
        assert val_a == val_b
        assert 0.0 <= val_a < 1.0

    def test_assign_reference_matches_across_independent_actor_instances(
        self, output_dir: str, ref_dataset: str
    ) -> None:
        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            actor_a = _build_stage(output_dir, ref_dataset)
            actor_a.setup()
            actor_b = _build_stage(output_dir, ref_dataset)
            actor_b.setup()

        _, ref_id_a = actor_a._assign_reference("Alice", "conv001")
        _, ref_id_b = actor_b._assign_reference("Alice", "conv001")
        assert ref_id_a == ref_id_b

    def test_assign_reference_independent_of_call_order(
        self, output_dir: str, ref_dataset: str
    ) -> None:
        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            actor_a = _build_stage(output_dir, ref_dataset)
            actor_a.setup()
            actor_b = _build_stage(output_dir, ref_dataset)
            actor_b.setup()

        alice_on_a = actor_a._assign_reference("Alice", "conv001")[1]
        bob_on_a = actor_a._assign_reference("Bob", "conv001")[1]
        bob_on_b = actor_b._assign_reference("Bob", "conv001")[1]
        alice_on_b = actor_b._assign_reference("Alice", "conv001")[1]

        assert alice_on_a == alice_on_b
        assert bob_on_a == bob_on_b

    def test_assign_reference_matches_across_actors_mls_layout(
        self, output_dir: str, ref_dataset_mls: str
    ) -> None:
        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            actor_a = _build_stage(output_dir, ref_dataset_mls, language="ru")
            actor_a.setup()
            actor_b = _build_stage(output_dir, ref_dataset_mls, language="ru")
            actor_b.setup()

        assert actor_a._assign_reference("Alice", "conv001")[1] == actor_b._assign_reference(
            "Alice", "conv001"
        )[1]

    def test_get_exaggeration_matches_across_independent_actor_instances(
        self, output_dir: str, ref_dataset: str
    ) -> None:
        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            actor_a = _build_stage(output_dir, ref_dataset, exaggeration=[0.3, 0.9])
            actor_a.setup()
            actor_b = _build_stage(output_dir, ref_dataset, exaggeration=[0.3, 0.9])
            actor_b.setup()

        exag_a = actor_a._get_exaggeration("conv001")
        exag_b = actor_b._get_exaggeration("conv001")
        assert exag_a == exag_b
        assert 0.3 <= exag_a <= 0.9

    def test_process_batch_preserves_voice_across_actors_out_of_order(
        self, output_dir: str, ref_dataset: str
    ) -> None:
        turns = [
            ("Hi Bob", "Alice", "t1"),
            ("Hi Alice", "Bob", "t2"),
            ("How are you?", "Alice", "t3"),
            ("Good, thanks", "Bob", "t4"),
        ]

        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            actor_a = _build_stage(output_dir, ref_dataset)
            actor_a.setup()
            actor_b = _build_stage(output_dir, ref_dataset)
            actor_b.setup()

            tasks_a = [_make_task(text, spk, "conv001", task_id=tid) for text, spk, tid in [turns[0], turns[2]]]
            tasks_b = [_make_task(text, spk, "conv001", task_id=tid) for text, spk, tid in [turns[3], turns[1]]]

            results_a = actor_a.process_batch(tasks_a)
            results_b = actor_b.process_batch(tasks_b)

        alice_voices = {r.data["reference_voice"] for r in results_a}
        bob_voices = {r.data["reference_voice"] for r in results_b}
        assert len(alice_voices) == 1
        assert len(bob_voices) == 1
        assert alice_voices != bob_voices

        with patch.object(ChatterboxTTSStage, "_load_model", _inject_model):
            actor_c = _build_stage(output_dir, ref_dataset)
            actor_c.setup()
        assert actor_c._assign_reference("Alice", "conv001")[1] == next(iter(alice_voices))
        assert actor_c._assign_reference("Bob", "conv001")[1] == next(iter(bob_voices))
