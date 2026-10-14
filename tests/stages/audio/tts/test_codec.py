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

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from nemo_curator.stages.audio.tts.codec import (
    CodecCodesStage,
    CodecEncoder,
    codes_complete,
    codes_marker_path,
    codes_paths,
    extract_codes_for_registry,
    padded_length,
    resample_poly,
    shard_codes_complete,
)
from nemo_curator.stages.audio.tts.shar import build_shars_for_manifest, load_registry, read_cut_ids
from nemo_curator.tasks import FileGroupTask
from tests.stages.audio.tts.test_shar import make_manifest

REAL_CODEC = Path(os.environ.get("GRANARY_CODEC_PATH", "/nonexistent/25fps_spectral_codec.nemo"))


class FakeCodec(torch.nn.Module):
    """22.05 kHz codec stand-in: 882 samples/frame (25 fps), 3 codebooks, code = frame index."""

    sample_rate = 22050
    samples_per_frame = 882

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[int, ...]] = []

    def audio_encoder(self, audio: torch.Tensor, audio_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert int(audio.shape[-1]) % self.samples_per_frame == 0
        self.calls.append(tuple(audio.shape))
        frames = audio_len // self.samples_per_frame
        encoded = torch.zeros(audio.shape[0], 4, int(frames.max()))
        return encoded, frames

    def quantize(self, encoded: torch.Tensor, encoded_len: torch.Tensor) -> torch.Tensor:
        del encoded_len
        t = torch.arange(encoded.shape[-1]).expand(encoded.shape[0], 3, -1)
        return t.clone()


def test_padded_length() -> None:
    assert padded_length(1, 882) == 882
    assert padded_length(882, 882) == 882
    assert padded_length(883, 882) == 1764
    assert padded_length(5, 0) == 5


def test_resample_poly_matches_scipy_and_is_torchaudio_independent() -> None:
    from scipy.signal import resample_poly as scipy_resample_poly

    audio = np.random.default_rng(0).standard_normal((1, 44100)).astype(np.float32)
    out = resample_poly(audio, 44100, 16000)
    assert out.dtype == np.float32
    assert out.shape == (1, 16000)
    np.testing.assert_allclose(out, scipy_resample_poly(audio, up=160, down=441, axis=-1), rtol=1e-6)
    assert resample_poly(audio, 16000, 16000) is not None


def test_encoder_trims_to_encoded_length() -> None:
    codec = FakeCodec()
    codes = CodecEncoder(codec, device="cpu").encode([np.zeros(882 * 3, np.float32), np.zeros(100, np.float32)])
    assert [c.shape for c in codes] == [(3, 3), (3, 1)]
    assert codes[0].dtype == np.uint16
    assert codec.calls == [(2, 882 * 3)]


def build_registry(tmp_path: Path) -> Path:
    manifest = make_manifest(tmp_path, n_videos=2, n_segments=3)
    return build_shars_for_manifest(manifest, tmp_path / "shars", dataset="D", max_per_shard=4)


def test_extract_codes_writes_aligned_temporal_arrays(tmp_path: Path) -> None:
    from lhotse import CutSet

    registry_path = build_registry(tmp_path)
    counts = extract_codes_for_registry(registry_path, CodecEncoder(FakeCodec(), "cpu"), batch_size=2)
    assert counts == {"encoded_shards": 2, "reused_shards": 0, "cuts": 6}
    shard = load_registry(registry_path)["shards"][0]
    rate_dir = Path(shard["dir"])
    target, context = codes_paths(rate_dir, 0, "25fpsSpectralCodecBWE")
    ids = read_cut_ids(Path(shard["cuts"]))
    assert codes_complete(target, ids)
    assert codes_complete(context, ids)
    assert not list(rate_dir.rglob("*.tmp.*"))

    cuts = CutSet.from_shar(
        fields={
            "cuts": [shard["cuts"]],
            "target_codes": [str(target)],
            "context_codes": [str(context)],
        }
    )
    for cut in cuts:
        codes = cut.load_target_codes()
        assert codes.dtype == np.uint16
        assert codes.shape[0] == 3
        assert codes.shape[1] == -(-round(cut.duration * 22050) // 882)  # ceil(duration * 25)
        assert cut.target_codes.frame_shift == pytest.approx(0.04)
        assert cut.load_context_codes().shape[0] == 3

    again = extract_codes_for_registry(registry_path, CodecEncoder(FakeCodec(), "cpu"))
    assert again == {"encoded_shards": 0, "reused_shards": 2, "cuts": 0}


def test_codes_reuse_keyed_on_audio_digest(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path, n_videos=2, n_segments=3)
    out = tmp_path / "shars"
    registry_path = build_shars_for_manifest(manifest, out, dataset="D", max_per_shard=4)
    extract_codes_for_registry(registry_path, CodecEncoder(FakeCodec(), "cpu"))

    # annotation-only change: the shard is rebuilt but its audio bytes are identical -> codes are reused
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[0]["segments"][0]["tts_flags"]["valid_ipa"] = False
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    registry = load_registry(build_shars_for_manifest(manifest, out, dataset="D", max_per_shard=4))
    assert registry["counts"]["built_shards"] == 1
    assert extract_codes_for_registry(registry_path, CodecEncoder(FakeCodec(), "cpu"))["reused_shards"] == 2

    # audio change (different context pairing for one target) -> that shard's codes are re-encoded
    seg0, seg2 = rows[0]["segments"][0], rows[0]["segments"][2]
    meta = seg2["GranaryHifi"]["AudioMetadata"]
    seg0["context"].update(
        context_recording_id=meta["tar_member"],
        context_audio_duration=seg2["duration"],
        context_audio_offset=meta["tar_member_offset"],
        context_audio_bytes=meta["tar_member_bytes"],
    )
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    build_shars_for_manifest(manifest, out, dataset="D", max_per_shard=4)
    counts = extract_codes_for_registry(registry_path, CodecEncoder(FakeCodec(), "cpu"))
    assert (counts["encoded_shards"], counts["reused_shards"]) == (1, 1)

    # a codes tar without its marker is not trusted
    shard = load_registry(registry_path)["shards"][0]
    codes_marker_path(Path(shard["dir"]), 0, "25fpsSpectralCodecBWE").unlink()
    assert not shard_codes_complete(shard, "25fpsSpectralCodecBWE")


def test_stage_with_injected_model(tmp_path: Path) -> None:
    registry_path = build_registry(tmp_path)
    stage = CodecCodesStage(codec_model=FakeCodec(), device="cpu")
    stage.setup()
    assert stage.frame_rate == 25.0
    task = FileGroupTask(dataset_name="d", data=[str(registry_path)])
    assert stage.process(task).data == [str(registry_path)]
    with pytest.raises(ValueError, match="model_path"):
        CodecCodesStage()


def test_stage_loads_public_hf_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from nemo_curator.stages.audio.tts import codec

    loaded: list[str] = []

    def fake_from_pretrained(model_name: str) -> FakeCodec:
        loaded.append(model_name)
        return FakeCodec()

    monkeypatch.setattr(codec, "load_pretrained_codec_model", fake_from_pretrained)
    stage = CodecCodesStage(model_path=codec.PUBLIC_CODEC_MODEL, device="cpu")
    stage.setup()
    assert loaded == [codec.PUBLIC_CODEC_MODEL]
    assert stage.codec_name == codec.PUBLIC_CODEC_NAME
    assert stage.frame_rate == 25.0


def test_stage_rejects_conflicting_model_sources_and_frame_rate() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        CodecCodesStage(model_path="private.nemo", codec_model=FakeCodec())

    stage = CodecCodesStage(codec_model=FakeCodec(), frame_rate=21.5, device="cpu")
    with pytest.raises(ValueError, match="does not match codec frame rate"):
        stage.setup()


@pytest.mark.skipif(not REAL_CODEC.is_file(), reason="25 fps codec .nemo not available")
def test_real_codec_strict_false_ignores_only_discriminator() -> None:
    from nemo_curator.stages.audio.tts.codec import check_codec_keys, checkpoint_keys, restore_codec_model

    model = restore_codec_model(str(REAL_CODEC), check_keys=False)
    ignored = checkpoint_keys(str(REAL_CODEC)) - set(model.state_dict())
    assert ignored
    assert all(key.startswith("discriminator.") for key in ignored)
    check_codec_keys(model, str(REAL_CODEC))
    model.register_buffer("extra_buffer_not_in_checkpoint", torch.zeros(1))
    with pytest.raises(RuntimeError, match="missing"):
        check_codec_keys(model, str(REAL_CODEC))


@pytest.mark.gpu
@pytest.mark.skipif(not REAL_CODEC.is_file(), reason="25 fps codec .nemo not available")
def test_real_codec_shapes() -> None:
    from nemo_curator.stages.audio.tts.codec import restore_codec_model

    model = restore_codec_model(str(REAL_CODEC)).to("cuda").eval()
    assert model.sample_rate == 16000  # encoder input rate (bandwidth extension decodes to 22.05 kHz)
    assert model.samples_per_frame == 640  # 25 fps
    encoder = CodecEncoder(model, "cuda")
    one_second = np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.1
    codes = encoder.encode([one_second, one_second[:8000]])
    assert codes[0].dtype == np.uint16
    assert codes[0].shape[1] == 25
    assert codes[1].shape[1] == 13
    assert codes[0].shape[0] == codes[1].shape[0]
