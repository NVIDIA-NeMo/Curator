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

import io
import tarfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from nemo_curator.stages.audio.tts.audio_io import (
    SegmentAudioError,
    TarAudioConfig,
    iter_segments,
    load_segment_audio,
    resolve_tar_path,
    sample_id,
    segment_text,
)
from nemo_curator.stages.audio.tts.speaker_embedding import TitaNetEmbedder, l2_normalize


def _wav_bytes(seconds: float, sample_rate: int, freq: float = 440.0) -> bytes:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    buf = io.BytesIO()
    sf.write(buf, (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sample_rate, format="WAV")
    return buf.getvalue()


def make_tar_segments(tmp_path: Path, specs: list[tuple[str, float, int]]) -> list[dict]:
    """Write a tar with one WAV member per spec and return Granary segments addressing them."""
    tar_path = tmp_path / "shard_0.tar"
    with tarfile.open(tar_path, "w") as tar:
        for name, seconds, sample_rate in specs:
            payload = _wav_bytes(seconds, sample_rate)
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    segments = []
    with tarfile.open(tar_path) as tar:
        for member, (_, seconds, _) in zip(tar.getmembers(), specs, strict=True):
            segments.append(
                {
                    "duration": seconds,
                    "GranaryV2": {"pnc_text": f"Text of {member.name}.", "tn_raw": f"text of {member.name}"},
                    "GranaryHifi": {
                        "AudioMetadata": {
                            "tar_key": str(tar_path),
                            "tar_member": member.name,
                            "tar_member_offset": member.offset_data,
                            "tar_member_bytes": member.size,
                        }
                    },
                }
            )
    return segments


def test_load_tar_member_native_and_resampled(tmp_path: Path) -> None:
    segment = make_tar_segments(tmp_path, [("a.wav", 1.0, 22050)])[0]
    audio, sr = load_segment_audio(segment)
    assert sr == 22050
    assert audio.dtype == np.float32
    assert audio.shape == (22050,)
    audio16, sr16 = load_segment_audio(segment, 16000)
    assert sr16 == 16000
    assert abs(audio16.shape[0] - 16000) <= 1


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda m: m.pop("tar_member"), "missing_tar_member"),
        (lambda m: m.pop("tar_key"), "missing_tar_key"),
        (lambda m: m.update(tar_member_offset=None), "missing_tar_offsets"),
        (lambda m: m.update(tar_key="/nonexistent/x.tar"), "missing_audio_tar"),
        (lambda m: m.update(tar_member_offset=0), "audio_load_failed"),
    ],
)
def test_error_ladder(tmp_path: Path, mutate: object, code: str) -> None:
    segment = make_tar_segments(tmp_path, [("a.wav", 0.5, 16000)])[0]
    mutate(segment["GranaryHifi"]["AudioMetadata"])
    with pytest.raises(SegmentAudioError) as exc_info:
        load_segment_audio(segment)
    assert exc_info.value.code == code


def test_load_audio_filepath_with_offset(tmp_path: Path) -> None:
    path = tmp_path / "x.wav"
    path.write_bytes(_wav_bytes(2.0, 16000))
    audio, sr = load_segment_audio({"audio_filepath": str(path), "offset": 0.5, "duration": 1.0})
    assert sr == 16000
    assert audio.shape == (16000,)


def test_resolve_tar_path_prefix_stripping() -> None:
    cfg = TarAudioConfig(tar_root="/data/ytc", strip_tar_key_prefixes=("bucket/prefix",))
    assert resolve_tar_path("bucket/prefix/it/shard_0.tar", cfg) == Path("/data/ytc/it/shard_0.tar")
    assert resolve_tar_path("other/shard.tar", cfg) == Path("/data/ytc/other/shard.tar")
    assert resolve_tar_path("/abs/shard.tar", cfg) == Path("/abs/shard.tar")
    assert resolve_tar_path(None, cfg) is None


def test_row_helpers() -> None:
    row = {"orig_sample_id": "vid", "segments": [{"a": 1}, "junk", {"GranaryV2": {"tn_raw": " "}, "text": "t"}]}
    assert [i for i, _ in iter_segments(row)] == [0, 2]
    assert list(iter_segments({"x": 1})) == [(0, {"x": 1})]
    assert segment_text(row["segments"][2], "GranaryV2.tn_raw", ("text",)) == "t"
    assert sample_id(row) == "vid"
    assert sample_id({}, 3) == "sample_00000003"


class _FakeTitaNet:
    def eval(self) -> None:
        pass

    def forward(self, input_signal: object, input_signal_length: object) -> tuple[None, object]:
        import torch

        # embedding = [mean, length] so different inputs give different vectors
        return None, torch.stack([input_signal.mean(dim=1), input_signal_length.float()], dim=1)


def test_embedder_batches_and_reports_errors(tmp_path: Path) -> None:
    segments = make_tar_segments(tmp_path, [("a.wav", 0.5, 16000), ("b.wav", 1.0, 16000), ("c.wav", 0.25, 16000)])
    segments[1]["GranaryHifi"]["AudioMetadata"]["tar_member_offset"] = None
    embedder = TitaNetEmbedder(model=_FakeTitaNet(), batch_size=1, device="cpu")
    embedder.load()
    embeddings, errors = embedder.embed_segments(segments)
    assert errors == [None, "missing_tar_offsets", None]
    assert embeddings[1] is None
    assert embeddings[0][1] == 8000
    assert embeddings[2][1] == 4000
    normed = l2_normalize(np.stack([embeddings[0], embeddings[2]]))
    np.testing.assert_allclose(np.linalg.norm(normed, axis=1), 1.0, rtol=1e-6)


def test_truncated_tar_is_a_load_failure(tmp_path: Path) -> None:
    segment = make_tar_segments(tmp_path, [("a.wav", 0.5, 16000)])[0]
    meta = segment["GranaryHifi"]["AudioMetadata"]
    meta["tar_member_bytes"] = Path(meta["tar_key"]).stat().st_size  # reads past EOF
    with pytest.raises(SegmentAudioError) as exc_info:
        load_segment_audio(segment)
    assert exc_info.value.code == "audio_load_failed"
    assert "short read" in str(exc_info.value)
