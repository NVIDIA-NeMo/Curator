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

"""FFmpeg installer capability checks and CPU audio decoding regression."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

INSTALLER = Path(__file__).resolve().parents[3] / "docker/common/install_ffmpeg.sh"


@pytest.mark.parametrize(
    ("codec", "missing"),
    [("flac", "decoders"), ("flac", "demuxers"), ("pcm_f32le", "encoders"), ("pcm_f32le", "decoders"), (None, None)],
)
@pytest.mark.parametrize("check_only", [False, True])
def test_audio_capabilities_are_required_without_unnecessary_installation(
    tmp_path: Path, codec: str | None, missing: str | None, check_only: bool
) -> None:
    ffmpeg = tmp_path / "ffmpeg"
    ffmpeg.write_text(
        "#!/bin/bash\n"
        'case "$2" in\n'
        "  -encoders) echo 'rawvideo libvpx-vp9 h264_nvenc hevc_nvenc av1_nvenc libopenh264 pcm_s16le' ;;\n"
        "  -decoders) echo 'rawvideo libvpx-vp9 vp9 vp8 h264_cuvid hevc_cuvid av1_cuvid mpeg1video mpeg2video mpeg4 h264 hevc av1 pcm_s16le' ;;\n"
        "  -muxers) echo wav ;;\n"
        "  -demuxers) echo wav ;;\n"
        "  -filters) echo aresample ;;\n"
        "esac | tr ' ' '\\n' | sed 's/^/ . /'\n"
        'if [[ "$2" =~ ^-(decoders|demuxers)$ && ! ( "$CODEC" == flac && "$2" == "-${MISSING}" ) ]]; then echo " . flac"; fi\n'
        'if [[ "$2" =~ ^-(encoders|decoders)$ && ! ( "$CODEC" == pcm_f32le && "$2" == "-${MISSING}" ) ]]; then echo " . pcm_f32le"; fi\n'
    )
    ffmpeg.chmod(0o755)
    for name, body in {
        "ffprobe": "exit 0",
        "apt-get": 'echo "repair requested" >&2; exit 99',
        "id": "echo 0",
    }.items():
        command = tmp_path / name
        command.write_text(f"#!/bin/bash\n{body}\n")
        command.chmod(0o755)

    result = subprocess.run(  # noqa: S603
        ["/bin/bash", str(INSTALLER), *(["--check"] if check_only else [])],
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "MISSING": missing or "", "CODEC": codec or ""},
        capture_output=True,
        text=True,
        check=False,
    )

    assert (result.returncode == 0) is (missing is None)
    if missing:
        assert f"ffmpeg {missing[:-1]} not found: {codec}" in result.stderr
    assert ("repair requested" in result.stderr) is (missing is not None and not check_only)


@pytest.mark.parametrize(("codec", "subtype"), [("pcm_s16le", "PCM_16"), ("pcm_f32le", "FLOAT")])
def test_torchcodec_decodes_flac_and_resampled_pcm(tmp_path: Path, codec: str, subtype: str) -> None:
    # These optional audio dependencies are needed only for the real decoder test.
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    torch = pytest.importorskip("torch")
    decoders = pytest.importorskip("torchcodec.decoders")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("FFmpeg CLI is required for WAV/resampling verification")

    flac = tmp_path / "audio.flac"
    sf.write(flac, np.linspace(-0.5, 0.5, 1600), 16000, subtype="PCM_16")
    expected, sample_rate = sf.read(flac, dtype="float32", always_2d=True)
    decoded = decoders.AudioDecoder(str(flac)).get_all_samples()
    assert decoded.sample_rate == sample_rate
    torch.testing.assert_close(decoded.data, torch.from_numpy(expected.T), rtol=0, atol=1e-6)

    wav = tmp_path / "resampled.wav"
    subprocess.run(  # noqa: S603
        [ffmpeg, "-v", "error", "-i", str(flac), "-ar", "8000", "-c:a", codec, str(wav)],
        check=True,
    )
    resampled = decoders.AudioDecoder(str(wav)).get_all_samples()
    assert resampled.sample_rate == 8000
    assert resampled.data.shape == (1, 800)
    assert sf.info(wav).subtype == subtype
