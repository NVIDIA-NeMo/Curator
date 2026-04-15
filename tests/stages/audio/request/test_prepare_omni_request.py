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

"""Tests for media format encoding in PrepareOmniRequestStage.

Covers all modality combinations (audio, image, text) x both format modes
(data_url, input_data), verifying correct JSON structure for each API target.
"""

import io
import struct
from pathlib import Path

import pandas as pd
import pytest

from nemo_curator.stages.audio.request.prepare_omni_request import (
    PrepareOmniRequestStage,
    resolve_media_content_type,
)
from nemo_curator.tasks import DocumentBatch

# ---------------------------------------------------------------------------
# Fixtures: tiny audio and image files
# ---------------------------------------------------------------------------


def _make_wav_bytes(num_samples: int = 160, sample_rate: int = 16000) -> bytes:
    """Create a minimal valid WAV file (PCM16 mono) in memory."""
    data = struct.pack(f"<{num_samples}h", *([0] * num_samples))
    data_size = len(data)
    # RIFF header
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    # fmt chunk
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))  # chunk size
    buf.write(struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16))
    # data chunk
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(data)
    return buf.getvalue()


def _make_jpeg_bytes() -> bytes:
    """Create a minimal valid JPEG file (1x1 white pixel)."""
    # Minimal JFIF: SOI + APP0 + DQT + SOF0 + DHT + SOS + data + EOI
    # Easier: use a known minimal JPEG byte sequence
    return (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
        b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
        b'\x1f\x1e\x1d\x1a\x1c\x1c $.\x27 ",.+\x1c\x1c(7),01444\x1f\x27'
        b"9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
        b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00"
        b"\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08"
        b"\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03"
        b"\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12"
        b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00T\xdb\xae\x8e\xed\xb1"
        b"\xff\xd9"
    )


@pytest.fixture
def wav_file(tmp_path: Path) -> Path:
    p = tmp_path / "test.wav"
    p.write_bytes(_make_wav_bytes())
    return p


@pytest.fixture
def jpg_file(tmp_path: Path) -> Path:
    p = tmp_path / "test.jpg"
    p.write_bytes(_make_jpeg_bytes())
    return p


def _make_batch(rows: list[dict]) -> DocumentBatch:
    return DocumentBatch(data=pd.DataFrame(rows), dataset_name="test", task_id="test")


def _get_user_content(messages: list[dict]) -> list[dict]:
    """Extract user content parts from messages list."""
    for msg in messages:
        if msg["role"] == "user":
            return msg["content"] if isinstance(msg["content"], list) else []
    return []


# ---------------------------------------------------------------------------
# Tests for resolve_media_content_type
# ---------------------------------------------------------------------------


class TestResolveMediaContentType:
    def test_data_url_audio(self):
        assert resolve_media_content_type("data_url", image=False) == "audio_url"

    def test_input_data_audio(self):
        assert resolve_media_content_type("input_data", image=False) == "input_audio"

    def test_image_always_image_url_data_url(self):
        assert resolve_media_content_type("data_url", image=True) == "image_url"

    def test_image_always_image_url_input_data(self):
        assert resolve_media_content_type("input_data", image=True) == "image_url"

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid format"):
            resolve_media_content_type("bad_value", image=False)

    def test_invalid_format_still_returns_image_url(self):
        # Even with invalid format, image=True returns image_url
        assert resolve_media_content_type("bad_value", image=True) == "image_url"


# ---------------------------------------------------------------------------
# Tests for PrepareOmniRequestStage: all modality combos x both formats
# ---------------------------------------------------------------------------


class TestAudioOnly:
    """Audio file only, no image, no text."""

    def test_data_url_format(self, wav_file: Path):
        stage = PrepareOmniRequestStage(format="data_url", system_prompt="", user_prompt="")
        batch = _make_batch([{"audio_filepath": str(wav_file)}])
        result = stage.process(batch)
        messages = result.to_pandas().iloc[0]["messages"]
        parts = _get_user_content(messages)
        assert len(parts) == 1
        assert parts[0]["type"] == "audio_url"
        assert "audio_url" in parts[0]
        url = parts[0]["audio_url"]["url"]
        assert url.startswith("data:audio/")
        assert ";base64," in url

    def test_input_data_format(self, wav_file: Path):
        stage = PrepareOmniRequestStage(format="input_data", system_prompt="", user_prompt="")
        batch = _make_batch([{"audio_filepath": str(wav_file)}])
        result = stage.process(batch)
        messages = result.to_pandas().iloc[0]["messages"]
        parts = _get_user_content(messages)
        assert len(parts) == 1
        assert parts[0]["type"] == "input_audio"
        assert "input_audio" in parts[0]
        assert "data" in parts[0]["input_audio"]
        assert "format" in parts[0]["input_audio"]
        assert parts[0]["input_audio"]["format"] == "wav"


class TestImageOnly:
    """Image file only, no audio, no text."""

    def test_data_url_format(self, jpg_file: Path):
        stage = PrepareOmniRequestStage(
            format="data_url",
            system_prompt="",
            user_prompt="",
            image_filepath_key="image_url",
        )
        batch = _make_batch([{"image_url": str(jpg_file)}])
        result = stage.process(batch)
        messages = result.to_pandas().iloc[0]["messages"]
        parts = _get_user_content(messages)
        assert len(parts) == 1
        assert parts[0]["type"] == "image_url"
        url = parts[0]["image_url"]["url"]
        assert url.startswith("data:")
        assert ";base64," in url

    def test_input_data_format_same_as_data_url(self, jpg_file: Path):
        """Images produce identical output regardless of format setting."""
        stage_du = PrepareOmniRequestStage(
            format="data_url",
            system_prompt="",
            user_prompt="",
            image_filepath_key="image_url",
        )
        stage_id = PrepareOmniRequestStage(
            format="input_data",
            system_prompt="",
            user_prompt="",
            image_filepath_key="image_url",
        )
        batch = _make_batch([{"image_url": str(jpg_file)}])
        result_du = stage_du.process(batch)
        result_id = stage_id.process(batch)
        parts_du = _get_user_content(result_du.to_pandas().iloc[0]["messages"])
        parts_id = _get_user_content(result_id.to_pandas().iloc[0]["messages"])
        # Both must be image_url with identical structure
        assert parts_du[0]["type"] == "image_url"
        assert parts_id[0]["type"] == "image_url"
        assert parts_du[0] == parts_id[0]


class TestAudioImageText:
    """Audio + image + text together."""

    def test_data_url_format(self, wav_file: Path, jpg_file: Path):
        stage = PrepareOmniRequestStage(
            format="data_url",
            system_prompt="You are helpful.",
            user_prompt="Describe what you see and hear.",
            image_filepath_key="image_url",
        )
        batch = _make_batch(
            [
                {
                    "audio_filepath": str(wav_file),
                    "image_url": str(jpg_file),
                }
            ]
        )
        result = stage.process(batch)
        messages = result.to_pandas().iloc[0]["messages"]

        # System message present
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."

        parts = _get_user_content(messages)
        types = [p["type"] for p in parts]
        assert "image_url" in types
        assert "audio_url" in types
        assert "text" in types

    def test_input_data_format(self, wav_file: Path, jpg_file: Path):
        stage = PrepareOmniRequestStage(
            format="input_data",
            system_prompt="",
            user_prompt="Describe.",
            image_filepath_key="image_url",
        )
        batch = _make_batch(
            [
                {
                    "audio_filepath": str(wav_file),
                    "image_url": str(jpg_file),
                }
            ]
        )
        result = stage.process(batch)
        messages = result.to_pandas().iloc[0]["messages"]
        parts = _get_user_content(messages)
        types = [p["type"] for p in parts]
        # Audio should be input_audio, image should still be image_url
        assert "input_audio" in types
        assert "image_url" in types
        assert "text" in types


class TestImageText:
    """Image + text, no audio."""

    def test_data_url_format(self, jpg_file: Path):
        stage = PrepareOmniRequestStage(
            format="data_url",
            system_prompt="",
            user_prompt="What is this?",
            image_filepath_key="image_url",
        )
        batch = _make_batch([{"image_url": str(jpg_file)}])
        result = stage.process(batch)
        parts = _get_user_content(result.to_pandas().iloc[0]["messages"])
        types = [p["type"] for p in parts]
        assert types == ["image_url", "text"]


class TestTextOnly:
    """Text only, no audio, no image."""

    def test_data_url_format(self):
        stage = PrepareOmniRequestStage(
            format="data_url",
            system_prompt="",
            user_prompt="Hello world",
        )
        batch = _make_batch([{"some_field": "value"}])
        result = stage.process(batch)
        messages = result.to_pandas().iloc[0]["messages"]
        # No system prompt → messages is [user_msg]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 1
        assert content[0] == {"type": "text", "text": "Hello world"}

    def test_input_data_format(self):
        stage = PrepareOmniRequestStage(
            format="input_data",
            system_prompt="",
            user_prompt="Hello world",
        )
        batch = _make_batch([{"some_field": "value"}])
        result = stage.process(batch)
        messages = result.to_pandas().iloc[0]["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "Hello world"}


# ---------------------------------------------------------------------------
# Tests for HTTP URL handling
# ---------------------------------------------------------------------------


class TestHttpUrlHandling:
    """HTTP URLs should pass through for audio_url/image_url but fail for input_audio."""

    def test_audio_http_url_data_url_passthrough(self):
        stage = PrepareOmniRequestStage(format="data_url", system_prompt="", user_prompt="")
        batch = _make_batch([{"audio_filepath": "https://example.com/audio.wav"}])
        result = stage.process(batch)
        parts = _get_user_content(result.to_pandas().iloc[0]["messages"])
        assert parts[0]["type"] == "audio_url"
        assert parts[0]["audio_url"]["url"] == "https://example.com/audio.wav"

    def test_audio_http_url_input_data_raises(self):
        stage = PrepareOmniRequestStage(format="input_data", system_prompt="", user_prompt="")
        batch = _make_batch([{"audio_filepath": "https://example.com/audio.wav"}])
        with pytest.raises(ValueError, match="Cannot pass HTTP URL"):
            stage.process(batch)

    def test_image_http_url_data_url_passthrough(self):
        stage = PrepareOmniRequestStage(
            format="data_url",
            system_prompt="",
            user_prompt="",
            image_filepath_key="image_url",
        )
        batch = _make_batch([{"image_url": "https://example.com/photo.jpg"}])
        result = stage.process(batch)
        parts = _get_user_content(result.to_pandas().iloc[0]["messages"])
        assert parts[0]["type"] == "image_url"
        assert parts[0]["image_url"]["url"] == "https://example.com/photo.jpg"

    def test_image_http_url_input_data_passthrough(self):
        """Images use image_url in both formats, so HTTP URLs always pass through."""
        stage = PrepareOmniRequestStage(
            format="input_data",
            system_prompt="",
            user_prompt="",
            image_filepath_key="image_url",
        )
        batch = _make_batch([{"image_url": "https://example.com/photo.jpg"}])
        result = stage.process(batch)
        parts = _get_user_content(result.to_pandas().iloc[0]["messages"])
        assert parts[0]["type"] == "image_url"
        assert parts[0]["image_url"]["url"] == "https://example.com/photo.jpg"


# ---------------------------------------------------------------------------
# Tests for missing/NaN fields
# ---------------------------------------------------------------------------


class TestMissingFields:
    """Missing or NaN values should be skipped, not crash."""

    def test_nan_audio_skipped(self, jpg_file: Path):
        stage = PrepareOmniRequestStage(
            format="data_url",
            system_prompt="",
            user_prompt="hi",
            image_filepath_key="image_url",
        )
        batch = _make_batch([{"audio_filepath": float("nan"), "image_url": str(jpg_file)}])
        result = stage.process(batch)
        parts = _get_user_content(result.to_pandas().iloc[0]["messages"])
        types = [p["type"] for p in parts]
        assert "audio_url" not in types
        assert "image_url" in types

    def test_empty_string_audio_skipped(self):
        stage = PrepareOmniRequestStage(
            format="data_url",
            system_prompt="",
            user_prompt="hi",
        )
        batch = _make_batch([{"audio_filepath": ""}])
        result = stage.process(batch)
        parts = _get_user_content(result.to_pandas().iloc[0]["messages"])
        types = [p["type"] for p in parts]
        assert "audio_url" not in types

    def test_none_image_skipped(self, wav_file: Path):
        stage = PrepareOmniRequestStage(
            format="data_url",
            system_prompt="",
            user_prompt="",
            image_filepath_key="image_url",
        )
        batch = _make_batch([{"audio_filepath": str(wav_file), "image_url": None}])
        result = stage.process(batch)
        parts = _get_user_content(result.to_pandas().iloc[0]["messages"])
        types = [p["type"] for p in parts]
        assert "image_url" not in types
        assert "audio_url" in types
