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

"""Validate private PNG provenance across payload and serialization boundaries."""

import io
import json

import cloudpickle
import pytest
from PIL import Image

from nemo_curator.stages.interleaved.pdf.nemotron_parse import _image_transport as transport


@pytest.mark.parametrize("images", [[], [None], [b""], [b"first", None, b"second"]])
def test_provenance_survives_serialization(images: list[bytes | None]):
    metadata = {transport.PNG_METADATA_KEY: transport.png_batch_provenance(iter(images))}
    restored = cloudpickle.loads(cloudpickle.dumps((metadata, images)))
    assert transport.has_png_batch_provenance(*restored)
    assert transport.has_png_batch_provenance(json.loads(json.dumps(metadata)), iter(images))


@pytest.mark.parametrize(
    "changed",
    [
        [b"changed", None, b"second"],
        [b"second", None, b"first"],
        [b"first", b"second"],
        [b"first", None],
        [b"first", None, b"second", None],
        [b"first", b"", b"second"],
    ],
)
def test_mutation_reordering_and_splitting_invalidate_provenance(changed: list[bytes | None]):
    original = [b"first", None, b"second"]
    metadata = {transport.PNG_METADATA_KEY: transport.png_batch_provenance(original)}
    assert not transport.has_png_batch_provenance(metadata, changed)


@pytest.mark.parametrize(("left", "right"), [([None], [b""]), ([b"a", b"bc"], [b"ab", b"c"]), ([], [None])])
def test_payload_framing_is_unambiguous(left: list[bytes | None], right: list[bytes | None]):
    metadata = {transport.PNG_METADATA_KEY: transport.png_batch_provenance(left)}
    assert not transport.has_png_batch_provenance(metadata, right)


def test_recompression_invalidates_provenance_without_changing_pixels():
    with Image.new("RGB", (33, 21), "red") as image:
        uncompressed = io.BytesIO()
        compressed = io.BytesIO()
        image.save(uncompressed, format="PNG", compress_level=0)
        image.save(compressed, format="PNG", compress_level=9)
    assert uncompressed.getvalue() != compressed.getvalue()
    metadata = {transport.PNG_METADATA_KEY: transport.png_batch_provenance([uncompressed.getvalue()])}
    assert not transport.has_png_batch_provenance(metadata, [compressed.getvalue()])


@pytest.mark.parametrize("field", ["version", "pillow", "zlib", "sha256"])
def test_unrecognized_encoder_or_contract_is_ineligible(field: str):
    marker = transport.png_batch_provenance([b"png"])
    marker[field] = "different"
    assert not transport.has_png_batch_provenance({transport.PNG_METADATA_KEY: marker}, [b"png"])


def test_worker_codec_mismatch_is_ineligible(monkeypatch: pytest.MonkeyPatch):
    metadata = {transport.PNG_METADATA_KEY: transport.png_batch_provenance([b"png"])}
    monkeypatch.setattr(transport.features, "version_codec", lambda _codec: "different-worker-codec")
    assert not transport.has_png_batch_provenance(metadata, [b"png"])


@pytest.mark.parametrize("codec", [None, ""])
def test_unknown_codec_identity_falls_back_even_when_both_sides_match(
    monkeypatch: pytest.MonkeyPatch, codec: str | None
):
    monkeypatch.setattr(transport.features, "version_codec", lambda _codec: codec)
    metadata = {transport.PNG_METADATA_KEY: transport.png_batch_provenance([b"png"])}
    assert not transport.has_png_batch_provenance(metadata, [b"png"])


@pytest.mark.parametrize("marker", [None, False, "untrusted", {}, {"version": "1"}])
def test_missing_or_malformed_marker_is_ineligible(marker: object):
    assert not transport.has_png_batch_provenance({transport.PNG_METADATA_KEY: marker}, [b"png"])
    assert not transport.has_png_batch_provenance({}, [b"png"])


def test_provenance_remains_compact_for_larger_batches():
    one = transport.png_batch_provenance([b"png"])
    many = transport.png_batch_provenance(b"png" for _ in range(1000))
    assert len(json.dumps(one)) == len(json.dumps(many))
    assert all(isinstance(value, str) for value in many.values())
    assert one != many


def test_invalid_payload_does_not_become_valid_provenance():
    with pytest.raises(TypeError, match="bytes or None"):
        transport.png_batch_provenance(["not bytes"])
    metadata = {transport.PNG_METADATA_KEY: transport.png_batch_provenance([b"png"])}
    assert not transport.has_png_batch_provenance(metadata, ["not bytes"])
