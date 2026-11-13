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

"""Private provenance for unchanged PNG payloads from PDF preprocessing."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import PIL
from PIL import features

if TYPE_CHECKING:
    from collections.abc import Iterable

PNG_METADATA_KEY = "_nemotron_parse_png"
_PNG_PROVENANCE_VERSION = "1"


def png_batch_provenance(images: Iterable[bytes | None]) -> dict[str, str]:
    """Bind the encoder identity and ordered page bytes without copying payloads.

    This marker identifies output from the trusted preprocessing stage; it does
    not authenticate externally supplied rows. Framing keeps absent, empty and
    concatenated page payloads distinct.
    """
    digest = hashlib.sha256(b"nemo-curator-png-batch-v1\0")
    count = 0
    for image in images:
        count += 1
        if image is None:
            digest.update(b"\0")
        elif isinstance(image, bytes):
            digest.update(b"\1")
            digest.update(len(image).to_bytes(8, "big"))
            digest.update(image)
        else:
            msg = "PNG batch payloads must be bytes or None"
            raise TypeError(msg)
    digest.update(count.to_bytes(8, "big"))
    return {
        "version": _PNG_PROVENANCE_VERSION,
        "pillow": PIL.__version__,
        "zlib": features.version_codec("zlib") or "",
        "sha256": digest.hexdigest(),
    }


def has_png_batch_provenance(metadata: dict, images: Iterable[bytes | None]) -> bool:
    """Check that the producer's marker still covers this exact ordered batch."""
    marker = metadata.get(PNG_METADATA_KEY)
    if not isinstance(marker, dict) or not marker.get("pillow") or not marker.get("zlib"):
        return False
    try:
        return marker == png_batch_provenance(images)
    except TypeError:
        return False
