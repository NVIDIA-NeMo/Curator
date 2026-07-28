# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Stable identifiers for waveform payloads."""

import hashlib
import os
from urllib.parse import urlparse


def audio_item_id_from_path(audio_path: str) -> str:
    parsed = urlparse(str(audio_path))
    basename = os.path.basename(parsed.path if parsed.scheme else str(audio_path))
    stem = os.path.splitext(basename)[0] or "audio"
    path_hash = hashlib.sha256(str(audio_path).encode()).hexdigest()[:8]
    return f"{stem}_{path_hash}"
