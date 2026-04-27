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

"""Shared fixtures for audio tagging stage tests."""

import pytest

from nemo_curator.tasks import AudioTask


@pytest.fixture
def audio_task():
    """Factory for AudioTask instances. Each task wraps a single dict."""

    def _make(**kwargs) -> AudioTask:
        return AudioTask(data=kwargs)

    return _make
