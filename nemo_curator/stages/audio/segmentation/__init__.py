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

"""Audio segmentation stages."""

import importlib
import logging

_logger = logging.getLogger(__name__)

__all__: list[str] = []


def _try_import(module_path: str, names: list[str]) -> None:
    try:
        mod = importlib.import_module(module_path)
        for name in names:
            globals()[name] = getattr(mod, name)
            __all__.append(name)
    except Exception as exc:  # noqa: BLE001
        _logger.debug("Skipping %s: %s", module_path, exc)


_try_import("nemo_curator.stages.audio.segmentation.speaker_separation", ["SpeakerSeparationStage"])
_try_import("nemo_curator.stages.audio.segmentation.vad_segmentation", ["VADSegmentationStage"])
_try_import("nemo_curator.stages.audio.segmentation.segment_extractor", ["SegmentExtractorStage"])
