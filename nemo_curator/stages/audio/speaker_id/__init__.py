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

"""Speaker ID for ASR data.

Curator stages and supporting libraries for speaker embedding extraction,
clustering, and per-utterance speaker confidence scoring.

Heavy submodules (``data``, ``embedding``, ``multigpu``, ``utils``,
``clustering``) are exposed as importable subpackages but are NOT eagerly
imported here -- they pull in optional GPU / model / boto3 / sklearn
dependencies that not every consumer needs.
"""

__version__ = "0.2.0"

from nemo_curator.stages.audio.speaker_id.speaker_clustering_and_scoring import SpeakerClusteringStage
from nemo_curator.stages.audio.speaker_id.speaker_embedding_lhotse import SpeakerEmbeddingLhotseStage
from nemo_curator.stages.audio.speaker_id.speaker_embedding_request import SpeakerEmbeddingRequestStage

__all__ = [
    "SpeakerClusteringStage",
    "SpeakerEmbeddingLhotseStage",
    "SpeakerEmbeddingRequestStage",
    "__version__",
]
