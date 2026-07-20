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

# Audio-modality path filter shared by review-curator-audio-pr scripts.
# Fern paths qualify only when their path is explicitly audio-scoped. Generic
# navigation files (for example versions/main.yml) remain visible in a qualifying
# PR but do not make an unrelated Fern PR audio-specific by themselves.
AUDIO_PATH_REGEX='^(nemo_curator/stages/audio/|nemo_curator/tasks/audio_task\.py|tutorials/audio/|tests/stages/audio/|tests/tasks/test_audio|benchmarking/.*([Aa]udio|ALM|alm)|fern/versions/[^/]+/pages/(get-started/audio\.mdx|curate-audio/|about/concepts/audio/|api-reference/tasks/audio-task\.mdx))'
AUDIO_MODALITY_LABEL='audio'
