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

"""TensorRT bundle validation for AI4Bharat IndicConformer encoders."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nemo_curator.stages.audio.inference.tensorrt_encoder import load_engine_metadata as _load_engine_metadata

if TYPE_CHECKING:
    from pathlib import Path


def load_engine_metadata(engine_dir: str | Path) -> dict[str, Any]:
    """Load and validate an IndicConformer TensorRT bundle manifest."""
    return _load_engine_metadata(
        engine_dir,
        model_type="indic_conformer_hybrid",
        required_positive_ints=("sample_rate", "feature_count", "subsampling_factor", "encoder_dim"),
    )
