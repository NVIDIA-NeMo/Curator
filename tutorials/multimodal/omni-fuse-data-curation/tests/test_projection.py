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

import numpy as np
import torch

from omnifuse_tutorial.projection import _recall_torch


def test_chunked_recall_matches_full_similarity_matrix() -> None:
    torch.manual_seed(0)
    annotations = torch.randn(10, 4)
    projected = annotations + 0.5 * torch.randn(10, 4)
    full, targets = annotations @ projected.T, torch.arange(10).unsqueeze(1)
    expected = [(s.topk(2, dim=1).indices == targets).any(dim=1).float().mean().item() for s in (full, full.T)]
    recall = _recall_torch(annotations, projected, k=2, chunk=3)
    np.testing.assert_allclose([recall["annotation_to_raw"], recall["raw_to_annotation"]], expected)
