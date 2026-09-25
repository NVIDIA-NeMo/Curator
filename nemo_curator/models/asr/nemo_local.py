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

"""Local ``.nemo`` checkpoint support for the shared NeMo ASR adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nemo_curator.models.asr.nemo_asr import NeMoASRAdapter, _nemo_asr_module


class LocalNeMoASRAdapter(NeMoASRAdapter):
    """Restore a local ``.nemo`` file while reusing #1967's generic ASR stage."""

    @staticmethod
    def _checkpoint_path(model_id: str) -> Path:
        path = Path(model_id)
        if path.suffix != ".nemo":
            msg = f"Local NeMo ASR checkpoint must end in .nemo: {path}"
            raise ValueError(msg)
        if not path.is_file():
            msg = f"Local NeMo ASR checkpoint does not exist: {path}"
            raise FileNotFoundError(msg)
        return path

    @classmethod
    def download_weights_on_node(cls, model_id: str, revision: str | None = None) -> None:
        """Validate the node-local checkpoint without downloading anything."""
        cls._reject_revision(revision)
        cls._checkpoint_path(model_id)

    def _load_checkpoint(self, device: Any) -> Any:  # noqa: ANN401
        path = self._checkpoint_path(self.model_id)
        return _nemo_asr_module().models.ASRModel.restore_from(
            restore_path=str(path),
            map_location=device,
        )
