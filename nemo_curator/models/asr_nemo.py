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

"""Model lifecycle adapter for NeMo ASR checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nemo.collections.asr as nemo_asr
import torch

from nemo_curator.models.base import ModelInterface


class NeMoASRAdapter(ModelInterface):
    """Load NeMo ASR from the pretrained registry or a local ``.nemo`` file."""

    def __init__(
        self,
        *,
        model_name: str = "",
        model_path: str | None = None,
        cache_dir: str | None = None,
        map_location: torch.device | str = "cpu",
        model: Any | None = None,  # noqa: ANN401
    ) -> None:
        if not model_name and not model_path and model is None:
            msg = "One of model_name, model_path, or model is required"
            raise ValueError(msg)
        if model_name and model_path:
            msg = "model_name and model_path are mutually exclusive"
            raise ValueError(msg)
        self.model_name = model_name
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.map_location = map_location
        self.model = model

    @property
    def model_id_names(self) -> list[str]:
        identifier = self.model_path or self.model_name
        return [identifier] if identifier else []

    def download_weights_on_node(self) -> None:
        if self.model is not None:
            return
        if self.model_path:
            path = Path(self.model_path)
            if path.suffix != ".nemo":
                msg = f"Local NeMo ASR checkpoint must end in .nemo: {path}"
                raise ValueError(msg)
            if not path.is_file():
                msg = f"Local NeMo ASR checkpoint does not exist: {path}"
                raise FileNotFoundError(msg)
            return

        kwargs: dict[str, Any] = {
            "model_name": self.model_name,
            "return_model_file": True,
        }
        if self.cache_dir is not None:
            kwargs["cache_dir"] = self.cache_dir
        nemo_asr.models.ASRModel.from_pretrained(**kwargs)

    def setup(self) -> None:
        if self.model is not None:
            return
        if self.model_path:
            self.model = nemo_asr.models.ASRModel.restore_from(
                restore_path=self.model_path,
                map_location=self.map_location,
            )
            return

        kwargs: dict[str, Any] = {
            "model_name": self.model_name,
            "map_location": self.map_location,
        }
        if self.cache_dir is not None:
            kwargs["cache_dir"] = self.cache_dir
        self.model = nemo_asr.models.ASRModel.from_pretrained(**kwargs)

    def teardown(self) -> None:
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _normalize_transcriptions(outputs: Any) -> list[str]:  # noqa: ANN401
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if not outputs:
            return []
        if isinstance(outputs[0], list):
            first_items = [inner[0] for inner in outputs]
            return [str(getattr(item, "text", item)) for item in first_items]
        return [str(getattr(output, "text", output)) for output in outputs]

    def transcribe(self, files: list[str]) -> list[str]:
        if self.model is None:
            msg = "NeMoASRAdapter.setup() must run before transcribe()"
            raise RuntimeError(msg)
        return self._normalize_transcriptions(self.model.transcribe(files))
