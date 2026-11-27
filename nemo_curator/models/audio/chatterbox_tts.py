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

"""Chatterbox TTS (English and multilingual) adapter.

The adapter owns the pinned weight download and the model lifecycle,
including the process-global attention settings the multilingual model needs.
``ChatterboxTTSStage`` owns voice assignment, caching and audio output.
``chatterbox`` itself is imported lazily: it only exists in the stage's
isolated ``runtime_env``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from huggingface_hub import snapshot_download
from loguru import logger

_CHATTERBOX_REPO_ID = "ResembleAI/chatterbox"
# Pin the model repository independently of the Python package.  Loading a
# moving ``main`` revision would make cache entries irreproducible even with an
# exact chatterbox-tts version.
_CHATTERBOX_MODEL_REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"  # pragma: allowlist secret
_ENGLISH_MODEL_FILES = (
    "ve.safetensors",
    "t3_cfg.safetensors",
    "s3gen.safetensors",
    "tokenizer.json",
    "conds.pt",
)
_MULTILINGUAL_MODEL_FILES = (
    "ve.pt",
    "t3_mtl23ls_v2.safetensors",
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "conds.pt",
    "Cangjie5_TC.json",
)

# The multilingual model requires eager attention. Loading it mutates
# process-global state (this env var and chatterbox's LLAMA_CONFIGS), which
# unload_model() restores so other transformer stages in the same worker are
# unaffected. ``_UNSET`` marks state that was absent before we touched it.
_ATTN_ENV = "TRANSFORMERS_ATTN_IMPLEMENTATION"
_UNSET = object()


@dataclass
class ChatterboxTTSAdapter:
    """Load ``ChatterboxTTS`` (``language=None``) or ``ChatterboxMultilingualTTS`` on ``device``."""

    language: str | None = None
    device: str = "cuda"
    cache_dir: str | None = None
    model: Any = field(default=None, init=False, repr=False)
    snapshot_dir: str | None = field(default=None, init=False, repr=False)
    _prev_attn_env: str | None = field(default=None, init=False, repr=False)
    _llama_cfg_restore: list[tuple[dict, Any]] = field(default_factory=list, init=False, repr=False)
    _global_state_modified: bool = field(default=False, init=False, repr=False)

    def download_weights_on_node(self) -> None:
        """Download the pinned checkpoint files into the Hugging Face cache."""
        model_files = _MULTILINGUAL_MODEL_FILES if self.language else _ENGLISH_MODEL_FILES
        self.snapshot_dir = snapshot_download(
            repo_id=_CHATTERBOX_REPO_ID,
            repo_type="model",
            revision=_CHATTERBOX_MODEL_REVISION,
            allow_patterns=list(model_files),
            cache_dir=self.cache_dir,
            token=os.getenv("HF_TOKEN"),
        )
        logger.info(f"Pre-downloaded Chatterbox weights from {_CHATTERBOX_REPO_ID}@{_CHATTERBOX_MODEL_REVISION}")

    def load_model(self, *, num_gpus: int) -> None:  # noqa: ARG002 (placement follows ``device``)
        """Load the model from the downloaded snapshot (downloading only if node setup did not)."""
        if self.snapshot_dir is None:
            self.download_weights_on_node()

        if self.language:
            from chatterbox.models.t3 import llama_configs as _llama_cfgs
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            self._prev_attn_env = os.environ.get(_ATTN_ENV, None)
            self._llama_cfg_restore = [
                (cfg, cfg.get("attn_implementation", _UNSET)) for cfg in _llama_cfgs.LLAMA_CONFIGS.values()
            ]
            self._global_state_modified = True

            os.environ[_ATTN_ENV] = "eager"
            for cfg in _llama_cfgs.LLAMA_CONFIGS.values():
                cfg["attn_implementation"] = "eager"

            self.model = ChatterboxMultilingualTTS.from_local(self.snapshot_dir, device=self.device)
            logger.info(f"Loaded ChatterboxMultilingualTTS (language={self.language})")
        else:
            from chatterbox.tts import ChatterboxTTS

            self.model = ChatterboxTTS.from_local(self.snapshot_dir, device=self.device)
            logger.info("Loaded ChatterboxTTS (English)")

    def unload_model(self) -> None:
        """Release the model and undo the multilingual load's process-global changes."""
        self.model = None
        if not self._global_state_modified:
            return

        if self._prev_attn_env is None:
            os.environ.pop(_ATTN_ENV, None)
        else:
            os.environ[_ATTN_ENV] = self._prev_attn_env

        for cfg, prev in self._llama_cfg_restore:
            if prev is _UNSET:
                cfg.pop("attn_implementation", None)
            else:
                cfg["attn_implementation"] = prev

        self._llama_cfg_restore = []
        self._prev_attn_env = None
        self._global_state_modified = False
