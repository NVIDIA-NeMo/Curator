# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import annotations

import ast
import os
import urllib.request
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata

_FASTTEXT_MODEL_URLS: dict[str, str] = {
    "lid.176.bin": "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
    "lid.176.ftz": "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz",
}
_DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/nemo_curator/fasttext")


@dataclass
class FastTextLIDStage(ProcessingStage[AudioTask, AudioTask]):
    """Language identification using FastText; flags non-target-language entries.

    When an entry is flagged, ``skip_me`` is set to a descriptive string:

    - ``"Wrong language"`` — detected language differs from ``target_lang``.
    - ``"Low probability of language"`` — correct language but confidence
      below ``min_lang_prob``.
    - ``"Empty text"`` — text is blank after stripping.

    An already non-empty ``skip_me`` value is never overwritten.

    ``model_path`` can be:
    - An absolute path to a local ``.bin`` or ``.ftz`` file.
    - A known model name (``lid.176.bin`` or ``lid.176.ftz``), which is
      downloaded to ``~/.cache/nemo_curator/fasttext/`` on first use.
    """

    model_path: str = ""
    target_lang: str = "en"
    min_lang_prob: float = 0.8
    text_key: str = "pred_text"
    skip_me_key: str = "skip_me"
    name: str = "FastTextLID"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    _lid: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.model_path:
            msg = "model_path is required for FastTextLIDStage"
            raise ValueError(msg)

    def _resolve_model_path(self) -> str:
        if os.path.isfile(self.model_path):
            return self.model_path
        if self.model_path in _FASTTEXT_MODEL_URLS:
            cache_path = os.path.join(_DEFAULT_CACHE_DIR, self.model_path)
            if os.path.isfile(cache_path):
                return cache_path
            os.makedirs(_DEFAULT_CACHE_DIR, exist_ok=True)
            url = _FASTTEXT_MODEL_URLS[self.model_path]
            logger.info(f"FastTextLIDStage: downloading {self.model_path} from {url}")
            urllib.request.urlretrieve(url, cache_path)  # noqa: S310
            return cache_path
        msg = (
            f"model_path '{self.model_path}' is not a valid file path and not a known model name. "
            f"Known names: {list(_FASTTEXT_MODEL_URLS)}"
        )
        raise ValueError(msg)

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        from nemo_curator.stages.text.filters.fasttext.fasttext_filters import FastTextLangId

        resolved = self._resolve_model_path()
        self._lid = FastTextLangId(model_path=resolved, min_langid_score=0.0)
        self._lid.load_model()
        logger.info(f"FastTextLIDStage: loaded model from {resolved}")

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.text_key, self.skip_me_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.skip_me_key]

    def process(self, task: AudioTask) -> AudioTask:
        if self._lid is None:
            logger.warning(
                f"FastTextLIDStage ({self.name}): setup() was not called before process(). "
                "Calling setup() now — check that your executor invokes setup() on each worker."
            )
            self.setup()
        text = task.data[self.text_key]
        if not isinstance(text, str):
            return task
        text = text.strip().replace("\n", " ")
        if not text:
            if not task.data[self.skip_me_key]:
                task.data[self.skip_me_key] = "Empty text"
            return task
        result_str = self._lid.score_document(text)
        score_list = ast.literal_eval(result_str)
        prob = float(score_list[0])
        lang = str(score_list[1]).lower()
        if not task.data[self.skip_me_key]:
            if lang != self.target_lang.lower():
                task.data[self.skip_me_key] = "Wrong language"
            elif prob < self.min_lang_prob:
                task.data[self.skip_me_key] = "Low probability of language"
        return task
